import math
from typing import Optional
import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertEncoder
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from models.hf_models.config import CehrBertConfig
from transformers.utils import logging

logger = logging.get_logger("transformers")


class PositionalEncodingLayer(nn.Module):
    def __init__(
            self,
            embedding_size: int,
            max_sequence_length: int
    ):
        super(PositionalEncodingLayer, self).__init__()

        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_sequence_length, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(
            self,
            visit_concept_orders: torch.IntTensor
    ) -> torch.Tensor:
        # Normalize the visit_orders using the smallest visit_concept_orders
        # Take the absolute value to make sure the padded values are not negative after
        # normalization
        min_vals = torch.min(visit_concept_orders, dim=-1)[0]
        visit_concept_orders = torch.abs(
            visit_concept_orders - min_vals.unsqueeze(-1)
        )
        # Get the same positional encodings for the concepts with the same visit_order
        positional_embeddings = self.pe[visit_concept_orders]
        return positional_embeddings


class TimeEmbeddingLayer(nn.Module):
    def __init__(
            self,
            n_time_embd,
            is_time_delta=False
    ):
        super(TimeEmbeddingLayer, self).__init__()
        self.n_time_embd = n_time_embd
        self.is_time_delta = is_time_delta
        self.w = nn.Parameter(torch.randn(1, self.n_time_embd))
        self.phi = nn.Parameter(torch.randn(1, self.n_time_embd))

    def forward(
            self,
            time_stamps: torch.Tensor
    ) -> torch.Tensor:
        time_stamps = time_stamps.to(torch.float)
        if self.is_time_delta:
            time_stamps = torch.cat(
                [torch.zeros(time_stamps[..., 0:1].shape),
                 time_stamps[..., 1:] - time_stamps[..., :-1]],
                dim=-1
            )
        next_input = time_stamps.unsqueeze(-1) * self.w + self.phi
        return torch.sin(next_input)


class ConceptValueTransformationLayer(nn.Module):
    def __init__(self, config):
        super(ConceptValueTransformationLayer, self).__init__()
        self.merge_value_transformation_layer = nn.Linear(
            config.n_embd + 1,
            config.n_embd
        )

    def forward(
            self,
            concept_embeddings: torch.Tensor,
            concept_values: Optional[torch.FloatTensor] = None,
            concept_value_masks: Optional[torch.FloatTensor] = None,
    ):
        if concept_values is None or concept_value_masks is None:
            logger.warning('concept_values and concept_value_masks are ignored')
            return concept_embeddings

        # (batch_size, seq_length, 1)
        concept_values = concept_values.unsqueeze(-1)
        # (batch_size, seq_length, 1)
        concept_value_masks = concept_value_masks.unsqueeze(-1)
        # (batch_size, seq_length, 1 + embedding_size)
        concept_embeddings_with_val = torch.cat(
            [concept_embeddings, concept_values], dim=-1
        )
        # Run through a dense layer to bring the dimension back to embedding_size
        concept_embeddings_with_val = self.merge_value_transformation_layer(
            concept_embeddings_with_val
        )

        merged = torch.where(
            concept_value_masks.to(torch.bool),
            concept_embeddings_with_val,
            concept_embeddings
        )

        return merged


class CehrBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CehrBertConfig
    base_model_prefix = "cehr_bert"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["BertLayer"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.concept_embedding_layer = nn.Embedding(config.vocab_size, self.embed_dim)
        self.visit_segment_layer = nn.Embedding(config.n_visit_segments, self.embed_dim)
        self.time_embedding_layer = TimeEmbeddingLayer(config.n_time_embd)
        self.age_embedding_layer = TimeEmbeddingLayer(config.n_time_embd)
        self.concept_value_transformation_layer = ConceptValueTransformationLayer(config)
        self.encoder = BertEncoder(config)
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer)))
