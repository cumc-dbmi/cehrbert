import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as f
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead
from transformers import PreTrainedModel
from models.hf_models.config import CehrBertConfig
from models.hf_models.hf_modeling_outputs import CehrBertModelOutput
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
            embedding_size,
            is_time_delta=False
    ):
        super(TimeEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta
        self.w = nn.Parameter(torch.randn(1, self.embedding_size))
        self.phi = nn.Parameter(torch.randn(1, self.embedding_size))

    def forward(
            self,
            dates: torch.Tensor
    ) -> torch.Tensor:
        dates = dates.to(torch.float)
        if self.is_time_delta:
            dates = torch.cat(
                [torch.zeros(dates[..., 0:1].shape),
                 dates[..., 1:] - dates[..., :-1]],
                dim=-1
            )
        next_input = dates.unsqueeze(-1) * self.w + self.phi
        return torch.sin(next_input)


class ConceptValueTransformationLayer(nn.Module):
    def __init__(self, embedding_size):
        super(ConceptValueTransformationLayer, self).__init__()
        self.merge_value_transformation_layer = nn.Linear(
            embedding_size + 1,
            embedding_size
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


class CehrBertEmbeddings(nn.Module):
    def __init__(self, config: CehrBertConfig):
        super(CehrBertEmbeddings, self).__init__()
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.visit_segment_embeddings = nn.Embedding(config.n_visit_segments, config.hidden_size)
        self.time_embedding_layer = TimeEmbeddingLayer(config.n_time_embd)
        self.age_embedding_layer = TimeEmbeddingLayer(config.n_time_embd)
        self.positional_embedding_layer = PositionalEncodingLayer(config.n_time_embd, config.max_position_embeddings)
        self.concept_value_transformation_layer = ConceptValueTransformationLayer(config.hidden_size)
        self.linear_proj = nn.Linear(config.hidden_size + 3 * config.n_time_embd, config.hidden_size)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            ages: Optional[torch.LongTensor] = None,
            dates: Optional[torch.LongTensor] = None,
            visit_concept_orders: Optional[torch.LongTensor] = None,
            concept_values: Optional[torch.FloatTensor] = None,
            concept_value_masks: Optional[torch.FloatTensor] = None,
            visit_segments: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        # Get the concept embeddings
        x = self.concept_embeddings(input_ids)
        # Combine values with the concept embeddings
        x = self.concept_value_transformation_layer(
            x,
            concept_values,
            concept_value_masks
        )
        age_embeddings = self.age_embedding_layer(ages)
        time_embeddings = self.age_embedding_layer(dates)
        positional_embeddings = self.positional_embedding_layer(visit_concept_orders)
        x = self.linear_proj(
            torch.cat([x, time_embeddings, age_embeddings, positional_embeddings], dim=-1)
        )
        x = f.gelu(x)
        x += self.visit_segment_embeddings(visit_segments)
        return x


class CehrBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CehrBertConfig
    base_model_prefix = "cehr_bert"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["BertLayer"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
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


class CehrBert(CehrBertPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    def __init__(self, config: CehrBertConfig):
        super().__init__(config)

        self.cehr_bert_embeddings = CehrBertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            ages: Optional[torch.LongTensor] = None,
            dates: Optional[torch.LongTensor] = None,
            visit_concept_orders: Optional[torch.LongTensor] = None,
            concept_values: Optional[torch.FloatTensor] = None,
            concept_value_masks: Optional[torch.FloatTensor] = None,
            visit_segments: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None
    ) -> CehrBertModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.cehr_bert_embeddings(
            input_ids=input_ids,
            ages=ages,
            dates=dates,
            visit_concept_orders=visit_concept_orders,
            concept_values=concept_values,
            concept_value_masks=concept_value_masks,
            visit_segments=visit_segments
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return CehrBertModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            attentions=encoder_outputs.attentions,
            pooler_output=pooled_output
        )


class CehrBertForPreTraining(CehrBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: CehrBertConfig):
        super().__init__(config)

        self.bert = CehrBert(config)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.bert.cehr_bert_embeddings.concept_embeddings

    def set_input_embeddings(self, value):
        self.bert.cehr_bert_embeddings.concept_embeddings = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: torch.Tensor,
            ages: Optional[torch.LongTensor] = None,
            dates: Optional[torch.LongTensor] = None,
            visit_concept_orders: Optional[torch.LongTensor] = None,
            concept_values: Optional[torch.FloatTensor] = None,
            concept_value_masks: Optional[torch.FloatTensor] = None,
            visit_segments: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            labels: Optional[torch.LongTensor] = None
    ) -> CehrBertModelOutput:
        cehrbert_output = self.bert(
            input_ids,
            attention_mask,
            ages,
            dates,
            visit_concept_orders,
            concept_values,
            concept_value_masks,
            visit_segments,
            output_attentions,
            output_hidden_states
        )

        prediction_scores = self.cls(cehrbert_output.last_hidden_state)

        total_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss

        return CehrBertModelOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            last_hidden_state=cehrbert_output.last_hidden_state,
            attentions=cehrbert_output.attentions,
            pooler_output=cehrbert_output.pooler_output
        )