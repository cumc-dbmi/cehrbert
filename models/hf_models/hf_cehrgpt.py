import warnings
from typing import Tuple, Optional, Union, List, Dict

import torch
import math
from torch import nn
import torch.nn.functional as f
from transformers import PreTrainedModel
from transformers.pytorch_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.activations import gelu_new
from transformers.utils import logging
from torch.nn import CrossEntropyLoss
from torch.distributions import Gamma

from transformers.generation.stopping_criteria import validate_stopping_criteria, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

from models.hf_models.hf_modeling_outputs import (
    CehrGptOutputWithPast, CehrGptCausalLMOutput, CehrGptGenerateDecoderOnlyOutput
)
from models.hf_models.config import CEHRGPTConfig

logger = logging.get_logger(__name__)


class WeibullModel(nn.Module):
    def __init__(self, input_dim):
        super(WeibullModel, self).__init__()
        self.linear1 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            gelu_new,
            nn.Linear(input_dim // 2, 1)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            gelu_new,
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        lambda_param = f.softplus(self.linear1(x))  # Ensure scale is positive
        k_param = f.softplus(self.linear2(x))  # Ensure shape is positive
        # Check for NaN values
        if torch.isnan(lambda_param).any():
            logger.warning(f"NaN values found in scale_param. x: {x}")
        if torch.isnan(k_param).any():
            logger.warning(f"NaN values found in k_param. x: {x}")
        return lambda_param, k_param
    #
    # def sample(self, x, num_samples=1):
    #     lambda_param, k_param = self.forward(x)
    #     dist = Weibull(k_param, lambda_param)
    #     samples = dist.sample((num_samples,))
    #     return samples


class ConceptValuePredictionLayer(nn.Module):
    def __init__(self, embedding_size):
        super(ConceptValuePredictionLayer, self).__init__()
        self.embedding_size = embedding_size
        self.concept_value_decoder_layer = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            gelu_new,
            nn.Linear(embedding_size // 2, 1),
        )

    def forward(
            self,
            hidden_states: Optional[torch.FloatTensor]
    ):
        # (batch_size, context_window, 1)
        concept_vals = self.concept_value_decoder_layer(hidden_states)
        return concept_vals


class ConceptValueTransformationLayer(nn.Module):
    def __init__(self, embedding_size):
        super(ConceptValueTransformationLayer, self).__init__()
        self.embedding_size = embedding_size
        self.merge_value_transformation_layer = nn.Sequential(
            nn.Linear(embedding_size + 1, embedding_size)  # +1 for the concept_values concatenation
        )

    def forward(
            self,
            concept_embeddings: Optional[torch.FloatTensor],
            value_indicators: Optional[torch.BoolTensor] = None,
            concept_values: Optional[torch.FloatTensor] = None
    ):
        if value_indicators is None or concept_values is None:
            return concept_embeddings
        # Expand dimensions for concept_values and concept_value_masks
        concept_values = concept_values.unsqueeze(-1)
        value_indicators = value_indicators.unsqueeze(-1)

        # Concatenate concept_embeddings and concept_values
        concept_embeddings_with_val = torch.cat([concept_embeddings, concept_values], dim=-1)

        # Transform concatenated embeddings back to embedding_size
        transformed_embeddings = self.merge_value_transformation_layer(concept_embeddings_with_val)

        # Apply mask using torch.where
        concept_embeddings = torch.where(value_indicators, transformed_embeddings, concept_embeddings)

        return concept_embeddings


class CehrGptEmbedding(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            sub_time_vocab_size: int,
            embedding_dim: int,
            time_token_mapping: Dict[int, List[int]]
    ):
        super(CehrGptEmbedding, self).__init__()
        assert embedding_dim % 3 == 0, "embedding_size needs to be a multiple of three"
        assert sub_time_vocab_size > 0, "sub_time_vocab_size nees to be greater than one"
        # Add default value
        if -1 not in time_token_mapping:
            time_token_mapping[-1] = [0, 0, 0]
        self.vocab_size = vocab_size
        self.sub_time_vocab_size = sub_time_vocab_size
        self.embedding_dim = embedding_dim
        self.sub_time_embedding_dim = embedding_dim // 3
        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        # self.time_interval_embedding_layer = nn.Embedding(self.sub_time_vocab_size, self.sub_time_embedding_dim)
        # self.linear = nn.Sequential(
        #     nn.Linear(self.embedding_dim, self.embedding_dim),
        #     gelu_new
        # )
        self.register_buffer(
            'time_tokens',
            torch.tensor(list(time_token_mapping.keys()), dtype=torch.int64)
        )
        self.register_buffer(
            'mapped_sub_time_tokens',
            torch.tensor(list(time_token_mapping.values()), dtype=torch.int64)
        )

    def generate_output(self, tokens: torch.Tensor):
        time_token_indicators = torch.isin(tokens, self.time_tokens)
        masked_tokens = tokens.clone()
        masked_tokens[~time_token_indicators] = -1
        # Get the index of the sub_time_tokens from the time_tokens tensor
        sub_time_token_indices = torch.argmax(
            (masked_tokens.unsqueeze(-1) == self.time_tokens.unsqueeze(0).unsqueeze(0)).to(torch.int32),
            dim=-1
        )
        sub_time_tokens = self.mapped_sub_time_tokens[sub_time_token_indices]
        return time_token_indicators, sub_time_tokens

    def forward(self, tokens: torch.LongTensor):
        # Identify all time tokens
        # time_token_indicators = torch.isin(tokens, self.time_tokens)
        # masked_tokens = tokens.clone()
        # masked_tokens[~time_token_indicators] = -1
        # # Get the index of the sub_time_tokens from the time_tokens tensor
        # sub_time_token_indices = torch.argmax(
        #     (masked_tokens.unsqueeze(-1) == self.time_tokens.unsqueeze(0).unsqueeze(0)).to(torch.int32),
        #     dim=-1
        # )
        # sub_time_tokens = self.mapped_sub_time_tokens[sub_time_token_indices]
        # time_interval_embeddings = self.time_interval_embedding_layer(sub_time_tokens).flatten(-2)
        # time_interval_embeddings = self.linear(time_interval_embeddings)
        # embeddings = self.embedding_layer(tokens)
        # merged_embeddings = torch.where(
        #     time_token_indicators.unsqueeze(-1),
        #     time_interval_embeddings,
        #     embeddings
        # )
        # return merged_embeddings
        return self.embedding_layer(tokens)

    @property
    def weight(self):
        # weight = self.embedding_layer.weight
        # valid_time_token_indicators = self.time_tokens >= 0
        # time_interval_embeddings = self.time_interval_embedding_layer(
        #     self.mapped_sub_time_tokens[valid_time_token_indicators]
        # )
        # # Use torch.no_grad() to perform the assignment without tracking gradients
        # with torch.no_grad():
        #     weight[self.time_tokens[valid_time_token_indicators]] = time_interval_embeddings.flatten(-2)
        # return weight
        return self.embedding_layer.weight


class CEHRGPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = CEHRGPTConfig
    base_model_prefix = "cehrgpt"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPT2Block"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

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


class CEHRGPT2Model(CEHRGPTPreTrainedModel):
    def __init__(self, config: CEHRGPTConfig):
        super().__init__(config)

        self.exclude_position_ids = config.exclude_position_ids
        self.include_values = config.include_values
        self.include_ttv_prediction = config.include_ttv_prediction
        self.embed_dim = config.hidden_size

        if config.use_sub_time_tokenization:
            self.wte = CehrGptEmbedding(
                vocab_size=config.vocab_size,
                sub_time_vocab_size=config.time_token_vocab_size,
                embedding_dim=self.embed_dim,
                time_token_mapping=config.token_to_time_token_mapping
            )
        else:
            self.wte = nn.Embedding(config.vocab_size, self.embed_dim)

        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        self.concept_value_transformation_layer = ConceptValueTransformationLayer(self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        # Check validity of device_map
        warnings.warn(
            "`CEHRGPT2Model.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your"
            " model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'h.0': 0, 'h.1': 1,"
            " ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.wte.embedding_layer = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor],
            value_indicators: Optional[torch.BoolTensor],
            values: Optional[torch.FloatTensor],
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CehrGptOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        batch_size = input_ids.shape[0]

        device = input_ids.device

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None and not self.exclude_position_ids:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        input_embeddings = self.wte(input_ids)

        if self.include_values:
            # Combine the value and concept embeddings together
            input_embeddings = self.concept_value_transformation_layer(
                concept_embeddings=input_embeddings,
                value_indicators=value_indicators,
                concept_values=values
            )

        if not self.exclude_position_ids:
            position_embeds = self.wpe(position_ids)
            hidden_states = input_embeddings + position_embeds
        else:
            hidden_states = input_embeddings

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    None,
                    None,
                    use_cache,
                    output_attentions,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions]
                if v is not None
            )

        return CehrGptOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions
        )


class CEHRGPT2LMHeadModel(CEHRGPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: CEHRGPTConfig):
        super().__init__(config)
        self.cehrgpt = CEHRGPT2Model(config)
        if self.config.include_values:
            self.concept_value_decoder_layer = ConceptValuePredictionLayer(config.n_embd)

        if self.config.include_ttv_prediction:
            self.tte_head = WeibullModel(config.n_embd)

        if self.config.use_sub_time_tokenization:
            self.time_token_lm_head = nn.Linear(config.n_embd // 3, config.time_token_vocab_size, bias=False)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def parallelize(self, device_map=None):
        warnings.warn(
            "`GPT2LMHeadModel.parallelize` is deprecated and will be removed in v5 of Transformers, you should load"
            " your model with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'transformer.h.0':"
            " 0, 'transformer.h.1': 1, ...}",
            FutureWarning,
        )
        self.device_map = (
            get_device_map(len(self.cehrgpt.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.cehrgpt.h))
        self.cehrgpt.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.cehrgpt.first_device)
        if self.config.include_values:
            self.concept_value_decoder_layer = self.concept_value_decoder_layer.to(self.cehrgpt.first_device)
        if self.config.include_ttv_prediction:
            self.tte_head = self.tte_head.to(self.cehrgpt.first_device)
        self.model_parallel = True

    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.cehrgpt.deparallelize()
        self.cehrgpt = self.cehrgpt.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        if self.config.include_values:
            self.concept_value_decoder_layer = self.concept_value_decoder_layer.to("cpu")
        if self.config.include_ttv_prediction:
            self.tte_head = self.tte_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):

        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if self.cehrgpt.include_values:
            value_indicators = kwargs.get("value_indicators", torch.zeros_like(input_ids).to(torch.bool))
            values = kwargs.get("values", torch.zeros_like(input_ids, dtype=torch.float32))

            # Omit tokens covered by past_key_values
            if past_key_values:
                past_length = past_key_values[0][0].shape[2]
                # Some generation methods already pass only the last input ID
                if value_indicators.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # Default to old behavior: keep only final ID
                    remove_prefix_length = value_indicators.shape[1] - 1
                value_indicators = value_indicators[:, remove_prefix_length:]
                values = values[:, remove_prefix_length:]

            model_inputs.update(
                {
                    "value_indicators": value_indicators,
                    "values": values
                }
            )

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask
            }
        )

        return model_inputs

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            value_indicators: Optional[torch.BoolTensor] = None,
            values: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            true_value_indicators: Optional[torch.BoolTensor] = None,
            true_values: Optional[torch.FloatTensor] = None,
            time_to_visits: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CehrGptCausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.cehrgpt(
            input_ids,
            value_indicators=value_indicators,
            values=values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.cehrgpt.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        if self.cehrgpt.include_values:
            lm_logits = self.lm_head(hidden_states)
            value_preds = self.concept_value_decoder_layer(hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)
            value_preds = None

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # We add another loss term when use_sub_time_tokenization is enabled, we need to recover the sub time token
            # predictions for year/month/token
            if self.config.use_sub_time_tokenization:
                if not isinstance(self.cehrgpt.wte, CehrGptEmbedding):
                    raise AttributeError(f"self.cehrgpt.wte should be an instance of CehrGptEmbedding")
                # Split the last dimensions into three parts
                time_loss_fct = CrossEntropyLoss(reduction="none")
                time_token_logits = self.time_token_lm_head(torch.unflatten(lm_logits, 2, (3, -1)))
                shifted_time_token_logits = time_token_logits[..., :-1, :, :].contiguous()
                shifted_time_token_indicators, shifted_time_token_labels = self.cehrgpt.wte.generate_output(
                    shift_labels
                )
                time_token_loss = time_loss_fct(
                    shifted_time_token_logits.view(-1, self.config.time_token_vocab_size),
                    shifted_time_token_labels.view(-1)
                )

                time_token_loss = (
                        time_token_loss.view(-1, 3) * shifted_time_token_indicators.view(-1, 1).to(torch.float32)
                )
                loss += torch.mean(time_token_loss)

        if time_to_visits is not None:
            lambda_param, k_param = self.tte_head(lm_logits)
            shifted_lambda_param = lambda_param[..., :-1, :].contiguous()
            shifted_k_param = k_param[..., :-1, :].contiguous()
            shift_time_to_visits = time_to_visits[..., 1:].contiguous()
            shift_time_to_visits = shift_time_to_visits.to(lm_logits.device)
            time_to_visit_indicator = (shift_time_to_visits >= 0).to(torch.float32)
            dist = Gamma(shifted_k_param.squeeze(-1), shifted_lambda_param.squeeze(-1))
            log_probs = dist.log_prob(torch.clamp(shift_time_to_visits, min=0.0) + 1e-6)
            log_probs *= time_to_visit_indicator
            loss += -log_probs.mean()

        if true_values is not None and true_value_indicators is not None:
            true_values = true_values.to(value_preds.device)
            shift_value_preds = value_preds.squeeze(-1)[..., :-1].contiguous()
            shift_value_indicators = true_value_indicators[..., :-1].contiguous()
            shift_next_values = true_values[..., 1:].contiguous()
            num_items = torch.sum(shift_value_indicators.to(torch.float32), dim=-1) + 1e-6
            masked_mse = torch.sum((shift_next_values - shift_value_preds) ** 2 * shift_value_indicators,
                                   dim=-1) / num_items
            loss += torch.mean(masked_mse)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CehrGptCausalLMOutput(
            loss=loss,
            logits=lm_logits,
            next_values=value_preds,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions
        )

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )

    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            logits_warper: Optional[LogitsProcessorList] = None,
            max_length: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[Union[int, List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            output_logits: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            synced_gpus: bool = False,
            streamer: Optional["BaseStreamer"] = None,
            **model_kwargs,
    ) -> Union[CehrGptGenerateDecoderOnlyOutput, torch.LongTensor]:
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_logits = output_logits if output_logits is not None else self.generation_config.output_logits
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape
        if "inputs_embeds" in model_kwargs:
            cur_len = model_kwargs["inputs_embeds"].shape[1]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs["cache_position"] = torch.arange(cur_len, device=input_ids.device)
        lab_token_ids = torch.tensor(
            [] if self.config.lab_token_ids is None else self.config.lab_token_ids, dtype=torch.int32
        )
        value_indicators = torch.zeros_like(input_ids).to(torch.bool)
        values = torch.zeros_like(input_ids, dtype=torch.float32)
        model_kwargs['value_indicators'] = value_indicators
        model_kwargs['values'] = values
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            if self.cehrgpt.include_values:
                next_value_indicators = torch.isin(
                    next_tokens,
                    lab_token_ids.to(next_tokens.device)
                )
                next_values = outputs.next_values[:, -1]

                # update value_indicators
                value_indicators = torch.cat(
                    [value_indicators, next_value_indicators[:, None]], dim=-1
                )

                # update values
                values = torch.cat(
                    [values, next_values], dim=-1
                )

                model_kwargs['value_indicators'] = value_indicators
                model_kwargs['values'] = values

            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        return CehrGptGenerateDecoderOnlyOutput(
            sequences=input_ids,
            sequence_val_masks=value_indicators.to(torch.bool) if self.cehrgpt.include_values else None,
            sequence_vals=values.to(torch.float32) if self.cehrgpt.include_values else None,
            scores=scores,
            logits=raw_logits,
            attentions=decoder_attentions,
            hidden_states=decoder_hidden_states,
            past_key_values=model_kwargs.get("past_key_values"),
        )
