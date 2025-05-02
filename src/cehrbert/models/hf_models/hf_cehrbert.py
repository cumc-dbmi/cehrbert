import math
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import PreTrainedModel
from transformers.activations import gelu_new
from transformers.models.bert import modeling_bert
from transformers.models.bert.modeling_bert import BertEncoder, BertOnlyMLMHead, BertPooler, BertSelfAttention
from transformers.utils import is_flash_attn_2_available, logging

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from cehrbert.models.hf_models.config import CehrBertConfig
from cehrbert.models.hf_models.hf_modeling_outputs import CehrBertModelOutput, CehrBertSequenceClassifierOutput

logger = logging.get_logger("transformers")
LARGE_POSITION_VALUE = 1000000


def create_sample_packing_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Create a block-diagonal attention mask for packed sequences within a batch.

    Args:
        attention_mask (torch.Tensor): (batch_size, seq_len) binary mask where 1 = token, 0 = padding

    Returns:
        torch.Tensor: (batch_size, seq_len, seq_len) attention mask where entries are 1 if tokens
                      can attend to each other (within same packed segment), 0 otherwise.
    """
    # Step 1: Identify segments within each sample
    cumsum_mask = (attention_mask == 0).cumsum(dim=-1)
    segment_ids = cumsum_mask * attention_mask  # zeros remain zero

    # Step 2: Compare segment IDs pairwise per batch element
    # Shape: (batch_size, seq_len, seq_len)
    attn_matrix = (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).int()

    # Step 3: Mask out padding tokens
    mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
    attn_matrix = attn_matrix * mask

    return attn_matrix


def is_sample_pack(attention_mask: torch.Tensor) -> bool:
    """
    Determines whether any sequence in the batch is likely sample-packed.

    A sample-packed sequence is one where there are non-padding (1) tokens
    after a padding (0) token, indicating multiple sequences packed together
    with padding as a separator.

    Args:
        attention_mask (torch.Tensor): A tensor of shape (batch_size, seq_len)
            where 1 indicates a real token and 0 indicates padding.

    Returns:
        bool: True if any sample in the batch is sample-packed, False otherwise.
    """
    nonzero_counts = attention_mask.sum(dim=1)
    max_token_positions = torch.argmax(attention_mask.flip(dims=[1]), dim=1)
    max_indices = attention_mask.shape[1] - 1 - max_token_positions
    return torch.any(nonzero_counts < (max_indices + 1)).item()


def flash_attention_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    dropout=0.0,
    softmax_scale=None,
    is_causal=False,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token.

    first unpad the input, then computes the attention scores and pad the final attention scores.
    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        is_causal (`bool`, *optional*):
    """
    dtype = query_states.dtype
    batch_size, query_length, n_heads, head_dim = query_states.shape
    query_states = query_states.to(torch.bfloat16)
    key_states = key_states.to(torch.bfloat16)
    value_states = value_states.to(torch.bfloat16)
    # Contains at least one padding token in the sequence
    if attention_mask is not None:
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = upad_input(query_states, key_states, value_states, attention_mask, query_length)

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
        # (batch, seq_length, n_heads, head_dim)
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=is_causal,
        )
    return attn_output.reshape(batch_size, query_length, n_heads * head_dim).to(dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2._upad_input
def get_unpad_data(attention_mask):
    # This infers sample packing
    if is_sample_pack(attention_mask):
        # Assume input: attention_mask shape = (batch, seq_len)
        attention_mask = attention_mask.flatten()  # shape: (seq_len,)

        # Compute max_index of the last non-zero element
        nonzero = torch.nonzero(attention_mask, as_tuple=False).flatten()
        max_index = nonzero[-1].item()

        # Pad the truncated attention mask
        padded_attention_mask = F.pad(attention_mask[: max_index + 1], (0, 1), value=0)

        # Indices of all tokens
        indices = torch.nonzero(attention_mask, as_tuple=False).flatten()

        # Find where 0s occur (segment boundaries)
        cumsum_seqlens_in_batch = torch.cumsum(padded_attention_mask, dim=0)[padded_attention_mask == 0]

        # Compute seqlens per segment
        seqlens_in_batch = (cumsum_seqlens_in_batch - F.pad(cumsum_seqlens_in_batch, (1, 0), value=0)[:-1]).to(
            torch.int
        )

        max_seqlen_in_batch = seqlens_in_batch.max().item() if seqlens_in_batch.numel() > 0 else 0
        cu_seqlens = F.pad(cumsum_seqlens_in_batch, (1, 0)).to(torch.int)
    else:
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def upad_input(query_layer, key_layer, value_layer, attention_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = get_unpad_data(attention_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
            indices_k,
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


class BertSelfFlashAttention(BertSelfAttention):

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:

        dtype = hidden_states.dtype
        query = self.query(hidden_states).to(torch.bfloat16)
        key = self.key(hidden_states).to(torch.bfloat16)
        value = self.value(hidden_states).to(torch.bfloat16)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.split_heads(key)
            value_layer = self.split_heads(value)
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.split_heads(key)
            value_layer = self.split_heads(value)
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.split_heads(key)
            value_layer = self.split_heads(value)

        query_layer = self.split_heads(query)
        attn_dropout = self.dropout.p if self.training else 0.0
        # Flash Attention forward pass
        attn_output = flash_attention_forward(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            attn_dropout,
            softmax_scale=None,
            is_causal=False,
        )
        attn_output = attn_output.to(dtype)
        # The BertLayer expects a tuple
        return (attn_output,)


modeling_bert.BERT_SELF_ATTENTION_CLASSES.update({"flash_attention_2": BertSelfFlashAttention})


class PositionalEncodingLayer(nn.Module):
    def __init__(self, embedding_size: int, max_sequence_length: int):
        super(PositionalEncodingLayer, self).__init__()

        self.max_sequence_length = max_sequence_length
        position = torch.arange(max_sequence_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * (-math.log(10000.0) / embedding_size))
        pe = torch.zeros(max_sequence_length, embedding_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, visit_concept_orders: torch.IntTensor) -> torch.Tensor:
        # Normalize the visit_orders using the smallest visit_concept_orders
        masked_visit_concept_orders = torch.where(
            visit_concept_orders > 0,
            visit_concept_orders,
            torch.tensor(LARGE_POSITION_VALUE),
        )
        first_vals = torch.min(masked_visit_concept_orders, dim=1).values.unsqueeze(dim=-1)
        visit_concept_orders = visit_concept_orders - first_vals
        visit_concept_orders = torch.maximum(visit_concept_orders, torch.zeros_like(visit_concept_orders))
        visit_concept_orders = torch.minimum(visit_concept_orders, torch.tensor(self.max_sequence_length) - 1)
        # Get the same positional encodings for the concepts with the same visit_order
        positional_embeddings = self.pe[visit_concept_orders]
        return positional_embeddings


class TimeEmbeddingLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        is_time_delta: bool = False,
        scaling_factor: float = 1.0,
    ):
        super(TimeEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.scaling_factor = scaling_factor
        self.is_time_delta = is_time_delta
        self.w = nn.Parameter(torch.randn(1, self.embedding_size))
        self.phi = nn.Parameter(torch.randn(1, self.embedding_size))

    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        dates = dates.to(torch.float)
        dates = dates / self.scaling_factor
        if self.is_time_delta:
            dates = torch.cat(
                [torch.zeros(dates[..., 0:1].shape), dates[..., 1:] - dates[..., :-1]],
                dim=-1,
            )
        next_input = dates.unsqueeze(-1) * self.w + self.phi
        return torch.sin(next_input)


class ConceptValueTransformationLayer(nn.Module):
    def __init__(self, embedding_size):
        super(ConceptValueTransformationLayer, self).__init__()
        self.merge_value_transformation_layer = nn.Linear(embedding_size + 1, embedding_size)

    def forward(
        self,
        concept_embeddings: torch.Tensor,
        concept_values: Optional[torch.FloatTensor] = None,
        concept_value_masks: Optional[torch.FloatTensor] = None,
    ):
        if concept_values is None or concept_value_masks is None:
            logger.warning("concept_values and concept_value_masks are ignored")
            return concept_embeddings

        # (batch_size, seq_length, 1)
        concept_values = torch.clamp(concept_values.unsqueeze(-1), min=-10, max=10)
        # (batch_size, seq_length, 1)
        concept_value_masks = concept_value_masks.unsqueeze(-1)
        # (batch_size, seq_length, 1 + embedding_size)
        concept_embeddings_with_val = torch.cat([concept_embeddings, concept_values], dim=-1)
        # Run through a dense layer to bring the dimension back to embedding_size
        concept_embeddings_with_val = self.merge_value_transformation_layer(concept_embeddings_with_val)

        merged = torch.where(
            concept_value_masks.to(torch.bool),
            gelu_new(concept_embeddings_with_val),
            concept_embeddings,
        )

        return merged


class ConceptValuePredictionLayer(nn.Module):
    def __init__(self, embedding_size, layer_norm_eps):
        super(ConceptValuePredictionLayer, self).__init__()
        self.embedding_size = embedding_size
        self.concept_value_decoder_layer = nn.Sequential(
            nn.Linear(embedding_size, embedding_size // 2),
            gelu_new,
            nn.LayerNorm(embedding_size // 2, eps=layer_norm_eps),
            nn.Linear(embedding_size // 2, 1),
            gelu_new,
        )

    def forward(self, hidden_states: Optional[torch.FloatTensor]):
        # (batch_size, context_window, 1)
        concept_vals = self.concept_value_decoder_layer(hidden_states)
        return concept_vals


class CehrBertEmbeddings(nn.Module):
    def __init__(self, config: CehrBertConfig):
        super(CehrBertEmbeddings, self).__init__()
        self.concept_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.visit_segment_embeddings = nn.Embedding(config.n_visit_segments, config.hidden_size)
        self.time_embedding_layer = TimeEmbeddingLayer(
            config.n_time_embd, scaling_factor=config.time_embedding_scaling_factor
        )
        self.age_embedding_layer = TimeEmbeddingLayer(
            config.n_time_embd, scaling_factor=config.age_embedding_scaling_factor
        )
        max_position_embeddings = config.max_position_embeddings
        if config.max_position_embeddings < config.sample_packing_max_positions:
            max_position_embeddings = config.sample_packing_max_positions
        self.positional_embedding_layer = PositionalEncodingLayer(config.n_time_embd, max_position_embeddings)
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
        visit_segments: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Get the concept embeddings
        x = self.concept_embeddings(input_ids)
        # Combine values with the concept embeddings
        x = self.concept_value_transformation_layer(x, concept_values, concept_value_masks)
        age_embeddings = self.age_embedding_layer(ages)
        time_embeddings = self.time_embedding_layer(dates)
        positional_embeddings = self.positional_embedding_layer(visit_concept_orders)
        x = self.linear_proj(torch.cat([x, time_embeddings, age_embeddings, positional_embeddings], dim=-1))
        x = gelu_new(x)
        x += self.visit_segment_embeddings(visit_segments)
        return x


class CehrBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading.

    and loading pretrained models.
    """

    config_class = CehrBertConfig
    base_model_prefix = "cehrbert"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["BertLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        """Initialize the weights."""
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
        output_hidden_states: Optional[bool] = None,
    ) -> CehrBertModelOutput:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
        input_shape = input_ids.size()

        # We can provide a self-attention mask of dimensions
        # [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        # The flash attention requires the original attention_mask
        if not getattr(self.config, "_attn_implementation", "eager") == "flash_attention_2":
            if is_sample_pack(attention_mask):
                attention_mask = create_sample_packing_attention_mask(attention_mask)[:, None, :, :]
                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            else:
                attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.cehr_bert_embeddings(
            input_ids=input_ids,
            ages=ages,
            dates=dates,
            visit_concept_orders=visit_concept_orders,
            concept_values=concept_values,
            concept_value_masks=concept_value_masks,
            visit_segments=visit_segments,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        return CehrBertModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            attentions=encoder_outputs.attentions,
            pooler_output=pooled_output,
        )


class CehrBertForPreTraining(CehrBertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config: CehrBertConfig):
        super().__init__(config)

        self.bert = CehrBert(config)
        if self.config.include_value_prediction:
            self.concept_value_decoder_layer = ConceptValuePredictionLayer(config.hidden_size, config.layer_norm_eps)
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
        labels: Optional[torch.LongTensor] = None,
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
            output_hidden_states,
        )

        prediction_scores = self.cls(cehrbert_output.last_hidden_state)

        total_loss = None
        if labels is not None:
            # Skip the MLM predictions for label concepts if include_value_prediction is disabled
            if not self.config.include_value_prediction:
                labels = torch.where(concept_value_masks.to(torch.bool), -100, labels)
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss

            # In addition to MLM, we also predict the values associated with the masked concepts
            if self.config.include_value_prediction:
                mlm_masks = labels != -100
                predicted_values = self.concept_value_decoder_layer(cehrbert_output.last_hidden_state)
                total_loss += torch.mean(
                    (predicted_values.squeeze(-1) - concept_values) ** 2 * concept_value_masks * mlm_masks
                )

        return CehrBertModelOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            last_hidden_state=cehrbert_output.last_hidden_state,
            attentions=cehrbert_output.attentions,
            pooler_output=cehrbert_output.pooler_output,
        )


class CehrBertForClassification(CehrBertPreTrainedModel):

    def __init__(self, config: CehrBertConfig):
        super().__init__(config)

        self.bert = CehrBert(config)
        self.age_batch_norm = torch.nn.BatchNorm1d(1)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense_layer = nn.Linear(config.hidden_size + 1, config.hidden_size // 2)
        self.dense_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size // 2, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def _apply_age_norm(
        self,
        age_at_index: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Applies batch normalization to the input age tensor.

        If the batch contains more than one example,
        standard batch normalization is applied. If the batch size is 1, batch normalization is applied
        without updating the running statistics, ensuring that the normalization uses the stored running
        mean and variance without modification.

        Args:
            age_at_index (torch.FloatTensor): A tensor containing the age values to normalize.
            The tensor has shape `(batch_size, num_features)` where `batch_size` is the number of samples in the batch.

        Returns:
            torch.FloatTensor: A tensor with the normalized age values.
        """
        if age_at_index.shape[0] > 1:
            normalized_age = self.age_batch_norm(age_at_index)
        else:
            self.age_batch_norm.eval()
            # Apply batch norm without updating running stats
            with torch.no_grad():  # Prevent tracking gradients, since we don't want to update anything
                normalized_age = self.age_batch_norm(age_at_index)
            # Optionally, set the layer back to training mode if needed later
            self.age_batch_norm.train()
        return normalized_age

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        age_at_index: torch.FloatTensor,
        ages: Optional[torch.LongTensor] = None,
        dates: Optional[torch.LongTensor] = None,
        visit_concept_orders: Optional[torch.LongTensor] = None,
        concept_values: Optional[torch.FloatTensor] = None,
        concept_value_masks: Optional[torch.FloatTensor] = None,
        visit_segments: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        classifier_label: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CehrBertSequenceClassifierOutput:

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
            output_hidden_states,
        )
        if is_sample_pack(attention_mask):
            features = cehrbert_output.last_hidden_state[:, (input_ids == self.config.cls_token_id).squeeze(0), :]
            assert features.shape[1] == classifier_label.shape[1], (
                "the length of the features need to be the same as the length of classifier_label. "
                f"features.shape[1]: {features.shape[1]}, "
                f"classifier_label.shape[1]: {classifier_label.shape[1]}"
            )
            assert features.shape[1] == age_at_index.shape[1], (
                "the length of the features need to be the same as the length of age_at_index. "
                f"features.shape[1]: {features.shape[1]}, "
                f"age_at_index.shape[1]: {age_at_index.shape[1]}"
            )
            num_samples = age_at_index.shape[1]
            features = features.view((num_samples, -1))
            classifier_label = classifier_label.view((num_samples, -1))
            with torch.autocast(device_type="cuda", enabled=False):
                normalized_age = self._apply_age_norm(age_at_index.view((num_samples, 1)))
        else:
            features = cehrbert_output.pooler_output
            # Disable autocasting for precision-sensitive operations
            with torch.autocast(device_type="cuda", enabled=False):
                normalized_age = self._apply_age_norm(age_at_index)

        # In case the model is in bfloat16
        if features.dtype != normalized_age.dtype:
            normalized_age = normalized_age.to(cehrbert_output.last_hidden_state.dtype)

        next_input = self.dropout(features)
        next_input = torch.cat([next_input, normalized_age], dim=1)
        next_input = self.dense_layer(next_input)
        next_input = nn.functional.relu(next_input)
        next_input = self.dense_dropout(next_input)
        logits = self.classifier(next_input)

        loss = None
        if classifier_label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, classifier_label)

        return CehrBertSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=cehrbert_output.last_hidden_state,
            attentions=cehrbert_output.attentions,
        )


class CehrBertLstmForClassification(CehrBertPreTrainedModel):

    def __init__(self, config: CehrBertConfig):
        super().__init__(config)

        self.bert = CehrBert(config)
        self.age_batch_norm = torch.nn.BatchNorm1d(1)

        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dense_layer = nn.Linear(config.hidden_size * (1 + config.bidirectional) + 1, config.hidden_size // 2)
        self.dense_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size // 2, 1)

        # Initialize weights and apply final processing
        self.post_init()

    def _apply_age_norm(
        self,
        age_at_index: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Applies batch normalization to the input age tensor.

        If the batch contains more than one example,
        standard batch normalization is applied. If the batch size is 1, batch normalization is applied
        without updating the running statistics, ensuring that the normalization uses the stored running
        mean and variance without modification.

        Args:
            age_at_index (torch.FloatTensor): A tensor containing the age values to normalize.
            The tensor has shape `(batch_size, num_features)` where `batch_size` is the number of samples in the batch.

        Returns:
            torch.FloatTensor: A tensor with the normalized age values.
        """
        if age_at_index.shape[0] > 1:
            normalized_age = self.age_batch_norm(age_at_index)
        else:
            self.age_batch_norm.eval()
            # Apply batch norm without updating running stats
            with torch.no_grad():  # Prevent tracking gradients, since we don't want to update anything
                normalized_age = self.age_batch_norm(age_at_index)
            # Optionally, set the layer back to training mode if needed later
            self.age_batch_norm.train()
        return normalized_age

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        age_at_index: torch.FloatTensor,
        ages: Optional[torch.LongTensor] = None,
        dates: Optional[torch.LongTensor] = None,
        visit_concept_orders: Optional[torch.LongTensor] = None,
        concept_values: Optional[torch.FloatTensor] = None,
        concept_value_masks: Optional[torch.FloatTensor] = None,
        visit_segments: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        classifier_label: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CehrBertSequenceClassifierOutput:

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
            output_hidden_states,
        )

        # Disable autocasting for precision-sensitive operations
        with torch.autocast(device_type="cuda", enabled=False):
            normalized_age = self._apply_age_norm(age_at_index)

        # In case the model is in bfloat16
        if cehrbert_output.last_hidden_state.dtype != normalized_age.dtype:
            normalized_age = normalized_age.to(cehrbert_output.last_hidden_state.dtype)

        lengths = torch.sum(attention_mask, dim=-1)
        packed_input = pack_padded_sequence(
            cehrbert_output.last_hidden_state,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, c_n) = self.lstm(packed_input)
        next_input = self.dropout(h_n.transpose(1, 0).reshape([h_n.shape[1], -1]))
        next_input = torch.cat([next_input, normalized_age], dim=1)
        next_input = self.dense_layer(next_input)
        next_input = nn.functional.relu(next_input)
        next_input = self.dense_dropout(next_input)
        logits = self.classifier(next_input)

        loss = None
        if classifier_label is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, classifier_label)

        return CehrBertSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=cehrbert_output.last_hidden_state,
            attentions=cehrbert_output.attentions,
        )
