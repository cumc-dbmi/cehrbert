# +
import tensorflow as tf

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding

from models.custom_layers import (VisitEmbeddingLayer, TimeSelfAttention, Encoder, DecoderLayer,
                                  TemporalEncoder, PositionalEncodingLayer,
                                  TimeEmbeddingLayer)
from utils.model_utils import create_concept_mask


def transformer_bert_model_visit_prediction(max_seq_length: int,
                                            concept_vocab_size: int,
                                            visit_vocab_size: int,
                                            embedding_size: int,
                                            depth: int,
                                            num_heads: int,
                                            transformer_dropout: float = 0.1,
                                            embedding_dropout: float = 0.6,
                                            l2_reg_penalty: float = 1e-4,
                                            use_time_embedding: bool = False):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations
    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    masked_concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                               name='masked_concept_ids')

    visit_segments = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                           name='visit_segments')

    visit_concept_orders = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                                 name='visit_concept_orders')

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    concept_mask = create_concept_mask(mask, max_seq_length)

    masked_visit_concepts = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                                  name='masked_visit_concepts')

    mask_visit = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                       name='mask_visit')

    default_inputs = [masked_concept_ids, visit_segments, visit_concept_orders, mask,
                      masked_visit_concepts, mask_visit]

    mask_visit_expanded = create_concept_mask(mask_visit, max_seq_length)

    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)

    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size, embedding_size,
        input_length=max_seq_length,
        name='concept_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)

    visit_embedding_layer = ReusableEmbedding(
        visit_vocab_size, embedding_size,
        input_length=max_seq_length,
        name='visit_embeddings',
        embeddings_regularizer=l2_regularizer)

    visit_segment_layer = VisitEmbeddingLayer(visit_order_size=3,
                                              embedding_size=embedding_size)

    positional_encoding_layer = PositionalEncodingLayer(max_sequence_length=max_seq_length,
                                                        embedding_size=embedding_size)

    encoder = Encoder(name='encoder',
                      num_layers=depth,
                      d_model=embedding_size,
                      num_heads=num_heads,
                      dropout_rate=transformer_dropout)

    decoder_layer = DecoderLayer(d_model=embedding_size, num_heads=num_heads, dff=512)

    output_layer_1 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    output_layer_2 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='visit_prediction_logits')

    concept_softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')

    visit_softmax_layer = tf.keras.layers.Softmax(name='visit_predictions')

    next_step_input, concept_embedding_matrix = concept_embedding_layer(masked_concept_ids)

    next_step_input = positional_encoding_layer(next_step_input, visit_concept_orders)
    # Building a Vanilla Transformer (described in
    # "Attention is all you need", 2017)
    next_step_input = visit_segment_layer([visit_segments, next_step_input])

    if use_time_embedding:
        time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                            name='time_stamps')
        default_inputs.append(time_stamps)
        time_embedding_layer = TimeEmbeddingLayer(embedding_size=embedding_size)
        next_step_input = time_embedding_layer([next_step_input, time_stamps])

    next_step_input, _ = encoder(next_step_input, concept_mask)

    concept_predictions = concept_softmax_layer(
        output_layer_1([next_step_input, concept_embedding_matrix]))

    input_for_decoder, visit_embedding_matrix = visit_embedding_layer(masked_visit_concepts)

    decoder_output, _, _ = decoder_layer(input_for_decoder, next_step_input, concept_mask,
                                         mask_visit_expanded)
    visit_predictions = visit_softmax_layer(
        output_layer_2([decoder_output, visit_embedding_matrix]))

    model = tf.keras.Model(
        inputs=default_inputs,
        outputs=[concept_predictions, visit_predictions])

    return model


def transformer_temporal_bert_model_visit_prediction(
        max_seq_length: int,
        time_window_size: int,
        concept_vocab_size: int,
        visit_vocab_size: int,
        embedding_size: int,
        depth: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4,
        time_attention_trainable=True):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations
    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    masked_concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                               name='masked_concept_ids')

    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='concept_ids')

    time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='time_stamps')

    visit_segments = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                           name='visit_segments')

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    concept_mask = create_concept_mask(mask, max_seq_length)

    masked_visit_concepts = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                                  name='masked_visit_concepts')

    mask_visit = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                       name='mask_visit')

    mask_visit_expanded = create_concept_mask(mask_visit, max_seq_length)

    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)

    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size, embedding_size,
        input_length=max_seq_length,
        name='bpe_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer)

    visit_embedding_layer = ReusableEmbedding(
        visit_vocab_size, embedding_size,
        input_length=max_seq_length,
        name='visit_embeddings',
        embeddings_regularizer=l2_regularizer)

    visit_segment_layer = VisitEmbeddingLayer(visit_order_size=3,
                                              embedding_size=embedding_size)

    time_attention_layer = TimeSelfAttention(vocab_size=concept_vocab_size,
                                             target_seq_len=max_seq_length,
                                             context_seq_len=max_seq_length,
                                             time_window_size=time_window_size,
                                             return_logits=True,
                                             self_attention_return_logits=True,
                                             trainable=time_attention_trainable)

    temporal_encoder = TemporalEncoder(name='temporal_encoder',
                                       num_layers=depth,
                                       d_model=embedding_size,
                                       num_heads=num_heads,
                                       dropout_rate=transformer_dropout)

    decoder_layer = DecoderLayer(d_model=embedding_size, num_heads=num_heads, dff=512)

    output_layer_1 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    output_layer_2 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='visit_prediction_logits')

    concept_softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')

    visit_softmax_layer = tf.keras.layers.Softmax(name='visit_predictions')

    next_step_input, embedding_matrix = concept_embedding_layer(masked_concept_ids)

    # Building a Vanilla Transformer (described in
    # "Attention is all you need", 2017)
    next_step_input = visit_segment_layer([visit_segments, next_step_input])
    # shape = (batch_size, seq_len, seq_len)
    time_attention = time_attention_layer([concept_ids, time_stamps, mask])
    # pad a dimension to accommodate the head split
    time_attention = tf.expand_dims(time_attention, axis=1)

    next_step_input, _, _ = temporal_encoder(next_step_input, concept_mask, time_attention)

    concept_predictions = concept_softmax_layer(
        output_layer_1([next_step_input, embedding_matrix]))

    input_for_decoder, visit_embedding_matrix = visit_embedding_layer(masked_visit_concepts)

    decoder_output, _, _ = decoder_layer(input_for_decoder, next_step_input, concept_mask,
                                         mask_visit_expanded)
    visit_predictions = visit_softmax_layer(
        output_layer_2([decoder_output, visit_embedding_matrix]))

    model = tf.keras.Model(
        inputs=[masked_concept_ids, concept_ids, time_stamps, visit_segments, mask,
                masked_visit_concepts, mask_visit],
        outputs=[concept_predictions, visit_predictions])

    return model
