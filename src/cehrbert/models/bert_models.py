import tensorflow as tf

from cehrbert.keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from cehrbert.models.layers.custom_layers import (
    ConceptValueTransformationLayer,
    Encoder,
    PositionalEncodingLayer,
    TimeEmbeddingLayer,
    VisitEmbeddingLayer,
)
from cehrbert.utils.model_utils import create_concept_mask


def transformer_bert_model(
    max_seq_length: int,
    vocab_size: int,
    embedding_size: int,
    depth: int,
    num_heads: int,
    transformer_dropout: float = 0.1,
    embedding_dropout: float = 0.6,
    l2_reg_penalty: float = 1e-4,
    use_time_embedding: bool = False,
    time_embeddings_size: int = 16,
    use_behrt: bool = False,
    include_prolonged_length_stay: bool = False,
):
    """
    Builds a BERT-based model (Bidirectional Encoder Representations.

    from Transformers) following paper "BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding"
    (https://arxiv.org/abs/1810.04805)

    Depending on the value passed with `use_universal_transformer` argument,
    this function applies either an Adaptive Universal Transformer (2018)
    or a vanilla Transformer (2017) to do the job (the original paper uses
    vanilla Transformer).
    """
    masked_concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="masked_concept_ids")

    visit_segments = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="visit_segments")

    visit_concept_orders = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="visit_concept_orders")

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="mask")

    concept_value_masks = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="concept_value_masks")
    concept_values = tf.keras.layers.Input(shape=(max_seq_length,), dtype="float32", name="concept_values")

    concept_mask = create_concept_mask(mask, max_seq_length)

    default_inputs = [
        masked_concept_ids,
        visit_segments,
        visit_concept_orders,
        mask,
        concept_value_masks,
        concept_values,
    ]

    l2_regularizer = tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None

    embedding_layer = ReusableEmbedding(
        vocab_size,
        embedding_size,
        input_length=max_seq_length,
        name="concept_embeddings",
        embeddings_regularizer=l2_regularizer,
    )

    visit_segment_layer = VisitEmbeddingLayer(visit_order_size=3, embedding_size=embedding_size, name="visit_segment")

    concept_value_transformation_layer = ConceptValueTransformationLayer(
        embedding_size=embedding_size, name="concept_value_transformation_layer"
    )

    encoder = Encoder(
        name="encoder",
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout,
    )

    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        add_biases=use_time_embedding,
        name="concept_prediction_logits",
    )

    softmax_layer = tf.keras.layers.Softmax(name="concept_predictions")

    next_step_input, embedding_matrix = embedding_layer(masked_concept_ids)

    # Transform the concept embeddings by combining their concept embeddings with the
    # corresponding val
    next_step_input = concept_value_transformation_layer(
        concept_embeddings=next_step_input,
        concept_values=concept_values,
        concept_value_masks=concept_value_masks,
    )

    if use_behrt:
        ages = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="ages")
        default_inputs.extend([ages])
        age_embedding_layer = TimeEmbeddingLayer(embedding_size=embedding_size)
        next_step_input = next_step_input + age_embedding_layer(ages)
        positional_encoding_layer = PositionalEncodingLayer(embedding_size=embedding_size)
        next_step_input += positional_encoding_layer(visit_concept_orders)

    elif use_time_embedding:
        time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="time_stamps")
        ages = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="ages")
        default_inputs.extend([time_stamps, ages])
        # # define the time embedding layer for absolute time stamps (since 1970)
        time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size, name="time_embedding_layer")
        # define the age embedding layer for the age w.r.t the medical record
        age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size, name="age_embedding_layer")
        positional_encoding_layer = PositionalEncodingLayer(
            embedding_size=time_embeddings_size, name="positional_encoding_layer"
        )

        scale_back_concat_layer = tf.keras.layers.Dense(embedding_size, activation="tanh", name="scale_pat_seq_layer")
        time_embeddings = time_embedding_layer(time_stamps)
        age_embeddings = age_embedding_layer(ages)
        positional_encodings = positional_encoding_layer(visit_concept_orders)
        next_step_input = scale_back_concat_layer(
            tf.concat(
                [
                    next_step_input,
                    time_embeddings,
                    age_embeddings,
                    positional_encodings,
                ],
                axis=-1,
            )
        )
    else:
        positional_encoding_layer = PositionalEncodingLayer(embedding_size=embedding_size)
        next_step_input += positional_encoding_layer(visit_concept_orders)

    # Building a Vanilla Transformer (described in
    # "Attention is all you need", 2017)
    next_step_input = visit_segment_layer([visit_segments, next_step_input])

    next_step_input, _ = encoder(next_step_input, concept_mask)

    concept_predictions = softmax_layer(output_layer([next_step_input, embedding_matrix]))

    outputs = [concept_predictions]

    if include_prolonged_length_stay:
        mask_embeddings = tf.tile(
            tf.expand_dims(mask == 0, -1, name="bert_expand_prolonged"),
            [1, 1, embedding_size],
            name="bert_tile_prolonged",
        )
        mask_embeddings = tf.cast(mask_embeddings, dtype=tf.float32, name="bert_cast_prolonged")
        contextualized_embeddings = tf.math.multiply(next_step_input, mask_embeddings, name="bert_multiply_prolonged")
        summed_contextualized_embeddings = tf.reduce_sum(contextualized_embeddings, axis=-1)
        prolonged_length_stay_prediction = tf.keras.layers.Dense(1, name="prolonged_length_stay", activation="sigmoid")

        outputs.append(prolonged_length_stay_prediction(summed_contextualized_embeddings))

    model = tf.keras.Model(inputs=default_inputs, outputs=outputs)

    return model
