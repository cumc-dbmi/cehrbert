import tensorflow as tf

from cehrbert.keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from cehrbert.models.layers.custom_layers import (
    ConceptValueTransformationLayer,
    DecoderLayer,
    Encoder,
    PositionalEncodingLayer,
    TimeEmbeddingLayer,
    VisitEmbeddingLayer,
)


def transformer_bert_model_visit_prediction(
    max_seq_length: int,
    concept_vocab_size: int,
    visit_vocab_size: int,
    embedding_size: int,
    depth: int,
    num_heads: int,
    transformer_dropout: float = 0.1,
    embedding_dropout: float = 0.6,
    l2_reg_penalty: float = 1e-4,
    use_time_embedding: bool = False,
    use_behrt: bool = False,
    time_embeddings_size: int = 16,
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
    masked_concept_ids = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="masked_concept_ids")
    visit_segments = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="visit_segments")
    visit_concept_orders = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="visit_concept_orders")
    mask = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="mask")
    masked_visit_concepts = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="masked_visit_concepts")
    mask_visit = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="mask_visit")
    concept_value_masks = tf.keras.Input(shape=[max_seq_length], dtype="int32", name="concept_value_masks")
    concept_values = tf.keras.Input(shape=[max_seq_length], dtype="float32", name="concept_values")

    default_inputs = [
        masked_concept_ids,
        visit_segments,
        visit_concept_orders,
        mask,
        masked_visit_concepts,
        mask_visit,
        concept_value_masks,
        concept_values,
    ]

    l2_regularizer = tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None

    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size,
        embedding_size,
        input_length=max_seq_length,
        name="concept_embeddings",
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=l2_regularizer,
    )

    visit_embedding_layer = ReusableEmbedding(
        visit_vocab_size,
        embedding_size,
        input_length=max_seq_length,
        name="visit_embeddings",
        embeddings_regularizer=l2_regularizer,
    )

    visit_segment_layer = VisitEmbeddingLayer(
        visit_order_size=3, embedding_size=embedding_size, name="visit_segment_layer"
    )

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

    decoder_layer = DecoderLayer(d_model=embedding_size, num_heads=num_heads, dff=512)

    output_layer_1 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name="concept_prediction_logits",
    )

    output_layer_2 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name="visit_prediction_logits",
    )

    concept_softmax_layer = tf.keras.layers.Softmax(name="concept_predictions")

    visit_softmax_layer = tf.keras.layers.Softmax(name="visit_predictions")

    # embeddings for encoder input
    input_for_encoder, concept_embedding_matrix = concept_embedding_layer(masked_concept_ids)

    # Transform the concept embeddings by combining their concept embeddings with the
    # corresponding val
    input_for_encoder = concept_value_transformation_layer(
        concept_embeddings=input_for_encoder,
        concept_values=concept_values,
        concept_value_masks=concept_value_masks,
    )

    # embeddings for decoder input
    input_for_decoder, visit_embedding_matrix = visit_embedding_layer(masked_visit_concepts)

    # Building a Vanilla Transformer (described in
    # "Attention is all you need", 2017)
    input_for_encoder = visit_segment_layer([visit_segments, input_for_encoder])

    if use_behrt:
        ages = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="ages")
        default_inputs.extend([ages])
        age_embedding_layer = TimeEmbeddingLayer(embedding_size=embedding_size)
        input_for_encoder = input_for_encoder + age_embedding_layer(ages)
        positional_encoding_layer = PositionalEncodingLayer(embedding_size=embedding_size)
        input_for_encoder += positional_encoding_layer(visit_concept_orders)

    elif use_time_embedding:
        # additional inputs with time embeddings
        time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="time_stamps")
        ages = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="ages")
        default_inputs.extend([time_stamps, ages])

        # # define the time embedding layer for absolute time stamps (since 1970)
        time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size, name="time_embedding_layer")
        # define the age embedding layer for the age w.r.t the medical record
        age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size, name="age_embedding_layer")
        positional_encoding_layer = PositionalEncodingLayer(
            embedding_size=embedding_size, name="positional_encoding_layer"
        )

        # dense layer for rescale the patient sequence embeddings back to the original size
        scale_back_patient_seq_concat_layer = tf.keras.layers.Dense(
            embedding_size, activation="tanh", name="scale_pat_seq_layer"
        )
        # dense layer for rescale the visit sequence embeddings back to the original size
        scale_back_visit_seq_concat_layer = tf.keras.layers.Dense(
            embedding_size, activation="tanh", name="scale_visit_seq_layer"
        )

        time_embeddings = time_embedding_layer(time_stamps)
        age_embeddings = age_embedding_layer(ages)
        positional_encodings = positional_encoding_layer(visit_concept_orders)

        input_for_encoder = scale_back_patient_seq_concat_layer(
            tf.concat(
                [
                    input_for_encoder,
                    time_embeddings,
                    age_embeddings,
                    positional_encodings,
                ],
                axis=-1,
                name="concat_for_encoder",
            )
        )
        input_for_decoder = scale_back_visit_seq_concat_layer(
            tf.concat(
                [input_for_decoder, time_embeddings, age_embeddings],
                axis=-1,
                name="concat_for_decoder",
            )
        )
    else:
        positional_encoding_layer = PositionalEncodingLayer(embedding_size=embedding_size)
        input_for_encoder += positional_encoding_layer(visit_concept_orders)

    input_for_encoder, att_weights = encoder(
        input_for_encoder,
        mask[:, tf.newaxis, :],
    )

    concept_predictions = concept_softmax_layer(output_layer_1([input_for_encoder, concept_embedding_matrix]))

    decoder_output, _, _ = decoder_layer(
        input_for_decoder,
        input_for_encoder,
        mask[:, tf.newaxis, :],
        mask_visit[:, tf.newaxis, :],
    )
    visit_predictions = visit_softmax_layer(output_layer_2([decoder_output, visit_embedding_matrix]))

    model = tf.keras.Model(inputs=default_inputs, outputs=[concept_predictions, visit_predictions])

    return model
