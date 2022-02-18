import tensorflow as tf

from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding

from models.custom_layers import (VisitEmbeddingLayer, Encoder, PositionalEncodingLayer,
                                  TimeEmbeddingLayer, MultiHeadAttention)


def transformer_bert_model_visit_prediction(max_seq_length: int,
                                            concept_vocab_size: int,
                                            embedding_size: int,
                                            depth: int,
                                            num_heads: int,
                                            transformer_dropout: float = 0.1,
                                            embedding_dropout: float = 0.6,
                                            l2_reg_penalty: float = 1e-4,
                                            time_embeddings_size: int = 16,
                                            num_hidden_state: int = 20):
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
    masked_concept_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype='int32',
        name='masked_concept_ids'
    )

    visit_segments = tf.keras.layers.Input(
        shape=(max_seq_length,),
        dtype='int32',
        name='visit_segments'
    )

    mask = tf.keras.layers.Input(
        shape=(max_seq_length,),
        dtype='int32',
        name='mask'
    )

    # additional inputs with time embeddings
    visit_concept_orders = tf.keras.layers.Input(
        shape=(max_seq_length,),
        dtype='int32',
        name='visit_concept_orders'
    )
    time_stamps = tf.keras.layers.Input(
        shape=(max_seq_length,),
        dtype='int32',
        name='time_stamps')
    ages = tf.keras.layers.Input(
        shape=(max_seq_length,),
        dtype='int32',
        name='ages')

    # num_hidden_state, embedding_size
    phenotype_matrix = tf.Variable(
        tf.random_uniform_initializer(
            minval=-1.,
            maxval=1.
        )(
            shape=[num_hidden_state, embedding_size],
            dtype=tf.float32
        ),
        name='phenotype_embeddings'
    )

    concept_mask = mask[:, tf.newaxis, tf.newaxis, :]

    default_inputs = [masked_concept_ids, visit_segments, visit_concept_orders,
                      mask, time_stamps, ages]

    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)

    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size, embedding_size,
        input_length=max_seq_length,
        name='concept_embeddings',
        embeddings_regularizer=l2_regularizer)

    visit_segment_layer = VisitEmbeddingLayer(
        visit_order_size=3,
        embedding_size=embedding_size,
        name='visit_segment_layer')

    encoder = Encoder(
        name='encoder',
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout)

    output_layer_1 = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    # embeddings for encoder input
    temporal_concept_embeddings, concept_embedding_matrix = concept_embedding_layer(
        masked_concept_ids
    )

    temporal_concept_embeddings = visit_segment_layer(
        [visit_segments, temporal_concept_embeddings]
    )
    # # define the time embedding layer for absolute time stamps (since 1970)
    time_embedding_layer = TimeEmbeddingLayer(
        embedding_size=time_embeddings_size,
        name='time_embedding_layer')
    # define the age embedding layer for the age w.r.t the medical record
    age_embedding_layer = TimeEmbeddingLayer(
        embedding_size=time_embeddings_size,
        name='age_embedding_layer')
    positional_encoding_layer = PositionalEncodingLayer(
        max_sequence_length=max_seq_length,
        embedding_size=embedding_size,
        name='positional_encoding_layer')

    # dense layer for rescale the patient sequence embeddings back to the original size
    temporal_transformation_layer = tf.keras.layers.Dense(
        embedding_size,
        activation='tanh',
        name='temporal_transformation')

    time_embeddings = time_embedding_layer(time_stamps)
    age_embeddings = age_embedding_layer(ages)
    positional_encodings = positional_encoding_layer(visit_concept_orders)

    temporal_concept_embeddings = temporal_transformation_layer(
        tf.concat(
            [temporal_concept_embeddings, time_embeddings, age_embeddings, positional_encodings],
            axis=-1
        )
    )

    # (batch_size, num_hidden_state, embedding_size)
    phenotype_embeddings = tf.tile(
        phenotype_matrix[tf.newaxis, :, :],
        [tf.shape(temporal_concept_embeddings)[0], 1, 1]
    )

    phenotype_mha_layer = MultiHeadAttention(
        d_model=embedding_size,
        num_heads=num_heads
    )

    layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    dropout = tf.keras.layers.Dropout(transformer_dropout)

    # Generate contextualized phenotype embeddings (batch_size, num_hidden_state, embedding_size)
    phenotype_embeddings, _ = phenotype_mha_layer(
        temporal_concept_embeddings,
        temporal_concept_embeddings,
        phenotype_embeddings,
        concept_mask
    )

    phenotype_embeddings = layernorm(
        dropout(
            phenotype_embeddings
        )
    )

    # This gives us the phenotype embeddings
    phenotype_hidden_state_layer = tf.keras.layers.Dense(
        units=1
    )

    # (batch_size, num_hidden_state)
    phenotype_probability_dist = tf.nn.softmax(
        tf.squeeze(
            phenotype_hidden_state_layer(
                phenotype_embeddings
            )
        )
    )

    # (batch_size, max_seq, embedding_size)
    contextualized_embeddings, _ = encoder(
        temporal_concept_embeddings,
        concept_mask
    )

    # (batch_size, num_hidden_state, max_seq, embedding_size + embedding_size)
    phenotype_concept_embeddings = tf.concat(
        [tf.tile(
            phenotype_embeddings[:, :, tf.newaxis, :],
            [1, 1, max_seq_length, 1]
        ),
            tf.tile(
                contextualized_embeddings[:, tf.newaxis, :, :],
                [1, num_hidden_state, 1, 1]
            )
        ],
        axis=-1
    )

    # (batch_size, num_hidden_state, max_seq, embedding_size)
    scale_layer = tf.keras.layers.Dense(
        embedding_size,
        activation='tanh',
        name='scale_layer'
    )

    phenotype_concept_embeddings = scale_layer(
        phenotype_concept_embeddings
    )

    reshaped_phenotype_concept_embeddings = tf.reshape(
        phenotype_concept_embeddings,
        [-1, num_hidden_state * max_seq_length, embedding_size]
    )

    # (batch_size, num_hidden_state, max_seq, vocab_size)
    concept_predictions = tf.reshape(
        tf.nn.softmax(
            output_layer_1(
                [reshaped_phenotype_concept_embeddings, concept_embedding_matrix]
            )
        ),
        [-1, num_hidden_state, max_seq_length, concept_vocab_size]
    )

    squeeze_layer = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x),
        name='concept_predictions'
    )

    weighted_concept_predictions = squeeze_layer(
        tf.reduce_sum(
            phenotype_probability_dist[:, :, tf.newaxis, tf.newaxis] * concept_predictions,
            axis=1
        )
    )

    model = tf.keras.Model(
        inputs=default_inputs,
        outputs=[weighted_concept_predictions])

    return model
