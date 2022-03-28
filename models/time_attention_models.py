import tensorflow as tf

from models.layers.custom_layers import TimeAttention


def time_attention_cbow_negative_sampling_model(max_seq_length: int,
                                                vocabulary_size: int,
                                                concept_embedding_size: int,
                                                time_window_size: int):
    """

    :param max_seq_length:
    :param vocabulary_size:
    :param concept_embedding_size:
    :param time_window_size:
    :return:
    """
    target_concepts = tf.keras.layers.Input(shape=(1,), dtype='int32', name='target_concepts')

    target_time_stamps = tf.keras.layers.Input(shape=(1,), dtype='int32', name='target_time_stamps')

    context_concepts = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                             name='context_concepts')

    context_time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                                name='context_time_stamps')

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    embedding_layer = tf.keras.layers.Embedding(vocabulary_size, concept_embedding_size,
                                                name='embedding_layer',
                                                mask_zero=True)

    time_attention_layer = TimeAttention(vocab_size=vocabulary_size,
                                         target_seq_len=1,
                                         context_seq_len=max_seq_length,
                                         time_window_size=time_window_size)

    dot_layer = tf.keras.layers.Dot(axes=2)

    sigmoid_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    # shape = (batch_size, 1, embedding_size)
    target_concept_embeddings = embedding_layer(target_concepts)

    # shape = (batch_size, seq_len, embedding_size)
    context_concept_embeddings = embedding_layer(context_concepts)

    # shape = (batch_size, 1, seq_len)
    time_attentions = time_attention_layer([target_concepts,
                                            target_time_stamps,
                                            context_time_stamps,
                                            mask])

    # shape = (batch_size, 1, embedding_size)
    combined_embeddings = tf.matmul(time_attentions, context_concept_embeddings)

    # shape = (batch_size, 1, 1)
    concept_predictions = sigmoid_layer(dot_layer([target_concept_embeddings, combined_embeddings]))

    model = tf.keras.Model(
        inputs=[target_concepts, target_time_stamps, context_concepts, context_time_stamps, mask],
        outputs=[concept_predictions])

    return model


def time_attention_cbow_model(max_seq_length: int,
                              vocabulary_size: int,
                              concept_embedding_size: int,
                              time_window_size: int):
    """

    :param max_seq_length:
    :param vocabulary_size:
    :param concept_embedding_size:
    :param time_window_size:
    :return:
    """
    target_concepts = tf.keras.layers.Input(shape=(1,), dtype='int32', name='target_concepts')

    target_time_stamps = tf.keras.layers.Input(shape=(1,), dtype='int32', name='target_time_stamps')

    context_concepts = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                             name='context_concepts')

    context_time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                                name='context_time_stamps')

    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    embedding_layer = tf.keras.layers.Embedding(vocabulary_size, concept_embedding_size,
                                                name='embedding_layer',
                                                mask_zero=True)

    time_embedding_layer = TimeAttention(vocab_size=vocabulary_size,
                                         target_seq_len=1,
                                         context_seq_len=max_seq_length,
                                         time_window_size=time_window_size,
                                         trainable=True)

    dense_layer = tf.keras.layers.Dense(vocabulary_size)

    softmax_layer = tf.keras.layers.Softmax(name='concept_predictions')

    # shape = (batch_size, seq_len, embedding_size)
    concept_embeddings = embedding_layer(context_concepts)

    if mask is not None:
        concept_embeddings = concept_embeddings * tf.cast(tf.expand_dims(mask == 0, axis=-1),
                                                          dtype=tf.float32)

    # shape = (batch_size, 1, seq_len)
    time_embeddings = time_embedding_layer([target_concepts,
                                            target_time_stamps,
                                            context_time_stamps,
                                            mask])

    # shape = (batch_size, 1, embedding_size)
    combined_embeddings = tf.matmul(time_embeddings, concept_embeddings)

    # shape = (batch_size, 1, vocab_size)
    concept_predictions = softmax_layer(dense_layer(combined_embeddings))

    model = tf.keras.Model(
        inputs=[target_concepts, target_time_stamps, context_concepts, context_time_stamps, mask],
        outputs=[concept_predictions])

    return model
