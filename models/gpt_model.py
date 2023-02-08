import tensorflow as tf

from models.layers.custom_layers import (
    GptDecoder,
    TokenAndPositionEmbedding
)


def create_model(
        context_window_size,
        vocab_size,
        embedding_size,
        num_heads,
        depth
):
    """
    model = create_model(
        max_len=100,
        vocab_size=100,
        embed_dim=128,
        num_heads=16,
        num_of_layers=5
    )
    :param context_window_size:
    :param vocab_size:
    :param embedding_size:
    :param num_heads:
    :param depth:
    :return:
    """
    concept_inputs = tf.keras.layers.Input(
        shape=(context_window_size,),
        dtype=tf.int32,
        name='concept_ids'
    )

    look_ahead_mask_base = tf.cast(
        1 - tf.linalg.band_part(tf.ones((context_window_size, context_window_size)), -1, 0),
        dtype=tf.int32
    )[tf.newaxis, tf.newaxis, :, :]

    embedding_layer = TokenAndPositionEmbedding(context_window_size, vocab_size, embedding_size)
    x = embedding_layer(concept_inputs)

    transformer_block = GptDecoder(depth, embedding_size, num_heads)
    x, _ = transformer_block(x, look_ahead_mask_base)

    concept_prediction_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    outputs = tf.keras.layers.Dense(vocab_size)(x)

    outputs = concept_prediction_layer(outputs)

    return tf.keras.Model(inputs=[concept_inputs], outputs=[outputs])
