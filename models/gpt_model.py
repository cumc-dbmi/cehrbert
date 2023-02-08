import tensorflow as tf

from models.layers.custom_layers import (
    GptDecoder,
    TokenAndPositionEmbedding
)


def create_model(
        max_len,
        vocab_size,
        embed_dim,
        num_heads,
        num_of_layers
):
    """
    model = create_model(
        max_len=100,
        vocab_size=100,
        embed_dim=128,
        num_heads=16,
        num_of_layers=5
    )
    :param max_len:
    :param vocab_size:
    :param embed_dim:
    :param num_heads:
    :param num_of_layers:
    :return:
    """
    concept_inputs = tf.keras.layers.Input(
        shape=(max_len,),
        dtype=tf.int32,
        name='concept_ids'
    )

    look_ahead_mask_base = tf.cast(
        1 - tf.linalg.band_part(tf.ones((max_len, max_len)), -1, 0),
        dtype=tf.int32
    )[tf.newaxis, tf.newaxis, :, :]

    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(concept_inputs)

    transformer_block = GptDecoder(num_of_layers, embed_dim, num_heads)
    x, _ = transformer_block(x, look_ahead_mask_base)

    outputs = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs=[concept_inputs], outputs=[outputs])
