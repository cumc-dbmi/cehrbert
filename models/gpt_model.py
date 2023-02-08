import tensorflow as tf

from models.layers.custom_layers import (
    Encoder, GptDecoder,
    TokenAndPositionEmbedding
)
from utils.model_utils import create_concept_mask

vocab_size = 20000  # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 8  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer


def create_model():
    concept_inputs = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)
    decoder_mask_inputs = tf.keras.layers.Input(shape=(maxlen,), dtype=tf.int32)

    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(concept_inputs)

    transformer_block = GptDecoder(6, embed_dim, num_heads)
    x = transformer_block(x, decoder_mask_inputs)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=[concept_inputs, decoder_mask_inputs], outputs=[outputs])
    return model
