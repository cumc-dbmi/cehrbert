import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects


class TransformerCoordinateEmbedding(tf.keras.layers.Layer):
    """
    Represents trainable positional embeddings for the Transformer model:

    1. word position embeddings - one for each position in the sequence.
    2. depth embeddings - one for each block of the model
    Calling the layer with the Transformer's input will return a new input
    with those embeddings added.
    """

    def __init__(self, max_transformer_depth: int, **kwargs):
        self.max_depth = max_transformer_depth
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["max_transformer_depth"] = self.max_depth
        return config

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):
        sequence_length, d_model = input_shape[-2:]
        self.word_position_embeddings = self.add_weight(
            shape=(sequence_length, d_model),
            initializer="uniform",
            name="word_position_embeddings",
            trainable=True,
        )
        self.depth_embeddings = self.add_weight(
            shape=(self.max_depth, d_model),
            initializer="uniform",
            name="depth_position_embeddings",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        depth = kwargs.get("step")
        if depth is None:
            raise ValueError("Please, provide current Transformer's step" "using 'step' keyword argument.")
        result = inputs + self.word_position_embeddings
        if depth is not None:
            result = result + self.depth_embeddings[depth]
        return result


get_custom_objects().update(
    {
        "TransformerCoordinateEmbedding": TransformerCoordinateEmbedding,
    }
)
