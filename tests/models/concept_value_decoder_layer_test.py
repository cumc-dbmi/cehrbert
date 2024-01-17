import tensorflow as tf
from models.layers.custom_layers import ConceptValueDecoderLayer
import unittest


class TestConceptValueDecoderLayer(unittest.TestCase):

    def test_initialization(self):
        """ Test the initialization of ConceptValueDecoderLayer. """
        embedding_size = 128
        layer = ConceptValueDecoderLayer(embedding_size)
        self.assertEqual(layer.embedding_size, embedding_size)

    def test_get_config(self):
        """ Test the get_config method. """
        embedding_size = 128
        layer = ConceptValueDecoderLayer(embedding_size)
        config = layer.get_config()
        self.assertEqual(config['embedding_size'], embedding_size)

    def test_call(self):
        """ Test the call method of the layer. """
        embedding_size = 128
        layer = ConceptValueDecoderLayer(embedding_size)

        # Create dummy data for testing
        batch_size = 2
        context_window = 3
        concept_val_embeddings = tf.random.normal((batch_size, context_window, embedding_size))
        concept_value_masks = tf.random.uniform((batch_size, context_window), minval=0, maxval=2, dtype=tf.int32)

        # Test the call method
        concept_embeddings, concept_values = layer(concept_val_embeddings, concept_value_masks)
        self.assertEqual(concept_embeddings.shape, (batch_size, context_window, embedding_size))
        self.assertEqual(concept_values.shape, (batch_size, context_window))

        inverse_mask = tf.cast(
            tf.logical_not(tf.cast(concept_value_masks[..., tf.newaxis], dtype=tf.bool)),
            dtype=tf.float32
        )

        # This is to test whether the positions without a val are equal between concept_embeddings and
        # concept_val_embeddings
        self.assertTrue(
            tf.reduce_all(tf.equal(inverse_mask * concept_embeddings, inverse_mask * concept_val_embeddings).numpy())
        )


if __name__ == '__main__':
    unittest.main()
