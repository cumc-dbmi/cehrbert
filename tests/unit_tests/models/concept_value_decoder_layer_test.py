import unittest

import tensorflow as tf

from cehrbert.models.layers.custom_layers import ConceptValuePredictionLayer


class TestConceptValuePredictionLayer(unittest.TestCase):

    def test_layer_initialization(self):
        """Test if the layer initializes with the correct embedding size."""
        embedding_size = 64
        layer = ConceptValuePredictionLayer(embedding_size)
        self.assertEqual(layer.embedding_size, embedding_size)

    def test_get_config(self):
        """Test if the get_config method returns the correct configuration."""
        embedding_size = 64
        layer = ConceptValuePredictionLayer(embedding_size)
        config = layer.get_config()
        self.assertEqual(config["embedding_size"], embedding_size)

    def test_call(self):
        """Test the call method of the layer."""
        embedding_size = 64
        layer = ConceptValuePredictionLayer(embedding_size)

        # Create mock data for testing
        batch_size = 2
        context_window = 3
        original_concept_embeddings = tf.random.normal((batch_size, context_window, embedding_size))
        concept_val_embeddings = tf.random.normal((batch_size, context_window, embedding_size))
        concept_value_masks = tf.random.uniform((batch_size, context_window), minval=0, maxval=2, dtype=tf.int32)

        # Test the call method
        concept_vals = layer(original_concept_embeddings, concept_val_embeddings, concept_value_masks)

        # Check the shape of the output
        self.assertEqual(concept_vals.shape, (batch_size, context_window, 1))


if __name__ == "__main__":
    unittest.main()
