import unittest
import tensorflow as tf
from models.custom_layers import PositionalEncodingLayer


class PositionalEncodingLayerTest(unittest.TestCase):
    def setUp(self):
        # Define the input tensor shape
        self.batch_size = 4
        self.max_sequence_length = 10
        self.embedding_size = 16

        # Create an instance of the PositionalEncodingLayer
        self.positional_encoding_layer = PositionalEncodingLayer(self.max_sequence_length, self.embedding_size)

    def test_positional_encoding_layer_shape(self):
        """
        Tests the output shape of the positional encoding layer.
        """
        # Create random input tensor
        visit_concept_orders = tf.random.uniform(
            shape=(self.batch_size, self.max_sequence_length),
            maxval=self.max_sequence_length,
            dtype=tf.int32
        )

        # Sort the visit concept orders in ascending order
        visit_concept_orders = tf.sort(visit_concept_orders, axis=-1)

        # Call the PositionalEncodingLayer
        positional_embeddings = self.positional_encoding_layer(visit_concept_orders)

        # Ensure the output shape is correct
        self.assertEqual(
            positional_embeddings.shape.as_list(),
            [self.batch_size, self.max_sequence_length, self.embedding_size]
        )

    def test_positional_encoding_layer_embeddings(self):
        """
        Tests that the positional embeddings are the same for concepts with the same visit order and different for
        concepts with different visit orders.
        """
        # Create random input tensor
        n_repeats = self.max_sequence_length // 2
        visit_concept_orders = tf.reshape(tf.tile(tf.constant([[1], [2]]), (1, n_repeats)), (1, -1))

        # Call the PositionalEncodingLayer
        positional_embeddings = self.positional_encoding_layer(visit_concept_orders)

        # Ensure the positional embeddings are the same for 0 and 4
        self.assertTrue(tf.reduce_all(positional_embeddings[:, 0] == positional_embeddings[:, 4]))

        # Ensure the positional embeddings are the same for 5 and 9
        self.assertTrue(tf.reduce_all(positional_embeddings[:, 5] == positional_embeddings[:, 9]))

        # Ensure the positional embeddings are different for concepts with different visit orders
        self.assertFalse(tf.reduce_all(positional_embeddings[:, 0] == positional_embeddings[:, 9]))

    def test_large_visit_concept_orders_layer(self):
        """
        Tests the positional encoding layer with large visit concept orders to ensure it can handle such inputs
        without errors.
        """
        # Create random input tensor
        visit_concept_orders = tf.random.uniform(
            shape=(self.batch_size, self.max_sequence_length - 1),
            minval=self.max_sequence_length * 10,
            maxval=self.max_sequence_length * 10 + self.max_sequence_length,
            dtype=tf.int32
        )
        # Sort the visit concept orders in ascending order
        visit_concept_orders = tf.sort(visit_concept_orders, axis=-1)

        # Create pads
        pads = tf.tile([[self.max_sequence_length]], (self.batch_size, 1))
        # Reproduce visit_concept_orders inputs e.g. [3000, 3001, 3002, 512]
        visit_concept_orders = tf.concat([visit_concept_orders, pads], axis=-1)

        try:
            # Call the PositionalEncodingLayer
            self.positional_encoding_layer(visit_concept_orders)
        except RuntimeError as e:
            self.fail(e)


if __name__ == '__main__':
    unittest.main()
