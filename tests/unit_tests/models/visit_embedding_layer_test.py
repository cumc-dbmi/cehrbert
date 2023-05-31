import tensorflow as tf
import unittest
from models.custom_layers import VisitEmbeddingLayer


class TestVisitEmbeddingLayer(unittest.TestCase):

    def setUp(self):
        visit_order_size = 10
        embedding_size = 32
        self.layer = VisitEmbeddingLayer(visit_order_size, embedding_size)
        self.concept_embeddings = tf.random.uniform((2, 3, embedding_size))

    def test_call(self):
        visit_orders = tf.constant([[1, 1, 2], [1, 1, 2]])

        expected_output = self.layer.visit_embedding_layer(visit_orders) + self.concept_embeddings

        output = self.layer.call([visit_orders, self.concept_embeddings])

        self.assertTrue(tf.reduce_all(tf.equal(output, expected_output)))

    def test_get_config(self):
        expected_config = {'visit_order_size': 10, 'embedding_size': 32}
        config = self.layer.get_config()
        config_subset = {k: config[k] for k in expected_config.keys()}
        self.assertDictEqual(config_subset, expected_config)


if __name__ == '__main__':
    unittest.main()
