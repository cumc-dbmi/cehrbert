import tensorflow as tf
import unittest
from models.custom_layers import TimeEmbeddingLayer


class TestTimeEmbeddingLayer(unittest.TestCase):

    def setUp(self):
        self.embedding_size = 32
        self.is_time_delta = True
        self.layer = TimeEmbeddingLayer(self.embedding_size, self.is_time_delta)

    def test_call(self):
        time_stamps = tf.constant([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]])
        expected_output_shape = tf.TensorShape([2, 4, self.embedding_size])
        output = self.layer(time_stamps)

        self.assertEqual(output.shape, expected_output_shape)
        self.assertTrue(tf.reduce_all(
            tf.math.logical_and(tf.greater_equal(output, -1.0), tf.less_equal(output, 1.0)))
        )


if __name__ == '__main__':
    unittest.main()
