import unittest
import tensorflow as tf
from models.custom_layers import Encoder


class EncoderTestCase(unittest.TestCase):
    def setUp(self):
        self.num_layers = 2
        self.d_model = 32
        self.num_heads = 4
        self.dff = 64
        self.dropout_rate = 0.2
        self.batch_size = 16
        self.input_seq_len = 10

    def test_encoder_output_shape(self):
        encoder = Encoder(self.num_layers, self.d_model, self.num_heads, self.dff, self.dropout_rate)
        x = tf.random.normal((self.batch_size, self.input_seq_len, self.d_model))
        mask = tf.sort(
            tf.random.uniform((self.batch_size, 1, 1, self.input_seq_len), minval=0, maxval=2, dtype=tf.int32),
            axis=-1
        )

        output, attention_weights = encoder(x, mask)

        expected_output_shape = tf.TensorShape([self.batch_size, self.input_seq_len, self.d_model])
        expected_attention_weights_shape = tf.TensorShape([
            self.num_layers, self.batch_size, self.num_heads, self.input_seq_len, self.input_seq_len
        ])

        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(attention_weights.shape, expected_attention_weights_shape)


if __name__ == '__main__':
    unittest.main()
