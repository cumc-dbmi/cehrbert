import unittest
import tensorflow as tf
from models.custom_layers import EncoderLayer


class EncoderLayerTest(unittest.TestCase):
    def setUp(self):
        self.d_model = 512
        self.num_heads = 8
        self.dff = 2048
        self.rate = 0.1
        self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)

    def test_call(self):
        batch_size = 32
        input_seq_len = 10

        x = tf.random.uniform((batch_size, input_seq_len, self.d_model))
        mask = tf.random.uniform((batch_size, 1, 1, input_seq_len), minval=0, maxval=2, dtype=tf.int32)

        output, attn_weights = self.encoder_layer.call(x, mask)

        # Assert output shape
        self.assertEqual(output.shape, tf.TensorShape([batch_size, input_seq_len, self.d_model]))

        # Assert attn_weights shape
        self.assertEqual(attn_weights.shape, tf.TensorShape([batch_size, self.num_heads, input_seq_len, input_seq_len]))

        # Add additional assertions based on the behavior of your EncoderLayer


if __name__ == '__main__':
    unittest.main()
