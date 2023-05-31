import unittest
import tensorflow as tf
from models.custom_layers import EncoderLayer


class EncoderLayerTest(unittest.TestCase):
    def setUp(self):
        self.batch_size = 32
        self.input_seq_len = 10
        self.d_model = 512
        self.num_heads = 8
        self.dff = 2048
        self.rate = 0.1
        self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.dff, self.rate)

    def test_call(self):
        x = tf.random.uniform((self.batch_size, self.input_seq_len, self.d_model))
        mask = tf.random.uniform((self.batch_size, 1, 1, self.input_seq_len), minval=0, maxval=2,
                                 dtype=tf.int32)

        output, attn_weights = self.encoder_layer.call(x, mask)

        # Assert output shape
        self.assertEqual(output.shape,
                         tf.TensorShape([self.batch_size, self.input_seq_len, self.d_model]))

        # Assert attn_weights shape
        self.assertEqual(
            attn_weights.shape,
            tf.TensorShape(
                [self.batch_size, self.num_heads, self.input_seq_len, self.input_seq_len])
        )

    def test_encoder_layer_mask(self):
        x = tf.random.normal((self.batch_size, self.input_seq_len, self.d_model))
        # Block the attention to the last two positions of the sequence in the batch
        encoder_mask = tf.tile(
            tf.expand_dims(
                tf.constant([0] * (self.input_seq_len - 2) + [1, 1]), axis=0
            ),
            [self.batch_size, 1]
        )[:, tf.newaxis, tf.newaxis, :]

        output, attn_weights_block = self.encoder_layer(
            x,
            encoder_mask
        )
        # Attention weights should be 0 since they are masked
        self.assertEqual(0, tf.reduce_sum(attn_weights_block[:, :, :, -2:]))


if __name__ == '__main__':
    unittest.main()
