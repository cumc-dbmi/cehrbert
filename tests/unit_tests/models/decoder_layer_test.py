import unittest
import tensorflow as tf
from models.custom_layers import DecoderLayer


class DecoderLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.d_model = 32
        self.num_heads = 4
        self.dff = 64
        self.rate = 0.2
        self.batch_size = 16
        self.target_seq_len = 10
        self.input_seq_len = 20
        self.decoder_layer = DecoderLayer(self.d_model, self.num_heads, self.dff, self.rate)

    def test_decoder_layer_output_shape(self):
        x = tf.random.normal((self.batch_size, self.target_seq_len, self.d_model))
        enc_output = tf.random.normal((self.batch_size, self.input_seq_len, self.d_model))
        decoder_mask = tf.sort(
            tf.random.uniform(
                (self.batch_size, 1, 1, self.target_seq_len), maxval=2, dtype=tf.int32
            ),
            axis=-1
        )
        encoder_mask = tf.sort(
            tf.random.uniform(
                (self.batch_size, 1, 1, self.input_seq_len), maxval=2, dtype=tf.int32
            ),
            axis=-1
        )
        output, attn_weights_block1, attn_weights_block2 = self.decoder_layer(
            x,
            enc_output,
            decoder_mask,
            encoder_mask
        )
        expected_output_shape = tf.TensorShape(
            [self.batch_size, self.target_seq_len, self.d_model])
        expected_attn_weights_block1_shape = tf.TensorShape(
            [self.batch_size, self.num_heads, self.target_seq_len, self.target_seq_len])
        expected_attn_weights_block2_shape = tf.TensorShape(
            [self.batch_size, self.num_heads, self.target_seq_len, self.input_seq_len])

        self.assertEqual(output.shape, expected_output_shape)
        self.assertEqual(attn_weights_block1.shape, expected_attn_weights_block1_shape)
        self.assertEqual(attn_weights_block2.shape, expected_attn_weights_block2_shape)

    def test_decoder_layer_mask(self):
        x = tf.random.normal((self.batch_size, self.target_seq_len, self.d_model))
        enc_output = tf.random.normal((self.batch_size, self.input_seq_len, self.d_model))

        # Block the attention to the last position of the sequence in the batch
        decoder_mask = tf.tile(
            tf.expand_dims(
                tf.constant([0] * (self.target_seq_len - 1) + [1]), axis=0
            ),
            [self.batch_size, 1]
        )[:, tf.newaxis, tf.newaxis, :]

        # Block the attention to the last two positions of the sequence in the batch
        encoder_mask = tf.tile(
            tf.expand_dims(
                tf.constant([0] * (self.input_seq_len - 2) + [1, 1]), axis=0
            ),
            [self.batch_size, 1]
        )[:, tf.newaxis, tf.newaxis, :]

        output, attn_weights_block1, attn_weights_block2 = self.decoder_layer(
            x,
            enc_output,
            decoder_mask,
            encoder_mask
        )
        # Attention weights should be 0 since they are masked
        self.assertEqual(0, tf.reduce_sum(attn_weights_block1[:, :, :, -1]))
        self.assertEqual(0, tf.reduce_sum(attn_weights_block2[:, :, :, -2:]))


if __name__ == '__main__':
    unittest.main()
