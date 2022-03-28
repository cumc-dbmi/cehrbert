import tensorflow as tf
import numpy as np

from tensorflow.keras.utils import get_custom_objects
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from keras_transformer.bert import MaskedPenalizedSparseCategoricalCrossentropy

from utils.model_utils import create_concept_mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (tf.cast(mask, dtype='float32') * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits,
                                      axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)

        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        return config

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def split_heads_query_key_value(self, batch_size, k, q, v):
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        return k, q, v

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        k, q, v = self.split_heads_query_key_value(batch_size, k, q, v)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1,
                                              3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1,
                                       self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(EncoderLayer, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(self, x, mask, **kwargs):
        attn_output, attn_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=kwargs.get('training'))
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=kwargs.get('training'))
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attn_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff=2148, dropout_rate=0.1, *args,
                 **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate, name='transformer' + str(i))
            for i in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config['num_layers'] = self.num_layers
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, x, mask, **kwargs):
        attention_weights = []
        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, mask, **kwargs)
            attention_weights.append(attn_weights)
        return x, tf.stack(attention_weights, axis=0)  # (batch_size, input_seq_len, d_model)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(DecoderLayer, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config['d_model'] = self.d_model
        config['num_heads'] = self.num_heads
        config['dff'] = self.dff
        config['rate'] = self.rate
        return config

    def call(self, x, enc_output, decoder_mask, encoder_mask, **kwargs):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               decoder_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                               encoder_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, **kwargs)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, **kwargs)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, *args, **kwargs):
        super(PositionalEncodingLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        # TODO: change this to dynamic in the future
        self.pos_encoding = tf.squeeze(positional_encoding(10000, self.embedding_size))

    def get_config(self):
        config = super().get_config()
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, visit_concept_orders):
        # Normalize the visit_orders using the smallest visit_concept_orders
        # Take the absolute value to make sure the padded values are not negative after
        # normalization
        visit_concept_orders = tf.abs(visit_concept_orders - tf.expand_dims(
            tf.math.reduce_min(visit_concept_orders, axis=1), axis=-1))
        # Get the same positional encodings for the concepts with the same visit_order
        positional_embeddings = tf.gather(self.pos_encoding, visit_concept_orders, axis=0)
        return positional_embeddings


class TimeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, is_time_delta=False, *args, **kwargs):
        super(TimeEmbeddingLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta
        self.w = self.add_weight(shape=(1, self.embedding_size),
                                 trainable=True,
                                 initializer=tf.keras.initializers.GlorotNormal(),
                                 name=f'time_embedding_weight_{self.name}')
        self.phi = self.add_weight(shape=(1, self.embedding_size),
                                   trainable=True,
                                   initializer=tf.keras.initializers.GlorotNormal(),
                                   name=f'time_embedding_phi_{self.name}')

    def get_config(self):
        config = super().get_config()
        config['embedding_size'] = self.embedding_size
        config['is_time_delta'] = self.is_time_delta
        return config

    def call(self, time_stamps):
        time_stamps = tf.cast(time_stamps, tf.float32)
        if self.is_time_delta:
            time_stamps = tf.concat(
                [time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]], axis=-1)
        next_input = tf.expand_dims(time_stamps, axis=-1) * self.w + self.phi
        return tf.sin(next_input)


class VisitEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, visit_order_size: int,
                 embedding_size: int, *args, **kwargs):
        super(VisitEmbeddingLayer, self).__init__(*args, **kwargs)
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size

        self.visit_embedding_layer = tf.keras.layers.Embedding(self.visit_order_size,
                                                               self.embedding_size)

    def get_config(self):
        config = super().get_config()
        config['visit_order_size'] = self.visit_order_size
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, inputs, **kwargs):
        visit_orders, concept_embeddings = inputs
        return self.visit_embedding_layer(visit_orders, **kwargs) + concept_embeddings


class ConceptValueTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, *args, **kwargs):
        super(ConceptValueTransformationLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.merge_value_transformation_layer = tf.keras.layers.Dense(
            embedding_size,
            name='merge_value_transformation_layer'
        )

    def get_config(self):
        config = super().get_config()
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, concept_embeddings, concept_values, concept_value_masks):
        # Mask out the concept embeddings without a value
        # Combine the concept embeddings with concept_values

        # (batch_size, num_of_visits, num_of_concepts, 1)
        concept_values = tf.expand_dims(
            concept_values,
            axis=-1
        )
        # (batch_size, num_of_visits, num_of_concepts, 1)
        concept_value_masks = tf.expand_dims(
            concept_value_masks,
            axis=-1
        )
        # (batch_size, num_of_visits, num_of_concepts, 1 + embedding_size)
        concept_embeddings_with_val = tf.concat(
            [concept_embeddings, concept_values],
            axis=-1
        )
        # Run through a dense layer to bring the dimension back to embedding_size
        concept_embeddings_with_val = self.merge_value_transformation_layer(
            concept_embeddings_with_val
        )
        # Zero out the positions without a val
        concept_embeddings_with_val = tf.multiply(
            concept_embeddings_with_val,
            tf.cast(concept_value_masks, dtype=tf.float32)
        )
        # Derive the inverse concept value masks for zeroing out the embeddings without a val
        inverse_concept_value_masks = tf.cast(
            tf.logical_not(
                tf.cast(concept_value_masks, dtype=tf.bool)
            ),
            dtype=tf.float32
        )

        # Zero out the position of concept embeddings with a val
        concept_embeddings_without_val = tf.multiply(
            inverse_concept_value_masks,
            concept_embeddings
        )

        # Merge two sets of concept embeddings
        concept_embeddings = concept_embeddings_without_val + concept_embeddings_with_val

        return concept_embeddings


class TemporalTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, time_embeddings_size, embedding_size, *args, **kwargs):
        super(TemporalTransformationLayer, self).__init__(*args, **kwargs)

        self.time_embeddings_size = time_embeddings_size
        self.embedding_size = embedding_size

        # define the time embedding layer for absolute time stamps (since 1970)
        self.time_embedding_layer = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            name='time_embedding_layer'
        )
        # define the age embedding layer for the age w.r.t the medical record
        self.age_embedding_layer = TimeEmbeddingLayer(
            embedding_size=time_embeddings_size,
            name='age_embedding_layer'
        )

        # define positional encoding layer for visit numbers, the visit numbers are normalized
        # by subtracting visit numbers off the first visit number
        self.positional_encoding_layer = PositionalEncodingLayer(
            embedding_size=time_embeddings_size,
            name='positional_encoding_layer'
        )
        # Temporal transformation
        self.temporal_transformation_layer = tf.keras.layers.Dense(
            embedding_size,
            activation='tanh',
            name='temporal_transformation'
        )

    def get_config(self):
        config = super().get_config()
        config['time_embeddings_size'] = self.time_embeddings_size
        config['embedding_size'] = self.embedding_size
        return config

    def call(self, concept_embeddings, pat_seq_age, pat_seq_time, visit_rank_order, **kwargs):
        _, _, num_of_concepts = pat_seq_age.shape

        pt_seq_age_embeddings = self.age_embedding_layer(
            pat_seq_age,
            **kwargs
        )
        pt_seq_time_embeddings = self.time_embedding_layer(
            pat_seq_time,
            **kwargs
        )
        visit_positional_encoding = self.positional_encoding_layer(
            visit_rank_order,
            **kwargs
        )

        visit_positional_encoding = tf.tile(
            visit_positional_encoding[:, :, tf.newaxis, :], [1, 1, num_of_concepts, 1])

        # (batch, num_of_visits, num_of_concepts, embedding_size)
        temporal_concept_embeddings = self.temporal_transformation_layer(
            tf.concat(
                [concept_embeddings,
                 pt_seq_age_embeddings,
                 pt_seq_time_embeddings,
                 visit_positional_encoding],
                axis=-1
            )
        )

        return temporal_concept_embeddings


class TimeAttention(tf.keras.layers.Layer):

    def __init__(self, vocab_size: int,
                 target_seq_len: int,
                 context_seq_len: int,
                 time_window_size: int,
                 return_logits: bool = False,
                 *args, **kwargs):
        super(TimeAttention, self).__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.target_seq_len = target_seq_len
        self.context_seq_len = context_seq_len

        # Save the half window size
        self.half_time_window_size = int(time_window_size / 2)
        # Pad one for time zero, in which the index event occurred
        self.time_window_size = self.half_time_window_size * 2 + 1
        self.return_logits = return_logits

        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size,
                                                         self.time_window_size,
                                                         embeddings_initializer=tf.keras.initializers.zeros,
                                                         name='time_attention_embedding',
                                                         trainable=kwargs.get('trainable'))
        self.softmax_layer = tf.keras.layers.Softmax()

    def get_config(self):
        config = super().get_config()
        config['vocab_size'] = self.vocab_size
        config['target_seq_len'] = self.target_seq_len
        config['context_seq_len'] = self.context_seq_len
        config['time_window_size'] = self.time_window_size
        config['return_logits'] = self.return_logits
        return config

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        target_concepts = inputs[0]
        target_time_stamps = inputs[1]
        context_time_stamps = inputs[2]
        time_mask = inputs[3]

        # shape = (batch_size, target_seq_length, time_window_size)
        concept_time_embeddings = self.embedding_layer(target_concepts)

        # shape = (batch_size, context_seq_length, target_seq_len)
        multiplied_context_time_stamps = tf.tile(tf.expand_dims(context_time_stamps, axis=-1),
                                                 tf.constant([1, 1, self.target_seq_len]))

        # shape = (batch_size, target_seq_length, context_seq_length)
        time_delta = tf.transpose(
            multiplied_context_time_stamps - tf.expand_dims(target_time_stamps, axis=1),
            perm=[0, 2, 1])

        # Clip the time deltas to fit the time window. E.g. if the time window is 101,
        # the allowed time delta values are between -50 to 50
        time_delta_value_clipped = tf.clip_by_value(time_delta,
                                                    clip_value_min=-self.half_time_window_size,
                                                    clip_value_max=self.half_time_window_size)
        # shape = (batch_size, target_seq_length, context_seq_length, full_time_window_size)
        time_delta_one_hot = tf.one_hot(time_delta_value_clipped + self.half_time_window_size,
                                        self.time_window_size)

        # shape = (batch_size, target_seq_length, time_window_size, 1)
        concept_time_embeddings_expanded = tf.expand_dims(concept_time_embeddings, axis=-1)

        # shape = (batch_size, target_seq_length, context_seq_length)
        next_input = tf.squeeze(tf.matmul(time_delta_one_hot, concept_time_embeddings_expanded),
                                axis=-1)

        # add the mask to the scaled tensor.
        if time_mask is not None:
            next_input += (tf.cast(tf.expand_dims(time_mask, axis=1), dtype='float32') * -1e9)

        return next_input if self.return_logits else self.softmax_layer(next_input)


class TimeSelfAttention(TimeAttention):

    def __init__(self,
                 target_seq_len: int,
                 context_seq_len: int,
                 self_attention_return_logits: bool,
                 *args, **kwargs):
        assert target_seq_len == context_seq_len
        super(TimeSelfAttention, self).__init__(target_seq_len=target_seq_len,
                                                context_seq_len=context_seq_len,
                                                *args, **kwargs)
        self.self_attention_return_logits = self_attention_return_logits

    def get_config(self):
        config = super().get_config()
        config['self_attention_return_logits'] = self.self_attention_return_logits
        return config

    def call(self, inputs, **kwargs):
        """

        :param inputs:
        :param kwargs:
        :return:
        """
        concept_ids = inputs[0]
        time_stamps = inputs[1]
        time_mask = inputs[2]

        # shape = (batch_size, seq_len, seq_len)
        self_attention_logits = super().call([concept_ids, time_stamps, time_stamps, time_mask])

        # add the mask to the scaled tensor.
        if time_mask is not None:
            self_attention_logits += (
                    tf.cast(tf.expand_dims(time_mask, axis=1), dtype='float32') * -1e9)

        return self_attention_logits if self.self_attention_return_logits else self.softmax_layer(
            self_attention_logits)


class BertLayer(tf.keras.layers.Layer):

    def __init__(self, model_path: str, *args, **kwargs):
        super(BertLayer, self).__init__(*args, **kwargs)
        bert_model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())

        self.model_path = model_path
        self.concept_embedding_layer = bert_model.get_layer('concept_embeddings')
        self.visit_segment_layer = [layer for layer in bert_model.layers if
                                    layer.name in ['visit_embedding_layer',
                                                   'visit_segment_layer']][0]
        self.positional_encoding_layer = bert_model.get_layer('positional_encoding_layer')
        self.time_embedding_layer = bert_model.get_layer('time_embedding_layer')
        self.age_embedding_layer = bert_model.get_layer('age_embedding_layer')
        self.scale_pat_seq_layer = bert_model.get_layer('scale_pat_seq_layer')
        self.encoder_layer = bert_model.get_layer('encoder')
        #         self.conv_1d = tf.keras.layers.Conv1D(1, 1)
        self.attention_dense = tf.keras.layers.Dense(self.scale_pat_seq_layer.units,
                                                     activation='tanh')
        self.dense = tf.keras.layers.Dense(self.scale_pat_seq_layer.units, activation='tanh')

    def get_config(self):
        config = super().get_config()
        config['model_path'] = self.model_path
        return config

    def call(self, inputs, **kwargs):
        (local_concept_ids, local_visit_segments, local_visit_concept_orders,
         local_time_stamps, local_ages, local_mask) = inputs

        batch_size, max_seq_length = local_mask.get_shape().as_list()

        concept_embeddings, _ = self.concept_embedding_layer(local_concept_ids)
        time_embeddings = self.time_embedding_layer(local_time_stamps)
        age_embeddings = self.age_embedding_layer(local_ages)
        positional_encoddings = self.positional_encoding_layer(local_visit_concept_orders)
        concept_mask = create_concept_mask(local_mask, max_seq_length)

        input_for_encoder = self.scale_pat_seq_layer(
            tf.concat([concept_embeddings, time_embeddings, age_embeddings, positional_encoddings],
                      axis=-1))
        input_for_encoder = self.visit_segment_layer([local_visit_segments, input_for_encoder])
        contextualized_embeddings, _ = self.encoder_layer(input_for_encoder, concept_mask)
        _, _, embedding_size = contextualized_embeddings.get_shape().as_list()
        mask_embeddings = tf.tile(tf.expand_dims(local_mask == 0, -1), [1, 1, embedding_size])
        contextualized_embeddings = tf.math.multiply(contextualized_embeddings,
                                                     tf.cast(mask_embeddings, dtype=tf.float32))

        # (batch, seq_len, embeddings_size)
        multi_dim_att = tf.nn.softmax(self.attention_dense(contextualized_embeddings)
                                      + (tf.cast(tf.expand_dims(local_mask, axis=-1),
                                                 dtype='float32') * -1e9), axis=1)
        context_representation = tf.reduce_sum(multi_dim_att * contextualized_embeddings, axis=1)

        #         conv_output = self.conv_1d(contextualized_embeddings)
        #         conv_output += (tf.cast(tf.expand_dims(local_mask, axis=-1), dtype='float32') * -1e9)
        #         context_representation = tf.reshape(
        #             tf.transpose(tf.nn.softmax(conv_output, axis=1), [0, 2, 1]) @ contextualized_embeddings,
        #             (-1, self.conv_1d.filters * embedding_size))

        return self.dense(context_representation)


class ConvolutionBertLayer(tf.keras.layers.Layer):

    def __init__(self,
                 model_path: str,
                 seq_len: int,
                 context_window: int,
                 stride: int, *args, **kwargs):
        super(ConvolutionBertLayer, self).__init__(*args, **kwargs)
        self.model_path = model_path
        self.seq_len = seq_len
        self.context_window = context_window
        self.stride = stride
        self.step = (seq_len - context_window) // stride + 1
        self.bert_layer = BertLayer(model_path=model_path)
        #         self.conv_1d = tf.keras.layers.Conv1D(1, 1)
        self.attention_dense = tf.keras.layers.Dense(self.bert_layer.scale_pat_seq_layer.units,
                                                     activation='tanh')

        assert (self.step - 1) * self.stride + self.context_window == self.seq_len

    def get_config(self):
        config = super().get_config()
        config['model_path'] = self.model_path
        config['seq_len'] = self.seq_len
        config['context_window'] = self.context_window
        config['stride'] = self.stride
        return config

    def call(self, inputs, **kwargs):
        concept_ids, visit_segments, visit_concept_orders, time_stamps, ages, mask = inputs

        bert_outputs = []
        bert_output_masking = []
        for i in range(self.step):
            start_index = i * self.stride
            end_index = i * self.stride + self.context_window

            concept_ids_step = concept_ids[:, start_index:end_index]
            visit_segments_step = visit_segments[:, start_index:end_index]
            time_stamps_step = time_stamps[:, start_index:end_index]
            ages_step = ages[:, start_index:end_index]
            visit_concept_orders_step = visit_concept_orders[:, start_index:end_index]
            mask_step = mask[:, start_index:end_index]

            inputs_step = [concept_ids_step,
                           visit_segments_step,
                           visit_concept_orders_step,
                           time_stamps_step,
                           ages_step,
                           mask_step]

            output_masking = tf.cast(tf.reduce_all(mask_step == 1, axis=-1), dtype=tf.int32)

            output_step = self.bert_layer(inputs_step)
            bert_outputs.append(output_step)
            bert_output_masking.append(output_masking)

        # (batch, step, embedding_size)
        bert_output_tensor = tf.stack(bert_outputs, axis=1)
        # (batch, step)
        bert_output_masking_tensor = tf.stack(bert_output_masking, axis=1)
        # (batch, step, 1)
        #         conv_output = self.conv_1d(bert_output_tensor)

        attn = self.attention_dense(bert_output_tensor)

        attn += (tf.cast(tf.expand_dims(bert_output_masking_tensor, axis=-1),
                         dtype='float32') * -1e9)

        _, _, embedding_size = bert_output_tensor.get_shape().as_list()

        context_representation = tf.reduce_sum(tf.nn.softmax(attn, axis=1) * bert_output_tensor,
                                               axis=1)

        #         context_representation = tf.reshape(
        #             tf.transpose(tf.nn.softmax(conv_output, axis=1), [0, 2, 1]) @ bert_output_tensor,
        #             (-1, self.conv_1d.filters * embedding_size))

        return context_representation


class HiddenPhenotypeLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_unit: int,
                 embedding_size: int,
                 num_heads: int,
                 dropout_rate: float = 0.1,
                 *args, **kwargs):
        super(HiddenPhenotypeLayer, self).__init__(*args, **kwargs)
        self.hidden_unit = hidden_unit
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        # num_hidden_state, embedding_size
        self.hidden_unit_embedding = self.add_weight(
            shape=(hidden_unit, embedding_size),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='phenotype_embeddings'
        )

        self.mha_layer = MultiHeadAttention(
            d_model=embedding_size,
            num_heads=num_heads
        )

        self.layer_norm_layer = tf.keras.layers.LayerNormalization(
            epsilon=1e-6
        )
        self.dropout_layer = tf.keras.layers.Dropout(dropout_rate)

        self.phenotype_hidden_state_layer = tf.keras.layers.Dense(
            units=1
        )

    def get_config(self):
        config = super().get_config()
        config['hidden_unit'] = self.hidden_unit
        config['embedding_size'] = self.embedding_size
        config['num_heads'] = self.num_heads
        config['dropout_rate'] = self.dropout_rate
        return config

    def call(self, inputs, **kwargs):
        seq_embeddings, mask = inputs
        # Use broadcasting to copy hidden_unit_embedding
        # (batch_size, num_hidden_state, embedding_size)
        expanded_phenotype_embeddings = tf.ones_like(
            seq_embeddings
        )[:, 0:1, 0:1] * self.hidden_unit_embedding[tf.newaxis, :, :]

        # (batch_size, num_hidden_state, embedding_size)
        context_phenotype_embeddings, _ = self.mha_layer(
            seq_embeddings,
            seq_embeddings,
            expanded_phenotype_embeddings,
            mask,
        )

        context_phenotype_embeddings = self.dropout_layer(
            context_phenotype_embeddings,
            **kwargs
        )

        context_phenotype_embeddings = self.layer_norm_layer(
            expanded_phenotype_embeddings + context_phenotype_embeddings,
            **kwargs
        )

        # (batch_size, num_hidden_state)
        phenotype_probability_dist = tf.nn.softmax(
            tf.squeeze(
                self.phenotype_hidden_state_layer(
                    context_phenotype_embeddings
                )
            )
        )

        phenotype_prob_entropy = -tf.reduce_sum(
            phenotype_probability_dist * tf.math.log(phenotype_probability_dist),
            axis=-1
        )
        # self.add_loss(
        #     tf.reduce_mean(
        #         phenotype_prob_entropy
        #     )
        # )

        self.add_metric(
            phenotype_prob_entropy,
            name='phenotype_probability_entropy'
        )

        return context_phenotype_embeddings, phenotype_probability_dist


class VisitPhenotypeLayer(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_unit: int,
                 embedding_size: int,
                 depth: int,
                 num_heads: int,
                 num_of_visits: int,
                 transformer_dropout: float = 0.1,
                 *args, **kwargs):
        super(VisitPhenotypeLayer, self).__init__(*args, **kwargs)
        self.hidden_unit = hidden_unit
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_of_visits = num_of_visits
        self.transformer_dropout = transformer_dropout
        self.depth = depth

        # Insert the att embeddings between the visit embeddings using the following trick
        self.identity = tf.constant(
            np.insert(
                np.identity(num_of_visits),
                obj=range(1, num_of_visits),
                values=0,
                axis=1
            ),
            dtype=tf.float32
        )

        # Create the inverse "identity" matrix for inserting att embeddings
        self.identity_inverse = tf.constant(
            np.insert(
                np.identity(num_of_visits - 1),
                obj=range(0, num_of_visits),
                values=0,
                axis=1),
            dtype=tf.float32)

        # num_hidden_state, embedding_size
        # Second bert applied at the patient level to the visit embeddings
        self.visit_encoder = Encoder(
            name='visit_encoder',
            num_layers=depth,
            d_model=embedding_size,
            num_heads=num_heads,
            dropout_rate=transformer_dropout)

        self.phenotype_hidden_state_layer = tf.keras.layers.Dense(
            units=hidden_unit * embedding_size
        )
        # Apply tanh to the input, otherwise we will run into the exploding gradient problem
        self.phenotype_hidden_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-6,
            name='phenotype_hidden_normalization'
        )
        # Standard practice to randomly drop out some neurons to avoid overfitting
        self.dropout_layer = tf.keras.layers.Dropout(
            transformer_dropout,
            name='phenotype_hidden_dropout'
        )
        self.phenotype_probability_layer = tf.keras.layers.Dense(
            units=1
        )

    def get_config(self):
        config = super().get_config()
        config['num_of_visits'] = self.num_of_visits
        config['hidden_unit'] = self.hidden_unit
        config['embedding_size'] = self.embedding_size
        config['num_heads'] = self.num_heads
        config['depth'] = self.depth
        config['transformer_dropout'] = self.transformer_dropout
        return config

    def call(self, inputs, **kwargs):
        visit_embeddings, att_embeddings, visit_mask = inputs

        # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
        expanded_visit_embeddings = tf.transpose(
            tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ self.identity,
            perm=[0, 2, 1]
        )

        # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
        expanded_att_embeddings = tf.transpose(
            tf.transpose(att_embeddings, perm=[0, 2, 1]) @ self.identity_inverse,
            perm=[0, 2, 1]
        )

        # Insert the att embeddings between visit embeddings
        # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
        augmented_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

        # Expand dimension for masking MultiHeadAttention in Visit Encoder
        visit_mask_with_att = (tf.reshape(
            tf.stack([visit_mask, visit_mask], axis=2),
            shape=(-1, self.num_of_visits * 2)
        )[:, 1:])[:, tf.newaxis, tf.newaxis, :]

        num_of_visits_with_att = tf.shape(visit_mask_with_att)[1]

        # (num_of_visits_with_att, num_of_visits_with_att)
        look_ahead_mask = tf.cast(
            1 - tf.linalg.band_part(tf.ones((num_of_visits_with_att, num_of_visits_with_att)), -1,
                                    0),
            dtype=tf.int32
        )

        # (batch_size, 1, num_of_visits_with_att, num_of_visits_with_att)
        look_ahead_mask = tf.maximum(visit_mask_with_att, look_ahead_mask)

        # (batch_size, 2 * num_of_visits - 1, embedding_size)
        context_visit_embeddings, attn_weights = self.visit_encoder(
            augmented_visit_embeddings,
            look_ahead_mask,
        )

        # (batch_size, num_of_visits, num_hidden_units * embedding_size)
        phenotype_embeddings = self.phenotype_hidden_state_layer(
            self.identity @ context_visit_embeddings
        )
        # Apply dropout to avoid overfitting and apply Layer normalization across all hidden
        # units to avoid the exploding gradient problem
        phenotype_embeddings = self.phenotype_hidden_norm(
            self.dropout_layer(
                phenotype_embeddings,
                **kwargs
            )
        )
        # (batch_size, num_of_visits, hidden_unit, embedding_size)
        reshaped_phenotype_embeddings = tf.reshape(
            phenotype_embeddings,
            (-1, self.num_of_visits, self.hidden_unit, self.embedding_size)
        )
        # (batch_size, num_of_visits, hidden_unit)
        phenotype_probability = tf.nn.softmax(
            tf.squeeze(
                self.phenotype_probability_layer(
                    reshaped_phenotype_embeddings
                )
            )
        )

        phenotype_prob_entropy = -tf.reduce_sum(
            phenotype_probability * tf.math.log(phenotype_probability),
            axis=-1
        )

        self.add_metric(
            phenotype_prob_entropy,
            name='phenotype_probability_entropy'
        )

        return (reshaped_phenotype_embeddings, phenotype_probability,
                context_visit_embeddings, attn_weights)


get_custom_objects().update({
    'MultiHeadAttention': MultiHeadAttention,
    'Encoder': Encoder,
    'EncoderLayer': EncoderLayer,
    'DecoderLayer': DecoderLayer,
    'TimeAttention': TimeAttention,
    'TimeSelfAttention': TimeSelfAttention,
    'PairwiseTimeAttention': TimeSelfAttention,
    'VisitEmbeddingLayer': VisitEmbeddingLayer,
    'PositionalEncodingLayer': PositionalEncodingLayer,
    'TimeEmbeddingLayer': TimeEmbeddingLayer,
    'TemporalTransformationLayer': TemporalTransformationLayer,
    'ConceptValueTransformationLayer': ConceptValueTransformationLayer,
    'ReusableEmbedding': ReusableEmbedding,
    'TiedOutputEmbedding': TiedOutputEmbedding,
    'MaskedPenalizedSparseCategoricalCrossentropy': MaskedPenalizedSparseCategoricalCrossentropy,
    'BertLayer': BertLayer,
    'ConvolutionBertLayer': ConvolutionBertLayer,
    'HiddenPhenotypeLayer': HiddenPhenotypeLayer,
    'VisitPhenotypeLayer': VisitPhenotypeLayer,
    'ConvolutionBertLayer': ConvolutionBertLayer
})
