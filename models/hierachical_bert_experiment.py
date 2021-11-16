#!/usr/bin/env python
# coding: utf-8
# %%
import tensorflow as tf


# %%
from models.custom_layers import *


# %%
num_concept_per_v = 50
num_visit = 20
num_seq = num_concept_per_v*num_visit


vocab_size = 40000
embeddinig_size = 128
time_embeddings_size = 16
depth = 16
num_heads = 8
transformer_dropout: float = 0.1


# %%
pt_seq = tf.keras.layers.Input(shape=(num_seq,), dtype='int32',
name='patient_seq')
pt_seq_age = tf.keras.layers.Input(shape=(num_seq,), dtype='int32',
name='patient_seq_age')
pt_seq_time = tf.keras.layers.Input(shape=(num_seq,), dtype='int32',
name='patient_seq_time')
mask = tf.keras.layers.Input(shape=(num_seq,), dtype='int32', name='mask')

visit_seq_age = tf.keras.layers.Input(shape=(num_visit,), dtype='int32',
name='visit_seq_age')
visit_seq_time = tf.keras.layers.Input(shape=(num_visit,), dtype='int32',
name='visit_seq_time')
visit_seq_time_delta = tf.keras.layers.Input(shape=(num_visit - 1,), dtype='int32',
name='visit_seq_time_delta')

pt_seq = tf.reshape(pt_seq, (-1, num_concept_per_v, num_visit))
pt_seq_age = tf.reshape(pt_seq_age, (-1, num_concept_per_v, num_visit))
pt_seq_time = tf.reshape(pt_seq_time, (-1, num_concept_per_v, num_visit))


# %%
concept_embedding_layer = tf.keras.layers.Embedding(vocab_size, embeddinig_size)
 # # define the time embedding layer for absolute time stamps (since 1970)
time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                                  name='time_embedding_layer')
        # define the age embedding layer for the age w.r.t the medical record
age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size,
                                                 name='age_embedding_layer')


# %%
pt_seq_concept_embeddings = concept_embedding_layer(pt_seq)
pt_seq_age_embeddings = time_embedding_layer(pt_seq_age)
pt_seq_time_embeddings = time_embedding_layer(pt_seq_time)

# dense layer for rescale the patient sequence embeddings back to the original size
scale_back_patient_seq_concat_layer = tf.keras.layers.Dense(embeddinig_size,
                                                            activation='tanh',
                                                            name='scale_pat_seq_layer')


# %%
input_for_encoder = scale_back_patient_seq_concat_layer(
            tf.concat([pt_seq_concept_embeddings, pt_seq_age_embeddings, pt_seq_time_embeddings],
                      axis=-1, name='concat_for_encoder'))


# %%
tf.shape(input_for_encoder)


# %%
def create_concept_mask(mask, max_seq_length):
    # mask the third dimension
    concept_mask_1 = tf.tile(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1),
                             [1, 1, 1, max_seq_length])
    # mask the fourth dimension
    concept_mask_2 = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
    concept_mask = tf.cast((concept_mask_1 + concept_mask_2) > 0, dtype=tf.int32)
    return concept_mask


# %%
concept_mask = create_concept_mask(mask, num_seq)


# %%
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

    def call(self, v, k, q, mask, time_attention_logits):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        k, q, v = self.split_heads_query_key_value(batch_size, k, q, v)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask,
                                                                           time_attention_logits)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1,
                                              3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1,
                                       self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


# %%
mha = MultiHeadAttention(embeddinig_size, num_heads)

# %%
tf.shape(input_for_encoder)

# %%
# mask = tf.ones((1, 1000))
# concept_mask = create_concept_mask(mask, num_seq)
x = tf.random.uniform((1, 50, 20, 128))

# %%
tf.shape(x)

# %%
x_reshape = tf.reshape(tf.transpose(x, perm=[0, 2, 1, 3]), (-1, num_concepts, embeddinig_size))

# %%
mha(x_reshape, x_reshape, x_reshape, None, None)

# %%
batch_size, num_concepts, num_visits, _ = tf.shape(x)


# %%
x = tf.reshape(x, (batch_size, num_concepts, num_visits, 8, 16))


# %%
tf.transpose(x, perm=[0, 2, 3, 1, 4])

# %%
encoder = Encoder(name='encoder',
                  num_layers=depth,
                  d_model=embeddinig_size,
                  num_heads=num_heads,
                  dropout_rate=transformer_dropout)


# %%
contextualized_visit_emebddings, attention_weights = encoder(x_reshape, None)

# %%
tf.shape(contextualized_visit_emebddings)

# %%
contextualized_visit_emebddings = tf.reshape(contextualized_visit_emebddings, (-1, 20, 50, embeddinig_size))

# %%
visit_embeddings = contextualized_visit_emebddings[:, 0]

# %%
visit_embeddings

# %%
