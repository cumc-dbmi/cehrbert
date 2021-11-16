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
concept_mask = create_concept_mask(mask, num_seq)


# %%
concept_mask


# %%

