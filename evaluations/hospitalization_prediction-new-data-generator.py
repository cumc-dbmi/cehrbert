# +
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import datetime
import pickle
from itertools import islice
# -

import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import spark_apps.parameters as p
from models.loss_schedulers import CosineLRSchedule
from evaluations.evaluation import *
from evaluations.model_evaluators import *

from data_generators.data_generator_base import *
from data_generators.learning_objective import post_pad_pre_truncate

# +
# from models.custom_layers import TemporalPositionalEncodingLayer
# -

import matplotlib.pyplot as plt

input_folder = '/data/research_ops/omops/omop_2020q2/cohorts_new/'
# input_folder = '/data/research_ops/omops/omop_2020q2/cohorts_new/'
time_attention_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_1985/time_attention'
vanilla_bert_model_folder = '/data/research_ops/omops/omop_2020q2/cohort_tests/vanilla_bert'
# vanilla_bert_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_1985/vanilla_bert_only_without_visit_prediction'
# vanilla_bert_model_folder = '/data/research_ops/omops/omop_2020q2/sine_time_embedding/vanilla_bert_time_embedding_sum/'
temporal_bert_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_1985/temporal_bert_test_time_attention'

# +
hospitalization_folder = os.path.join(input_folder, 'hf_readmit')

tokenizer_path = os.path.join(time_attention_model_folder, p.tokenizer_path)
bert_tokenizer_path = os.path.join(vanilla_bert_model_folder, p.tokenizer_path)
bert_visit_tokenizer_path = os.path.join(vanilla_bert_model_folder, p.visit_tokenizer_path)
temporal_bert_tokenizer_path = os.path.join(temporal_bert_model_folder, p.tokenizer_path)

time_aware_model_path = os.path.join(time_attention_model_folder, p.time_attention_model_path)
bert_model_path = os.path.join(vanilla_bert_model_folder, p.bert_model_path)
temporal_bert_model_path = os.path.join(temporal_bert_model_folder, p.temporal_bert_model_path)

# +
merged_dataset = pd.read_parquet(hospitalization_folder)

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
bert_tokenizer = pickle.load(open(bert_tokenizer_path, 'rb'))
# bert_visit_tokenizer = pickle.load(open(bert_visit_tokenizer_path, 'rb'))
temporal_bert_tokenizer = pickle.load(open(temporal_bert_tokenizer_path, 'rb'))

# +
# merged_dataset['age'] = ((merged_dataset['age'] - merged_dataset['age'].mean()) / merged_dataset['age'].std()).apply(lambda c: [c])

# +
positive_cases = merged_dataset[merged_dataset['label'] == 1.0]
negative_cases = merged_dataset[merged_dataset['label'] == 0.0]

# positive_cases = positive_cases[positive_cases.concept_ids.apply(lambda concept_ids: len(concept_ids)) >= 10]
# negative_cases = negative_cases[negative_cases.concept_ids.apply(lambda concept_ids: len(concept_ids)) >= 10]
# -

training_data = pd.concat([negative_cases, positive_cases]).sample(frac=1.0)
training_data.concept_ids = training_data.concept_ids.apply(lambda concept_ids: concept_ids.tolist())
# training_data.visit_concept_ids = training_data.visit_concept_ids.apply(lambda concept_ids: concept_ids.tolist())

training_data.groupby('label')['label'].count()

# +
from sklearn.model_selection import train_test_split

max_seq_length = 300
batch_size = 128
embedding_size = 128

merged_dataset = merged_dataset.sample(frac=1.0, random_state=0)

token_ids = bert_tokenizer.encode(merged_dataset.concept_ids.apply(lambda concept_ids: concept_ids.tolist()))
visit_segments = merged_dataset.visit_segments
time_stamps = merged_dataset.dates
# visit_orders = merged_dataset.concept_id_visit_orders
ages = ((merged_dataset['age'] - merged_dataset['age'].mean()) / merged_dataset['age'].std()).astype(float).apply(lambda c: [c]).tolist()
labels = merged_dataset.label

padded_token_ides = post_pad_pre_truncate(token_ids, bert_tokenizer.get_unused_token_id(), max_seq_length)
padded_visit_segments = post_pad_pre_truncate(visit_segments, 0, max_seq_length)
# padded_visit_orders = post_pad_pre_truncate(visit_orders, 0, max_seq_length)
padded_time_stamps = post_pad_pre_truncate(time_stamps, 0, max_seq_length)
mask = (padded_token_ides == bert_tokenizer.get_unused_token_id()).astype(int)

inputs = {
    'concept_ids': padded_token_ides,
    'masked_concept_ids': padded_token_ides,
    'visit_segments': padded_visit_segments,
    'time_stamps': padded_time_stamps,
    'mask': mask,
    'age': ages
}
dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))


# -

def get_metrics():
    """
    Standard metrics used for compiling the models
    :return:
    """
    
    return ['binary_accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(name='auc')]


total = len(merged_dataset)

training = dataset.take(int(total * 0.8)).cache().batch(batch_size)
testing = dataset.skip(int(total * 0.8)).cache().batch(batch_size)


# +
def transform(x, y):
    subset = {
        'concept_ids': x['masked_concept_ids'],
        'time_stamps': x['time_stamps'],
        'age': x['age']
    }
    return (subset, y)

lstm_training = training.map(transform)
lstm_testing = testing.map(transform)


# +
def get_concept_embeddings(time_aware_model_path):
    another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0") #enforce to use first(0) cpu
    with another_strategy.scope():
        time_aware_model = tf.keras.models.load_model(time_aware_model_path,
                                                      custom_objects=dict(
                                                          **get_custom_objects()))
        embedding_layer = time_aware_model.get_layer('embedding_layer')
    return embedding_layer.get_weights()[0] #concept embedding layer to initiate the LSTM embedding layer

embeddings = get_concept_embeddings(time_aware_model_path)
vocab_size, dim = np.shape(embeddings)
strategy = tf.distribute.MirroredStrategy() #model will be distributed into two gpus
with strategy.scope(): 
    lstm_model = create_bi_lstm_model(max_seq_length, vocab_size=vocab_size, embedding_size=dim, concept_embeddings=embeddings)
    # in models/evaluation_models
    lstm_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=get_metrics())

# +
# history = lstm_model.fit(lstm_training, validation_data=lstm_testing, epochs=20, verbose=1)

# +
# RETAIN Model
# def create_vanilla_bert_retain_model(vanilla_bert_model_path):

#     vanilla_bert_model = tf.keras.models.load_model(bert_model_path, custom_objects=dict(**get_custom_objects()))

#     bert_inputs = vanilla_bert_model.inputs[0:3]

#     contextualized_embeddings, _ = vanilla_bert_model.get_layer('encoder').output

#     _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

#     mask_input = bert_inputs[-1]
#     mask_embeddings = tf.tile(tf.expand_dims(mask_input == 0, -1, name='expand_mask'),
#                               [1, 1, embedding_size], name='tile_mask')
#     contextualized_embeddings = tf.math.multiply(contextualized_embeddings,
#                                                  tf.cast(mask_embeddings, dtype=tf.float32,
#                                                          name='cast_mask'))

#     _, max_seq_length = bert_inputs[0].get_shape().as_list()

#     age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

#     visit_orders = tf.keras.layers.Input(name='visit_orders', shape=(max_seq_length,), dtype=tf.int32)

#     max_visit_order = tf.reduce_max(visit_orders * tf.cast(mask_input == 0, tf.int32))

#     reversed_visit_orders = tf.reverse_sequence(visit_orders, tf.argmax(mask_input, axis=1), seq_axis=1, batch_axis=0)

#     reversed_concept_embeddings = tf.reverse_sequence(contextualized_embeddings, tf.argmax(mask_input, axis=1), seq_axis=1, batch_axis=0)

#     visit_embeddings = tf.matmul(tf.transpose(tf.one_hot(max_visit_order - reversed_visit_orders, max_visit_order), [0, 2, 1]), reversed_concept_embeddings)

#     gru_a = tf.keras.layers.GRU(64, return_sequences=True)

#     gru_b = tf.keras.layers.GRU(64, return_sequences=True)

#     dense_a = tf.keras.layers.Dense(1)

#     dense_b = tf.keras.layers.Dense(embedding_size)

#     output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    
#     attn_a = tf.nn.softmax(dense_a(gru_a(visit_embeddings)), axis=1)
    
#     attn_b = tf.nn.tanh(dense_b(gru_b(visit_embeddings)))
    
#     context_embeddings = tf.reduce_sum(visit_embeddings * attn_b * attn_a, axis=1)

#     next_input = tf.keras.layers.concatenate([context_embeddings, age_of_visit_input])

#     output = output_layer(next_input)

#     model = Model(inputs=bert_inputs + [visit_orders, age_of_visit_input],
#                                        outputs=output, name='Vanilla_BERT_PLUS_RETAIN')

#     return model

# +
# starter_learning_rate = 1e-4
# end_learning_rate = 1e-8
# decay_steps = 10
# learning_rate_fn = tf.keras.optimizers.schedules.ExponentialDecay(
#     starter_learning_rate,
#     decay_steps,
#     decay_rate=0.1)
# learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_fn, verbose=1)
# -

#error not allowed to use the same name for multiple layers, need to rerun it
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    bert_lstm_model = create_vanilla_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path=bert_model_path)
    bert_lstm_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=get_metrics())

bert_history = bert_lstm_model.fit(training, validation_data=testing, epochs=10, verbose=1)

# + active=""
#
# -

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    temp_bert_lstm_model = create_temporal_bert_bi_lstm_model(max_seq_length, temporal_bert_model_path=temporal_bert_model_path)
    temp_bert_lstm_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=get_metrics())

temp_bert_history = temp_bert_lstm_model.fit(training, validation_data=testing, epochs=10, verbose=1)




