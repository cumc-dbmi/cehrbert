# +
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import numpy as np
import datetime
import pickle
from itertools import islice

# +
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import keras_metrics
# -

import spark_apps.parameters as p
from models.loss_schedulers import CosineLRSchedule
from evaluations.evaluation import *
from evaluations.model_evaluators import *

from data_generators.data_generator_base import *
from data_generators.learning_objective import post_pad_pre_truncate

import matplotlib.pyplot as plt

input_folder = '/data/research_ops/omops/omop_2020q2/training_data_2010/'
time_attention_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_2015/time_attention'
vanilla_bert_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_1985/vanilla_bert_only'
temporal_bert_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_1985/vanilla_bert_only'

# +
hospitalization_folder = os.path.join(input_folder, 'hospitalization')

tokenizer_path = os.path.join(time_attention_model_folder, p.tokenizer_path)
bert_tokenizer_path = os.path.join(vanilla_bert_model_folder, p.tokenizer_path)
bert_visit_tokenizer_path = os.path.join(vanilla_bert_model_folder, p.visit_tokenizer_path)

time_aware_model_path = os.path.join(time_attention_model_folder, p.time_attention_model_path)
bert_model_path = os.path.join(vanilla_bert_model_folder, p.bert_model_path)
temporal_bert_model_path = os.path.join(temporal_bert_model_folder, p.temporal_bert_model_path)

# +
merged_dataset = pd.read_parquet(hospitalization_folder)

tokenizer = pickle.load(open(tokenizer_path, 'rb'))
bert_tokenizer = pickle.load(open(bert_tokenizer_path, 'rb'))
bert_visit_tokenizer = pickle.load(open(bert_visit_tokenizer_path, 'rb'))

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

pd.concat([training_data.concept_ids.apply(lambda concept_ids: len(concept_ids)), training_data.label], axis=1).groupby('label') \
    ['concept_ids'].quantile(0.5)

training_data.groupby('label')['label'].count()

# +
from sklearn.model_selection import train_test_split

max_seq_length = 512
batch_size = 64
embedding_size = 128

token_ids = bert_tokenizer.encode(merged_dataset.concept_ids.apply(lambda concept_ids: concept_ids.tolist()))
visit_segments = merged_dataset.visit_segments
time_stamps = merged_dataset.dates
ages = ((merged_dataset['age'] - merged_dataset['age'].mean()) / merged_dataset['age'].std()).astype(float).apply(lambda c: [c]).tolist()
labels = merged_dataset.label

padded_token_ides = post_pad_pre_truncate(token_ids, bert_tokenizer.get_unused_token_id(), max_seq_length)
padded_visit_segments = post_pad_pre_truncate(visit_segments, 0, max_seq_length)
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
        'age': x['age']
    }
    return (subset, y)


lstm_training = training.map(transform)
lstm_testing = testing.map(transform)
# -

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    lstm_model = create_bi_lstm_model(max_seq_length, vocab_size=bert_tokenizer.get_vocab_size(), embedding_size=128, concept_embeddings=None)
    lstm_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=get_metrics())

history = lstm_model.fit(lstm_training, validation_data=lstm_testing, epochs=5, verbose=1)

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    bert_lstm_model = create_vanilla_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path=bert_model_path)
    bert_lstm_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=get_metrics())

bert_history = bert_lstm_model.fit(training, validation_data=testing, epochs=5, verbose=1)

temporal_bert_model = tf.keras.models.load_model(temporal_bert_model_path,custom_objects=dict(**get_custom_objects()))

strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    temp_bert_lstm_model = create_temporal_bert_bi_lstm_model(max_seq_length, temporal_bert_model_path=temporal_bert_model_path)
    temp_bert_lstm_model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=get_metrics())

temp_bert_history = temp_bert_lstm_model.fit(training, validation_data=testing, epochs=5, verbose=1)
