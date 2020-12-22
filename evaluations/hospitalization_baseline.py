import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pandas as pd
import numpy as np
import datetime
import pickle
from itertools import islice, chain

# +
import tensorflow as tf
#import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer

import keras_metrics
# -

import spark_apps.parameters as p
# from utils.utils import CosineLRSchedule
from models.evaluation_models import *
from models.bert_models import *
from models.time_attention_models import *
from models.custom_layers import get_custom_objects, TimeAttention
from keras_transformer.bert import get_custom_objects as get_custom_objects_add

from sklearn import metrics

import matplotlib.pyplot as plt


def compute_metrics(model_to_validate, val_dataset):
    step = 0
    predictions = []
    labels = []
    for next_batch in val_dataset:
        x, y = next_batch
        prediction_batch = model_to_validate.predict(x)
        predictions.extend(prediction_batch.flatten().tolist())
        labels.extend(y.numpy().tolist())
        step += 1
        if step >= (val_size // batch_size):
            break
    
    lr_precision, lr_recall, _ = metrics.precision_recall_curve(labels, np.asarray(predictions))
    
    print(f'Recall: {metrics.recall_score(labels, (np.asarray(predictions) > 0.5).astype(int), average="binary")}')
    print(f'Precision: {metrics.precision_score(labels, (np.asarray(predictions) > 0.5).astype(int), average="binary")}')
    print(f'F1 score: {metrics.f1_score(labels, (np.asarray(predictions) > 0.5).astype(int), average="binary")}')
    print(f'Precision-Recall auc: {metrics.auc(lr_recall, lr_precision)}')


input_folder = '/data/research_ops/omops/omop_2020q2/training_data_2015/'
time_attention_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_2015/time_attention'
vanilla_bert_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_2015/vanilla_bert/'
temporal_bert_model_folder = '/data/research_ops/omops/omop_2020q2/training_data_2015/temporal_bert_512_context/'

heart_failure_folder = '/data/research_ops/omops/omop_2020q2/cohorts/re_admit_rollup_frequency'

merged_dataset = pd.read_parquet(heart_failure_folder)

# +
# merged_dataset['age'] = ((merged_dataset['age'] - merged_dataset['age'].mean()) / merged_dataset['age'].std()).apply(lambda c: c)

# +
positive_cases = merged_dataset[merged_dataset['label'] == 1.0]
negative_cases = merged_dataset[merged_dataset['label'] == 0.0]

# positive_cases = positive_cases[positive_cases.concept_ids.apply(lambda concept_ids: len(concept_ids)) >= 10]
# negative_cases = negative_cases[negative_cases.concept_ids.apply(lambda concept_ids: len(concept_ids)) >= 10]
# -

training_data = pd.concat([negative_cases, positive_cases]).sample(frac=1.0)
training_data.concept_ids = training_data.concept_ids.apply(lambda concept_ids: concept_ids.tolist())
training_data.race_concept_id = training_data.race_concept_id.astype(str)
training_data.gender_concept_id = training_data.gender_concept_id.astype(str)

# +
# training_data = training_data[training_data.concept_ids.apply(lambda concept_ids: len(concept_ids)) >= 10]
# -

training_data.groupby('label')['label'].count()

len(training_data.concept_ids.explode().unique())

pd.concat([training_data.concept_ids.apply(lambda concept_ids: len(concept_ids)), training_data.label], axis=1).groupby('label') \
    ['concept_ids'].quantile(0.5)

tokenizer = Tokenizer(filters='', lower=False)

tokenizer.fit_on_texts(training_data['concept_ids'])

training_data['token_ids'] = tokenizer.texts_to_sequences(training_data['concept_ids'])

training_data = training_data.reset_index().reset_index()

training_data['row_index'] = training_data[['token_ids', 'level_0']].apply(lambda tup: [tup[1]] * len(tup[0]), axis=1)

training_data.groupby('label')['age'].hist()

from scipy.sparse import csr_matrix, vstack, hstack
from sklearn.preprocessing import normalize, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

row_index = list(chain(*training_data['row_index'].tolist()))
col_index = list(chain(*training_data['token_ids'].tolist()))
values = list(chain(*training_data['frequencies'].tolist()))

data_size = len(training_data)
vocab_size = len(tokenizer.word_index) + 1

row_index, col_index, values = zip(*sorted(zip(row_index, col_index, values), key=lambda tup: (tup[0], tup[1])))

concept_freq_count = csr_matrix((values, (row_index, col_index)), shape=(data_size, vocab_size))
normalized_concept_freq_count = normalize(concept_freq_count)

one_hot_gender_race = OneHotEncoder(handle_unknown='ignore') \
    .fit_transform(training_data[['gender_concept_id', 'race_concept_id']].to_numpy())
scaled_age = StandardScaler().fit_transform(training_data[['age']].to_numpy())

# +
# X = hstack([concept_freq_count, one_hot_gender_race, scaled_age])
X = hstack([normalized_concept_freq_count, scaled_age])

y = training_data['label'].to_numpy()

x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# +
# sparse_tensor = tf.sparse.to_dense(tf.cast(tf.SparseTensor(indices=np.asarray(list(zip(row_index, col_index))), values=values, dense_shape=[data_size, vocab_size]), dtype='float32'))

# +
# dataset = dataset.shuffle(1000).batch(128).prefetch(1)

# +
# _, input_size = X.get_shape()

# +
# model = tf.keras.Sequential([
#     tf.keras.layers.Input(shape=(None, input_size), dtype='float32'),
#     tf.keras.layers.Dense(32, activation='relu'),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# +
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy', 
#                         tf.keras.metrics.Recall(), 
#                         tf.keras.metrics.Precision(), 
#                         tf.keras.metrics.AUC(curve='PR')])

# +
# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=1024,
#     epochs=50,
#     validation_data=(x_val, y_val)
# )
# -

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.model_selection import train_test_split

logisticRegr = LogisticRegression(random_state=0, n_jobs=20, verbose=1)

# %timeit logisticRegr.fit(x_train, y_train)

predictions = logisticRegr.predict(x_val)
probabilities = logisticRegr.predict_proba(x_val)[:, 1]

lr_precision, lr_recall, _ = metrics.precision_recall_curve(y_val, probabilities)

print("Accuracy: %.2f%%" % (metrics.accuracy_score(y_val, predictions) * 100.0))
print(f'Recall: {metrics.recall_score(y_val, (np.asarray(predictions) > 0.5).astype(int), average="binary")}')
print(f'Precision: {metrics.precision_score(y_val, (np.asarray(predictions) > 0.5).astype(int), average="binary")}')
print(f'F1 score: {metrics.f1_score(y_val, (np.asarray(predictions) > 0.5).astype(int), average="binary")}')
print(f'Precision-Recall auc: {metrics.auc(lr_recall, lr_precision)}')
print(f'AUC: {metrics.roc_auc_score(y_val, probabilities)}')

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

model = XGBClassifier()

model.fit(x_train, y_train)

predictions = model.predict(x_val)
probabilities = model.predict_proba(x_val)[:, 1]

lr_precision, lr_recall, _ = metrics.precision_recall_curve(y_val, probabilities)

print("Accuracy: %.2f%%" % (accuracy_score(y_val, predictions) * 100.0))
print(f'Recall: {metrics.recall_score(y_val, predictions, average="binary")}')
print(f'Precision: {metrics.precision_score(y_val, predictions.astype(int), average="binary")}')
print(f'F1 score: {metrics.f1_score(y_val, predictions, average="binary")}')
print(f'Precision-Recall auc: {metrics.auc(lr_recall, lr_precision)}')
print(f'AUC: {metrics.roc_auc_score(y_val, probabilities)}')

fpr, tpr, _ = metrics.roc_curve(y_val, probabilities)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()




