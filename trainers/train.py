import pickle

import pandas as pd
import os

from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)

from keras_transformer.bert import get_custom_objects as get_custom_objects_addition

from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

from models.bert_models_visit_prediction import *
from utils.utils import CosineLRSchedule
from models.custom_layers import get_custom_objects

from data_generators.data_generator_base import BertVisitPredictionDataGenerator
from data_generators.tokenizer import ConceptTokenizer

CONFIDENCE_PENALTY = 0.1
BERT_SPECIAL_TOKENS = ['[MASK]', '[UNUSED]']
MAX_LEN = 100
BATCH_SIZE = 128
LEARNING_RATE = 2e-4
CONCEPT_EMBEDDING = 128
EPOCH = 10


def compile_new_model():
    optimizer = optimizers.Adam(
        lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999)

    if os.path.exists(bert_model_path):
        _model = tf.keras.models.load_model(bert_model_path,
                                            custom_objects=dict(**get_custom_objects(),
                                                                **get_custom_objects_addition()))
    else:
        _model = transformer_bert_model_visit_prediction(max_seq_length=MAX_LEN,
                                                         concept_vocab_size=tokenizer.get_vocab_size(),
                                                         visit_vocab_size=visit_tokenizer.get_vocab_size(),
                                                         embedding_size=CONCEPT_EMBEDDING, depth=5,
                                                         num_heads=8)

    _model.compile(
        optimizer,
        loss={
            'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY),
            'visit_predictions': MaskedPenalizedSparseCategoricalCrossentropy(CONFIDENCE_PENALTY)
        },
        metrics={'concept_predictions': masked_perplexity})

    return _model


# +
data_folder = '/data/research_ops/omops/omop_2020q1/training_data_2019'
bert_model_folder = '/data/research_ops/omops/omop_2020q1/training_data_2019/vanilla_bert'

bert_model_path = os.path.join(bert_model_folder, 'vanilla_bert.h5')
training_data_path = os.path.join(data_folder, 'patient_sequence.pickle')
tokenizer_path = os.path.join(data_folder, 'tokenizer.pickle')
visit_tokenizer_path = os.path.join(data_folder, 'visit_tokenizer.pickle')
# -

sequence_data = pd.read_parquet(
    '/data/research_ops/omops/omop_2020q1/training_data_2019/patient_sequence')
sequence_data.concept_ids = sequence_data.concept_ids.apply(
    lambda concept_ids: concept_ids.tolist())
sequence_data.visit_concept_ids = sequence_data.visit_concept_ids.apply(
    lambda visit_concept_ids: visit_concept_ids.tolist())
sequence_data = sequence_data[sequence_data['concept_ids'].apply(len) > 1]
sequence_data.to_pickle(training_data_path)

training_data = pd.read_pickle(training_data_path)

visit_tokenizer = ConceptTokenizer(oov_token='-1')
visit_tokenizer.fit_on_concept_sequences(training_data.visit_concept_ids)
encoded_visit_concept_ids = visit_tokenizer.encode(training_data.visit_concept_ids)
training_data['visit_token_ids'] = encoded_visit_concept_ids
pickle.dump(visit_tokenizer, open(visit_tokenizer_path, 'wb'))

tokenizer = ConceptTokenizer(oov_token='0')
tokenizer.fit_on_concept_sequences(training_data.concept_ids)
encoded_sequences = tokenizer.encode(training_data.concept_ids)
training_data['token_ids'] = encoded_sequences
pickle.dump(tokenizer, open(tokenizer_path, 'wb'))

data_generator = BertVisitPredictionDataGenerator(training_data=training_data,
                                                  batch_size=BATCH_SIZE,
                                                  max_seq_len=MAX_LEN,
                                                  min_num_of_concepts=10,
                                                  concept_tokenizer=tokenizer,
                                                  visit_tokenizer=visit_tokenizer)

dataset = tf.data.Dataset.from_generator(data_generator.create_batch_generator,
                                         output_types=(data_generator.get_tf_dataset_schema()))

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    model = compile_new_model()

lr_scheduler = callbacks.LearningRateScheduler(
    CosineLRSchedule(lr_high=LEARNING_RATE, lr_low=1e-8,
                     initial_period=10),
    verbose=1)

model_callbacks = [
    callbacks.ModelCheckpoint(
        filepath=bert_model_path,
        save_best_only=True,
        verbose=1),
    lr_scheduler,
]

model.fit(
    dataset,
    steps_per_epoch=data_generator.get_steps_per_epoch() + 1,
    epochs=EPOCH,
    #     callbacks=model_callbacks,
    validation_data=dataset.shard(10, 1),
    validation_steps=10,
)
