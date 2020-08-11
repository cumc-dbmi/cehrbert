import argparse

import os
import itertools
import pickle

import pandas as pd

from models.custom_layers import get_custom_objects
from data_generators.data_generator import BatchGenerator
from data_generators.tokenizer import ConceptTokenizer
from models.time_attention_models import time_attention_cbow_model
from utils.utils import CosineLRSchedule

import tensorflow as tf


class Trainer:
    def __init__(self,
                 input_folder,
                 output_folder,
                 concept_embedding_size,
                 max_seq_length,
                 time_window_size,
                 batch_size,
                 epochs,
                 learning_rate,
                 tf_board_log_path):

        self.input_folder = input_folder
        self.output_folder = output_folder
        self.concept_embedding_size = concept_embedding_size
        self.max_seq_length = max_seq_length
        self.time_window_size = time_window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tf_board_log_path = tf_board_log_path

        self.raw_input_data_path = os.path.join(self.input_folder, 'visit_event_sequence_v2')
        self.training_data_path = os.path.join(self.input_folder, 'patient_event_sequence.pickle')
        self.tokenizer_path = os.path.join(self.output_folder, 'tokenizer.pickle')
        self.model_path = os.path.join(self.output_folder, 'model_time_aware_embeddings.h5')

    def run(self):

        training_data = self.process_raw_input()
        # shuffle the training data
        training_data = training_data.sample(frac=1).reset_index(drop=True)

        tokenizer, training_data = self.tokenize_concept_sequences(training_data)

        dataset, steps_per_epoch = self.create_tf_dataset(tokenizer, training_data)

        self.train(vocabulary_size=tokenizer.get_vocab_size(),
                   dataset=dataset,
                   val_dataset=dataset.shard(10, 1),
                   steps_per_epoch=steps_per_epoch + 1,
                   val_steps_per_epoch=100)

    def create_tf_dataset(self, tokenizer, training_data):

        unused_token_id = tokenizer.get_unused_token_id()
        batch_generator = BatchGenerator(patient_event_sequence=training_data,
                                         max_sequence_length=self.max_seq_length,
                                         batch_size=self.batch_size,
                                         unused_token_id=unused_token_id)
        dataset = tf.data.Dataset.from_generator(batch_generator.batch_generator,
                                                 output_types=({'target_concepts': tf.int32,
                                                                'target_time_stamps': tf.float32,
                                                                'context_concepts': tf.int32,
                                                                'context_time_stamps': tf.float32,
                                                                'mask': tf.int32}, tf.int32))
        dataset = dataset.take(batch_generator.get_steps_per_epoch()).cache().repeat()
        dataset = dataset.shuffle(5).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset, batch_generator.get_steps_per_epoch()

    def process_raw_input(self):
        """
        Process the raw input data
        :return: save and return the training dataset
        """
        if os.path.exists(self.training_data_path):
            training_data = pd.read_pickle(self.training_data_path)
        else:
            visit_concepts = pd.read_parquet(self.raw_input_data_path)
            visit_concepts['concept_id_visit_orders'] = visit_concepts['concept_ids'].apply(len) * visit_concepts[
                'visit_rank_order'].apply(
                lambda v: [v])

            patient_concept_ids = visit_concepts.sort_values(['person_id', 'visit_rank_order']) \
                .groupby('person_id')['concept_ids'].apply(lambda x: list(itertools.chain(*x))).reset_index()

            patient_visit_ids = visit_concepts.sort_values(['person_id', 'visit_rank_order']) \
                .groupby('person_id')['concept_id_visit_orders'].apply(
                lambda x: list(itertools.chain(*x))).reset_index()

            event_dates = visit_concepts.sort_values(['person_id', 'visit_rank_order']) \
                .groupby('person_id')['dates'].apply(lambda x: list(itertools.chain(*x))).reset_index()

            concept_positions = visit_concepts.sort_values(['person_id', 'visit_rank_order']) \
                .groupby('person_id')['concept_positions'].apply(lambda x: list(itertools.chain(*x))).reset_index()

            training_data = patient_concept_ids.merge(patient_visit_ids).merge(event_dates).merge(concept_positions)
            training_data = training_data[training_data['concept_ids'].apply(len) > 1]
            training_data.to_pickle(self.training_data_path)

        return training_data

    def tokenize_concept_sequences(self, training_data):
        """

        :param training_data:
        :return:
        """
        tokenizer = ConceptTokenizer()
        tokenizer.fit_on_concept_sequences(training_data.concept_ids)
        encoded_sequences = tokenizer.encode(training_data.concept_ids)
        training_data['token_ids'] = encoded_sequences
        pickle.dump(tokenizer, open(self.tokenizer_path, 'wb'))
        return tokenizer, training_data

    def train(self, vocabulary_size, dataset, val_dataset, steps_per_epoch, val_steps_per_epoch):
        """

        :param vocabulary_size:
        :param dataset:
        :param val_dataset:
        :param steps_per_epoch:
        :param val_steps_per_epoch:
        :return:
        """
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

        with strategy.scope():
            if os.path.exists(self.model_path):
                model = tf.keras.models.load_model(self.model_path, custom_objects=get_custom_objects())
            else:
                optimizer = tf.keras.optimizers.Adam(
                    lr=self.learning_rate, beta_1=0.9, beta_2=0.999)

                model = time_attention_cbow_model(max_seq_length=self.max_seq_length,
                                                  vocabulary_size=vocabulary_size,
                                                  concept_embedding_size=self.concept_embedding_size,
                                                  time_window_size=self.time_window_size)
                model.compile(
                    optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=tf.keras.metrics.sparse_categorical_accuracy)

        model_callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.tf_board_log_path),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self.model_path,
                save_best_only=True,
                verbose=1),
            tf.keras.callbacks.LearningRateScheduler(
                CosineLRSchedule(
                    lr_high=self.learning_rate,
                    lr_low=1e-8,
                    initial_period=10), verbose=1)
        ]

        model.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            callbacks=model_callbacks,
            validation_data=val_dataset,
            validation_steps=val_steps_per_epoch
        )


def main(args):
    trainer = Trainer(input_folder=args.input_folder,
                      output_folder=args.output_folder,
                      concept_embedding_size=args.concept_embedding_size,
                      max_seq_length=args.max_seq_length,
                      time_window_size=args.time_window_size,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      tf_board_log_path=args.tf_board_log_path)

    trainer.run()


def create_parse_args():
    parser = argparse.ArgumentParser(description='Arguments for concept embedding model')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the raw data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The output folder that stores the domain tables download destination',
                        required=True)
    parser.add_argument('-m',
                        '--max_seq_length',
                        dest='max_seq_length',
                        action='store',
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument('-t',
                        '--time_window_size',
                        dest='time_window_size',
                        action='store',
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument('-c',
                        '--concept_embedding_size',
                        dest='concept_embedding_size',
                        action='store',
                        type=int,
                        default=128,
                        required=False)
    parser.add_argument('-e',
                        '--epochs',
                        dest='epochs',
                        action='store',
                        type=int,
                        default=50,
                        required=False)
    parser.add_argument('-b',
                        '--batch_size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=128,
                        required=False)
    parser.add_argument('-lr',
                        '--learning_rate',
                        dest='learning_rate',
                        action='store',
                        type=float,
                        default=2e-4,
                        required=False)
    parser.add_argument('-bl',
                        '--tf_board_log_path',
                        dest='tf_board_log_path',
                        action='store',
                        default='./logs',
                        required=False)
    return parser


if __name__ == "__main__":
    main(create_parse_args().parse_args())
