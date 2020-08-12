import os
import pickle

import pandas as pd
import tensorflow as tf

from config.model_configs import Config, create_time_attention_model_config
from config.parse_args import create_parse_args
from data_generators.data_generator import BatchGenerator
from data_generators.tokenizer import ConceptTokenizer
from models.custom_layers import get_custom_objects
from models.time_attention_models import time_attention_cbow_model
from utils.utils import CosineLRSchedule


class Trainer:
    def __init__(self, config: Config):

        self.parquet_data_path = config.parquet_data_path
        self.feather_data_path = config.feather_data_path
        self.tokenizer_path = config.tokenizer_path
        self.model_path = config.model_path

        self.concept_embedding_size = config.concept_embedding_size
        self.max_seq_length = config.max_seq_length
        self.time_window_size = config.time_window_size
        self.batch_size = config.batch_size
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate
        self.tf_board_log_path = config.tf_board_log_path

    def run(self):

        # shuffle the training data
        shuffled_sequence_data = self.create_if_not_exists().sample(frac=1).reset_index(drop=True)

        tokenizer, shuffled_sequence_data = self.tokenize_concept_sequences(shuffled_sequence_data)

        dataset, steps_per_epoch = self.create_tf_dataset(shuffled_sequence_data, tokenizer)

        self.train(vocabulary_size=tokenizer.get_vocab_size(),
                   dataset=dataset,
                   val_dataset=dataset.shard(10, 1),
                   steps_per_epoch=steps_per_epoch + 1,
                   val_steps_per_epoch=100)

    def create_if_not_exists(self):
        """
        Process the raw input data
        :return: save and return the training dataset
        """
        if not os.path.exists(self.feather_data_path):
            sequence_data = pd.read_parquet(self.parquet_data_path)
            sequence_data.concept_ids = sequence_data.concept_ids.apply(lambda concept_ids: concept_ids.tolist())
            sequence_data = sequence_data[sequence_data['concept_ids'].apply(len) > 1]
            sequence_data.to_pickle(self.feather_data_path)
        return pd.read_pickle(self.feather_data_path)

    def create_tf_dataset(self, sequence_data, tokenizer):
        """
        Create a tensorflow dataset
        :param sequence_data:
        :param tokenizer:
        :return:
        """

        unused_token_id = tokenizer.get_unused_token_id()
        batch_generator = BatchGenerator(patient_event_sequence=sequence_data,
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
    Trainer(create_time_attention_model_config(args)).run()


if __name__ == "__main__":
    main(create_parse_args().parse_args())
