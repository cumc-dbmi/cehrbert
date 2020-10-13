import os

import tensorflow as tf
from tensorflow.python.keras import Model

from config.model_configs import create_time_attention_model_config
from config.parse_args import create_parse_args
from trainers.model_trainer import AbstractConceptEmbeddingTrainer
from utils.model_utils import tokenize_concepts
from data_generators.data_generator_base import TimeAttentionDataGenerator
from models.custom_layers import get_custom_objects
from models.time_attention_models import time_attention_cbow_model


class TimeAttentionConceptEmbeddingTrainer(AbstractConceptEmbeddingTrainer):

    def __init__(self,
                 tokenizer_path: str,
                 embedding_size: int,
                 context_window_size: int,
                 time_window_size: int,
                 *args, **kwargs):

        self._tokenizer_path = tokenizer_path
        self._embedding_size = embedding_size
        self._context_window_size = context_window_size
        self._time_window_size = time_window_size

        super(TimeAttentionConceptEmbeddingTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f'tokenizer_path: {tokenizer_path}\n'
            f'embedding_size: {embedding_size}\n'
            f'context_window_size: {context_window_size}\n'
            f'time_window_size: {time_window_size}\n')

    def _load_dependencies(self):
        self._tokenizer = tokenize_concepts(self._training_data, 'concept_ids', 'token_ids',
                                            self._tokenizer_path)

    def create_dataset(self):
        data_generator = TimeAttentionDataGenerator(training_data=self._training_data,
                                                    time_window_size=self._time_window_size,
                                                    max_seq_len=self._context_window_size,
                                                    concept_tokenizer=self._tokenizer,
                                                    batch_size=self._batch_size,
                                                    min_num_of_concepts=self.min_num_of_concepts)

        dataset = tf.data.Dataset.from_generator(data_generator.create_batch_generator,
                                                 output_types=data_generator.get_tf_dataset_schema()
                                                 )

        dataset = dataset.take(data_generator.get_steps_per_epoch()).cache().repeat()
        dataset = dataset.shuffle(5).prefetch(tf.data.experimental.AUTOTUNE)

        return dataset, data_generator.get_steps_per_epoch()

    def _create_model(self) -> Model:

        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info(f'Number of devices: {strategy.num_replicas_in_sync}')

        with strategy.scope():
            if os.path.exists(self._model_path):
                self.get_logger().info(
                    f'The {self} model will be loaded from {self._model_path}')
                model = tf.keras.models.load_model(self._model_path,
                                                   custom_objects=get_custom_objects())
            else:
                optimizer = tf.keras.optimizers.Adam(
                    lr=self._learning_rate, beta_1=0.9, beta_2=0.999)

                model = time_attention_cbow_model(max_seq_length=self._context_window_size,
                                                  vocabulary_size=self._tokenizer.get_vocab_size(),
                                                  concept_embedding_size=self._embedding_size,
                                                  time_window_size=self._time_window_size)
                model.compile(
                    optimizer,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                    metrics=tf.keras.metrics.sparse_categorical_accuracy)
        return model

    def eval_model(self):
        raise NotImplemented('Not implemented!')


def main(args):
    config = create_time_attention_model_config(args)
    TimeAttentionConceptEmbeddingTrainer(training_data_parquet_path=config.parquet_data_path,
                                         model_path=config.model_path,
                                         tokenizer_path=config.tokenizer_path,
                                         embedding_size=config.concept_embedding_size,
                                         context_window_size=config.max_seq_length,
                                         time_window_size=config.time_window_size,
                                         batch_size=config.batch_size, epochs=config.epochs,
                                         learning_rate=config.learning_rate,
                                         tf_board_log_path=config.tf_board_log_path).train_model()


if __name__ == "__main__":
    main(create_parse_args().parse_args())
