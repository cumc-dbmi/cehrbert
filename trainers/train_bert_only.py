import os

import tensorflow as tf

from config.model_configs import create_bert_model_config
from config.parse_args import create_parse_args_base_bert
from trainers.model_trainer import AbstractConceptEmbeddingTrainer
from utils.model_utils import tokenize_concepts
from models.bert_models_visit_prediction import transformer_bert_model_visit_prediction
from models.bert_models import transformer_bert_model
from models.custom_layers import get_custom_objects
from data_generators.data_generator_base import *

from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)

from tensorflow.keras import optimizers


class VanillaBertTrainer(AbstractConceptEmbeddingTrainer):
    confidence_penalty = 0.1

    def __init__(self,
                 tokenizer_path: str,
                 visit_tokenizer_path: str,
                 embedding_size: int,
                 context_window_size: int,
                 depth: int,
                 num_heads: int,
                 include_visit_prediction: bool,
                 *args, **kwargs):

        self._tokenizer_path = tokenizer_path
        self._visit_tokenizer_path = visit_tokenizer_path
        self._embedding_size = embedding_size
        self._context_window_size = context_window_size
        self._depth = depth
        self._num_heads = num_heads
        self._include_visit_prediction = include_visit_prediction

        super(VanillaBertTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f'{self} will be trained with the following parameters:\n'
            f'tokenizer_path: {tokenizer_path}\n'
            f'visit_tokenizer_path: {visit_tokenizer_path}\n'
            f'embedding_size: {embedding_size}\n'
            f'context_window_size: {context_window_size}\n'
            f'depth: {depth}\n'
            f'num_heads: {num_heads}\n'
            f'include_visit_prediction: {include_visit_prediction}\n')

    def _load_dependencies(self):

        self._tokenizer = tokenize_concepts(self._training_data, 'concept_ids', 'token_ids',
                                            self._tokenizer_path)
        self._visit_tokenizer = tokenize_concepts(self._training_data, 'visit_concept_ids',
                                                  'visit_token_ids', self._visit_tokenizer_path,
                                                  oov_token='-1')

    def create_dataset(self):

        if self._include_visit_prediction:
            data_generator = BertVisitPredictionDataGenerator(training_data=self._training_data,
                                                              batch_size=self._batch_size,
                                                              max_seq_len=self._context_window_size,
                                                              min_num_of_concepts=self.min_num_of_concepts,
                                                              concept_tokenizer=self._tokenizer,
                                                              visit_tokenizer=self._visit_tokenizer)
        else:
            data_generator = GenericBertDataGenerator(training_data=self._training_data,
                                                      batch_size=self._batch_size,
                                                      max_seq_len=self._context_window_size,
                                                      min_num_of_concepts=self.min_num_of_concepts,
                                                      concept_tokenizer=self._tokenizer,
                                                      visit_tokenizer=self._visit_tokenizer)

            learning_objectives = [
                MaskedLanguageModelLearningObjective(concept_tokenizer=self._tokenizer,
                                                     max_seq_len=self._context_window_size),
            ]

            data_generator.set_learning_objectives(learning_objectives)

        dataset = tf.data.Dataset.from_generator(data_generator.create_batch_generator,
                                                 output_types=(
                                                     data_generator.get_tf_dataset_schema()))

        return dataset, data_generator.get_steps_per_epoch()

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            if os.path.exists(self._model_path):
                self.get_logger().info(
                    f'The {self} model will be loaded from {self._model_path}')
                model = tf.keras.models.load_model(self._model_path,
                                                   custom_objects=get_custom_objects())
            else:
                optimizer = optimizers.Adam(
                    lr=self._learning_rate, beta_1=0.9, beta_2=0.999)

                if self._include_visit_prediction:
                    model = transformer_bert_model_visit_prediction(
                        max_seq_length=self._context_window_size,
                        concept_vocab_size=self._tokenizer.get_vocab_size(),
                        visit_vocab_size=self._visit_tokenizer.get_vocab_size(),
                        embedding_size=self._embedding_size,
                        depth=self._depth,
                        num_heads=self._num_heads)

                    losses = {
                        'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty),
                        'visit_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty)
                    }
                else:
                    model = transformer_bert_model(
                        max_seq_length=self._context_window_size,
                        vocabulary_size=self._tokenizer.get_vocab_size(),
                        concept_embedding_size=self._embedding_size,
                        depth=self._depth,
                        num_heads=self._num_heads)

                    losses = {
                        'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty)
                    }

                model.compile(optimizer, loss=losses,
                              metrics={'concept_predictions': masked_perplexity})
        return model

    def eval_model(self):
        pass


def main(args):
    config = create_bert_model_config(args)
    VanillaBertTrainer(training_data_parquet_path=config.parquet_data_path,
                       model_path=config.model_path,
                       tokenizer_path=config.tokenizer_path,
                       visit_tokenizer_path=config.visit_tokenizer_path,
                       embedding_size=config.concept_embedding_size,
                       context_window_size=config.max_seq_length,
                       depth=config.depth,
                       num_heads=config.num_heads,
                       include_visit_prediction=False,
                       batch_size=config.batch_size,
                       epochs=config.epochs,
                       learning_rate=config.learning_rate,
                       tf_board_log_path=config.tf_board_log_path).train_model()


if __name__ == "__main__":
    main(create_parse_args_base_bert().parse_args())
