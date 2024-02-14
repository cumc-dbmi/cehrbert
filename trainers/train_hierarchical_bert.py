import os

import tensorflow as tf
from tensorflow.keras import optimizers

from data_generators.data_classes import TokenizeFieldInfo
from data_generators.data_generator_base import *
from keras_transformer.bert import (
    masked_perplexity, MaskedPenalizedSparseCategoricalCrossentropy, SequenceCrossentropy)
from models.hierachical_bert_model_v2 import transformer_hierarchical_bert_model
from models.layers.custom_layers import get_custom_objects
from models.parse_args import create_parse_args_hierarchical_bert
from trainers.model_trainer import AbstractConceptEmbeddingTrainer
from utils.model_utils import tokenize_one_field, tokenize_multiple_fields


class HierarchicalBertTrainer(AbstractConceptEmbeddingTrainer):
    confidence_penalty = 0.1

    def __init__(
            self,
            concept_similarity_path: str,
            concept_similarity_type: str,
            embedding_size: int,
            depth: int,
            max_num_visits: int,
            max_num_concepts: int,
            min_num_of_concepts: int,
            min_num_of_visits: int,
            num_heads: int,
            time_embeddings_size: int,
            include_att_prediction: bool,
            include_visit_prediction: bool,
            include_readmission: bool,
            include_prolonged_length_stay: bool,
            random_mask_prob: float = 0.0,
            warmup_step: int = -1,
            *args, **kwargs
    ):
        self._concept_similarity_path = concept_similarity_path
        self._concept_similarity_type = concept_similarity_type
        self._embedding_size = embedding_size
        self._depth = depth
        self._max_num_visits = max_num_visits
        self._max_num_concepts = max_num_concepts
        self._min_num_of_concepts = min_num_of_concepts
        self._min_num_of_visits = min_num_of_visits
        self._num_heads = num_heads
        self._time_embeddings_size = time_embeddings_size
        self._include_att_prediction = include_att_prediction
        self._include_visit_prediction = include_visit_prediction
        self._include_readmission = include_readmission
        self._include_prolonged_length_stay = include_prolonged_length_stay
        self._random_mask_prob = random_mask_prob
        self._warmup_step = warmup_step

        super(HierarchicalBertTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f'{self} will be trained with the following parameters:\n'
            f'model_name: {self.get_model_name()}\n'
            f'tokenizer_path: {self.get_tokenizer_path()}\n'
            f'visit_tokenizer_path: {self.get_visit_tokenizer_path()}\n'
            f'concept_similarity_table: {concept_similarity_path}\n'
            f'concept_similarity_type: {concept_similarity_type}\n'
            f'embedding_size: {embedding_size}\n'
            f'depth: {depth}\n'
            f'max_num_visits: {max_num_visits}\n'
            f'max_num_concepts: {max_num_concepts}\n'
            f'min_num_of_visits: {min_num_of_visits}\n'
            f'min_num_of_concepts: {min_num_of_concepts}\n'
            f'num_heads: {num_heads}\n'
            f'time_embeddings_size: {time_embeddings_size}\n'
            f'include_att_prediction: {include_att_prediction}\n'
            f'include_visit_prediction: {include_visit_prediction}\n'
            f'include_prolonged_length_stay: {include_prolonged_length_stay}\n'
            f'include_readmission: {include_readmission}\n'
            f'random_mask_prob: {random_mask_prob}\n'
            f'warmup_step: {warmup_step}\n'
        )

    def _load_dependencies(self):

        self._training_data['patient_concept_ids'] = self._training_data.concept_ids \
            .apply(lambda visit_concepts: np.hstack(visit_concepts))
        tokenize_fields_info = [
            TokenizeFieldInfo(column_name='patient_concept_ids'),
            TokenizeFieldInfo(column_name='time_interval_atts')
        ]
        self._tokenizer = tokenize_multiple_fields(
            self._training_data,
            tokenize_fields_info,
            self.get_tokenizer_path(),
            encode=False
        )
        self._visit_tokenizer = tokenize_one_field(
            self._training_data,
            'visit_concept_ids',
            'visit_token_ids',
            self.get_visit_tokenizer_path()
        )

    def create_data_generator(self) -> HierarchicalBertDataGenerator:

        parameters = {
            'training_data': self._training_data,
            'concept_tokenizer': self._tokenizer,
            'visit_tokenizer': self._visit_tokenizer,
            'batch_size': self._batch_size,
            'max_num_of_visits': self._max_num_visits,
            'max_num_of_concepts': self._max_num_concepts,
            'include_att_prediction': self._include_att_prediction,
            'include_visit_prediction': self._include_visit_prediction,
            'concept_similarity_path': self._concept_similarity_path,
            'concept_similarity_type': self._concept_similarity_type,
            'min_num_of_concepts': self._min_num_of_concepts,
            'min_num_of_visits': self._min_num_of_visits,
            'warmup_step': self._warmup_step
        }

        data_generator_class = HierarchicalBertDataGenerator

        if self.has_secondary_learning_objectives():
            # parameters['visit_tokenizer'] = self._visit_tokenizer
            parameters.update({
                'include_readmission': self._include_readmission,
                'include_prolonged_length_stay': self._include_prolonged_length_stay,
                'random_mask_prob': self._random_mask_prob
            })
            data_generator_class = HierarchicalBertMultiTaskDataGenerator

        return data_generator_class(**parameters)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(
            strategy.num_replicas_in_sync))
        with strategy.scope():
            if self.checkpoint_exists():
                model = self.restore_from_checkpoint()
            else:
                optimizer = optimizers.Adam(
                    lr=self._learning_rate,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-5,
                    clipnorm=1.0
                )
                visit_vocab_size = (
                    self._visit_tokenizer.get_vocab_size() if
                    self._include_visit_prediction else None
                )

                model = transformer_hierarchical_bert_model(
                    num_of_visits=self._max_num_visits,
                    num_of_concepts=self._max_num_concepts,
                    concept_vocab_size=self._tokenizer.get_vocab_size(),
                    visit_vocab_size=visit_vocab_size,
                    embedding_size=self._embedding_size,
                    depth=self._depth,
                    num_heads=self._num_heads,
                    time_embeddings_size=self._time_embeddings_size,
                    include_att_prediction=self._include_att_prediction,
                    include_visit_type_prediction=self._include_visit_prediction,
                    include_readmission=self._include_readmission,
                    include_prolonged_length_stay=self._include_prolonged_length_stay
                )

                losses = {
                    'concept_predictions':
                        MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty)
                }

                if self._include_att_prediction:
                    losses['att_predictions'] = (
                        MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty)
                    )

                if self._include_visit_prediction:
                    losses['visit_predictions'] = (
                        MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty)
                    )

                if self._include_readmission:
                    losses['is_readmission'] = SequenceCrossentropy()

                if self._include_prolonged_length_stay:
                    losses['visit_prolonged_stay'] = SequenceCrossentropy()

                model.compile(
                    optimizer,
                    loss=losses,
                    metrics={'concept_predictions': masked_perplexity}
                )
        return model

    def has_secondary_learning_objectives(self):
        return (
                self._include_visit_prediction
                or self._include_readmission
                or self._include_prolonged_length_stay
        )

    def get_model_name(self):
        return 'HIERARCHICAL_CEHR_BERT'

    def get_model_config(self):
        model_config = super().get_model_config()
        model_config['visit_tokenizer'] = self.get_visit_tokenizer_name()
        return model_config


def main(args):
    HierarchicalBertTrainer(
        training_data_parquet_path=args.training_data_parquet_path,
        model_folder=args.output_folder,
        checkpoint_name=args.checkpoint_name,
        concept_similarity_path=args.concept_similarity_path,
        concept_similarity_type=args.concept_similarity_type,
        embedding_size=args.embedding_size,
        depth=args.depth,
        max_num_visits=args.max_num_visits,
        max_num_concepts=args.max_num_concepts,
        min_num_of_visits=args.min_num_of_visits,
        min_num_of_concepts=args.min_num_of_concepts,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        include_att_prediction=args.include_att_prediction,
        include_visit_prediction=args.include_visit_prediction,
        include_prolonged_length_stay=args.include_prolonged_length_stay,
        include_readmission=args.include_readmission,
        random_mask_prob=args.random_mask_prob,
        warmup_step=args.warmup_step,
        time_embeddings_size=args.time_embeddings_size,
        use_dask=args.use_dask,
        tf_board_log_path=args.tf_board_log_path
    ).train_model()


if __name__ == "__main__":
    main(create_parse_args_hierarchical_bert().parse_args())
