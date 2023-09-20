import os

import tensorflow as tf
from tensorflow.keras import optimizers

from data_generators.data_generator_base import *
from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)
from models.gpt_model import create_model, ComputeMarginalDistribution
from models.layers.custom_layers import get_custom_objects
from models.model_parameters import ModelPathConfig
from models.parse_args import create_parse_args_gpt
from trainers.model_trainer import AbstractConceptEmbeddingTrainer
from utils.model_utils import tokenize_one_field


class GptModelTrainer(AbstractConceptEmbeddingTrainer):
    confidence_penalty = 0.1

    def __init__(
            self,
            tokenizer_path: str,
            concept_path: str,
            embedding_size: int,
            context_window_size: int,
            depth: int,
            num_heads: int,
            min_num_of_visits: int,
            max_num_of_visits: int,
            min_num_of_concepts: int,
            print_every: int,
            num_of_patients: int,
            sampling_batch_size: int,
            including_long_sequence: bool,
            *args, **kwargs
    ):
        self._tokenizer_path = tokenizer_path
        self._concept_path = concept_path
        self._embedding_size = embedding_size
        self._context_window_size = context_window_size
        self._depth = depth
        self._num_heads = num_heads
        self._min_num_of_concepts = min_num_of_concepts
        self._min_num_of_visits = min_num_of_visits
        self._max_num_of_visits = max_num_of_visits
        self._print_every = print_every
        self._num_of_patients = num_of_patients
        self._sampling_batch_size = sampling_batch_size
        self._including_long_sequence = including_long_sequence

        super(GptModelTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f'{self} will be trained with the following parameters:\n'
            f'tokenizer_path: {tokenizer_path}\n'
            f'concept_path: {concept_path}\n'
            f'embedding_size: {embedding_size}\n'
            f'context_window_size: {context_window_size}\n'
            f'depth: {depth}\n'
            f'num_heads: {num_heads}\n'
            f'min_num_of_visits: {min_num_of_visits}\n'
            f'max_num_of_visits: {max_num_of_visits}\n'
            f'print_every: {print_every}\n'
            f'min_num_of_concepts: {min_num_of_concepts}\n'
            f'num_of_patients:{num_of_patients}\n'
            f'sampling_batch_size: {sampling_batch_size}\n'
            f'including_long_sequence: {including_long_sequence}'
        )

    def _load_dependencies(self):
        self._tokenizer = tokenize_one_field(
            self._training_data,
            'concept_ids',
            'token_ids',
            self._tokenizer_path
        )
        self._concept_map = dict()
        concept_ids = self._tokenizer.tokenizer.word_index.keys()
        concept = pd.read_parquet(self._concept_path)
        for t in concept.itertuples():
            if str(t.concept_id) in concept_ids:
                self._concept_map[str(t.concept_id)] = t.concept_name

        print(f'Extracting demographic prompts from the training data')
        demographic_info = self._training_data.concept_ids.apply(lambda concept_list: concept_list[0:4])
        self._demographic_info = self._tokenizer.encode(map(list, demographic_info))

    def create_data_generator(self) -> GptDataGenerator:
        parameters = {
            'training_data': self._training_data,
            'batch_size': self._batch_size,
            'max_seq_len': self._context_window_size,
            'concept_tokenizer': self._tokenizer,
            'min_num_of_visits': self._min_num_of_visits,
            'max_num_of_visits': self._max_num_of_visits,
            'min_num_of_concepts': self._min_num_of_concepts,
            'including_long_sequence': self._including_long_sequence
        }

        return GptDataGenerator(**parameters)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            existing_model_path = os.path.join(self.get_model_folder(), 'bert_model.h5')
            if os.path.exists(existing_model_path):
                self.get_logger().info(
                    f'The {self} model will be loaded from {existing_model_path}')
                model = tf.keras.models.load_model(
                    existing_model_path, custom_objects=get_custom_objects())
            else:
                optimizer = optimizers.Adam(
                    lr=self._learning_rate, beta_1=0.9, beta_2=0.999)

                model = create_model(
                    context_window_size=self._context_window_size,
                    vocab_size=self._tokenizer.get_vocab_size(),
                    embedding_size=self._embedding_size,
                    num_heads=self._num_heads,
                    depth=self._depth
                )

                losses = {
                    'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                        self.confidence_penalty)
                }

                model.compile(
                    optimizer,
                    loss=losses,
                    metrics={'concept_predictions': masked_perplexity}
                )
        return model

    def eval_model(self):
        pass

    def _get_callbacks(self):
        call_backs = super()._get_callbacks()
        call_backs.append(
            ComputeMarginalDistribution(
                demographic_info=self._demographic_info,
                max_seq=self._context_window_size,
                concept_tokenizer=self._tokenizer,
                concept_map=self._concept_map,
                print_every=self._print_every,
                batch_size=self._sampling_batch_size,
                num_of_patients=self._num_of_patients
            )
        )
        return call_backs


def main(args):
    config = ModelPathConfig(args.input_folder, args.output_folder)
    GptModelTrainer(
        training_data_parquet_path=config.parquet_data_path,
        model_path=config.model_path,
        tokenizer_path=config.tokenizer_path,
        concept_path=args.concept_path,
        embedding_size=args.embedding_size,
        context_window_size=args.max_seq_length,
        depth=args.depth,
        num_heads=args.num_heads,
        min_num_of_visits=args.min_num_of_visits,
        max_num_of_visits=args.max_num_of_visits,
        min_num_of_concepts=args.min_num_of_concepts,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_dask=args.use_dask,
        tf_board_log_path=args.tf_board_log_path,
        print_every=args.print_every,
        num_of_patients=args.num_of_patients,
        sampling_batch_size=args.sampling_batch_size,
        including_long_sequence=args.including_long_sequence,
        save_checkpoint=args.save_checkpoint,
        save_freq=args.save_freq
    ).train_model()


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    main(create_parse_args_gpt().parse_args())
