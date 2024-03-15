import os
import tensorflow as tf
from tensorflow.keras import optimizers

from data_generators.data_generator_base import *
from data_generators.gpt_learning_objectives import CosineMaskRateScheduler
from keras_transformer.bert import (
    masked_perplexity,
    MaskedPenalizedSparseCategoricalCrossentropy,
    MaskedMeanSquaredError
)
from models.gpt_model import create_model
from models.parse_args import create_parse_args_gpt
from trainers.callbacks.gpt_callbacks import StepValidationCallBack, ComputeMarginalDistribution
from trainers.model_trainer import AbstractConceptEmbeddingTrainer
from utils.model_utils import tokenize_one_field


class GptModelTrainer(AbstractConceptEmbeddingTrainer):
    confidence_penalty = 0.1

    def __init__(
            self,
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
            sampling_dataset_enabled: bool = False,
            efficient_training: bool = False,
            is_random_cursor_long_sequence: bool = False,
            include_numeric_value: bool = False,
            low_rate: float = 0.5,
            high_rate: float = 1.0,
            period: int = 1000,
            total: int = np.infty,
            is_weighted_sample: bool = False,
            weighted_sample_scaling_factor: float = 2.0,
            num_steps: int = None,
            include_penalty: bool = False,
            *args, **kwargs
    ):
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
        self._sampling_dataset_enabled = sampling_dataset_enabled
        self._is_random_cursor_long_sequence = is_random_cursor_long_sequence
        self._include_numeric_value = include_numeric_value
        self._efficient_training = efficient_training
        self._is_weighted_sample = is_weighted_sample
        self._weighted_sample_scaling_factor = weighted_sample_scaling_factor
        self._num_steps = num_steps
        self._include_penalty = include_penalty
        self._cosine_mask_rate_scheduler = CosineMaskRateScheduler(
            low_rate=low_rate,
            high_rate=high_rate,
            period=period,
            total=total
        )

        if efficient_training:
            kwargs['shuffle_training_data'] = False

        super(GptModelTrainer, self).__init__(
            *args, **kwargs
        )

        self.get_logger().info(
            f'{self} will be trained with the following parameters:\n'
            f'model_name: {self.get_model_name()}\n'
            f'tokenizer_path: {self.get_tokenizer_path()}\n'
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
            f'including_long_sequence: {including_long_sequence}\n'
            f'sampling_dataset_enabled: {sampling_dataset_enabled}\n'
            f'is_random_cursor_long_sequence: {is_random_cursor_long_sequence}\n'
            f'include_numeric_value: {include_numeric_value}\n'
            f'low_rate: {low_rate}\n'
            f'high_rate: {high_rate}\n'
            f'period: {period}\n'
            f'total: {total}\n'
            f'efficient_training: {efficient_training}\n'
            f'is_weighted_sample: {is_weighted_sample}\n'
            f'weighted_sample_scaling_factor: {weighted_sample_scaling_factor}\n'
            f'num_steps: {num_steps}\n'
            f'include_penalty: {include_penalty}\n'
        )

    def _load_dependencies(self):
        self._tokenizer = tokenize_one_field(
            self._training_data,
            'concept_ids',
            'token_ids',
            self.get_tokenizer_path()
        )
        if self._val_data is not None:
            tokenize_one_field(
                self._val_data,
                'concept_ids',
                'token_ids',
                self.get_tokenizer_path()
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
            'including_long_sequence': self._including_long_sequence,
            'sampling_dataset_enabled': self._sampling_dataset_enabled,
            'is_random_cursor': self._is_random_cursor_long_sequence,
            'include_numeric_value': self._include_numeric_value,
            'mask_rate_scheduler': self._cosine_mask_rate_scheduler,
            'efficient_training': self._efficient_training,
            'shuffle_records': self._shuffle_records,
            'is_weighted_sample': self._is_weighted_sample,
            'weighted_sample_scaling_factor': self._weighted_sample_scaling_factor,
            'num_steps': self._num_steps
        }

        return GptDataGenerator(**parameters)

    def create_val_data_generator(self) -> GptDataGenerator:

        if self._val_data is not None:
            parameters = {
                'training_data': self._val_data,
                'batch_size': self._batch_size,
                'max_seq_len': self._context_window_size,
                'concept_tokenizer': self._tokenizer,
                'min_num_of_visits': self._min_num_of_visits,
                'max_num_of_visits': self._max_num_of_visits,
                'min_num_of_concepts': self._min_num_of_concepts,
                'including_long_sequence': self._including_long_sequence,
                'include_numeric_value': self._include_numeric_value,
                'mask_rate_scheduler': self._cosine_mask_rate_scheduler,
                'is_pretraining': False,
                'shuffle_records': False
            }
            return GptDataGenerator(**parameters)

        return None

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            if self.checkpoint_exists():
                model = self.restore_from_checkpoint()
            else:
                optimizer = optimizers.Adam(
                    lr=self._learning_rate, beta_1=0.9, beta_2=0.999)

                model = create_model(
                    context_window_size=self._context_window_size,
                    vocab_size=self._tokenizer.get_vocab_size(),
                    embedding_size=self._embedding_size,
                    num_heads=self._num_heads,
                    depth=self._depth,
                    include_numeric_value=self._include_numeric_value,
                    include_penalty=self._include_penalty
                )

                losses = {
                    'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                        self.confidence_penalty)
                }

                if self._include_numeric_value:
                    losses['next_value_predictions'] = MaskedMeanSquaredError()

                model.compile(
                    optimizer,
                    loss=losses,
                    metrics={'concept_predictions': masked_perplexity}
                )
        return model

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
        # if self._save_checkpoint:
        #     call_backs.append(
        #         StepValidationCallBack(
        #             val_data_generator=self.create_val_data_generator(),
        #             save_freq=self._save_freq,
        #             model_folder=self.get_model_folder()
        #         )
        #     )
        return call_backs

    def get_model_name(self):
        return 'CEHR_GPT'


def main(args):
    GptModelTrainer(
        training_data_parquet_path=args.training_data_parquet_path,
        val_data_parquet_path=args.val_data_parquet_path,
        model_folder=args.output_folder,
        checkpoint_name=args.checkpoint_name,
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
        save_freq=args.save_freq,
        sampling_dataset_enabled=args.sampling_dataset_enabled,
        is_random_cursor_long_sequence=args.is_random_cursor_long_sequence,
        efficient_training=args.efficient_training,
        include_numeric_value=args.include_numeric_value,
        low_rate=args.low_rate,
        high_rate=args.high_rate,
        period=args.period,
        total=args.total,
        shuffle_records=args.shuffle_records,
        is_weighted_sample=args.is_weighted_sample,
        weighted_sample_scaling_factor=args.weighted_sample_scaling_factor,
        num_steps=args.num_steps,
        include_penalty=args.include_penalty
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
