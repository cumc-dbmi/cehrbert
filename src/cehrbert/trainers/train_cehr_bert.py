import tensorflow as tf
from tensorflow.keras import optimizers

from cehrbert.data_generators.data_generator_base import (
    BertDataGenerator,
    BertVisitPredictionDataGenerator,
    MedBertDataGenerator,
)
from cehrbert.keras_transformer.bert import MaskedPenalizedSparseCategoricalCrossentropy, masked_perplexity
from cehrbert.models.bert_models import transformer_bert_model
from cehrbert.models.bert_models_visit_prediction import transformer_bert_model_visit_prediction
from cehrbert.models.parse_args import create_parse_args_base_bert
from cehrbert.trainers.model_trainer import AbstractConceptEmbeddingTrainer
from cehrbert.utils.model_utils import tokenize_one_field


class VanillaBertTrainer(AbstractConceptEmbeddingTrainer):
    confidence_penalty = 0.1

    def __init__(
        self,
        embedding_size: int,
        context_window_size: int,
        depth: int,
        num_heads: int,
        include_visit_prediction: bool,
        include_prolonged_length_stay: bool,
        use_time_embedding: bool,
        use_behrt: bool,
        time_embeddings_size: int,
        *args,
        **kwargs,
    ):
        self._embedding_size = embedding_size
        self._context_window_size = context_window_size
        self._depth = depth
        self._num_heads = num_heads
        self._include_visit_prediction = include_visit_prediction
        self._include_prolonged_length_stay = include_prolonged_length_stay
        self._use_time_embedding = use_time_embedding
        self._use_behrt = use_behrt
        self._time_embeddings_size = time_embeddings_size

        super(VanillaBertTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f"{self} will be trained with the following parameters:\n"
            f"model_name: {self.get_model_name()}\n"
            f"tokenizer_path: {self.get_tokenizer_path()}\n"
            f"visit_tokenizer_path: {self.get_visit_tokenizer_path()}\n"
            f"embedding_size: {embedding_size}\n"
            f"context_window_size: {context_window_size}\n"
            f"depth: {depth}\n"
            f"num_heads: {num_heads}\n"
            f"include_visit_prediction: {include_visit_prediction}\n"
            f"include_prolonged_length_stay: {include_prolonged_length_stay}\n"
            f"use_time_embeddings: {use_time_embedding}\n"
            f"use_behrt: {use_behrt}\n"
            f"time_embeddings_size: {time_embeddings_size}"
        )

    def _load_dependencies(self):

        self._tokenizer = tokenize_one_field(self._training_data, "concept_ids", "token_ids", self.get_tokenizer_path())

        if self._include_visit_prediction:
            self._visit_tokenizer = tokenize_one_field(
                self._training_data,
                "visit_concept_ids",
                "visit_token_ids",
                self.get_visit_tokenizer_path(),
                oov_token="-1",
            )

    def create_data_generator(self) -> BertDataGenerator:

        parameters = {
            "training_data": self._training_data,
            "batch_size": self._batch_size,
            "max_seq_len": self._context_window_size,
            "min_num_of_concepts": self.min_num_of_concepts,
            "concept_tokenizer": self._tokenizer,
            "is_random_cursor": True,
        }

        data_generator_class = BertDataGenerator

        if self._include_visit_prediction:
            parameters["visit_tokenizer"] = self._visit_tokenizer
            data_generator_class = BertVisitPredictionDataGenerator
        elif self._include_prolonged_length_stay:
            data_generator_class = MedBertDataGenerator

        return data_generator_class(**parameters)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            if self.checkpoint_exists():
                model = self.restore_from_checkpoint()
            else:
                optimizer = optimizers.Adam(lr=self._learning_rate, beta_1=0.9, beta_2=0.999)

                if self._include_visit_prediction:
                    model = transformer_bert_model_visit_prediction(
                        max_seq_length=self._context_window_size,
                        concept_vocab_size=self._tokenizer.get_vocab_size(),
                        visit_vocab_size=self._visit_tokenizer.get_vocab_size(),
                        embedding_size=self._embedding_size,
                        depth=self._depth,
                        num_heads=self._num_heads,
                        use_time_embedding=self._use_time_embedding,
                        time_embeddings_size=self._time_embeddings_size,
                    )

                    losses = {
                        "concept_predictions": MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty),
                        "visit_predictions": MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty),
                    }
                else:
                    model = transformer_bert_model(
                        max_seq_length=self._context_window_size,
                        vocab_size=self._tokenizer.get_vocab_size(),
                        embedding_size=self._embedding_size,
                        depth=self._depth,
                        num_heads=self._num_heads,
                        use_time_embedding=self._use_time_embedding,
                        time_embeddings_size=self._time_embeddings_size,
                        use_behrt=self._use_behrt,
                        include_prolonged_length_stay=self._include_prolonged_length_stay,
                    )

                    losses = {
                        "concept_predictions": MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty)
                    }

                    if self._include_prolonged_length_stay:
                        losses["prolonged_length_stay"] = tf.losses.BinaryCrossentropy()

                model.compile(
                    optimizer,
                    loss=losses,
                    metrics={"concept_predictions": masked_perplexity},
                )
        return model

    def get_model_name(self):
        return "CEHR_BERT"

    def get_model_config(self):
        model_config = super().get_model_config()
        if self._include_visit_prediction:
            model_config["visit_tokenizer"] = self.get_visit_tokenizer_name()
        return model_config


def main(args):
    VanillaBertTrainer(
        training_data_parquet_path=args.training_data_parquet_path,
        model_folder=args.output_folder,
        checkpoint_name=args.checkpoint_name,
        embedding_size=args.embedding_size,
        context_window_size=args.max_seq_length,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        include_visit_prediction=args.include_visit_prediction,
        include_prolonged_length_stay=args.include_prolonged_length_stay,
        use_time_embedding=args.use_time_embedding,
        time_embeddings_size=args.time_embeddings_size,
        use_behrt=args.use_behrt,
        use_dask=args.use_dask,
        tf_board_log_path=args.tf_board_log_path,
    ).train_model()


if __name__ == "__main__":
    main(create_parse_args_base_bert().parse_args())
