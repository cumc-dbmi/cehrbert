import pickle

import numpy as np
import tensorflow as tf

from cehrbert.data_generators.learning_objective import post_pad_pre_truncate
from cehrbert.evaluations.model_evaluators.model_evaluators import get_metrics
from cehrbert.evaluations.model_evaluators.sequence_model_evaluators import SequenceModelEvaluator
from cehrbert.models.evaluation_models import (
    create_probabilistic_bert_bi_lstm_model,
    create_random_vanilla_bert_bi_lstm_model,
    create_sliding_bert_model,
    create_temporal_bert_bi_lstm_model,
    create_vanilla_bert_bi_lstm_model,
    create_vanilla_feed_forward_model,
)


class BertLstmModelEvaluator(SequenceModelEvaluator):

    def __init__(
        self,
        max_seq_length: str,
        bert_model_path: str,
        tokenizer_path: str,
        is_temporal: bool = True,
        *args,
        **kwargs,
    ):
        self._max_seq_length = max_seq_length
        self._bert_model_path = bert_model_path
        self._tokenizer = pickle.load(open(tokenizer_path, "rb"))
        self._is_temporal = is_temporal

        self.get_logger().info(
            f"max_seq_length: {max_seq_length}\n"
            f"vanilla_bert_model_path: {bert_model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
            f"is_temporal: {is_temporal}\n"
        )

        super(BertLstmModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self, **kwargs):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            create_model_fn = (
                create_temporal_bert_bi_lstm_model if self._is_temporal else create_vanilla_bert_bi_lstm_model
            )
            try:
                model = create_model_fn(self._max_seq_length, self._bert_model_path, **kwargs)
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_model_fn(self._max_seq_length, self._bert_model_path, **kwargs)

            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=get_metrics(),
            )
            return model

    def extract_model_inputs(self):
        token_ids = self._tokenizer.encode(self._dataset.concept_ids.apply(lambda concept_ids: concept_ids.tolist()))
        visit_segments = self._dataset.visit_segments
        time_stamps = self._dataset.dates
        ages = self._dataset.ages
        visit_concept_orders = self._dataset.visit_concept_orders
        # index_age = np.asarray(
        #     ((self._dataset['age'] - self._dataset['age'].mean()) / self._dataset[
        #         'age'].std()).astype(float).apply(lambda c: [c]).tolist())
        labels = self._dataset.label.to_numpy()
        padded_token_ides = post_pad_pre_truncate(
            token_ids, self._tokenizer.get_unused_token_id(), self._max_seq_length
        )
        padded_visit_segments = post_pad_pre_truncate(visit_segments, 0, self._max_seq_length)
        mask = (padded_token_ides == self._tokenizer.get_unused_token_id()).astype(int)

        padded_time_stamps = post_pad_pre_truncate(time_stamps, 0, self._max_seq_length)
        padded_ages = post_pad_pre_truncate(ages, 0, self._max_seq_length)
        padded_visit_concept_orders = post_pad_pre_truncate(
            visit_concept_orders, self._max_seq_length, self._max_seq_length
        )

        # Retrieve the values associated with the concepts, this is mostly for measurements
        padded_concept_values = post_pad_pre_truncate(
            self._dataset.concept_values, -1.0, self._max_seq_length, d_type="float32"
        )

        padded_concept_value_masks = post_pad_pre_truncate(self._dataset.concept_value_masks, 0, self._max_seq_length)

        inputs = {
            "age": np.expand_dims(self._dataset.age, axis=-1),
            "concept_ids": padded_token_ides,
            "masked_concept_ids": padded_token_ides,
            "mask": mask,
            "visit_segments": padded_visit_segments,
            "time_stamps": padded_time_stamps,
            "ages": padded_ages,
            "visit_concept_orders": padded_visit_concept_orders,
            "concept_values": padded_concept_values,
            "concept_value_masks": padded_concept_value_masks,
        }
        return inputs, labels


class ProbabilisticBertModelEvaluator(BertLstmModelEvaluator):

    def __init__(self, *args, **kwargs):
        super(ProbabilisticBertModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            try:
                model = create_probabilistic_bert_bi_lstm_model(self._max_seq_length, self._bert_model_path)
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_probabilistic_bert_bi_lstm_model(self._max_seq_length, self._bert_model_path)
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=get_metrics(),
            )
            return model


class BertFeedForwardModelEvaluator(BertLstmModelEvaluator):

    def __init__(self, *args, **kwargs):
        super(BertFeedForwardModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            try:
                model = create_vanilla_feed_forward_model((self._bert_model_path))
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_vanilla_feed_forward_model((self._bert_model_path))
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=get_metrics(),
            )
            return model


class SlidingBertModelEvaluator(BertLstmModelEvaluator):

    def __init__(self, context_window: int, stride: int, *args, **kwargs):
        self._context_window = context_window
        self._stride = stride
        super(SlidingBertModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            try:
                model = create_sliding_bert_model(
                    model_path=self._bert_model_path,
                    max_seq_length=self._max_seq_length,
                    context_window=self._context_window,
                    stride=self._stride,
                )
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_sliding_bert_model(
                    model_path=self._bert_model_path,
                    max_seq_length=self._max_seq_length,
                    context_window=self._context_window,
                    stride=self._stride,
                )
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=get_metrics(),
            )
            return model


class RandomVanillaLstmBertModelEvaluator(BertLstmModelEvaluator):

    def __init__(
        self,
        embedding_size,
        depth,
        num_heads,
        use_time_embedding,
        time_embeddings_size,
        visit_tokenizer_path,
        *args,
        **kwargs,
    ):
        self._embedding_size = embedding_size
        self._depth = depth
        self._num_heads = num_heads
        self._use_time_embedding = use_time_embedding
        self._time_embeddings_size = time_embeddings_size
        self._visit_tokenizer = pickle.load(open(visit_tokenizer_path, "rb"))
        super(RandomVanillaLstmBertModelEvaluator, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f"embedding_size: {embedding_size}\n"
            f"depth: {depth}\n"
            f"num_heads: {num_heads}\n"
            f"use_time_embedding: {use_time_embedding}\n"
            f"time_embeddings_size: {time_embeddings_size}\n"
            f"visit_tokenizer_path: {visit_tokenizer_path}\n"
        )

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():

            try:
                model = create_random_vanilla_bert_bi_lstm_model(
                    max_seq_length=self._max_seq_length,
                    embedding_size=self._embedding_size,
                    depth=self._depth,
                    tokenizer=self._tokenizer,
                    visit_tokenizer=self._visit_tokenizer,
                    num_heads=self._num_heads,
                    use_time_embedding=self._use_time_embedding,
                    time_embeddings_size=self._time_embeddings_size,
                )
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_random_vanilla_bert_bi_lstm_model(
                    max_seq_length=self._max_seq_length,
                    embedding_size=self._embedding_size,
                    depth=self._depth,
                    tokenizer=self._tokenizer,
                    visit_tokenizer=self._visit_tokenizer,
                    num_heads=self._num_heads,
                    use_time_embedding=self._use_time_embedding,
                    time_embeddings_size=self._time_embeddings_size,
                )
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=get_metrics(),
            )
            return model
