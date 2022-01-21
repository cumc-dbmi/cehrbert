from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from data_generators.learning_objective import post_pad_pre_truncate
from evaluations.model_evaluators.model_evaluators import get_metrics
from evaluations.model_evaluators.sequence_model_evaluators import SequenceModelEvaluator
from models.evaluation_models import create_cher_bert_bi_lstm_model, \
    create_cher_bert_bi_lstm_model_with_model
from models.hierachical_bert_model import transformer_hierarchical_bert_model
from utils.model_utils import *


class HierarchicalBertEvaluator(SequenceModelEvaluator):
    def __init__(self,
                 bert_model_path: str,
                 tokenizer_path: str,
                 max_num_of_visits: int,
                 max_num_of_concepts: int,
                 *args, **kwargs):
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._bert_model_path = bert_model_path
        self._tokenizer = pickle.load(open(tokenizer_path, 'rb'))

        self.get_logger().info(f'max_num_of_visits: {max_num_of_visits}\n'
                               f'max_num_of_concepts: {max_num_of_concepts}\n'
                               f'vanilla_bert_model_path: {bert_model_path}\n'
                               f'tokenizer_path: {tokenizer_path}\n')

        super(HierarchicalBertEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            try:
                model = create_cher_bert_bi_lstm_model(self._bert_model_path)
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_cher_bert_bi_lstm_model(self._bert_model_path)

            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=get_metrics())
            return model

    def _concept_mask(self, concept_ids):
        return list(
            map(lambda c: (c == self._tokenizer.get_unused_token_id()).astype(int), concept_ids))

    def _pad(self, x, padded_token):
        return pad_sequences(np.asarray(x), maxlen=self._max_num_of_concepts, padding='post',
                             value=padded_token, dtype='int32')

    def extract_model_inputs(self):
        max_seq_len = self._max_num_of_concepts * self._max_num_of_visits
        unused_token_id = self._tokenizer.get_unused_token_id()

        # Process concept ids
        token_ids = self._dataset.concept_ids \
            .apply(convert_to_list_of_lists) \
            .apply(self._tokenizer.encode) \
            .apply(lambda tokens: self._pad(tokens, padded_token=unused_token_id))
        padded_token_ids = np.reshape(
            post_pad_pre_truncate(
                token_ids.apply(lambda d: d.flatten()),
                unused_token_id,
                max_seq_len
            ),
            (-1, self._max_num_of_visits, self._max_num_of_concepts)
        )

        # Generate the concept mask
        pat_mask = (padded_token_ids == unused_token_id).astype(int)

        # Process age sequence
        ages = self._dataset.ages.apply(
            convert_to_list_of_lists
        ).apply(
            lambda tokens: self._pad(tokens, padded_token=0)
        )
        padded_ages = np.reshape(
            post_pad_pre_truncate(ages.apply(lambda d: d.flatten()), 0, max_seq_len),
            (-1, self._max_num_of_visits, self._max_num_of_concepts)
        )

        # Process time sequence
        dates = self._dataset.dates \
            .apply(convert_to_list_of_lists) \
            .apply(lambda tokens: self._pad(tokens, padded_token=0))

        padded_dates = np.reshape(
            post_pad_pre_truncate(dates.apply(lambda d: d.flatten()), 0, max_seq_len),
            (-1, self._max_num_of_visits, self._max_num_of_concepts)
        )

        # Process att tokens
        att_tokens = self._tokenizer.encode(
            self._dataset.time_interval_atts.apply(lambda t: t.tolist()).tolist())
        padded_att_tokens = post_pad_pre_truncate(
            att_tokens,
            unused_token_id,
            self._max_num_of_visits
        )[:, 1:]

        # Process visit segments
        padded_visit_segments = post_pad_pre_truncate(
            self._dataset.visit_segments,
            pad_value=0,
            max_seq_len=self._max_num_of_visits
        )

        # Process visit_rank_orders
        padded_visit_rank_orders = post_pad_pre_truncate(
            self._dataset.visit_rank_orders,
            pad_value=0,
            max_seq_len=self._max_num_of_visits
        )

        padded_visit_mask = post_pad_pre_truncate(
            self._dataset.visit_masks,
            pad_value=1,
            max_seq_len=self._max_num_of_visits
        )

        inputs = {
            'pat_seq': padded_token_ids,
            'pat_mask': pat_mask,
            'pat_seq_time': padded_dates,
            'pat_seq_age': padded_ages,
            'visit_segment': padded_visit_segments,
            'visit_rank_order': padded_visit_rank_orders,
            'visit_time_delta_att': padded_att_tokens,
            'visit_mask': padded_visit_mask,
            'age': np.expand_dims(self._dataset.age, axis=-1)
        }
        labels = self._dataset.label

        return inputs, labels


class RandomHierarchicalBertEvaluator(HierarchicalBertEvaluator):

    def __init__(self,
                 num_of_exchanges,
                 embedding_size,
                 depth,
                 num_heads,
                 use_time_embedding,
                 time_embeddings_size,
                 *args, **kwargs):
        self._num_of_exchanges = num_of_exchanges
        self._embedding_size = embedding_size
        self._depth = depth
        self._num_heads = num_heads
        self._use_time_embedding = use_time_embedding
        self._time_embeddings_size = time_embeddings_size
        super(RandomHierarchicalBertEvaluator, self).__init__(*args, **kwargs)

        self.get_logger().info(f'num_of_exchanges: {num_of_exchanges}\n'
                               f'embedding_size: {embedding_size}\n'
                               f'depth: {depth}\n'
                               f'num_heads: {num_heads}\n'
                               f'use_time_embedding: {use_time_embedding}\n'
                               f'time_embeddings_size: {time_embeddings_size}\n')

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():

            try:
                cherbert_model = transformer_hierarchical_bert_model(
                    num_of_visits=self._max_num_of_visits,
                    num_of_concepts=self._max_num_of_concepts,
                    concept_vocab_size=self._tokenizer.get_vocab_size(),
                    embedding_size=self._embedding_size,
                    depth=self._depth,
                    num_heads=self._num_heads,
                    num_of_exchanges=self._num_of_exchanges,
                    time_embeddings_size=self._time_embeddings_size
                )
                model = create_cher_bert_bi_lstm_model_with_model(
                    cherbert_model
                )
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_cher_bert_bi_lstm_model_with_model(
                    cherbert_model
                )
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=get_metrics())
            return model
