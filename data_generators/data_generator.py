# +
# # +
import random
from itertools import islice

import numpy as np

from keras.preprocessing.sequence import pad_sequences


class BatchGenerator:

    def __init__(self, patient_event_sequence,
                 unused_token_id: int,
                 max_sequence_length: int,
                 batch_size: int,
                 time_window_size: int = 100,
                 minimum_num_of_concepts: int = 5):

        self.patient_event_sequence = patient_event_sequence
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.unused_token_id = unused_token_id
        self.time_window_size = time_window_size
        self.minimum_num_of_concepts = minimum_num_of_concepts

    def batch_generator(self):
        training_example_generator = self.data_generator()
        while True:
            next_bunch_of_examples = islice(training_example_generator, self.batch_size)
            target_concepts, target_time_stamps, context_concepts, context_time_stamps, labels = zip(
                *list(next_bunch_of_examples))

            target_concepts = np.asarray(target_concepts)
            target_time_stamps = np.asarray(target_time_stamps)
            context_concepts = pad_sequences(context_concepts, maxlen=self.max_sequence_length, padding='post',
                                             value=self.unused_token_id)
            context_time_stamps = pad_sequences(context_time_stamps, maxlen=self.max_sequence_length, padding='post',
                                                value=0, dtype='float32')
            mask = (context_concepts == self.unused_token_id).astype(int)

            yield ({'target_concepts': target_concepts,
                    'target_time_stamps': target_time_stamps,
                    'context_concepts': context_concepts,
                    'context_time_stamps': context_time_stamps,
                    'mask': mask}, labels)

    def data_generator(self):

        while True:
            for tup in self.patient_event_sequence.itertuples():
                concept_ids, dates = zip(*sorted(zip(tup.token_ids, tup.dates), key=lambda tup2: tup2[1]))
                sorted_tup = (concept_ids, dates)
                for i, concept_id in enumerate(concept_ids):

                    target_concepts = [concept_id]
                    target_time_stamps = [dates[i]]

                    is_qualified, context_concepts, context_time_stamps = self.extract_concepts_time_stamps(i,
                                                                                                            sorted_tup)

                    if is_qualified:
                        yield (
                            target_concepts, target_time_stamps, context_concepts,
                            context_time_stamps, target_concepts)

    def extract_concepts_time_stamps(self, i, sorted_tup):

        concept_ids, dates = sorted_tup
        left_index, right_index = self.compute_index_bounds(i)

        sequence = np.asarray(concept_ids[left_index: i] + concept_ids[i + 1: right_index])
        time_stamp_sequence = np.asarray(dates[left_index: i] + dates[i + 1: right_index])

        qualified_indexes = self.qualified_indexes(dates[i], time_stamp_sequence)

        if len(qualified_indexes) > self.minimum_num_of_concepts:
            return True, sequence[qualified_indexes], time_stamp_sequence[qualified_indexes]

        return False, None, None

    def qualified_indexes(self, date, time_stamp_sequence):
        half_time_window = self.get_half_time_window()
        time_deltas = time_stamp_sequence - date
        qualified_indexes = np.squeeze(np.argwhere(
            (time_deltas >= -half_time_window) & (time_deltas <= half_time_window)), axis=-1)
        return qualified_indexes

    def compute_index_bounds(self, i):
        half_window_size = int(self.max_sequence_length / 2)
        left_index = i - half_window_size if i - half_window_size > 0 else 0
        right_index = i + 1 + half_window_size
        return left_index, right_index

    def get_half_time_window(self):
        return int(self.time_window_size / 2)

    def get_steps_per_epoch(self):
        return self.estimate_data_size() // self.batch_size

    def estimate_data_size(self):
        return len(self.patient_event_sequence.token_ids.explode())


class NegativeSamplingBatchGenerator(BatchGenerator):

    def __init__(self,
                 num_of_negative_samples: int,
                 negative_sample_factor: float,
                 first_token_id: int,
                 last_token_id: int,
                 *args, **kwargs):
        super(NegativeSamplingBatchGenerator, self).__init__(*args, **kwargs)
        self.num_of_negative_samples = num_of_negative_samples
        self.first_token_id = first_token_id
        self.last_token_id = last_token_id

        # build the token negative sampling probability distribution
        all_tokens = self.patient_event_sequence.token_ids.explode()
        self.token_prob_dist = np.power(all_tokens.value_counts(), negative_sample_factor)
        self.token_prob_dist = self.token_prob_dist / np.sum(self.token_prob_dist)

    def data_generator(self):
        training_example_generator = super().data_generator()
        for positive_example in training_example_generator:
            negative_example_generator = self.negative_sample_generator(positive_example)
            for next_example in negative_example_generator:
                yield next_example

    def negative_sample_generator(self, next_example):

        target_concepts, target_time_stamps, context_concepts, context_time_stamps, labels = next_example
        # Yield the positive example
        yield (target_concepts, target_time_stamps, context_concepts, context_time_stamps, [1])

        all_token_ids = list(range(self.first_token_id, self.last_token_id + 1))
        samples = set()
        while len(samples) < self.num_of_negative_samples:
            candidates = np.random.choice(all_token_ids, self.num_of_negative_samples, False, self.token_prob_dist)
            samples.update(np.setdiff1d(candidates, target_concepts + context_concepts))

        # yield the negative examples
        for negative_sample in samples:
            yield ([negative_sample], target_time_stamps, context_concepts, context_time_stamps, [0])

    def estimate_data_size(self):
        return super().estimate_data_size() * (1 + self.num_of_negative_samples)


class BertBatchGenerator(BatchGenerator):
    """
    This class generates batches for a BERT-based language model
    in an abstract way, by using an external function sampling
    sequences of token IDs of a given length.
    """

    def __init__(self,
                 mask_token_id: int,
                 first_token_id: int,
                 last_token_id: int,
                 *args, **kwargs):
        super(BertBatchGenerator, self).__init__(*args, **kwargs)
        self.mask_token_id = mask_token_id
        self.first_token_id = first_token_id
        self.last_token_id = last_token_id

    def batch_generator(self):
        training_example_generator = self.data_generator()
        while True:
            next_bunch_of_examples = islice(training_example_generator, self.batch_size)
            output_mask, sequence, masked_sequence, time_stamp_sequence, visit_order_sequence = zip(
                *list(next_bunch_of_examples))

            sequence = pad_sequences(np.asarray(sequence), maxlen=self.max_sequence_length, padding='post',
                                     value=self.unused_token_id, dtype='int32')
            masked_sequence = pad_sequences(np.asarray(masked_sequence), maxlen=self.max_sequence_length,
                                            padding='post', value=self.unused_token_id, dtype='int32')
            time_stamp_sequence = pad_sequences(np.asarray(time_stamp_sequence), maxlen=self.max_sequence_length,
                                                padding='post', value=0, dtype='int32')
            visit_order_sequence = pad_sequences(np.asarray(visit_order_sequence), maxlen=self.max_sequence_length,
                                                 padding='post', value=0, dtype='int32')
            mask = (sequence == self.unused_token_id).astype(int)
            combined_label = np.stack([sequence, output_mask], axis=-1)

            yield ({'masked_concept_ids': masked_sequence,
                    'concept_ids': sequence,
                    'time_stamps': time_stamp_sequence,
                    'visit_orders': visit_order_sequence,
                    'mask': mask}, combined_label)

    def extract_concepts_time_stamps(self, i, tup):

        concept_ids, dates, concept_id_visit_orders = zip(
            *sorted(zip(tup.token_ids, tup.dates, tup.concept_id_visit_orders), key=lambda tup2: (tup2[1],
                                                                                                  tup2[2])))
        left_index, right_index = self.compute_index_bounds(i)
        right_index -= 1

        sequence = np.asarray(concept_ids[left_index: right_index])
        time_stamp_sequence = np.asarray(dates[left_index: right_index])
        visit_order_sequence = np.asarray(concept_id_visit_orders[left_index: right_index])
        qualified_indexes = self.qualified_indexes(dates[i], time_stamp_sequence)

        if len(qualified_indexes) > self.minimum_num_of_concepts:
            sequence = sequence[qualified_indexes]
            time_stamp_sequence = time_stamp_sequence[qualified_indexes]
            visit_order_sequence = visit_order_sequence[qualified_indexes]
            return True, sequence, time_stamp_sequence, visit_order_sequence

        return False, None, None, None

    def data_generator(self):

        while True:
            for tup in self.patient_event_sequence.itertuples():

                (is_qualified, sequence, time_stamp_sequence, visit_order_sequence) = self.extract_concepts_time_stamps(
                    random.randint(0, len(tup.concept_ids) - 1), tup)

                if is_qualified:
                    masked_sequence = sequence.copy()
                    output_mask = np.zeros((self.max_sequence_length,), dtype=int)

                    for word_pos in range(0, len(sequence)):
                        if sequence[word_pos] == self.unused_token_id:
                            break

                        if random.random() < 0.15:
                            dice = random.random()
                            if dice < 0.8:
                                masked_sequence[word_pos] = self.mask_token_id
                            elif dice < 0.9:
                                masked_sequence[word_pos] = random.randint(
                                    self.first_token_id, self.last_token_id)
                            # else: 10% of the time we just leave the word as is
                            output_mask[word_pos] = 1

                    yield (output_mask, sequence, masked_sequence, time_stamp_sequence, visit_order_sequence)

    def estimate_data_size(self):
        return len(self.patient_event_sequence)


class BertFineTuningBatchGenerator(BertBatchGenerator):

    def batch_generator(self):
        training_example_generator = self.data_generator()
        while True:
            sequence, masked_sequence, time_stamp_sequence, visit_order_sequence, labels = zip(
                *list(islice(training_example_generator, self.batch_size)))

            sequence = pad_sequences(np.asarray(sequence), maxlen=self.max_sequence_length, padding='post',
                                     value=self.unused_token_id, dtype='int32')
            masked_sequence = pad_sequences(np.asarray(masked_sequence), maxlen=self.max_sequence_length,
                                            padding='post', value=self.unused_token_id, dtype='int32')
            time_stamp_sequence = pad_sequences(np.asarray(time_stamp_sequence), maxlen=self.max_sequence_length,
                                                padding='post', value=0, dtype='int32')
            visit_order_sequence = pad_sequences(np.asarray(visit_order_sequence), maxlen=self.max_sequence_length,
                                                 padding='post', value=0, dtype='int32')
            mask = (sequence == self.unused_token_id).astype(int)

            yield ({'masked_concept_ids': masked_sequence,
                    'concept_ids': sequence,
                    'time_stamps': time_stamp_sequence,
                    'visit_orders': visit_order_sequence,
                    'mask': mask}, labels)

    def data_generator(self):
        while True:
            for tup in self.patient_event_sequence.itertuples():
                concept_ids, dates, concept_id_visit_orders = zip(
                    *sorted(zip(tup.token_ids, tup.dates, tup.concept_id_visit_orders), key=lambda tup2: (tup2[1],
                                                                                                          tup2[2])))
                yield (concept_ids, concept_ids, dates, concept_id_visit_orders, tup.labels)
