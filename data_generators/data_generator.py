# +
# # +
import random
from itertools import islice

import numpy as np

from keras.preprocessing.sequence import pad_sequences


class TimeAttentionDataGenerator:

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
        training_data_generator = self.data_generator()
        while True:
            next_bunch = islice(training_data_generator, self.batch_size)
            target_concepts, target_time_stamps, context_concepts, context_time_stamps, labels = zip(
                *list(next_bunch))

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
                    time_window_qualified_indexes = self.get_time_window_qualified_indexes(i, dates)
                    if len(time_window_qualified_indexes) > self.minimum_num_of_concepts:
                        context_concepts, context_time_stamps = self.generate_inputs(
                            i, sorted_tup, time_window_qualified_indexes)

                        yield (
                            [concept_id], [dates[i]], context_concepts,
                            context_time_stamps, [concept_id])

    def generate_inputs(self, i, sorted_tup, qualified_indexes):

        concept_ids, dates = sorted_tup
        sequence = self.get_inputs_for_index(concept_ids, i)
        time_stamp_sequence = self.get_inputs_for_index(dates, i)
        return sequence[qualified_indexes], time_stamp_sequence[qualified_indexes]

    def get_time_window_qualified_indexes(self, i, dates):

        time_stamps = self.get_inputs_for_index(dates, i)
        half_time_window = self.get_half_time_window()
        time_deltas = time_stamps - dates[i]
        return np.squeeze(np.argwhere(
            (time_deltas >= -half_time_window) & (time_deltas <= half_time_window)), axis=-1)

    def get_inputs_for_index(self, inputs, i):
        left_index, right_index = self.compute_index_range(i)
        time_stamps = np.asarray(inputs[left_index: i] + inputs[i + 1: right_index])
        return time_stamps

    def compute_index_range(self, i):
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


class NegativeSamplingBatchGenerator(TimeAttentionDataGenerator):

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


class TemporalBertDataGenerator(TimeAttentionDataGenerator):
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
        super(TemporalBertDataGenerator, self).__init__(*args, **kwargs)
        self.mask_token_id = mask_token_id
        self.first_token_id = first_token_id
        self.last_token_id = last_token_id

    def batch_generator(self):
        training_data_generator = self.data_generator()
        while True:
            next_batch = islice(training_data_generator, self.batch_size)
            output_mask, concepts, masked_concepts, time_stamps, visit_orders, visit_segments, concept_positions = zip(
                *list(next_batch))

            concepts = self.pad_input(concepts, self.unused_token_id)
            masked_concepts = self.pad_input(masked_concepts, self.unused_token_id)
            time_stamps = self.pad_input(time_stamps, 0)
            visit_orders = self.pad_input(visit_orders, 0)
            visit_segments = self.pad_input(visit_segments, 0)
            concept_positions = self.pad_input(concept_positions, 0)

            mask = (concepts == self.unused_token_id).astype(int)
            combined_label = np.stack([concepts, output_mask], axis=-1)

            yield ({'masked_concept_ids': masked_concepts,
                    'concept_ids': concepts,
                    'time_stamps': time_stamps,
                    'visit_orders': visit_orders,
                    'visit_segments': visit_segments,
                    'concept_positions': concept_positions,
                    'mask': mask}, combined_label)

    def data_generator(self):

        while True:
            for tup in self.patient_event_sequence.itertuples():
                # If the length of concept ids is less than the maximum allowed length
                if len(tup.concept_ids) <= self.max_sequence_length:
                    i = len(tup.concept_ids) // 2
                else:
                    # Randomly get an index
                    i = random.randint(0, len(tup.concept_ids) - 1)
                # Get the indexes that fall within the time window given the random index
                time_window_qualified_indexes = self.get_time_window_qualified_indexes(i, tup.dates)
                # Check if the number of indexes exceeds the minimum number of concepts
                if len(time_window_qualified_indexes) > self.minimum_num_of_concepts:
                    (concepts, time_stamps, visit_orders, visit_segments,
                     concept_positions) = self.generate_inputs(i, tup, time_window_qualified_indexes)

                    # Create the masked concepts and the corresponding mask
                    masked_concepts, output_mask = self.mask_concepts(concepts)

                    yield (output_mask, concepts, masked_concepts, time_stamps, visit_orders,
                           visit_segments, concept_positions)

    def mask_concepts(self, concepts):
        """
        Mask out 15% of the concepts
        :param concepts:
        :return:
        """
        masked_concepts = concepts.copy()
        output_mask = np.zeros((self.max_sequence_length,), dtype=int)
        for word_pos in range(0, len(concepts)):
            if concepts[word_pos] == self.unused_token_id:
                break

            if random.random() < 0.15:
                dice = random.random()
                if dice < 0.8:
                    masked_concepts[word_pos] = self.mask_token_id
                elif dice < 0.9:
                    masked_concepts[word_pos] = random.randint(
                        self.first_token_id, self.last_token_id)
                # else: 10% of the time we just leave the word as is
                output_mask[word_pos] = 1
        return masked_concepts, output_mask

    def generate_inputs(self, i, tup, time_window_qualified_indexes):

        iterator = zip(tup.token_ids, tup.dates, tup.concept_id_visit_orders, tup.visit_segments, tup.concept_positions)
        concept_ids, dates, visit_orders, visit_segments, concept_positions = zip(
            *sorted(iterator, key=lambda tup2: (tup2[1], tup2[2])))

        concepts = np.asarray(self.get_inputs_for_index(concept_ids, i))[time_window_qualified_indexes]
        time_stamps = np.asarray(self.get_inputs_for_index(dates, i))[time_window_qualified_indexes]
        visit_orders = np.asarray(self.get_inputs_for_index(visit_orders, i))[time_window_qualified_indexes]
        visit_segments = np.asarray(self.get_inputs_for_index(visit_segments, i))[time_window_qualified_indexes]
        concept_positions = np.asarray(self.get_inputs_for_index(concept_positions, i))[time_window_qualified_indexes]

        return concepts, time_stamps, visit_orders, visit_segments, concept_positions

    def get_inputs_for_index(self, inputs, i):
        left_index, right_index = self.compute_index_range(i)
        time_stamps = np.asarray(inputs[left_index: right_index])
        return time_stamps

    def compute_index_range(self, i):
        half_window_size = int(self.max_sequence_length / 2)
        left_index = i - half_window_size if i - half_window_size > 0 else 0
        right_index = i + half_window_size
        return left_index, right_index

    def estimate_data_size(self):
        return len(self.patient_event_sequence)

    def pad_input(self, inputs, pad_value):
        return pad_sequences(np.asarray(inputs), maxlen=self.max_sequence_length, padding='post',
                             value=pad_value, dtype='int32')


class BertDataGenerator(TemporalBertDataGenerator):

    def data_generator(self):

        while True:
            for tup in self.patient_event_sequence.itertuples():
                # If the length of concept ids is less than the maximum allowed length
                if len(tup.concept_ids) <= self.max_sequence_length:
                    i = len(tup.concept_ids) // 2
                else:
                    # Randomly get an index
                    i = random.randint(0, len(tup.concept_ids) - 1)

                concepts = self.get_inputs_for_index(tup.concepts, i)
                time_stamps = self.get_inputs_for_index(tup.dates, i)
                visit_orders = self.get_inputs_for_index(tup.concept_id_visit_orders, i)
                visit_segments = self.get_inputs_for_index(tup.visit_segments, i)
                concept_positions = self.get_inputs_for_index(tup.concept_positions, i)

                # Create the masked concepts and the corresponding mask
                masked_concepts, output_mask = self.mask_concepts(concepts)

                yield (output_mask, concepts, masked_concepts, time_stamps, visit_orders,
                       visit_segments, concept_positions)


class BertFineTuningDataGenerator(TemporalBertDataGenerator):

    def batch_generator(self):
        training_example_generator = self.data_generator()
        while True:
            concepts, masked_concepts, time_stamps, visit_orders, visit_segments, concept_positions, labels = zip(
                *list(islice(training_example_generator, self.batch_size)))

            concepts = self.pad_input(concepts, 0)
            masked_concepts = self.pad_input(masked_concepts, 0)
            time_stamps = self.pad_input(time_stamps, 0)
            visit_orders = self.pad_input(visit_orders, 0)
            visit_segments = self.pad_input(visit_segments, 0)
            concept_positions = self.pad_input(concept_positions, 0)

            mask = (concepts == 0).astype(int)

            yield ({'masked_concept_ids': masked_concepts,
                    'concept_ids': concepts,
                    'time_stamps': time_stamps,
                    'visit_orders': visit_orders,
                    'visit_segments': visit_segments,
                    'concept_positions': concept_positions,
                    'mask': mask}, labels)

    def data_generator(self):
        while True:
            for tup in self.patient_event_sequence.itertuples():
                concept_ids, dates, concept_id_visit_orders, visit_segments, concept_positions = zip(
                    *sorted(zip(tup.token_ids, tup.dates, tup.concept_id_visit_orders, tup.visit_segments,
                                tup.concept_positions), key=lambda tup2: (tup2[1],
                                                                          tup2[2])))
                yield (
                    concept_ids, concept_ids, dates, concept_id_visit_orders, visit_segments, concept_positions,
                    tup.labels)
