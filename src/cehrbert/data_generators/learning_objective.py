import random
from abc import ABC, abstractmethod
from itertools import islice
from typing import Dict, List

import numpy as np
import pandas as pd
from tensorflow.dtypes import DType, float32, int32
from tensorflow.keras.utils import pad_sequences

from cehrbert.data_generators.data_classes import RowSlicer
from cehrbert.data_generators.graph_sample_method import GraphSampler
from cehrbert.data_generators.tokenizer import ConceptTokenizer
from cehrbert.utils.model_utils import convert_to_list_of_lists


def validate_columns_decorator(function):
    """
    A decorator to validate whether the parameter rows passed to LearningObjective.process_batch.

    contain the required columns. It raises AttributeError if any of the required columns is
    missing from the rows

    :param function:
    :return:
    """

    def wrapper(self, rows: List[RowSlicer], *args, **kwargs):
        required_columns = self.get_required_columns()
        for row_slicer in rows:
            for column in required_columns:
                if not hasattr(row_slicer.row, column):
                    raise AttributeError(f"The required column {column} is missing for {self}")
            break

        return function(self, rows, *args, **kwargs)

    return wrapper


def post_pad_pre_truncate(inputs, pad_value, max_seq_len, d_type="int32"):
    """
    Post _pad and pre-truncate the sequence.

    :param inputs:
    :param pad_value:
    :param max_seq_len:
    :param d_type:
    :return:
    """
    return pad_sequences(inputs, maxlen=max_seq_len, padding="post", value=pad_value, dtype=d_type)


class LearningObjective(ABC):

    @property
    def required_columns(self):
        raise NotImplementedError

    @validate_columns_decorator
    @abstractmethod
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """

    @abstractmethod
    def get_tf_dataset_schema(self):
        """
        Get the schema for the input and output to the tensorflow Dataset.

        :return:
        """

    @classmethod
    def get_required_columns(cls):
        """
        Get the required columns for this learning objective.

        :return:
        """
        return cls.required_columns

    def __str__(self):
        return str(self.__class__.__name__)


class CustomLearningObjective(LearningObjective):
    required_columns = []

    def __init__(self, input_schema: Dict[str, DType], output_schema: Dict[str, DType]):
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_columns = input_schema.keys()
        self.output_columns = output_schema.keys()

    def get_tf_dataset_schema(self):
        return self.input_schema, self.output_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        input_dict = {}
        output_dict = {}
        for row_slicer in rows:

            for input_column in self.input_columns:
                if input_column not in input_dict:
                    input_dict[input_column] = []
                input_dict[input_column].append(getattr(row_slicer.row, input_column))

            for output_column in self.output_columns:
                if output_column not in output_dict:
                    output_dict[output_column] = []
                output_dict[output_column].append(getattr(row_slicer.row, output_column))

        return input_dict, output_dict


class BertFineTuningLearningObjective(LearningObjective):
    required_columns = ["label"]

    def get_tf_dataset_schema(self):
        output_dict_schema = {"label": int32}
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        labels = []
        for row_slicer in rows:
            labels.append(row_slicer.row.label)

        output_dict = {"label": labels}
        return {}, output_dict


class DemographicsLearningObjective(LearningObjective):
    required_columns = ["age", "gender_concept_id"]

    def get_tf_dataset_schema(self):
        input_dict_schema = {"age": int32, "gender": int32}
        return input_dict_schema, {}

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        age_input = []
        gender_input = []
        for row_slicer in rows:
            age_input.append(row_slicer.row.age)
            gender_input.append(row_slicer.row.gender_concept_id)

        input_dict = {"age": age_input, "gender": gender_input}

        return input_dict, {}


class ProlongedLengthStayLearningObjective(LearningObjective):
    required_columns = ["prolonged_length_stay"]

    def get_tf_dataset_schema(self):
        output_dict_schema = {"prolonged_length_stay": int32}
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        prolonged_length_stay_input = []
        for row_slicer in rows:
            prolonged_length_stay_input.append(row_slicer.row.prolonged_length_stay)

        output_dict = {"prolonged_length_stay": prolonged_length_stay_input}

        return {}, output_dict


class VisitPredictionLearningObjective(LearningObjective):
    required_columns = ["visit_token_ids", "visit_concept_orders"]

    def __init__(self, visit_tokenizer: ConceptTokenizer, max_seq_len: int):
        self._max_seq_len = max_seq_len
        self._visit_tokenizer = visit_tokenizer

    def get_tf_dataset_schema(self):
        input_dict_schema = {"masked_visit_concepts": int32, "mask_visit": int32}
        output_dict_schema = {"visit_predictions": int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):

        (output_mask, masked_visit_concepts, visit_concepts) = zip(*list(map(self._make_record, rows)))

        unused_token_id = self._visit_tokenizer.get_unused_token_id()

        visit_concepts = post_pad_pre_truncate(visit_concepts, unused_token_id, self._max_seq_len)
        masked_visit_concepts = post_pad_pre_truncate(masked_visit_concepts, unused_token_id, self._max_seq_len)

        # 1 indicates attention and 0 indicates mask
        visit_mask = (visit_concepts != unused_token_id).astype(int)

        combined_label = np.stack([visit_concepts, output_mask], axis=-1)

        input_dict = {
            "masked_visit_concepts": masked_visit_concepts,
            "mask_visit": visit_mask,
        }

        output_dict = {"visit_predictions": combined_label}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a namedtuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, left_index, right_index, *_ = row_slicer

        iterator = zip(row.visit_concept_orders, row.visit_token_ids)
        (dates, visit_concept_ids) = zip(*islice(sorted(iterator, key=lambda tup2: tup2[0]), left_index, right_index))

        masked_visit_concepts, output_mask = self._mask_visit_concepts(visit_concept_ids)

        return output_mask, masked_visit_concepts, visit_concept_ids

    def _mask_visit_concepts(self, visit_concepts):
        """
        Any visit has 50% chance to be masked.

        :param visit_concepts:
        :return:
        """
        masked_visit_concepts = np.asarray(visit_concepts).copy()
        output_mask = np.zeros((self._max_seq_len,), dtype=int)
        for word_pos in range(0, len(visit_concepts)):
            if random.random() < 0.5:
                output_mask[word_pos] = 1
                masked_visit_concepts[word_pos] = self._visit_tokenizer.get_mask_token_id()
        return masked_visit_concepts, output_mask


class MaskedLanguageModelLearningObjective(LearningObjective):
    required_columns = [
        "token_ids",
        "dates",
        "visit_segments",
        "ages",
        "visit_concept_orders",
        "concept_values",
        "concept_value_masks",
        "mlm_skip_values",
    ]

    def __init__(
        self,
        concept_tokenizer: ConceptTokenizer,
        max_seq_len: int,
        is_pretraining: bool,
    ):
        self._max_seq_len = max_seq_len
        self._concept_tokenizer = concept_tokenizer
        self._is_pretraining = is_pretraining

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            "masked_concept_ids": int32,
            "concept_ids": int32,
            "mask": int32,
            "time_stamps": int32,
            "visit_segments": int32,
            "ages": int32,
            "visit_concept_orders": int32,
            "concept_values": float32,
            "concept_value_masks": int32,
        }
        output_dict_schema = {"concept_predictions": int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):

        (
            output_mask,
            masked_concepts,
            concepts,
            time_stamps,
            visit_segments,
            ages,
            visit_concept_orders,
            concept_value_masks,
            concept_values,
        ) = zip(*list(map(self._make_record, rows)))

        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # The main inputs for bert
        masked_concepts = post_pad_pre_truncate(masked_concepts, unused_token_id, self._max_seq_len)
        concepts = post_pad_pre_truncate(concepts, unused_token_id, self._max_seq_len)
        concept_value_masks = post_pad_pre_truncate(concept_value_masks, 0, self._max_seq_len)
        concept_values = post_pad_pre_truncate(concept_values, -1.0, self._max_seq_len, d_type="float32")

        # 1 indicates attention and 0 indicates mask
        mask = (concepts != unused_token_id).astype(int)

        # The auxiliary inputs for bert
        visit_segments = post_pad_pre_truncate(visit_segments, 0, self._max_seq_len)
        time_stamps = post_pad_pre_truncate(time_stamps, 0, self._max_seq_len)
        ages = post_pad_pre_truncate(ages, 0, self._max_seq_len)
        visit_concept_orders = post_pad_pre_truncate(visit_concept_orders, self._max_seq_len, self._max_seq_len)

        input_dict = {
            "masked_concept_ids": masked_concepts,
            "concept_ids": concepts,
            "mask": mask,
            "time_stamps": time_stamps,
            "ages": ages,
            "visit_segments": visit_segments,
            "visit_concept_orders": visit_concept_orders,
            "concept_value_masks": concept_value_masks,
            "concept_values": concept_values,
        }

        output_dict = {"concept_predictions": np.stack([concepts, output_mask], axis=-1)}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, left_index, right_index, *_ = row_slicer

        sorting_columns = getattr(row, "orders") if hasattr(row, "orders") else row.dates

        iterator = zip(
            map(int, sorting_columns),
            row.token_ids,
            row.visit_segments,
            row.dates,
            row.ages,
            row.visit_concept_orders,
            row.concept_value_masks,
            row.concept_values,
            row.mlm_skip_values,
        )
        sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1]))

        (
            _,
            concepts,
            segments,
            dates,
            ages,
            visit_concept_orders,
            concept_value_masks,
            concept_values,
            mlm_skip_values,
        ) = zip(*list(islice(sorted_list, left_index, right_index)))

        masked_concepts, output_mask = self._mask_concepts(concepts, mlm_skip_values)

        return (
            output_mask,
            masked_concepts,
            concepts,
            dates,
            segments,
            ages,
            visit_concept_orders,
            concept_value_masks,
            concept_values,
        )

    def _mask_concepts(self, concepts, mlm_skip_values):
        """
        Mask out 15% of the concepts.

        :param concepts:
        :param mlm_skip_values:
        :return:
        """

        masked_concepts = np.asarray(concepts).copy()
        output_mask = np.zeros((self._max_seq_len,), dtype=int)

        if self._is_pretraining:
            for word_pos in range(0, len(concepts)):
                # Check if this position needs to be skipped
                if mlm_skip_values[word_pos] == 1:
                    continue
                if concepts[word_pos] == self._concept_tokenizer.get_unused_token_id():
                    break
                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_concepts[word_pos] = self._concept_tokenizer.get_mask_token_id()
                    elif dice < 0.9:
                        masked_concepts[word_pos] = random.randint(
                            self._concept_tokenizer.get_first_token_index(),
                            self._concept_tokenizer.get_last_token_index(),
                        )
                    # else: 10% of the time we just leave the word as is
                    output_mask[word_pos] = 1

        return masked_concepts, output_mask


class HierarchicalMaskedLanguageModelLearningObjective(LearningObjective):
    required_columns = [
        "concept_ids",
        "dates",
        "visit_segments",
        "ages",
        "visit_dates",
        "visit_masks",
        "visit_rank_orders",
        "concept_values",
        "concept_value_masks",
        "mlm_skip_values",
    ]

    def __init__(
        self,
        concept_tokenizer: ConceptTokenizer,
        max_num_of_visits: int,
        max_num_of_concepts: int,
        is_pretraining: bool,
        concept_similarity_path: str,
        concept_similarity_type: str,
    ):
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts
        self._is_pretraining = is_pretraining
        self._graph_sampler = GraphSampler(concept_similarity_path, concept_similarity_type)

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            "pat_seq": int32,
            "pat_seq_age": int32,
            "pat_seq_time": int32,
            "pat_mask": int32,
            "visit_mask": int32,
            "visit_rank_order": int32,
            "concept_values": float32,
            "concept_value_masks": int32,
        }
        output_dict_schema = {"concept_predictions": int32}
        return input_dict_schema, output_dict_schema

    def _pad(self, x, padded_token, maxlen, token_dtype="int32"):
        return pad_sequences(
            x,
            maxlen=maxlen,
            padding="post",
            truncating="post",
            value=padded_token,
            dtype=token_dtype,
        )

    def _concept_mask(self, concept_ids):
        return list(
            map(
                lambda c: (c == self._concept_tokenizer.get_unused_token_id()).astype(int),
                concept_ids,
            )
        )

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):

        (
            output_concept_masks,
            masked_concepts,
            concepts,
            dates,
            ages,
            visit_dates,
            visit_masks,
            visit_rank_orders,
            concept_values,
            concept_value_masks,
        ) = zip(*list(map(self._make_record, rows)))

        # Retrieve the unused token id to pad the visit sequences
        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # Calculate the max sequence length in 1-D
        max_seq_len = self._max_num_of_concepts * self._max_num_of_visits

        # Pad each visit with up to max_num_of_concepts
        masked_concepts = (
            pd.Series(masked_concepts)
            .apply(convert_to_list_of_lists)
            .apply(self._concept_tokenizer.encode)
            .apply(
                lambda tokens: self._pad(
                    tokens,
                    padded_token=unused_token_id,
                    maxlen=self._max_num_of_concepts,
                )
            )
        )
        # Post pad the sequence and pre-truncate the sequence
        padded_masked_concepts = np.reshape(
            post_pad_pre_truncate(
                masked_concepts.apply(lambda d: d.flatten()),
                unused_token_id,
                max_seq_len,
            ),
            (-1, self._max_num_of_visits, self._max_num_of_concepts),
        )

        # 1 indicates attention and 0 indicates mask
        padded_pat_mask = (padded_masked_concepts != unused_token_id).astype(int)

        # Process visit_rank_orders
        padded_visit_rank_orders = post_pad_pre_truncate(
            visit_rank_orders, pad_value=0, max_seq_len=self._max_num_of_visits
        )
        # Process visit_masks
        padded_visit_masks = post_pad_pre_truncate(visit_masks, pad_value=1, max_seq_len=self._max_num_of_visits)
        # 1 indicates attention and 0 indicates mask, therefore we need to flip it.
        padded_visit_masks = 1 - padded_visit_masks

        # The concept values for bert
        concept_values = (
            pd.Series(concept_values)
            .apply(convert_to_list_of_lists)
            .apply(
                lambda time_stamps: self._pad(
                    time_stamps,
                    padded_token=-1.0,
                    token_dtype="float32",
                    maxlen=self._max_num_of_concepts,
                )
            )
        )

        padded_concept_values = np.reshape(
            post_pad_pre_truncate(concept_values.apply(lambda d: d.flatten()), -1.0, max_seq_len),
            (-1, self._max_num_of_visits, self._max_num_of_concepts),
        )

        # The concept value masks for bert, this indicates which concept in the visit sequence
        # has a value associated
        concept_value_masks = (
            pd.Series(concept_value_masks)
            .apply(convert_to_list_of_lists)
            .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0, maxlen=self._max_num_of_concepts))
        )

        padded_concept_value_masks = np.reshape(
            post_pad_pre_truncate(concept_value_masks.apply(lambda d: d.flatten()), 0, max_seq_len),
            (-1, self._max_num_of_visits, self._max_num_of_concepts),
        )

        # The auxiliary inputs for bert
        dates = (
            pd.Series(dates)
            .apply(convert_to_list_of_lists)
            .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0, maxlen=self._max_num_of_concepts))
        )

        padded_dates = np.reshape(
            post_pad_pre_truncate(dates.apply(lambda d: d.flatten()), 0, max_seq_len),
            (-1, self._max_num_of_visits, self._max_num_of_concepts),
        )

        ages = (
            pd.Series(ages)
            .apply(convert_to_list_of_lists)
            .apply(lambda time_stamps: self._pad(time_stamps, padded_token=0, maxlen=self._max_num_of_concepts))
        )

        padded_ages = np.reshape(
            post_pad_pre_truncate(ages.apply(lambda d: d.flatten()), 0, max_seq_len),
            (-1, self._max_num_of_visits, self._max_num_of_concepts),
        )

        input_dict = {
            "pat_seq": padded_masked_concepts,
            "pat_mask": padded_pat_mask,
            "pat_seq_time": padded_dates,
            "pat_seq_age": padded_ages,
            "visit_mask": padded_visit_masks,
            "visit_rank_order": padded_visit_rank_orders,
            "concept_values": padded_concept_values,
            "concept_value_masks": padded_concept_value_masks,
        }

        # Create the targets for MLM
        # Pad each visit with up to max_num_of_concepts
        concepts = (
            pd.Series(concepts)
            .apply(convert_to_list_of_lists)
            .apply(self._concept_tokenizer.encode)
            .apply(
                lambda tokens: self._pad(
                    tokens,
                    padded_token=unused_token_id,
                    maxlen=self._max_num_of_concepts,
                )
            )
        )

        # Reshape this into 1-D for the MLM prediction
        padded_concepts = post_pad_pre_truncate(concepts.apply(lambda d: d.flatten()), unused_token_id, max_seq_len)

        output_concept_masks = (
            pd.Series(output_concept_masks)
            .apply(convert_to_list_of_lists)
            .apply(lambda masks: self._pad(masks, padded_token=0, maxlen=self._max_num_of_concepts))
        )

        # Reshape this into 1-D for the MLM prediction
        padded_output_concept_masks = post_pad_pre_truncate(
            output_concept_masks.apply(lambda d: d.flatten()),
            pad_value=0,
            max_seq_len=max_seq_len,
        )

        output_dict = {"concept_predictions": np.stack([padded_concepts, padded_output_concept_masks], axis=-1)}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, *_ = row_slicer

        # Get the concepts
        concepts = row.concept_ids[start_index:end_index]
        # We skip all the MLM since these padded visits are fake
        mlm_skip_values = row.mlm_skip_values[start_index:end_index]

        # Get the temporal information
        dates = row.dates[start_index:end_index]
        ages = row.ages[start_index:end_index]

        # Retrieve the values associated with the concepts, this is mostly for measurements
        concept_values = row.concept_values[start_index:end_index]
        concept_value_masks = row.concept_value_masks[start_index:end_index]

        visit_segments = row.visit_segments[start_index:end_index]
        visit_dates = row.visit_dates[start_index:end_index]
        visit_masks = row.visit_masks[start_index:end_index]
        visit_rank_orders = row.visit_rank_orders[start_index:end_index]

        masked_concepts, output_concept_masks = zip(*list(map(self._mask_concepts, zip(concepts, mlm_skip_values))))

        return (
            output_concept_masks,
            masked_concepts,
            concepts,
            dates,
            ages,
            visit_dates,
            visit_masks,
            visit_rank_orders,
            concept_values,
            concept_value_masks,
        )

    def _mask_concepts(self, concepts_tuple):
        """
        Mask out 15% of the concepts.

        :param concepts_tuple:
        :return:
        """
        concepts, mlm_skip_values = concepts_tuple

        masked_concepts = np.asarray(concepts).copy()
        output_mask = np.zeros((len(masked_concepts),), dtype=int)

        if self._is_pretraining:
            # the first position is reserved for cls, so we don't mask the first element
            for word_pos in range(1, len(concepts)):

                # Check if this position needs to be skipped
                if mlm_skip_values[word_pos] == 1:
                    continue
                # Do no mask the [UNKNOWN] token
                if concepts[word_pos] == "0":
                    continue

                if random.random() < 0.15:
                    dice = random.random()
                    if dice < 0.8:
                        masked_concepts[word_pos] = self._concept_tokenizer.get_mask_token_id()
                    elif dice < 0.9:
                        masked_concepts[word_pos] = random.randint(
                            self._concept_tokenizer.get_first_token_index(),
                            self._concept_tokenizer.get_last_token_index(),
                        )
                    # else: 10% of the time we just leave the word as is
                    output_mask[word_pos] = 1

                elif random.random() < 0.15:
                    # the concept will be replaced by the neighbor concept in the graph
                    masked_concepts[word_pos] = self._graph_sampler.sample_graph(masked_concepts[word_pos])

        return masked_concepts, output_mask


class HierarchicalVisitTypePredictionLearningObjective(HierarchicalMaskedLanguageModelLearningObjective):
    required_columns = ["visit_token_ids"]

    def __init__(
        self,
        visit_tokenizer: ConceptTokenizer,
        max_num_of_visits: int,
        is_pretraining: bool,
        include_visit_prediction: bool,
        warmup_step: int,
    ):
        self._visit_tokenizer = visit_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._is_pretraining = is_pretraining
        self._include_visit_prediction = include_visit_prediction
        self._warmup_step = warmup_step
        self._counter = 0

    def get_tf_dataset_schema(self):
        input_dict_schema = {"masked_visit_type": int32}
        output_dict_schema = {}
        if self._include_visit_prediction:
            output_dict_schema.update({"visit_predictions": int32})
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        (masked_visit_token_ids, output_mask, visit_token_ids) = zip(*list(map(self._make_record, rows)))

        padded_masked_visit_token_ids = self._pad(
            masked_visit_token_ids,
            padded_token=self._visit_tokenizer.get_unused_token_id(),
            maxlen=self._max_num_of_visits,
        )

        input_dict = {"masked_visit_type": padded_masked_visit_token_ids}
        output_dict = {}

        if self._include_visit_prediction:
            padded_visit_token_ids = self._pad(
                visit_token_ids,
                padded_token=self._visit_tokenizer.get_unused_token_id(),
                maxlen=self._max_num_of_visits,
            )

            padded_output_masks = self._pad(output_mask, padded_token=0, maxlen=self._max_num_of_visits)

            if self._counter < self._warmup_step:
                self._counter += 1
                padded_output_masks = np.zeros_like(padded_output_masks)

            output_dict["visit_predictions"] = np.stack([padded_visit_token_ids, padded_output_masks], axis=-1)

        return input_dict, output_dict

    def _pad(self, x, padded_token, maxlen):
        return pad_sequences(
            np.asarray(x, dtype=object),
            maxlen=maxlen,
            padding="post",
            value=padded_token,
            dtype="int32",
        )

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, *_ = row_slicer

        visit_token_ids = row.visit_token_ids[start_index:end_index]

        masked_visit_token_ids, output_mask = self._mask_visit_concepts(visit_token_ids)

        return masked_visit_token_ids, output_mask, visit_token_ids

    def _mask_visit_concepts(self, visit_concepts):
        """
        Any visit has 50% chance to be masked.

        :param visit_concepts:
        :return:
        """
        masked_visit_concepts = np.asarray(visit_concepts).copy()
        output_mask = np.zeros((self._max_num_of_visits,), dtype=int)
        if self._include_visit_prediction:
            for word_pos in range(0, len(visit_concepts)):
                if random.random() < 0.5:
                    output_mask[word_pos] = 1
                    masked_visit_concepts[word_pos] = self._visit_tokenizer.get_mask_token_id()
        return masked_visit_concepts, output_mask


class HierarchicalReadmissionLearningObjective(HierarchicalVisitTypePredictionLearningObjective):
    required_columns = ["is_readmissions", "is_inpatients"]

    def __init__(
        self,
        max_num_of_visits: int,
        is_pretraining: bool,
        random_mask_prob: float,
        warmup_step: int,
    ):
        self._max_num_of_visits = max_num_of_visits
        self._is_pretraining = is_pretraining
        self._random_mask_prob = random_mask_prob
        self._warmup_step = warmup_step
        self._counter = 0

    def get_tf_dataset_schema(self):
        output_dict_schema = {"is_readmission": int32}
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        is_readmissions, is_inpatients = zip(*list(map(self._make_record, rows)))

        padded_is_readmissions = self._pad(is_readmissions, padded_token=0, maxlen=self._max_num_of_visits)

        padded_is_inpatients = self._pad(is_inpatients, padded_token=0, maxlen=self._max_num_of_visits)

        # if _random_mask_prob=0.2, there is 20% chance of being masked
        random_mask = np.random.rand(*padded_is_inpatients.shape) < self._random_mask_prob
        mask = padded_is_inpatients & random_mask

        if self._counter < self._warmup_step:
            self._counter += 1
            mask = np.zeros_like(mask)

        output_dict = {"is_readmission": np.stack([padded_is_readmissions, mask], axis=-1)}

        return {}, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, *_ = row_slicer

        is_readmissions = row.is_readmissions[start_index:end_index].astype(int)
        is_inpatients = row.is_inpatients[start_index:end_index]

        return (is_readmissions, is_inpatients)


class HierarchicalProlongedLengthStayLearningObjective(HierarchicalVisitTypePredictionLearningObjective):
    required_columns = ["visit_prolonged_stays", "is_inpatients"]

    def __init__(
        self,
        max_num_of_visits: int,
        is_pretraining: bool,
        random_mask_prob: float,
        warmup_step: int,
    ):
        self._max_num_of_visits = max_num_of_visits
        self._is_pretraining = is_pretraining
        self._random_mask_prob = random_mask_prob
        self._warmup_step = warmup_step
        self._counter = 0

    def get_tf_dataset_schema(self):
        output_dict_schema = {"visit_prolonged_stay": int32}
        return {}, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        visit_prolonged_stays, is_inpatients = zip(*list(map(self._make_record, rows)))

        padded_visit_prolonged_stays = self._pad(visit_prolonged_stays, padded_token=0, maxlen=self._max_num_of_visits)

        padded_is_inpatients = self._pad(is_inpatients, padded_token=0, maxlen=self._max_num_of_visits)

        # if _random_mask_prob=0.2, there is 20% chance of being masked
        random_mask = np.random.rand(*padded_is_inpatients.shape) < self._random_mask_prob
        mask = padded_is_inpatients & random_mask

        if self._counter < self._warmup_step:
            self._counter += 1
            mask = np.zeros_like(mask)

        output_dict = {"visit_prolonged_stay": np.stack([padded_visit_prolonged_stays, mask], axis=-1)}

        return {}, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, *_ = row_slicer

        visit_prolonged_stays = row.visit_prolonged_stays[start_index:end_index].astype(int)
        is_inpatients = row.is_inpatients[start_index:end_index]

        return (visit_prolonged_stays, is_inpatients)


class HierarchicalArtificialTokenPredictionLearningObjective(HierarchicalMaskedLanguageModelLearningObjective):
    required_columns = ["time_interval_atts"]

    def __init__(
        self,
        concept_tokenizer: ConceptTokenizer,
        max_num_of_visits: int,
        include_att_prediction: bool,
    ):
        self._concept_tokenizer = concept_tokenizer
        self._max_num_of_visits = max_num_of_visits
        self._include_att_prediction = include_att_prediction

    def get_tf_dataset_schema(self):
        input_dict_schema = {"visit_time_delta_att": int32}
        output_dict_schema = {}

        # when att prediction is enabled, we update the output data schema
        if self._include_att_prediction:
            output_dict_schema.update({"att_predictions": int32})

        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        """
        Process a batch of rows to generate input and output data for the learning objective.

        :param rows:
        :return:
        """
        (output_mask, masked_time_interval_att_tokens, time_interval_att_tokens) = zip(
            *list(map(self._make_record, rows))
        )

        masked_time_interval_att_tokens = np.asarray(
            self._concept_tokenizer.encode(pd.Series(masked_time_interval_att_tokens).apply(lambda t: t.tolist()))
        )
        padded_masked_time_interval_att_tokens = post_pad_pre_truncate(
            masked_time_interval_att_tokens,
            self._concept_tokenizer.get_unused_token_id(),
            self._max_num_of_visits,
        )[:, 1:]

        input_dict = {"visit_time_delta_att": padded_masked_time_interval_att_tokens}

        output_dict = {}

        if self._include_att_prediction:
            time_interval_att_tokens = np.asarray(
                self._concept_tokenizer.encode(pd.Series(time_interval_att_tokens).apply(lambda t: t.tolist()))
            )
            padded_time_interval_att_tokens = post_pad_pre_truncate(
                time_interval_att_tokens,
                self._concept_tokenizer.get_unused_token_id(),
                self._max_num_of_visits,
            )[:, 1:]

            padded_output_mask = post_pad_pre_truncate(output_mask, 0, self._max_num_of_visits)[:, 1:]

            output_dict.update(
                {"att_predictions": np.stack([padded_time_interval_att_tokens, padded_output_mask], axis=-1)}
            )

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, start_index, end_index, *_ = row_slicer
        time_interval_att_tokens = row.time_interval_atts[start_index:end_index]

        masked_time_interval_att_tokens, output_mask = self._mask_visit_concepts(time_interval_att_tokens)
        return output_mask, masked_time_interval_att_tokens, time_interval_att_tokens

    def _mask_visit_concepts(self, time_interval_att_tokens):
        """
        Any visit has 50% chance to be masked when att prediction is enabled, otherwise just.

        return the time_interval_att_tokens as the masked_time_interval_att_tokens

        :param time_interval_att_tokens:
        :return:
        """
        masked_time_interval_att_tokens = np.asarray(time_interval_att_tokens).copy()
        output_mask = np.zeros_like(masked_time_interval_att_tokens).astype(int)

        # when att prediction is enabled, we need to generate the output associated with this
        # learning objective
        if self._include_att_prediction:

            for word_pos in range(0, len(time_interval_att_tokens)):

                # Do not mask the [UNUSED] token
                if time_interval_att_tokens[word_pos] == self._concept_tokenizer.get_unused_token():
                    break

                if random.random() < 0.15:
                    output_mask[word_pos] = 1
                    masked_time_interval_att_tokens[word_pos] = self._concept_tokenizer.get_att_mask_token_id()

        return masked_time_interval_att_tokens, output_mask


class TimeAttentionLearningObjective(LearningObjective):
    required_columns = ["token_ids", "dates"]

    def __init__(
        self,
        concept_tokenizer: ConceptTokenizer,
        max_seq_len: int,
        time_window_size: int,
    ):
        super(TimeAttentionLearningObjective, self).__init__()
        self._concept_tokenizer = concept_tokenizer
        self._max_seq_len = max_seq_len
        self._time_window_size = time_window_size

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            "target_concepts": int32,
            "target_time_stamps": int32,
            "context_concepts": int32,
            "context_time_stamps": int32,
            "mask": int32,
        }
        output_dict_schema = {"concept_predictions": int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        (target_concepts, target_dates, context_concepts, context_time_stamps) = zip(
            *list(map(self._make_record, rows))
        )

        target_concepts = np.asarray(target_concepts)
        target_time_stamps = np.asarray(target_dates)

        context_concepts = post_pad_pre_truncate(
            context_concepts,
            self._concept_tokenizer.get_unused_token_id(),
            self._max_seq_len,
        )

        context_time_stamps = post_pad_pre_truncate(context_time_stamps, 0, self._max_seq_len)
        mask = (context_concepts == self._concept_tokenizer.get_unused_token_id()).astype(int)

        input_dict = {
            "target_concepts": target_concepts,
            "target_time_stamps": target_time_stamps,
            "context_concepts": context_concepts,
            "context_time_stamps": context_time_stamps,
            "mask": mask,
        }

        output_dict = {"concept_predictions": target_concepts}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the time attention data generator to yield.

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        target_index = row_slicer.target_index
        start_index = row_slicer.start_index
        end_index = row_slicer.end_index
        concepts = np.asarray(row_slicer.row.token_ids)
        dates = np.asarray(row_slicer.row.dates)

        indexes = np.asarray(list(range(start_index, end_index + 1)))
        indexes = indexes[indexes != target_index]

        return (
            [concepts[target_index]],
            [dates[target_index]],
            concepts[indexes],
            dates[indexes],
        )
