import copy
from itertools import islice
from typing import List
from tensorflow.dtypes import int32, float32

import numpy as np

from data_generators.data_classes import RowSlicer
from data_generators.learning_objective import LearningObjective, validate_columns_decorator, post_pad_pre_truncate
from data_generators.tokenizer import ConceptTokenizer


class PredictNextValueLearningObjective(LearningObjective):
    required_columns = [
        'concept_values', 'concept_value_masks'
    ]

    def __init__(
            self,
            max_seq_len: int
    ):
        self._max_seq_len = max_seq_len

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'concept_values': float32,
            'concept_value_masks': int32
        }
        output_dict_schema = {'next_concept_values': float32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        (
            concept_value_masks, concept_values
        ) = zip(*list(map(self._make_record, rows)))

        concept_value_masks = post_pad_pre_truncate(
            concept_value_masks,
            0,
            self._max_seq_len
        )
        concept_values = post_pad_pre_truncate(
            concept_values,
            0.0,
            self._max_seq_len,
            d_type='float32'
        )

        input_dict = {
            'concept_value_masks': concept_value_masks,
            'concept_values': concept_values
        }

        output_dict = {'value_predictions': np.stack([concept_values, concept_value_masks], axis=-1)}

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield

        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, left_index, right_index, _ = row_slicer

        sorting_columns = getattr(row, 'orders') if hasattr(row, 'orders') else row.dates

        iterator = zip(
            map(int, sorting_columns), row.concept_value_masks, row.concept_values
        )
        sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1]))

        _, concept_value_masks, concept_values = zip(*list(islice(sorted_list, left_index, right_index)))

        return (
            concept_value_masks, concept_values
        )


class SequenceGenerationLearningObjective(LearningObjective):
    required_columns = [
        'token_ids',
        'visit_concept_orders'
    ]

    def __init__(
            self,
            concept_tokenizer: ConceptTokenizer,
            max_seq_len: int
    ):
        self._max_seq_len = max_seq_len
        self._concept_tokenizer = concept_tokenizer

    def get_tf_dataset_schema(self):
        input_dict_schema = {
            'concept_ids': int32,
            'visit_concept_orders': int32,
        }
        output_dict_schema = {'concept_predictions': int32}
        return input_dict_schema, output_dict_schema

    @validate_columns_decorator
    def process_batch(self, rows: List[RowSlicer]):
        (
            concepts, visit_concept_orders, shifted_concepts
        ) = zip(*list(map(self._make_record, rows)))

        unused_token_id = self._concept_tokenizer.get_unused_token_id()

        # The main inputs for bert
        concepts = post_pad_pre_truncate(
            concepts,
            unused_token_id,
            self._max_seq_len
        )

        # The main inputs for bert
        shifted_concepts = post_pad_pre_truncate(
            shifted_concepts,
            unused_token_id,
            self._max_seq_len
        )

        mask = (concepts != unused_token_id).astype(int)

        visit_concept_orders = post_pad_pre_truncate(
            visit_concept_orders,
            self._max_seq_len,
            self._max_seq_len
        )

        input_dict = {
            'concept_ids': concepts,
            'visit_concept_orders': visit_concept_orders
        }

        output_dict = {
            'concept_predictions': np.stack([shifted_concepts, mask], axis=-1)
        }

        return input_dict, output_dict

    def _make_record(self, row_slicer: RowSlicer):
        """
        A method for making a bert record for the bert data generator to yield
        :param row_slicer: a tuple containing a pandas row,
        left_index and right_index for slicing the sequences such as concepts

        :return:
        """

        row, left_index, right_index, _ = row_slicer

        sorting_columns = getattr(row, 'orders') if hasattr(row, 'orders') else row.dates

        iterator = zip(
            map(int, sorting_columns), row.token_ids, row.visit_concept_orders
        )
        sorted_list = sorted(iterator, key=lambda tup2: (tup2[0], tup2[1]))

        _, concept_list, visit_concept_orders = zip(*list(islice(sorted_list, left_index, right_index)))

        concept_list = [self._concept_tokenizer.get_start_token_id()] + list(concept_list)
        visit_concept_orders = [0] + list(visit_concept_orders)
        shifted_concept_list = copy.deepcopy(list(concept_list)[1:]) + [
            self._concept_tokenizer.get_end_token_id()]

        return concept_list, visit_concept_orders, shifted_concept_list
