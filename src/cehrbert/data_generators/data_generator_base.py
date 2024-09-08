import inspect
import logging
import random
from abc import ABC, abstractmethod
from collections import ChainMap
from itertools import chain, islice
from typing import List, Set

import numpy as np
import pandas as pd

from cehrbert.data_generators.data_classes import RowSlicer
from cehrbert.data_generators.learning_objective import (
    BertFineTuningLearningObjective,
    DemographicsLearningObjective,
    HierarchicalArtificialTokenPredictionLearningObjective,
    HierarchicalMaskedLanguageModelLearningObjective,
    HierarchicalProlongedLengthStayLearningObjective,
    HierarchicalReadmissionLearningObjective,
    HierarchicalVisitTypePredictionLearningObjective,
    LearningObjective,
    MaskedLanguageModelLearningObjective,
    ProlongedLengthStayLearningObjective,
    TimeAttentionLearningObjective,
    VisitPredictionLearningObjective,
)
from cehrbert.data_generators.tokenizer import ConceptTokenizer


def create_indexes_by_time_window(dates, cursor, max_seq_len, time_window_size):
    """
    Extract the start_index and end_index used for slicing the sequences e.g. concept_ids and dates.

    :param dates: a list of time stamps associated with the context
    :param cursor: the current index used as the center for slicing the sequence
    :param max_seq_len: the maximum sequence length
    :param time_window_size: the maximum time window allowed
    :return: start_index and end_index
    """
    seq_len = len(dates)
    half_context_window_size = int(max_seq_len / 2)
    start_index = max(0, cursor - half_context_window_size)
    end_index = min(cursor + half_context_window_size, seq_len)

    half_time_window_size = int(time_window_size / 2)
    context_dates = dates[start_index:end_index]
    time_deltas = context_dates - dates[cursor]
    context_indexes = np.squeeze(
        np.argwhere((time_deltas >= -half_time_window_size) & (time_deltas <= half_time_window_size)),
        axis=-1,
    )

    return np.min(context_indexes).item(), np.max(context_indexes).item()


def get_required_params(clazz: LearningObjective):
    """
    Get required parameters for the learning objective class.

    :param clazz:
    :return:
    """
    params = inspect.signature(clazz).parameters
    return [dict(name=name, required=param.default is inspect.Parameter.empty) for name, param in params.items()]


class AbstractDataGeneratorBase(ABC):
    default_min_num_of_concepts = 2
    default_required_column = "concept_ids"

    def __init__(
        self,
        training_data: pd.DataFrame,
        batch_size: int,
        max_seq_len: int,
        min_num_of_concepts: int,
        is_random_cursor: bool = False,
        is_pretraining: bool = True,
        num_steps: int = None,
        *args,
        **kwargs,
    ):

        self._training_data = training_data
        self._batch_size = batch_size
        self._max_seq_len = max_seq_len
        self._min_num_of_concepts = min_num_of_concepts
        self._is_random_cursor = is_random_cursor
        self._is_pretraining = is_pretraining
        self._num_steps = num_steps

        self.get_logger().info(
            f"batch_size: {batch_size}\n"
            f"max_seq_len: {max_seq_len}\n"
            f"min_num_of_concepts: {min_num_of_concepts}\n"
            f"is_random_cursor: {is_random_cursor}\n"
            f"is_pretraining: {is_pretraining}\n"
            f"num_of_steps: {num_steps}\n"
        )

        self._learning_objectives = self._initialize_learning_objectives(
            max_seq_len=max_seq_len, is_pretraining=is_pretraining, **kwargs
        )
        # validate the required columns in the training data
        self._validate_data_frame_columns()
        self._clean_dataframe()

    @abstractmethod
    def _get_learning_objective_classes(self) -> List[LearningObjective]:
        """
        Initialize a list of LearningObjectives used for generating the input and and output.

        :return:
        """

    def _initialize_learning_objectives(self, **kwargs) -> List[LearningObjective]:
        """
        Initialize a list of LearningObjectives used for generating the input and and output.

        :return:
        """

        def _initialize(learning_objective) -> LearningObjective:
            """
            Initialize one LearningObjective using the provided keyword arguments.

            from the parent method

            :param learning_objective:
            :return:
            """
            learning_object_input = dict()
            params = get_required_params(learning_objective)
            for required_param in [param["name"] for param in params if param["required"]]:
                if required_param in kwargs:
                    learning_object_input[required_param] = kwargs[required_param]
            return learning_objective(**learning_object_input)

        return list(map(_initialize, self._get_learning_objective_classes()))

    def _validate_data_frame_columns(self):
        """
        Validate if the training data has all required columns.

        :return:
        """
        dataframe_columns = self._training_data.columns.tolist()
        for required_column in self._get_required_columns():
            if not required_column in dataframe_columns:
                raise ValueError(f"The required column {required_column} does not exist in the training data")

    @abstractmethod
    def _clean_dataframe(self):
        """
        Clean the input data (_training_data) e.g. remove rows whose sequence length is less than.

        _minimum_num_of_concepts.

        Overload this method in the subclasses to overwrite the default behavior

        :return:
        """

    def create_batch_generator(self):
        """
        Create the batch generator for tf.dataset.from_generator to use.

        :return:
        """
        while True:
            # Get a new iterator
            iterator = self._create_iterator()
            # Slice out a batch of data for every step
            try:
                for _ in range(self.get_steps_per_epoch()):
                    rows = list(islice(iterator, self._batch_size))
                    input_dicts = []
                    output_dicts = []
                    for learning_objective in self._learning_objectives:
                        input_dict, output_dict = learning_objective.process_batch(list(rows))
                        input_dicts.append(input_dict)
                        output_dicts.append(output_dict)
                    yield dict(ChainMap(*input_dicts)), dict(ChainMap(*output_dicts))
            except (RuntimeError, ValueError) as e:
                print(f"Error caught: {e}")

            # Break out of the infinite loop in the non pretraining mode
            if not self._is_pretraining:
                break

    def set_learning_objectives(self, learning_objectives: List[LearningObjective]):
        """
        Overwrite the default learning objectives.

        :param learning_objectives:
        :return:
        """
        self._learning_objectives = learning_objectives

    @abstractmethod
    def _create_iterator(self) -> RowSlicer:
        pass

    @abstractmethod
    def get_data_size(self):
        pass

    def get_steps_per_epoch(self):
        """
        Calculate the number of steps required for one epoch to complete.

        Floor division + 1 if there is any modulo value
        :return:
        """
        num_of_steps = self.get_data_size() // self._batch_size + bool(self.get_data_size() % self._batch_size)

        if self._num_steps:
            return min(self._num_steps, num_of_steps)

        return num_of_steps

    def _get_required_columns(self) -> Set[str]:
        """
        Combine lists of required columns from multiple learning objectives into a unique set of.

        required columns

        :return:
        """
        learning_objective_required_columns = list(
            chain(*[learning_objective.get_required_columns() for learning_objective in self._learning_objectives])
        )
        return set(learning_objective_required_columns + [self.default_required_column])

    def get_tf_dataset_schema(self):
        """
        Combine the input and output tensorflow data schema from multiple learning objectives.

        :return:
        """
        input_dict_schemas = []
        output_dict_schemas = []
        for learning_objective in self._learning_objectives:
            input_dict_schema, output_dict_schema = learning_objective.get_tf_dataset_schema()
            input_dict_schemas.append(input_dict_schema)
            output_dict_schemas.append(output_dict_schema)
        return dict(ChainMap(*input_dict_schemas)), dict(ChainMap(*output_dict_schemas))

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


class BertDataGenerator(AbstractDataGeneratorBase):

    def __init__(self, concept_tokenizer: ConceptTokenizer, *args, **kwargs):
        super(BertDataGenerator, self).__init__(concept_tokenizer=concept_tokenizer, *args, **kwargs)
        self._concept_tokenizer = concept_tokenizer

    def _clean_dataframe(self):
        self._training_data = self._training_data[
            self._training_data[self.default_required_column].apply(len)
            >= max(self.default_min_num_of_concepts, self._min_num_of_concepts)
        ]

    def _get_learning_objective_classes(self):
        return [MaskedLanguageModelLearningObjective]

    def _create_iterator(self):
        """
        Create an iterator that will iterate through all training data.

        :return:
        """
        for row in self._training_data.sample(frac=1).itertuples():
            seq_length = len(row.token_ids)
            if self._is_pretraining:
                cursor = (
                    random.randint(0, seq_length - 1)
                    if self._is_random_cursor & (seq_length > self._max_seq_len)
                    else seq_length // 2
                )

                half_window_size = int(self._max_seq_len / 2)
                start_index = max(0, cursor - half_window_size)
                end_index = min(cursor + half_window_size, seq_length)

                if start_index < end_index:
                    yield RowSlicer(row, start_index, end_index)
            else:
                yield RowSlicer(row, 0, seq_length)

    def get_data_size(self):
        return len(self._training_data)


class BertVisitPredictionDataGenerator(BertDataGenerator):
    def __init__(self, visit_tokenizer: ConceptTokenizer, *args, **kwargs):
        super(BertDataGenerator, self).__init__(visit_tokenizer=visit_tokenizer, *args, **kwargs)
        self._visit_tokenizer = visit_tokenizer

    def _get_learning_objective_classes(self):
        return [MaskedLanguageModelLearningObjective, VisitPredictionLearningObjective]


class HierarchicalBertDataGenerator(AbstractDataGeneratorBase):
    def __init__(
        self,
        concept_tokenizer: ConceptTokenizer,
        visit_tokenizer: ConceptTokenizer,
        max_num_of_visits: int,
        max_num_of_concepts: int,
        include_att_prediction: bool,
        include_visit_prediction: bool,
        *args,
        min_num_of_concepts: int = 5,
        min_num_of_visits: int = 2,
        **kwargs,
    ):

        # The num of visits
        self._min_num_of_visits = min_num_of_visits
        self._max_num_of_visits = max_num_of_visits
        self._max_num_of_concepts = max_num_of_concepts

        super().__init__(
            concept_tokenizer=concept_tokenizer,
            visit_tokenizer=visit_tokenizer,
            max_num_of_visits=max_num_of_visits,
            max_num_of_concepts=max_num_of_concepts,
            max_seq_len=max_num_of_visits * max_num_of_concepts,
            min_num_of_concepts=min_num_of_concepts,
            include_att_prediction=include_att_prediction,
            include_visit_prediction=include_visit_prediction,
            *args,
            **kwargs,
        )

    def _clean_dataframe(self):
        """
        Remove the patients that don't have enough concepts to qualify.

        :return:
        """
        min_num_of_concepts = max(self.default_min_num_of_concepts, self._min_num_of_concepts)
        criteria = (self._training_data["num_of_concepts"] >= min_num_of_concepts) & (
            self._training_data["num_of_visits"] >= self._min_num_of_visits
        )
        self._training_data = self._training_data[criteria]

    def _get_learning_objective_classes(self):
        return [
            HierarchicalMaskedLanguageModelLearningObjective,
            HierarchicalArtificialTokenPredictionLearningObjective,
            HierarchicalVisitTypePredictionLearningObjective,
        ]

    def _create_iterator(self):
        """
        Create an iterator that will iterate through all training example.

        :return:
        """
        for row in self._training_data.itertuples():

            if self._is_pretraining:
                if self._max_num_of_visits >= row.num_of_visits:
                    start_index = 0
                    end_index = row.num_of_visits
                else:
                    start_index = random.randint(0, row.num_of_visits - self._max_num_of_visits)
                    end_index = start_index + self._max_num_of_visits

                assert start_index < end_index
                yield RowSlicer(row, start_index, end_index)

            else:
                # Return the entire patient history
                yield RowSlicer(row, 0, row.num_of_visits)

    def get_data_size(self):
        return len(self._training_data)


class HierarchicalBertMultiTaskDataGenerator(HierarchicalBertDataGenerator):
    def __init__(
        self,
        include_readmission: bool,
        include_prolonged_length_stay: bool,
        *args,
        **kwargs,
    ):
        self._include_readmission = include_readmission
        self._include_prolonged_length_stay = include_prolonged_length_stay
        super().__init__(*args, **kwargs)

    def _get_learning_objective_classes(self):

        learning_objectives = [
            HierarchicalMaskedLanguageModelLearningObjective,
            HierarchicalArtificialTokenPredictionLearningObjective,
            HierarchicalVisitTypePredictionLearningObjective,
        ]

        if self._include_readmission:
            learning_objectives.append(HierarchicalReadmissionLearningObjective)

        if self._include_prolonged_length_stay:
            learning_objectives.append(HierarchicalProlongedLengthStayLearningObjective)

        return learning_objectives


class MedBertDataGenerator(BertDataGenerator):
    def _get_learning_objective_classes(self):
        return [
            MaskedLanguageModelLearningObjective,
            ProlongedLengthStayLearningObjective,
        ]


class TimeAttentionDataGenerator(AbstractDataGeneratorBase):
    def __init__(
        self,
        concept_tokenizer: ConceptTokenizer,
        time_window_size: int,
        *args,
        **kwargs,
    ):
        super(TimeAttentionDataGenerator, self).__init__(
            concept_tokenizer=concept_tokenizer,
            time_window_size=time_window_size,
            *args,
            **kwargs,
        )
        self._concept_tokenizer = concept_tokenizer
        self._time_window_size = time_window_size

    def _get_learning_objective_classes(self):
        return [TimeAttentionLearningObjective]

    def _create_iterator(self):
        """
        Create an iterator that will iterate forever.

        :return:
        """
        while True:
            for row in self._training_data.itertuples():
                concept_ids, dates = zip(*sorted(zip(row.token_ids, row.dates), key=lambda tup2: tup2[1]))
                for i in range(len(concept_ids)):
                    # Only include the concepts whose time stamps are within -half_time_window and
                    # half_time_window from the target time stamp
                    start_index, end_index = create_indexes_by_time_window(
                        dates, i, self._max_seq_len, self._time_window_size
                    )
                    if start_index < end_index:
                        yield RowSlicer(row, start_index, end_index, i)

    def get_data_size(self):
        return len(self._training_data.token_ids.explode())


class FineTuningHierarchicalBertDataGenerator(HierarchicalBertDataGenerator):
    def _get_learning_objective_classes(self):
        return [
            HierarchicalMaskedLanguageModelLearningObjective,
            DemographicsLearningObjective,
            BertFineTuningLearningObjective,
        ]
