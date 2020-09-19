import os
import re
from abc import ABC, abstractmethod
from typing import List

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from utils.common import *

NEGATIVE_CONTROL_MATCH_QUERY_TEMPLATE = """
SELECT DISTINCT
    n.*
FROM global_temp.positives AS p
JOIN global_temp.negatives AS n
    ON p.gender_concept_id = n.gender_concept_id
        AND p.age BETWEEN n.age - {age_match_window} AND n.age + {age_match_window}
"""


def cohort_validator(required_columns_attribute):
    """
    Decorator for validating the cohort dataframe returned by create_cohort function in
    AbstractCohortBuilderBase
    :param required_columns_attribute: attribute for storing cohort_required_columns
    in :class:`spark_apps.spark_app_base.AbstractCohortBuilderBase`
    :return:
    """

    def cohort_validator_decorator(function):
        def wrapper(self, *args, **kwargs):
            cohort = function(self, *args, **kwargs)
            required_columns = getattr(self, required_columns_attribute)
            for required_column in required_columns:
                if required_column not in cohort.columns:
                    raise AssertionError(f'{required_column} is a required column in the cohort')
            return cohort

        return wrapper

    return cohort_validator_decorator


class AbstractCohortBuilderBase(ABC):
    cohort_required_columns = ['person_id', 'index_date']

    def __init__(self,
                 cohort_name: str,
                 input_folder: str,
                 output_folder: str,
                 date_lower_bound: str,
                 date_upper_bound: str,
                 age_lower_bound: int,
                 age_upper_bound: int,
                 observation_window: int,
                 prediction_window: int,
                 index_date_match_window: int,
                 ehr_table_list: List[str],
                 dependency_list: List[str],
                 include_ehr_records: bool = True):

        self._cohort_name = cohort_name
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._date_lower_bound = date_lower_bound
        self._date_upper_bound = date_upper_bound
        self._age_lower_bound = age_lower_bound
        self._age_upper_bound = age_upper_bound
        self._observation_window = observation_window
        self._prediction_window = prediction_window
        self._index_date_match_window = index_date_match_window
        self._ehr_table_list = ehr_table_list
        self._dependency_list = dependency_list
        self._output_data_folder = os.path.join(self._output_folder,
                                                re.sub('[^a-z0-9]+', '_',
                                                       self._cohort_name.lower()))
        self._include_ehr_records = include_ehr_records

        # Validate the input and output folders
        self._validate_folder(self._input_folder)
        self._validate_folder(self._output_folder)
        # Validate the age range, observation_window and prediction_window
        self._validate_integer_inputs()
        # Validate if the data folders exist
        self._validate_date_folder(self._ehr_table_list)
        self._validate_date_folder(self._dependency_list)

        self.spark = SparkSession.builder.appName(f'Generate {self._cohort_name}').getOrCreate()

        self._dependency_dict = self._instantiate_dependencies()

    def build(self):
        """
        Build the cohort and write the dataframe as parquet files to _output_data_folder
        """
        self.preprocess_dependencies()

        cohort = self.create_cohort()

        if self._include_ehr_records:
            ehr_records_for_cohorts = self.extract_ehr_records_for_cohort(cohort)
            cohort = cohort.select('person_id', 'label', 'age', 'gender_concept_id',
                                   'race_concept_id').join(ehr_records_for_cohorts, 'person_id')

        cohort.write.mode('overwrite').parquet(self._output_data_folder)

        self._destroy_dependencies()

        return self

    @cohort_validator('cohort_required_columns')
    @abstractmethod
    def create_cohort(self):
        pass

    @abstractmethod
    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        pass

    @abstractmethod
    def preprocess_dependencies(self):
        pass

    def _validate_integer_inputs(self):
        assert self._age_lower_bound >= 0
        assert self._age_upper_bound > 0
        assert self._age_lower_bound < self._age_upper_bound

        assert self._observation_window > 0
        assert self._prediction_window >= 0

    def _validate_date_folder(self, table_list):
        for domain_table_name in table_list:
            parquet_file_path = os.path.join(self._input_folder, domain_table_name)
            if not os.path.exists(parquet_file_path):
                raise FileExistsError(f'{parquet_file_path} does not exist')

    def _validate_folder(self, folder_path):
        if not os.path.exists(folder_path):
            raise FileExistsError(f'{folder_path} does not exist')

    def _validate_cohort_columns(self, cohort: DataFrame):
        """
        Validate the target cohort to see if it has all required columns
        :return:
        """
        for required_column in self.cohort_required_columns:
            if not required_column not in cohort.columns:
                raise AssertionError(f'{required_column} is a required column in the cohort')

    def _instantiate_dependencies(self):
        dependency_dict = dict()
        for domain_table_name in self._dependency_list:
            table = preprocess_domain_table(self.spark, self._input_folder, domain_table_name)
            table.createOrReplaceGlobalTempView(domain_table_name)
            dependency_dict[domain_table_name] = table
        return dependency_dict

    def _destroy_dependencies(self):
        for domain_table_name in self._dependency_dict:
            self.spark.sql(f'DROP VIEW global_temp.{domain_table_name}')

    def get_total_window(self):
        return self._observation_window + self._prediction_window

    def load_cohort(self):
        return self.spark.read.parquet(self._output_data_folder)


class AbstractCaseControlCohortBuilderBase(AbstractCohortBuilderBase):

    def create_cohort(self):
        """
        Create the cohort using a case-control design
        :return:
        """
        incident_cases = self.create_incident_cases()

        control_cases = self.create_control_cases()

        matched_control_cases = self.create_matching_control_cases(incident_cases, control_cases)

        cohort = incident_cases.union(matched_control_cases)

        return cohort

    @abstractmethod
    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        pass

    def create_matching_control_cases(self, incident_cases: DataFrame, control_cases: DataFrame):
        """
        Match control cases for the corresponding incident cases using the query provided by
        :func:`get_matching_control_query`

        :param incident_cases:
        :param control_cases:
        :return:
        """
        incident_cases.createOrReplaceGlobalTempView('positives')
        control_cases.createOrReplaceGlobalTempView('negatives')

        matched_negative_hf_cases = self.spark.sql(self.get_matching_control_query())

        self.spark.sql('DROP VIEW global_temp.positives')
        self.spark.sql('DROP VIEW global_temp.negatives')

        return matched_negative_hf_cases

    def get_matching_control_query(self):
        return NEGATIVE_CONTROL_MATCH_QUERY_TEMPLATE.format(
            age_match_window=self._index_date_match_window)


class RetrospectiveCohortBuilderBase(AbstractCaseControlCohortBuilderBase):

    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        ehr_records = extract_ehr_records(self.spark,
                                          self._input_folder,
                                          self._ehr_table_list)

        cohort_ehr_records = ehr_records.join(cohort, 'person_id') \
            .where(
            ehr_records['date'] <= F.date_sub(cohort['index_date'],
                                              self._prediction_window)).where(
            ehr_records['date'] >= F.date_sub(cohort['index_date'],
                                              self.get_total_window())).select(
            ehr_records['person_id'], ehr_records['standard_concept_id'],
            ehr_records['date'], ehr_records['visit_occurrence_id'],
            ehr_records['domain'])

        return create_sequence_data(cohort_ehr_records, None)

    @abstractmethod
    def preprocess_dependencies(self):
        pass

    @abstractmethod
    def create_incident_cases(self):
        pass

    @abstractmethod
    def create_control_cases(self):
        pass


class ProspectiveCohortBuilderBase(AbstractCaseControlCohortBuilderBase):

    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        ehr_records = extract_ehr_records(self.spark,
                                          self._input_folder,
                                          self._ehr_table_list)

        cohort_ehr_records = ehr_records.join(cohort, 'person_id') \
            .where(ehr_records['date'] >= cohort['index_date']) \
            .where(
            ehr_records['date'] <= F.date_add(cohort['index_date'], self._observation_window)) \
            .select(ehr_records['person_id'], ehr_records['standard_concept_id'],
                    ehr_records['date'], ehr_records['visit_occurrence_id'],
                    ehr_records['domain'])

        return create_sequence_data(cohort_ehr_records, None)

    @abstractmethod
    def create_incident_cases(self):
        pass

    @abstractmethod
    def create_control_cases(self):
        pass


class LastVisitCohortBuilderBase(AbstractCaseControlCohortBuilderBase):

    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        ehr_records = extract_ehr_records(self.spark,
                                          self._input_folder,
                                          self._ehr_table_list)

        cohort_ehr_records = ehr_records.join(cohort, 'visit_occurrence_id') \
            .select(ehr_records['person_id'], ehr_records['standard_concept_id'],
                    ehr_records['date'], ehr_records['visit_occurrence_id'],
                    ehr_records['domain'])

        return create_sequence_data(cohort_ehr_records, None)

    @abstractmethod
    def preprocess_dependencies(self):
        pass

    @abstractmethod
    def create_incident_cases(self):
        pass

    @abstractmethod
    def create_control_cases(self):
        pass


class NestedCohortBuilderBase(RetrospectiveCohortBuilderBase):

    def __init__(self, target_cohort: DataFrame, *args, **kwargs):
        super(NestedCohortBuilderBase, self).__init__(*args, **kwargs)
        self._target_cohort = target_cohort

    @abstractmethod
    def preprocess_dependencies(self):
        pass

    @abstractmethod
    def create_incident_cases(self):
        pass

    @abstractmethod
    def create_control_cases(self):
        pass
