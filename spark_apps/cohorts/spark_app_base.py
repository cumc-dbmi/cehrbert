import os
import re
from abc import ABC
from typing import List
from pandas import to_datetime

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession

from utils.spark_utils import *
from utils.logging_utils import *
from spark_apps.cohorts.query_builder import QueryBuilder

COHORT_TABLE_NAME = 'cohort'
PERSON = 'person'
OBSERVATION_PERIOD = 'observation_period'
DEFAULT_DEPENDENCY = ['person', 'observation_period', 'concept', 'concept_ancestor',
                      'concept_relationship']


def cohort_validator(required_columns_attribute):
    """
    Decorator for validating the cohort dataframe returned by build function in
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


def instantiate_dependencies(spark, input_folder, dependency_list):
    dependency_dict = dict()
    for domain_table_name in dependency_list + DEFAULT_DEPENDENCY:
        table = preprocess_domain_table(spark, input_folder, domain_table_name)
        table.createOrReplaceGlobalTempView(domain_table_name)
        dependency_dict[domain_table_name] = table
    return dependency_dict


def validate_date_folder(input_folder, table_list):
    for domain_table_name in table_list:
        parquet_file_path = os.path.join(input_folder, domain_table_name)
        if not os.path.exists(parquet_file_path):
            raise FileExistsError(f'{parquet_file_path} does not exist')


def validate_folder(folder_path):
    if not os.path.exists(folder_path):
        raise FileExistsError(f'{folder_path} does not exist')


class BaseCohortBuilder(ABC):
    cohort_required_columns = ['person_id', 'index_date', 'visit_occurrence_id']

    def __init__(self,
                 query_builder: QueryBuilder,
                 input_folder: str,
                 output_folder: str,
                 date_lower_bound: str,
                 date_upper_bound: str,
                 age_lower_bound: int,
                 age_upper_bound: int,
                 prior_observation_period: int,
                 post_observation_period: int):

        self._query_builder = query_builder
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._date_lower_bound = date_lower_bound
        self._date_upper_bound = date_upper_bound
        self._age_lower_bound = age_lower_bound
        self._age_upper_bound = age_upper_bound
        self._prior_observation_period = prior_observation_period
        self._post_observation_period = post_observation_period
        cohort_name = re.sub('[^a-z0-9]+', '_', self._query_builder.get_cohort_name().lower())
        self._output_data_folder = os.path.join(self._output_folder, cohort_name)

        self.get_logger().info(f'query_builder: {query_builder}\n'
                               f'input_folder: {input_folder}\n'
                               f'output_folder: {output_folder}\n'
                               f'date_lower_bound: {date_lower_bound}\n'
                               f'date_upper_bound: {date_upper_bound}\n'
                               f'age_lower_bound: {age_lower_bound}\n'
                               f'age_upper_bound: {age_upper_bound}\n'
                               f'prior_observation_period: {prior_observation_period}\n'
                               f'post_observation_period: {post_observation_period}\n')

        # Validate the age range, observation_window and prediction_window
        self._validate_integer_inputs()
        # Validate the input and output folders
        validate_folder(self._input_folder)
        validate_folder(self._output_folder)
        # Validate if the data folders exist
        validate_date_folder(self._input_folder, self._query_builder.get_dependency_list())

        self.spark = SparkSession.builder.appName(
            f'Generate {self._query_builder.get_cohort_name()}').getOrCreate()

        self._dependency_dict = instantiate_dependencies(self.spark, self._input_folder,
                                                         self._query_builder.get_dependency_list())

    @cohort_validator('cohort_required_columns')
    def create_cohort(self):
        """
        Create cohort
        :return:
        """
        # Build the ancestor tables for the main query to use  if the ancestor_table_specs are
        # available
        if self._query_builder.get_ancestor_table_specs():
            for ancestor_table_spec in self._query_builder.get_ancestor_table_specs():
                func = get_descendant_concept_ids if ancestor_table_spec.is_standard else build_ancestry_table_for
                ancestor_table = func(self.spark, ancestor_table_spec.ancestor_concept_ids)
                ancestor_table.createOrReplaceGlobalTempView(ancestor_table_spec.table_name)

        # Build the dependencies for the main query to use if the dependency_queries are available
        if self._query_builder.get_dependency_queries():
            for dependency_query in self._query_builder.get_dependency_queries():
                query = dependency_query.query_template.format(**dependency_query.parameters)
                dependency_table = self.spark.sql(query)
                dependency_table.createOrReplaceGlobalTempView(dependency_query.table_name)

        main_query = self._query_builder.get_query()
        cohort = self.spark.sql(main_query.query_template.format(**main_query.parameters))
        cohort.createOrReplaceGlobalTempView(main_query.table_name)

        # Post process the cohort if the post_process_queries are available
        if self._query_builder.get_post_process_queries():
            for post_query in self._query_builder.get_post_process_queries():
                cohort = self.spark.sql(post_query.query_template.format(**post_query.parameters))
                cohort.createOrReplaceGlobalTempView(main_query.table_name)

        return cohort

    def build(self):
        """
        Build the cohort and write the dataframe as parquet files to _output_data_folder
        """
        cohort = self.create_cohort()

        cohort = self._apply_observation_period(cohort)

        cohort = self._add_demographics(cohort)

        cohort = cohort.where(F.col('age').between(self._age_lower_bound, self._age_upper_bound)) \
            .where(F.col('index_date').between(to_datetime(self._date_lower_bound),
                                               to_datetime(self._date_upper_bound)))

        cohort.write.mode('overwrite').parquet(self._output_data_folder)

        return self

    def load_cohort(self):
        return self.spark.read.parquet(self._output_data_folder)

    @cohort_validator('cohort_required_columns')
    def _apply_observation_period(self, cohort: DataFrame):
        cohort.createOrReplaceGlobalTempView('cohort')

        qualified_cohort = self.spark.sql("""
        SELECT
            c.*
        FROM global_temp.cohort AS c 
        JOIN global_temp.observation_period AS p 
            ON c.person_id = p.person_id 
                AND DATE_ADD(c.index_date, -{prior_observation_period}) >= p.observation_period_start_date
                AND DATE_ADD(c.index_date, {post_observation_period}) <= p.observation_period_end_date
        """.format(prior_observation_period=self._prior_observation_period,
                   post_observation_period=self._post_observation_period))

        self.spark.sql(f'DROP VIEW global_temp.cohort')
        return qualified_cohort

    @cohort_validator('cohort_required_columns')
    def _add_demographics(self, cohort: DataFrame):
        return cohort.join(self._dependency_dict[PERSON], 'person_id') \
            .withColumn('age', F.year('index_date') - F.col('year_of_birth')) \
            .select(F.col('person_id'),
                    F.col('age'),
                    F.col('gender_concept_id'),
                    F.col('race_concept_id'),
                    F.col('index_date'),
                    F.col('visit_occurrence_id')).distinct()

    def _validate_integer_inputs(self):
        assert self._age_lower_bound >= 0
        assert self._age_upper_bound > 0
        assert self._age_lower_bound < self._age_upper_bound
        assert self._prior_observation_period >= 0
        assert self._post_observation_period >= 0

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


class NestedCohortBuilder:
    def __init__(self,
                 cohort_name: str,
                 input_folder: str,
                 output_folder: str,
                 target_cohort: DataFrame,
                 outcome_cohort: DataFrame,
                 ehr_table_list: List[str],
                 observation_window: int,
                 hold_off_window: int,
                 prediction_start_days: int,
                 prediction_window: int,
                 is_window_post_index: bool = False,
                 include_visit_type: bool = True,
                 is_feature_concept_frequency: bool = False,
                 is_roll_up_concept: bool = False,
                 is_new_patient_representation: bool = False):
        self._cohort_name = cohort_name
        self._input_folder = input_folder
        self._output_folder = output_folder
        self._target_cohort = target_cohort
        self._outcome_cohort = outcome_cohort
        self._ehr_table_list = ehr_table_list
        self._observation_window = observation_window
        self._hold_off_window = hold_off_window
        self._prediction_start_days = prediction_start_days
        self._prediction_window = prediction_window
        self._is_observation_post_index = is_window_post_index
        self._include_visit_type = include_visit_type
        self._is_feature_concept_frequency = is_feature_concept_frequency
        self._is_roll_up_concept = is_roll_up_concept
        self._is_new_patient_representation = is_new_patient_representation
        self._output_data_folder = os.path.join(self._output_folder,
                                                re.sub('[^a-z0-9]+', '_',
                                                       self._cohort_name.lower()))

        self.get_logger().info(f'cohort_name: {cohort_name}\n'
                               f'input_folder: {input_folder}\n'
                               f'output_folder: {output_folder}\n'
                               f'ehr_table_list: {ehr_table_list}\n'
                               f'observation_window: {observation_window}\n'
                               f'prediction_start_days: {prediction_start_days}\n'
                               f'prediction_window: {prediction_window}\n'
                               f'hold_off_window: {hold_off_window}\n'
                               f'is_window_post_index: {is_window_post_index}\n'
                               f'include_visit_type: {include_visit_type}\n'
                               f'is_feature_concept_frequency: {is_feature_concept_frequency}\n'
                               f'is_roll_up_concept: {is_roll_up_concept}\n'
                               f'is_new_patient_representation: {is_new_patient_representation}\n')

        self.spark = SparkSession.builder.appName(f'Generate {self._cohort_name}').getOrCreate()
        self._dependency_dict = instantiate_dependencies(self.spark, self._input_folder,
                                                         DEFAULT_DEPENDENCY)

        # Validate the input and output folders
        validate_folder(self._input_folder)
        validate_folder(self._output_folder)
        # Validate if the data folders exist
        validate_date_folder(self._input_folder, self._ehr_table_list)

    def build(self):
        self._target_cohort.createOrReplaceGlobalTempView('target_cohort')
        self._outcome_cohort.createOrReplaceGlobalTempView('outcome_cohort')

        prediction_start_days = self._prediction_start_days
        prediction_window = self._prediction_window

        if self._is_observation_post_index:
            prediction_start_days += self._observation_window + self._hold_off_window
            prediction_window += self._observation_window + self._hold_off_window

        cohort = self.spark.sql("""
            SELECT DISTINCT
                t.*,
                CAST(ISNOTNULL(o.person_id) AS INT) AS label
            FROM global_temp.target_cohort AS t 
            JOIN global_temp.observation_period AS op
                ON t.person_id = op.person_id
            LEFT JOIN global_temp.outcome_cohort AS o
                ON t.person_id = o.person_id
                    AND o.index_date BETWEEN DATE_ADD(t.index_date, {prediction_start_days}) 
                        AND DATE_ADD(t.index_date, {prediction_window})
            WHERE DATE_ADD(t.index_date, {prediction_window}) <= op.observation_period_end_date
                OR o.person_id IS NOT NULL
        """.format(prediction_start_days=prediction_start_days,
                   prediction_window=prediction_window))

        ehr_records_for_cohorts = self.extract_ehr_records_for_cohort(cohort)
        cohort = cohort.join(ehr_records_for_cohorts, 'person_id')
        cohort.write.mode('overwrite').parquet(self._output_data_folder)

    def extract_ehr_records_for_cohort(self, cohort: DataFrame):
        ehr_records = extract_ehr_records(self.spark, self._input_folder, self._ehr_table_list,
                                          self._include_visit_type, self._is_roll_up_concept)

        if self._is_observation_post_index:
            record_window_filter = ehr_records['date'].between(
                cohort['index_date'], F.date_add(cohort['index_date'], self._observation_window))
        else:
            record_window_filter = ehr_records['date'].between(
                F.date_sub(cohort['index_date'], self._observation_window),
                F.date_sub(cohort['index_date'], self._hold_off_window))

        cohort_ehr_records = ehr_records.join(cohort, 'person_id').where(record_window_filter) \
            .select([ehr_records[field_name] for field_name in ehr_records.schema.fieldNames()])

        if self._is_feature_concept_frequency:
            return create_concept_frequency_data(cohort_ehr_records, None)

        if self._is_new_patient_representation:
            validate_date_folder(self._input_folder, [VISIT_OCCURRENCE])
            visit_occurrence = preprocess_domain_table(self.spark,
                                                       self._input_folder,
                                                       VISIT_OCCURRENCE)
            return create_sequence_data_time_delta_embedded(cohort_ehr_records, visit_occurrence,
                                                            include_visit_type=self._include_visit_type)

        return create_sequence_data(cohort_ehr_records, None, self._include_visit_type)

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


def create_prediction_cohort(spark_args,
                             target_query_builder: QueryBuilder,
                             outcome_query_builder: QueryBuilder,
                             ehr_table_list):
    """
    TODO
    :param spark_args:
    :param target_query_builder:
    :param outcome_query_builder:
    :param ehr_table_list:
    :return:
    """
    cohort_name = spark_args.cohort_name
    input_folder = spark_args.input_folder
    output_folder = spark_args.output_folder
    date_lower_bound = spark_args.date_lower_bound
    date_upper_bound = spark_args.date_upper_bound
    age_lower_bound = spark_args.age_lower_bound
    age_upper_bound = spark_args.age_upper_bound
    observation_window = spark_args.observation_window
    prediction_start_days = spark_args.prediction_start_days
    prediction_window = spark_args.prediction_window
    hold_off_window = spark_args.hold_off_window
    include_visit_type = spark_args.include_visit_type
    is_feature_concept_frequency = spark_args.is_feature_concept_frequency
    is_roll_up_concept = spark_args.is_roll_up_concept
    is_window_post_index = spark_args.is_window_post_index
    is_new_patient_representation = spark_args.is_new_patient_representation

    # Toggle the prior/post observation_period depending on the is_window_post_index flag
    prior_observation_period = 0 if is_window_post_index else observation_window + hold_off_window
    post_observation_period = observation_window + hold_off_window if is_window_post_index else 0

    # Generate the target cohort
    target_cohort = BaseCohortBuilder(
        query_builder=target_query_builder,
        input_folder=input_folder,
        output_folder=output_folder,
        date_lower_bound=date_lower_bound,
        date_upper_bound=date_upper_bound,
        age_lower_bound=age_lower_bound,
        age_upper_bound=age_upper_bound,
        prior_observation_period=prior_observation_period,
        post_observation_period=post_observation_period).build().load_cohort()

    # Generate the outcome cohort
    outcome_cohort = BaseCohortBuilder(
        query_builder=outcome_query_builder,
        input_folder=input_folder,
        output_folder=output_folder,
        date_lower_bound=date_lower_bound,
        date_upper_bound=date_upper_bound,
        age_lower_bound=age_lower_bound,
        age_upper_bound=age_upper_bound,
        prior_observation_period=0,
        post_observation_period=0).build().load_cohort()

    NestedCohortBuilder(cohort_name=cohort_name,
                        input_folder=input_folder,
                        output_folder=output_folder,
                        target_cohort=target_cohort,
                        outcome_cohort=outcome_cohort,
                        ehr_table_list=ehr_table_list,
                        observation_window=observation_window,
                        hold_off_window=hold_off_window,
                        prediction_start_days=prediction_start_days,
                        prediction_window=prediction_window,
                        is_window_post_index=is_window_post_index,
                        include_visit_type=include_visit_type,
                        is_feature_concept_frequency=is_feature_concept_frequency,
                        is_roll_up_concept=is_roll_up_concept,
                        is_new_patient_representation=is_new_patient_representation).build()
