import math
from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, Window as W, types as T

from const.common import MEASUREMENT, CATEGORICAL_MEASUREMENT


class AttType(Enum):
    DAY = 'day'
    WEEK = 'week'
    MONTH = 'month'
    CEHR_BERT = 'cehr_bert'
    MIX = 'mix'


class PatientEventDecorator(ABC):
    @abstractmethod
    def _decorate(self, patient_events):
        pass

    def decorate(self, patient_events):
        decorated_patient_events = self._decorate(patient_events)
        self.validate(decorated_patient_events)
        return decorated_patient_events

    @classmethod
    def get_required_columns(cls):
        return set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
                    'visit_occurrence_id', 'domain', 'concept_value', 'visit_rank_order',
                    'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
                    'mlm_skip_value', 'age', 'visit_concept_id', 'visit_start_date'])

    def validate(self, patient_events: DataFrame):
        actual_column_set = set(patient_events.columns)
        expected_column_set = set(self.get_required_columns())
        if actual_column_set != expected_column_set:
            diff_left = actual_column_set - expected_column_set
            diff_right = expected_column_set - actual_column_set
            raise RuntimeError(
                f'{self}\n'
                f'actual_column_set - expected_column_set: {diff_left}\n'
                f'expected_column_set - actual_column_set: {diff_right}'
            )


class PatientEventBaseDecorator(
    PatientEventDecorator
):
    # output_columns = [
    #     'cohort_member_id', 'person_id', 'concept_ids', 'visit_segments', 'orders',
    #     'dates', 'ages', 'visit_concept_orders', 'num_of_visits', 'num_of_concepts',
    #     'concept_value_masks', 'concept_values', 'mlm_skip_values',
    #     'visit_concept_ids'
    # ]
    def __init__(self, visit_occurrence):
        self._visit_occurrence = visit_occurrence

    def _decorate(
            self,
            patient_events: DataFrame
    ):
        """
        patient_events contains the following columns (cohort_member_id, person_id,
        standard_concept_id, date, visit_occurrence_id, domain, concept_value)

        :param patient_events:
        :return:
        """

        # todo: create an assertion the dataframe contains the above columns

        # Add visit_start_date to the patient_events dataframe and create the visit rank
        visit_rank_udf = F.row_number().over(
            W.partitionBy('person_id').orderBy('visit_start_date')
        )
        visit_segment_udf = F.col('visit_rank_order') % F.lit(2) + 1
        visits = self._visit_occurrence.select(
            'visit_occurrence_id', 'visit_start_date', 'person_id'
        ).withColumn('visit_rank_order', visit_rank_udf) \
            .withColumn('visit_segment', visit_segment_udf) \
            .drop('person_id')

        # Add visit_rank_order, and visit_segment to patient_events
        patient_events = patient_events.join(visits, 'visit_occurrence_id')

        # Set the priority for the events.
        # Create the week since epoch UDF
        weeks_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')
        patient_events = patient_events \
            .withColumn('priority', F.lit(0)) \
            .withColumn('date_in_week', weeks_since_epoch_udf)

        # Create the concept_value_mask field to indicate whether domain values should be skipped
        # As of now only measurement has values, so other domains would be skipped.
        patient_events = patient_events \
            .withColumn('concept_value_mask', (F.col('domain') == MEASUREMENT).cast('int')) \
            .withColumn('mlm_skip_value',
                        (F.col('domain').isin([MEASUREMENT, CATEGORICAL_MEASUREMENT])).cast('int'))

        if 'concept_value' not in patient_events.schema.fieldNames():
            patient_events = patient_events.withColumn('concept_value', F.lit(-1.0))

        # (cohort_member_id, person_id, standard_concept_id, date, visit_occurrence_id, domain,
        # concept_value, visit_rank_order, visit_segment, priority, date_in_week,
        # concept_value_mask, mlm_skip_value, age)
        return patient_events


class PatientEventAttDecorator(PatientEventDecorator):
    def __init__(
            self,
            visit_occurrence,
            include_visit_type,
            exclude_visit_tokens,
            att_type: AttType
    ):
        self._visit_occurrence = visit_occurrence
        self._include_visit_type = include_visit_type
        self._exclude_visit_tokens = exclude_visit_tokens
        self._att_type = att_type

    def _decorate(
            self,
            patient_events: DataFrame
    ):
        # visits should the following columns (person_id,
        # visit_concept_id, visit_start_date, visit_occurrence_id, domain, concept_value)
        cohort_member_person_pair = patient_events.select('person_id', 'cohort_member_id').distinct()
        valid_visit_ids = patient_events.select('visit_occurrence_id').distinct()
        visits = self._visit_occurrence.select(
            'person_id',
            F.col('visit_start_date').cast(T.DateType()).alias('date'),
            F.col('visit_start_date').cast(T.DateType()).alias('visit_start_date'),
            'visit_concept_id',
            'visit_occurrence_id',
            F.lit('visit').alias('domain'),
            F.lit(-1).alias('concept_value'),
            F.lit(0).alias('concept_value_mask'),
            F.lit(0).alias('mlm_skip_value'),
            'age'
        ).join(cohort_member_person_pair, 'person_id') \
            .join(valid_visit_ids, 'visit_occurrence_id')

        # Add visit_rank and visit_segment to the visits dataframe and create the visit rank
        visit_rank_udf = F.row_number().over(
            W.partitionBy('person_id').orderBy('date')
        )
        visit_segment_udf = F.col('visit_rank_order') % F.lit(2) + 1
        weeks_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')
        visits = visits \
            .withColumn('visit_rank_order', visit_rank_udf) \
            .withColumn('visit_segment', visit_segment_udf) \
            .withColumn('date_in_week', weeks_since_epoch_udf)

        visits.cache()

        # (cohort_member_id, person_id, standard_concept_id, date, visit_occurrence_id, domain,
        # concept_value, visit_rank_order, visit_segment, priority, date_in_week,
        # concept_value_mask, mlm_skip_value)
        visit_start_events = visits \
            .withColumn('standard_concept_id', F.lit('VS')) \
            .withColumn('priority', F.lit(-2))

        visit_end_events = visits \
            .withColumn('standard_concept_id', F.lit('VE')) \
            .withColumn('priority', F.lit(1))

        # Get the prev days_since_epoch
        prev_date_udf = F.lag('date').over(
            W.partitionBy('person_id').orderBy('date', 'visit_occurrence_id')
        )

        # Compute the time difference between the current record and the previous record
        time_delta_udf = F.when(F.col('prev_date').isNull(), 0).otherwise(
            F.datediff('date', 'prev_date')
        )

        # Udf for calculating the time token
        if self._att_type == AttType.DAY:
            att_func = time_day_token
        elif self._att_type == AttType.WEEK:
            att_func = time_week_token
        elif self._att_type == AttType.MONTH:
            att_func = time_month_token
        elif self._att_type == AttType.MIX:
            att_func = time_mix_token
        else:
            att_func = time_token_func

        time_token_udf = F.udf(att_func, T.StringType())

        att_tokens = visits \
            .withColumn('prev_date', prev_date_udf) \
            .withColumn('time_delta', time_delta_udf) \
            .withColumn('standard_concept_id', time_token_udf('time_delta')) \
            .withColumn('priority', F.lit(-3)) \
            .where(F.col('prev_date').isNotNull()) \
            .drop('prev_date', 'time_delta')

        if self._exclude_visit_tokens:
            artificial_tokens = att_tokens
        else:
            artificial_tokens = visit_start_events.union(att_tokens).union(visit_end_events)

        if self._include_visit_type:
            # insert visit type after the VS token
            visit_type_tokens = visits \
                .withColumn('standard_concept_id', F.col('visit_concept_id')) \
                .withColumn('priority', F.lit(-1))
            artificial_tokens = artificial_tokens.union(visit_type_tokens)

        self.validate(patient_events)
        self.validate(artificial_tokens)

        # artificial_tokens = artificial_tokens.select(sorted(artificial_tokens.columns))

        return patient_events.unionByName(artificial_tokens)


class DemographicPromptDecorator(
    PatientEventDecorator
):
    def __init__(
            self,
            patient_demographic
    ):
        self._patient_demographic = patient_demographic

    def _decorate(
            self,
            patient_events: DataFrame
    ):
        if self._patient_demographic is None:
            return patient_events

        # set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
        #      'visit_occurrence_id', 'domain', 'concept_value', 'visit_rank_order',
        #      'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
        #      'mlm_skip_value', 'age', 'visit_concept_id'])

        # Get the first token of the patient history
        first_token_udf = F.row_number().over(
            W.partitionBy('cohort_member_id', 'person_id').orderBy(
                'visit_start_date',
                'visit_occurrence_id',
                'priority',
                'standard_concept_id')
        )

        # Identify the first token of each patient history
        patient_first_token = patient_events \
            .withColumn('token_order', first_token_udf) \
            .where('token_order = 1') \
            .drop('token_order')

        # Udf for identifying the earliest date associated with a visit_occurrence_id
        sequence_start_year_token = patient_first_token \
            .withColumn('standard_concept_id',
                        F.concat(F.lit('year:'), F.year('date').cast(T.StringType()))) \
            .withColumn('priority', F.lit(-10)) \
            .withColumn('visit_segment', F.lit(0)) \
            .withColumn('date_in_week', F.lit(0)) \
            .withColumn('age', F.lit(-1))

        sequence_start_year_token.cache()

        age_at_first_visit_udf = F.ceil(
            F.months_between(F.col('date'), F.col('birth_datetime')) / F.lit(12)
        )
        sequence_age_token = self._patient_demographic.select(
            F.col('person_id'),
            F.col('birth_datetime')
        ).join(
            sequence_start_year_token,
            'person_id'
        ).withColumn(
            'standard_concept_id',
            F.concat(F.lit('age:'), age_at_first_visit_udf.cast(T.StringType()))
        ).withColumn('priority', F.lit(-9)).drop('birth_datetime')

        sequence_gender_token = self._patient_demographic.select(
            F.col('person_id'),
            F.col('gender_concept_id')
        ).join(
            sequence_start_year_token,
            'person_id'
        ).withColumn(
            'standard_concept_id',
            F.col('gender_concept_id').cast(T.StringType())
        ).withColumn('priority', F.lit(-8)).drop('gender_concept_id')

        sequence_race_token = self._patient_demographic.select(
            F.col('person_id'),
            F.col('race_concept_id')
        ).join(
            sequence_start_year_token,
            'person_id'
        ).withColumn(
            'standard_concept_id',
            F.col('race_concept_id').cast(T.StringType())
        ).withColumn('priority', F.lit(-7)).drop('race_concept_id')

        patient_events = patient_events.unionByName(sequence_start_year_token)
        patient_events = patient_events.unionByName(sequence_age_token)
        patient_events = patient_events.unionByName(sequence_gender_token)
        patient_events = patient_events.unionByName(sequence_race_token)

        return patient_events


def time_token_func(time_delta):
    if np.isnan(time_delta):
        return None
    if time_delta < 0:
        return 'W-1'
    if time_delta < 28:
        return f'W{str(math.floor(time_delta / 7))}'
    if time_delta < 360:
        return f'M{str(math.floor(time_delta / 30))}'
    return 'LT'


def time_day_token(time_delta):
    if np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f'D{str(time_delta)}'
    return 'LT'


def time_week_token(time_delta):
    if np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f'W{str(math.floor(time_delta / 7))}'
    return 'LT'


def time_month_token(time_delta):
    if np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f'M{str(math.floor(time_delta / 30))}'
    return 'LT'


def time_mix_token(time_delta):
    #        WHEN day_diff <= 7 THEN CONCAT('D', day_diff)
    #         WHEN day_diff <= 30 THEN CONCAT('W', ceil(day_diff / 7))
    #         WHEN day_diff <= 360 THEN CONCAT('M', ceil(day_diff / 30))
    #         WHEN day_diff <= 720 THEN CONCAT('Q', ceil(day_diff / 90))
    #         WHEN day_diff <= 1440 THEN CONCAT('Y', ceil(day_diff / 360))
    #         ELSE 'LT'
    if np.isnan(time_delta):
        return None
    if time_delta <= 7:
        return f'D{str(time_delta)}'
    if time_delta <= 30:
        # e.g. 8 -> W2
        return f'W{str(math.ceil(time_delta / 7))}'
    if time_delta <= 360:
        # e.g. 31 -> M2
        return f'M{str(math.ceil(time_delta / 30))}'
    # if time_delta <= 720:
    #     # e.g. 361 -> Q5
    #     return f'Q{str(math.ceil(time_delta / 90))}'
    # if time_delta <= 1080:
    #     # e.g. 1081 -> Y2
    #     return f'Y{str(math.ceil(time_delta / 360))}'
    return 'LT'
