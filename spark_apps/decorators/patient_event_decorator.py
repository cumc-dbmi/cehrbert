import math
from abc import ABC, abstractmethod

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F, Window as W, types as T

from const.common import MEASUREMENT, CATEGORICAL_MEASUREMENT


class PatientEventDecorator(ABC):
    def __init__(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def decorate(self, patient_event):
        pass


class PatientEventBaseDecorator(
    PatientEventDecorator
):
    def decorate(
            self,
            patient_event: DataFrame
    ):
        # Convert Date to days since epoch
        days_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60)).cast('int')
        weeks_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60 * 7)).cast('int')
        visit_date_udf = F.first('days_since_epoch').over(
            W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy(
                'days_since_epoch'))

        visit_rank_udf = F.dense_rank().over(
            W.partitionBy('cohort_member_id', 'person_id').orderBy('visit_start_date'))
        visit_segment_udf = F.col('visit_rank_order') % F.lit(2) + 1

        patient_event = patient_event \
            .withColumn('priority', F.lit(0)) \
            .withColumn('days_since_epoch', days_since_epoch_udf) \
            .withColumn('date_in_week', weeks_since_epoch_udf) \
            .withColumn('visit_start_date', visit_date_udf) \
            .withColumn('visit_rank_order', visit_rank_udf) \
            .withColumn('visit_segment', visit_segment_udf) \
            .withColumn('concept_value_mask', (F.col('domain') == MEASUREMENT).cast('int')) \
            .withColumn('mlm_skip_value',
                        (F.col('domain').isin([MEASUREMENT, CATEGORICAL_MEASUREMENT])).cast('int'))
        return patient_event


class PatientEventAttDecorator(PatientEventDecorator):
    def __init__(
            self,
            include_visit_type,
            exclude_visit_tokens
    ):
        self._include_visit_type = include_visit_type
        self._exclude_visit_tokens = exclude_visit_tokens

    def decorate(
            self,
            patient_event: DataFrame
    ):
        # Udf for identifying the earliest date associated with a visit_occurrence_id
        visit_start_date_udf = F.first('date').over(
            W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy('date'))

        # Udf for identifying the latest date associated with a visit_occurrence_id
        visit_end_date_udf = F.first('date').over(
            W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy(
                F.col('date').desc()))

        # Udf for identifying the first concept
        first_concept_rank_udf = F.row_number().over(
            W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy('date')
        )

        # Udf for identifying the last concept
        last_concept_rank_udf = F.row_number().over(
            W.partitionBy('cohort_member_id', 'person_id', 'visit_occurrence_id').orderBy(
                F.desc('date'))
        )

        visit_start_events = patient_event.withColumn('date', visit_start_date_udf) \
            .withColumn('standard_concept_id', F.lit('VS')) \
            .withColumn('domain', F.lit('visit')) \
            .withColumn('concept_value', F.lit(-1)) \
            .withColumn('rank', first_concept_rank_udf) \
            .withColumn('priority', F.lit('-2')) \
            .where('rank = 1') \
            .drop('rank').distinct()

        visit_end_events = patient_event.withColumn('date', visit_end_date_udf) \
            .withColumn('standard_concept_id', F.lit('VE')) \
            .withColumn('domain', F.lit('visit')) \
            .withColumn('concept_value', F.lit(-1)) \
            .withColumn('rank', last_concept_rank_udf) \
            .withColumn('priority', F.lit('1')) \
            .where('rank = 1') \
            .drop('rank').distinct()

        # Convert Date to days since epoch
        days_since_epoch_udf = (F.unix_timestamp('date') / F.lit(24 * 60 * 60)).cast('int')
        # Get the prev days_since_epoch
        prev_days_since_epoch_udf = F.lag('days_since_epoch').over(
            W.partitionBy('cohort_member_id', 'person_id').orderBy('date', 'priority',
                                                                   'visit_occurrence_id'))
        # Compute the time difference between the current record and the previous record
        time_delta_udf = F.when(F.col('prev_days_since_epoch').isNull(), 0).otherwise(
            F.col('days_since_epoch') - F.col('prev_days_since_epoch'))

        # Udf for calculating the time token
        time_token_udf = F.udf(time_token_func, T.StringType())

        patient_event = patient_event.union(visit_start_events).union(visit_end_events) \
            .withColumn('days_since_epoch', days_since_epoch_udf) \
            .withColumn('prev_days_since_epoch', prev_days_since_epoch_udf) \
            .withColumn('time_delta', time_delta_udf) \
            .withColumn('time_token', time_token_udf('time_delta'))

        time_token_insertions = patient_event.where('standard_concept_id = "VS"') \
            .withColumn('standard_concept_id', F.col('time_token')) \
            .withColumn('priority', F.lit(-3)) \
            .withColumn('visit_segment', F.lit(0)) \
            .withColumn('date_in_week', F.lit(0)) \
            .withColumn('age', F.lit(-1)) \
            .withColumn('visit_concept_id', F.lit(0)) \
            .where('prev_days_since_epoch IS NOT NULL')

        patient_event = patient_event.union(time_token_insertions).distinct() \
            .drop('prev_days_since_epoch',
                  'time_delta',
                  'time_token')

        if self._include_visit_type:
            # insert visit type after the VS token
            visit_type_tokens = patient_event.where('standard_concept_id = "VS"') \
                .withColumn('standard_concept_id', F.col('visit_concept_id')) \
                .withColumn('priority', F.lit(-1))
            patient_event = patient_event.union(visit_type_tokens)

        if self._exclude_visit_tokens:
            patient_event = patient_event.filter(
                ~F.col('standard_concept_id').isin(['VS', 'VE']))

        return patient_event


class DemographicPromptDecorator(
    PatientEventDecorator
):
    def __init__(
            self,
            patient_demographic
    ):
        self._patient_demographic = patient_demographic

    def decorate(
            self,
            patient_event: DataFrame
    ):
        if self._patient_demographic is None:
            return patient_event

        # Get the first token of the patient history
        first_token_udf = F.row_number().over(
            W.partitionBy('cohort_member_id', 'person_id').orderBy('date')
        )

        patient_event = patient_event \
            .withColumn('token_order', first_token_udf) \
            .where('token_order = 1') \
            .drop('token_order')

        # Udf for identifying the earliest date associated with a visit_occurrence_id
        sequence_start_year_token = patient_event \
            .withColumn('standard_concept_id',
                        F.concat(F.lit('year:'), F.year('date').cast(T.StringType()))) \
            .withColumn('priority', F.lit(-10)) \
            .withColumn('visit_segment', F.lit(0)) \
            .withColumn('date_in_week', F.lit(0)) \
            .withColumn('age', F.lit(-1))

        age_at_first_visit_udf = F.ceil(
            F.months_between(F.col('date'), F.col('birth_datetime')) / F.lit(12)
        )
        sequence_age_token = patient_event \
            .withColumn('standard_concept_id',
                        F.concat(F.lit('age:'), age_at_first_visit_udf.cast(T.StringType()))) \
            .withColumn('priority', F.lit(-9))

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

        patient_event = patient_event.union(sequence_start_year_token)
        patient_event = patient_event.union(sequence_age_token)
        patient_event = patient_event.union(sequence_gender_token)
        patient_event = patient_event.union(sequence_race_token)

        return patient_event


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
