import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from pyspark.sql import types as T

from ...const.common import CATEGORICAL_MEASUREMENT, MEASUREMENT


class AttType(Enum):
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    CEHR_BERT = "cehrbert"
    MIX = "mix"
    NONE = "none"


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
        return set(
            [
                "cohort_member_id",
                "person_id",
                "standard_concept_id",
                "date",
                "datetime",
                "visit_occurrence_id",
                "domain",
                "concept_value",
                "visit_rank_order",
                "visit_segment",
                "priority",
                "date_in_week",
                "concept_value_mask",
                "mlm_skip_value",
                "age",
                "visit_concept_id",
                "visit_start_date",
                "visit_start_datetime",
                "visit_concept_order",
                "concept_order",
            ]
        )

    def validate(self, patient_events: DataFrame):
        actual_column_set = set(patient_events.columns)
        expected_column_set = set(self.get_required_columns())
        if actual_column_set != expected_column_set:
            diff_left = actual_column_set - expected_column_set
            diff_right = expected_column_set - actual_column_set
            raise RuntimeError(
                f"{self}\n"
                f"actual_column_set - expected_column_set: {diff_left}\n"
                f"expected_column_set - actual_column_set: {diff_right}"
            )


class PatientEventBaseDecorator(PatientEventDecorator):
    # output_columns = [
    #     'cohort_member_id', 'person_id', 'concept_ids', 'visit_segments', 'orders',
    #     'dates', 'ages', 'visit_concept_orders', 'num_of_visits', 'num_of_concepts',
    #     'concept_value_masks', 'concept_values', 'mlm_skip_values',
    #     'visit_concept_ids'
    # ]
    def __init__(self, visit_occurrence):
        self._visit_occurrence = visit_occurrence

    def _decorate(self, patient_events: DataFrame):
        """
        Patient_events contains the following columns (cohort_member_id, person_id,.

        standard_concept_id, date, visit_occurrence_id, domain, concept_value)

        :param patient_events:
        :return:
        """

        # todo: create an assertion the dataframe contains the above columns

        valid_visit_ids = patient_events.select(
            "visit_occurrence_id", "cohort_member_id"
        ).distinct()

        # Add visit_start_date to the patient_events dataframe and create the visit rank
        visit_rank_udf = F.row_number().over(
            W.partitionBy("person_id", "cohort_member_id").orderBy(
                "visit_start_datetime", "is_inpatient", "expired", "visit_occurrence_id"
            )
        )
        visit_segment_udf = F.col("visit_rank_order") % F.lit(2) + 1

        # The visit records are joined to the cohort members (there could be multiple entries for the same patient)
        # if multiple entries are present, we duplicate the visit records for those.
        visits = (
            self._visit_occurrence.join(valid_visit_ids, "visit_occurrence_id")
            .select(
                "person_id",
                "cohort_member_id",
                "visit_occurrence_id",
                "visit_end_date",
                F.col("visit_start_date").cast(T.DateType()).alias("visit_start_date"),
                F.to_timestamp("visit_start_datetime").alias("visit_start_datetime"),
                F.col("visit_concept_id")
                .cast("int")
                .isin([9201, 262, 8971, 8920])
                .cast("int")
                .alias("is_inpatient"),
                F.when(
                    F.col("discharged_to_concept_id").cast("int") == 4216643, F.lit(1)
                )
                .otherwise(F.lit(0))
                .alias("expired"),
            )
            .withColumn("visit_rank_order", visit_rank_udf)
            .withColumn("visit_segment", visit_segment_udf)
            .drop("person_id", "expired")
        )

        # Determine the concept order depending on the visit type. For outpatient visits, we assume the concepts to
        # have the same order, whereas for inpatient visits, the concept order is determined by the time stamp.
        # the concept order needs to be generated per each cohort member because the same visit could be used
        # in multiple cohort's histories of the same patient
        concept_order_udf = F.when(
            F.col("is_inpatient") == 1,
            F.dense_rank().over(
                W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy(
                    "datetime"
                )
            ),
        ).otherwise(F.lit(1))

        # Determine the global visit concept order for each patient, this takes both visit_rank_order and concept_order
        # into account when assigning this new order.
        # e.g. visit_rank_order = [1, 1, 2, 2], concept_order = [1, 1, 1, 2] -> visit_concept_order = [1, 1, 2, 3]
        visit_concept_order_udf = F.dense_rank().over(
            W.partitionBy("person_id", "cohort_member_id").orderBy(
                "visit_rank_order", "concept_order"
            )
        )

        # We need to set the visit_end_date as the visit_start_date for outpatient visits
        # For inpatient visits, we use the original visit_end_date if available, otherwise
        # we will infer the visit_end_date using the max(date) of the current visit
        visit_end_date_udf = F.when(
            F.col("is_inpatient") == 1,
            F.coalesce(
                F.col("visit_end_date"),
                F.max("date").over(
                    W.partitionBy("cohort_member_id", "visit_occurrence_id")
                ),
            ),
        ).otherwise(F.col("visit_start_date"))

        # We need to bound the medical event dates between visit_start_date and visit_end_date
        bound_medical_event_date = F.when(
            F.col("date") < F.col("visit_start_date"), F.col("visit_start_date")
        ).otherwise(
            F.when(
                F.col("date") > F.col("visit_end_date"), F.col("visit_end_date")
            ).otherwise(F.col("date"))
        )

        # We need to bound the medical event dates between visit_start_date and visit_end_date
        bound_medical_event_datetime = F.when(
            F.col("datetime") < F.col("visit_start_datetime"),
            F.col("visit_start_datetime"),
        ).otherwise(
            F.when(
                F.col("datetime") > F.col("visit_end_datetime"),
                F.col("visit_end_datetime"),
            ).otherwise(F.col("datetime"))
        )

        patient_events = (
            patient_events.join(visits, ["cohort_member_id", "visit_occurrence_id"])
            .withColumn("visit_end_date", visit_end_date_udf)
            .withColumn("visit_end_datetime", F.date_add("visit_end_date", 1))
            .withColumn(
                "visit_end_datetime", F.expr("visit_end_datetime - INTERVAL 1 MINUTE")
            )
            .withColumn("date", bound_medical_event_date)
            .withColumn("datetime", bound_medical_event_datetime)
            .withColumn("concept_order", concept_order_udf)
            .withColumn("visit_concept_order", visit_concept_order_udf)
            .drop("is_inpatient", "visit_end_date", "visit_end_datetime")
            .distinct()
        )

        # Set the priority for the events.
        # Create the week since epoch UDF
        weeks_since_epoch_udf = (
            F.unix_timestamp("date") / F.lit(24 * 60 * 60 * 7)
        ).cast("int")
        patient_events = patient_events.withColumn("priority", F.lit(0)).withColumn(
            "date_in_week", weeks_since_epoch_udf
        )

        # Create the concept_value_mask field to indicate whether domain values should be skipped
        # As of now only measurement has values, so other domains would be skipped.
        patient_events = patient_events.withColumn(
            "concept_value_mask", (F.col("domain") == MEASUREMENT).cast("int")
        ).withColumn(
            "mlm_skip_value",
            (F.col("domain").isin([MEASUREMENT, CATEGORICAL_MEASUREMENT])).cast("int"),
        )

        if "concept_value" not in patient_events.schema.fieldNames():
            patient_events = patient_events.withColumn("concept_value", F.lit(0.0))

        # (cohort_member_id, person_id, standard_concept_id, date, datetime, visit_occurrence_id, domain,
        # concept_value, visit_rank_order, visit_segment, priority, date_in_week,
        # concept_value_mask, mlm_skip_value, age)
        return patient_events


class PatientEventAttDecorator(PatientEventDecorator):
    def __init__(
        self,
        visit_occurrence,
        include_visit_type,
        exclude_visit_tokens,
        att_type: AttType,
        include_inpatient_hour_token: bool = False,
    ):
        self._visit_occurrence = visit_occurrence
        self._include_visit_type = include_visit_type
        self._exclude_visit_tokens = exclude_visit_tokens
        self._att_type = att_type
        self._include_inpatient_hour_token = include_inpatient_hour_token

    def _decorate(self, patient_events: DataFrame):
        if self._att_type == AttType.NONE:
            return patient_events

        # visits should the following columns (person_id,
        # visit_concept_id, visit_start_date, visit_occurrence_id, domain, concept_value)
        cohort_member_person_pair = patient_events.select(
            "person_id", "cohort_member_id"
        ).distinct()
        valid_visit_ids = patient_events.groupby(
            "cohort_member_id",
            "visit_occurrence_id",
            "visit_segment",
            "visit_rank_order",
        ).agg(
            F.min("visit_concept_order").alias("min_visit_concept_order"),
            F.max("visit_concept_order").alias("max_visit_concept_order"),
            F.min("concept_order").alias("min_concept_order"),
            F.max("concept_order").alias("max_concept_order"),
        )

        visit_occurrence = (
            self._visit_occurrence.select(
                "person_id",
                F.col("visit_start_date").cast(T.DateType()).alias("date"),
                F.col("visit_start_date").cast(T.DateType()).alias("visit_start_date"),
                F.col("visit_start_datetime")
                .cast(T.TimestampType())
                .alias("visit_start_datetime"),
                F.coalesce("visit_end_date", "visit_start_date")
                .cast(T.DateType())
                .alias("visit_end_date"),
                "visit_concept_id",
                "visit_occurrence_id",
                F.lit("visit").alias("domain"),
                F.lit(0.0).alias("concept_value"),
                F.lit(0).alias("concept_value_mask"),
                F.lit(0).alias("mlm_skip_value"),
                "age",
                "discharged_to_concept_id",
            )
            .join(valid_visit_ids, "visit_occurrence_id")
            .join(cohort_member_person_pair, ["person_id", "cohort_member_id"])
        )

        # We assume outpatient visits end on the same day, therefore we start visit_end_date to visit_start_date due
        # to bad date
        visit_occurrence = visit_occurrence.withColumn(
            "visit_end_date",
            F.when(
                F.col("visit_concept_id").isin([9201, 262, 8971, 8920]),
                F.col("visit_end_date"),
            ).otherwise(F.col("visit_start_date")),
        )

        weeks_since_epoch_udf = (
            F.unix_timestamp("date") / F.lit(24 * 60 * 60 * 7)
        ).cast("int")
        visit_occurrence = visit_occurrence.withColumn(
            "date_in_week", weeks_since_epoch_udf
        )

        # Cache visit for faster processing
        visit_occurrence.cache()

        visits = visit_occurrence.drop("discharged_to_concept_id")

        # (cohort_member_id, person_id, standard_concept_id, date, visit_occurrence_id, domain,
        # concept_value, visit_rank_order, visit_segment, priority, date_in_week,
        # concept_value_mask, mlm_skip_value, visit_end_date)
        visit_start_events = (
            visits.withColumn("date", F.col("visit_start_date"))
            .withColumn("datetime", F.to_timestamp("visit_start_date"))
            .withColumn("standard_concept_id", F.lit("VS"))
            .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
            .withColumn("concept_order", F.col("min_concept_order") - 1)
            .withColumn("priority", F.lit(-2))
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        visit_end_events = (
            visits.withColumn("date", F.col("visit_end_date"))
            .withColumn("datetime", F.date_add(F.to_timestamp("visit_end_date"), 1))
            .withColumn("datetime", F.expr("datetime - INTERVAL 1 MINUTE"))
            .withColumn("standard_concept_id", F.lit("VE"))
            .withColumn("visit_concept_order", F.col("max_visit_concept_order"))
            .withColumn("concept_order", F.col("max_concept_order") + 1)
            .withColumn("priority", F.lit(200))
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        # Get the prev days_since_epoch
        prev_visit_end_date_udf = F.lag("visit_end_date").over(
            W.partitionBy("person_id", "cohort_member_id").orderBy("visit_rank_order")
        )

        # Compute the time difference between the current record and the previous record
        time_delta_udf = F.when(F.col("prev_visit_end_date").isNull(), 0).otherwise(
            F.datediff("visit_start_date", "prev_visit_end_date")
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

        att_tokens = (
            visits.withColumn("datetime", F.to_timestamp("date"))
            .withColumn("prev_visit_end_date", prev_visit_end_date_udf)
            .where(F.col("prev_visit_end_date").isNotNull())
            .withColumn("time_delta", time_delta_udf)
            .withColumn(
                "time_delta",
                F.when(F.col("time_delta") < 0, F.lit(0)).otherwise(
                    F.col("time_delta")
                ),
            )
            .withColumn("standard_concept_id", time_token_udf("time_delta"))
            .withColumn("priority", F.lit(-3))
            .withColumn("visit_rank_order", F.col("visit_rank_order"))
            .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
            .withColumn("concept_order", F.lit(0))
            .drop("prev_visit_end_date", "time_delta")
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        if self._exclude_visit_tokens:
            artificial_tokens = att_tokens
        else:
            artificial_tokens = visit_start_events.unionByName(att_tokens).unionByName(
                visit_end_events
            )

        if self._include_visit_type:
            # insert visit type after the VS token
            visit_type_tokens = (
                visits.withColumn("standard_concept_id", F.col("visit_concept_id"))
                .withColumn("datetime", F.to_timestamp("date"))
                .withColumn("visit_concept_order", F.col("min_visit_concept_order"))
                .withColumn("concept_order", F.lit(0))
                .withColumn("priority", F.lit(-1))
                .drop("min_visit_concept_order", "max_visit_concept_order")
                .drop("min_concept_order", "max_concept_order")
            )

            artificial_tokens = artificial_tokens.unionByName(visit_type_tokens)

        artificial_tokens = artificial_tokens.drop("visit_end_date")

        # Retrieving the events that are ONLY linked to inpatient visits
        inpatient_visits = visit_occurrence.where(
            F.col("visit_concept_id").isin([9201, 262, 8971, 8920])
        ).select("visit_occurrence_id", "visit_end_date", "cohort_member_id")
        inpatient_events = patient_events.join(
            inpatient_visits, ["visit_occurrence_id", "cohort_member_id"]
        )

        # Fill in the visit_end_date if null (because some visits are still ongoing at the time of data extraction)
        # Bound the event dates within visit_start_date and visit_end_date
        # Generate a span rank to indicate the position of the group of events
        # Update the priority for each span
        inpatient_events = (
            inpatient_events.withColumn(
                "visit_end_date",
                F.coalesce(
                    "visit_end_date",
                    F.max("date").over(
                        W.partitionBy("cohort_member_id", "visit_occurrence_id")
                    ),
                ),
            )
            .withColumn(
                "date",
                F.when(
                    F.col("date") < F.col("visit_start_date"), F.col("visit_start_date")
                ).otherwise(
                    F.when(
                        F.col("date") > F.col("visit_end_date"), F.col("visit_end_date")
                    ).otherwise(F.col("date"))
                ),
            )
            .withColumn("priority", F.col("priority") + F.col("concept_order") * 0.1)
            .drop("visit_end_date")
        )

        discharge_events = (
            visit_occurrence.where(
                F.col("visit_concept_id").isin([9201, 262, 8971, 8920])
            )
            .withColumn(
                "standard_concept_id",
                F.coalesce(F.col("discharged_to_concept_id"), F.lit(0)),
            )
            .withColumn("visit_concept_order", F.col("max_visit_concept_order"))
            .withColumn("concept_order", F.col("max_concept_order") + 1)
            .withColumn("date", F.col("visit_end_date"))
            .withColumn("datetime", F.date_add(F.to_timestamp("visit_end_date"), 1))
            .withColumn("datetime", F.expr("datetime - INTERVAL 1 MINUTE"))
            .withColumn("priority", F.lit(100))
            .drop("discharged_to_concept_id", "visit_end_date")
            .drop("min_visit_concept_order", "max_visit_concept_order")
            .drop("min_concept_order", "max_concept_order")
        )

        # Add discharge events to the inpatient visits
        inpatient_events = inpatient_events.unionByName(discharge_events)

        # Get the prev days_since_epoch
        inpatient_prev_date_udf = F.lag("date").over(
            W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy(
                "concept_order"
            )
        )

        # Compute the time difference between the current record and the previous record
        inpatient_time_delta_udf = F.when(F.col("prev_date").isNull(), 0).otherwise(
            F.datediff("date", "prev_date")
        )

        if self._include_inpatient_hour_token:
            # Create ATT tokens within the inpatient visits
            inpatient_prev_datetime_udf = F.lag("datetime").over(
                W.partitionBy("cohort_member_id", "visit_occurrence_id").orderBy(
                    "concept_order"
                )
            )
            # Compute the time difference between the current record and the previous record
            inpatient_hour_delta_udf = F.when(
                F.col("prev_datetime").isNull(), 0
            ).otherwise(
                F.floor(
                    (F.unix_timestamp("datetime") - F.unix_timestamp("prev_datetime"))
                    / 3600
                )
            )
            inpatient_att_token = F.when(
                F.col("hour_delta") < 24, F.concat(F.lit("i-H"), F.col("hour_delta"))
            ).otherwise(F.concat(F.lit("i-"), time_token_udf("time_delta")))
            # Create ATT tokens within the inpatient visits
            inpatient_att_events = (
                inpatient_events.withColumn(
                    "is_span_boundary",
                    F.row_number().over(
                        W.partitionBy(
                            "cohort_member_id", "visit_occurrence_id", "concept_order"
                        ).orderBy("priority")
                    ),
                )
                .where(F.col("is_span_boundary") == 1)
                .withColumn("prev_date", inpatient_prev_date_udf)
                .withColumn("time_delta", inpatient_time_delta_udf)
                .withColumn("prev_datetime", inpatient_prev_datetime_udf)
                .withColumn("hour_delta", inpatient_hour_delta_udf)
                .where(F.col("prev_date").isNotNull())
                .where(F.col("hour_delta") > 0)
                .withColumn("standard_concept_id", inpatient_att_token)
                .withColumn("visit_concept_order", F.col("visit_concept_order"))
                .withColumn("priority", F.col("priority") - 0.01)
                .withColumn("concept_value_mask", F.lit(0))
                .withColumn("concept_value", F.lit(0.0))
                .drop("prev_date", "time_delta", "is_span_boundary")
                .drop("prev_datetime", "hour_delta")
            )
        else:
            # Create ATT tokens within the inpatient visits
            inpatient_att_events = (
                inpatient_events.withColumn(
                    "is_span_boundary",
                    F.row_number().over(
                        W.partitionBy(
                            "cohort_member_id", "visit_occurrence_id", "concept_order"
                        ).orderBy("priority")
                    ),
                )
                .where(F.col("is_span_boundary") == 1)
                .withColumn("prev_date", inpatient_prev_date_udf)
                .withColumn("time_delta", inpatient_time_delta_udf)
                .where(F.col("time_delta") != 0)
                .where(F.col("prev_date").isNotNull())
                .withColumn(
                    "standard_concept_id",
                    F.concat(F.lit("i-"), time_token_udf("time_delta")),
                )
                .withColumn("visit_concept_order", F.col("visit_concept_order"))
                .withColumn("priority", F.col("priority") - 0.01)
                .withColumn("concept_value_mask", F.lit(0))
                .withColumn("concept_value", F.lit(0.0))
                .drop("prev_date", "time_delta", "is_span_boundary")
            )

        self.validate(inpatient_events)
        self.validate(inpatient_att_events)

        # Retrieving the events that are NOT linked to inpatient visits
        other_events = patient_events.join(
            inpatient_visits.select("visit_occurrence_id", "cohort_member_id"),
            ["visit_occurrence_id", "cohort_member_id"],
            how="left_anti",
        )

        patient_events = inpatient_events.unionByName(inpatient_att_events).unionByName(
            other_events
        )

        self.validate(patient_events)
        self.validate(artificial_tokens)

        # artificial_tokens = artificial_tokens.select(sorted(artificial_tokens.columns))
        return patient_events.unionByName(artificial_tokens)


class DemographicPromptDecorator(PatientEventDecorator):
    def __init__(self, patient_demographic, use_age_group: bool = False):
        self._patient_demographic = patient_demographic
        self._use_age_group = use_age_group

    def _decorate(self, patient_events: DataFrame):
        if self._patient_demographic is None:
            return patient_events

        # set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
        #      'visit_occurrence_id', 'domain', 'concept_value', 'visit_rank_order',
        #      'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
        #      'mlm_skip_value', 'age', 'visit_concept_id'])

        # Get the first token of the patient history
        first_token_udf = F.row_number().over(
            W.partitionBy("cohort_member_id", "person_id").orderBy(
                "visit_start_datetime",
                "visit_occurrence_id",
                "priority",
                "standard_concept_id",
            )
        )

        # Identify the first token of each patient history
        patient_first_token = (
            patient_events.withColumn("token_order", first_token_udf)
            .withColumn("concept_value_mask", F.lit(0))
            .withColumn("concept_value", F.lit(0.0))
            .where("token_order = 1")
            .drop("token_order")
        )

        # Udf for identifying the earliest date associated with a visit_occurrence_id
        sequence_start_year_token = (
            patient_first_token.withColumn(
                "standard_concept_id",
                F.concat(F.lit("year:"), F.year("date").cast(T.StringType())),
            )
            .withColumn("priority", F.lit(-10))
            .withColumn("visit_segment", F.lit(0))
            .withColumn("date_in_week", F.lit(0))
            .withColumn("age", F.lit(-1))
            .withColumn("visit_rank_order", F.lit(0))
            .withColumn("visit_concept_order", F.lit(0))
            .withColumn("concept_order", F.lit(0))
        )

        sequence_start_year_token.cache()

        if self._use_age_group:
            calculate_age_group_at_first_visit_udf = F.ceil(
                F.floor(
                    F.months_between(F.col("date"), F.col("birth_datetime"))
                    / F.lit(12)
                    / 10
                )
            )
            age_at_first_visit_udf = F.concat(
                F.lit("age:"),
                (calculate_age_group_at_first_visit_udf * 10).cast(T.StringType()),
                F.lit("-"),
                ((calculate_age_group_at_first_visit_udf + 1) * 10).cast(
                    T.StringType()
                ),
            )
        else:
            calculate_age_at_first_visit_udf = F.ceil(
                F.months_between(F.col("date"), F.col("birth_datetime")) / F.lit(12)
            )
            age_at_first_visit_udf = F.concat(
                F.lit("age:"), calculate_age_at_first_visit_udf.cast(T.StringType())
            )

        sequence_age_token = (
            self._patient_demographic.select(
                F.col("person_id"), F.col("birth_datetime")
            )
            .join(sequence_start_year_token, "person_id")
            .withColumn("standard_concept_id", age_at_first_visit_udf)
            .withColumn("priority", F.lit(-9))
            .drop("birth_datetime")
        )

        sequence_gender_token = (
            self._patient_demographic.select(
                F.col("person_id"), F.col("gender_concept_id")
            )
            .join(sequence_start_year_token, "person_id")
            .withColumn(
                "standard_concept_id", F.col("gender_concept_id").cast(T.StringType())
            )
            .withColumn("priority", F.lit(-8))
            .drop("gender_concept_id")
        )

        sequence_race_token = (
            self._patient_demographic.select(
                F.col("person_id"), F.col("race_concept_id")
            )
            .join(sequence_start_year_token, "person_id")
            .withColumn(
                "standard_concept_id", F.col("race_concept_id").cast(T.StringType())
            )
            .withColumn("priority", F.lit(-7))
            .drop("race_concept_id")
        )

        patient_events = patient_events.unionByName(sequence_start_year_token)
        patient_events = patient_events.unionByName(sequence_age_token)
        patient_events = patient_events.unionByName(sequence_gender_token)
        patient_events = patient_events.unionByName(sequence_race_token)

        return patient_events


class DeathEventDecorator(PatientEventDecorator):
    def __init__(self, death, att_type):
        self._death = death
        self._att_type = att_type

    def _decorate(self, patient_events: DataFrame):
        if self._death is None:
            return patient_events

        death_records = patient_events.join(
            self._death.select("person_id", "death_date"), "person_id"
        )

        max_visit_occurrence_id = death_records.select(
            F.max("visit_occurrence_id").alias("max_visit_occurrence_id")
        )

        last_ve_record = (
            death_records.where(F.col("standard_concept_id") == "VE")
            .withColumn(
                "record_rank",
                F.row_number().over(
                    W.partitionBy("person_id", "cohort_member_id").orderBy(
                        F.desc("date")
                    )
                ),
            )
            .where(F.col("record_rank") == 1)
            .drop("record_rank")
        )

        last_ve_record.cache()
        last_ve_record.show()
        # set(['cohort_member_id', 'person_id', 'standard_concept_id', 'date',
        #      'visit_occurrence_id', 'domain', 'concept_value', 'visit_rank_order',
        #      'visit_segment', 'priority', 'date_in_week', 'concept_value_mask',
        #      'mlm_skip_value', 'age', 'visit_concept_id'])

        artificial_visit_id = F.row_number().over(
            W.partitionBy(F.lit(0)).orderBy("person_id", "cohort_member_id")
        ) + F.col("max_visit_occurrence_id")
        death_records = (
            last_ve_record.crossJoin(max_visit_occurrence_id)
            .withColumn("visit_occurrence_id", artificial_visit_id)
            .withColumn("standard_concept_id", F.lit("[DEATH]"))
            .withColumn("domain", F.lit("death"))
            .withColumn("visit_rank_order", F.lit(1) + F.col("visit_rank_order"))
            .withColumn("priority", F.lit(20))
            .drop("max_visit_occurrence_id")
        )

        vs_records = death_records.withColumn(
            "standard_concept_id", F.lit("VS")
        ).withColumn("priority", F.lit(15))

        ve_records = death_records.withColumn(
            "standard_concept_id", F.lit("VE")
        ).withColumn("priority", F.lit(30))

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

        att_records = death_records.withColumn(
            "death_date",
            F.when(F.col("death_date") < F.col("date"), F.col("date")).otherwise(
                F.col("death_date")
            ),
        )
        att_records = (
            att_records.withColumn("time_delta", F.datediff("death_date", "date"))
            .withColumn("standard_concept_id", time_token_udf("time_delta"))
            .withColumn("priority", F.lit(10))
            .drop("time_delta")
        )

        new_tokens = (
            att_records.unionByName(vs_records)
            .unionByName(death_records)
            .unionByName(ve_records)
        )
        new_tokens = new_tokens.drop("death_date")
        self.validate(new_tokens)

        return patient_events.unionByName(new_tokens)


def time_token_func(time_delta) -> Optional[str]:
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 0:
        return "W-1"
    if time_delta < 28:
        return f"W{str(math.floor(time_delta / 7))}"
    if time_delta < 360:
        return f"M{str(math.floor(time_delta / 30))}"
    return "LT"


def time_day_token(time_delta):
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"D{str(time_delta)}"
    return "LT"


def time_week_token(time_delta):
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"W{str(math.floor(time_delta / 7))}"
    return "LT"


def time_month_token(time_delta):
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta < 1080:
        return f"M{str(math.floor(time_delta / 30))}"
    return "LT"


def time_mix_token(time_delta):
    #        WHEN day_diff <= 7 THEN CONCAT('D', day_diff)
    #         WHEN day_diff <= 30 THEN CONCAT('W', ceil(day_diff / 7))
    #         WHEN day_diff <= 360 THEN CONCAT('M', ceil(day_diff / 30))
    #         WHEN day_diff <= 720 THEN CONCAT('Q', ceil(day_diff / 90))
    #         WHEN day_diff <= 1440 THEN CONCAT('Y', ceil(day_diff / 360))
    #         ELSE 'LT'
    if time_delta is None or np.isnan(time_delta):
        return None
    if time_delta <= 7:
        return f"D{str(time_delta)}"
    if time_delta <= 30:
        # e.g. 8 -> W2
        return f"W{str(math.ceil(time_delta / 7))}"
    if time_delta <= 360:
        # e.g. 31 -> M2
        return f"M{str(math.ceil(time_delta / 30))}"
    # if time_delta <= 720:
    #     # e.g. 361 -> Q5
    #     return f'Q{str(math.ceil(time_delta / 90))}'
    # if time_delta <= 1080:
    #     # e.g. 1081 -> Y2
    #     return f'Y{str(math.ceil(time_delta / 360))}'
    return "LT"


def get_att_function(att_type: Union[AttType, str]):
    # Convert the att_type str to the corresponding enum type
    if isinstance(att_type, str):
        att_type = AttType(att_type)

    if att_type == AttType.DAY:
        return time_day_token
    elif att_type == AttType.WEEK:
        return time_week_token
    elif att_type == AttType.MONTH:
        return time_month_token
    elif att_type == AttType.MIX:
        return time_mix_token
    elif att_type == AttType.CEHR_BERT:
        return time_token_func
    return None
