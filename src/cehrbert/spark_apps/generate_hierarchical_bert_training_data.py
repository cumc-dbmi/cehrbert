"""
This module generates hierarchical BERT training data based on domain tables from OMOP EHR data.

It processes patient event data, joins multiple domain tables, filters concepts based on a
minimum number of patients, and creates hierarchical sequence data for BERT training.

Key Functions:
    - preprocess_domain_table: Preprocesses domain tables for data extraction.
    - process_measurement: Handles special processing for measurement data.
    - join_domain_tables: Joins multiple domain tables into a unified DataFrame.
    - create_hierarchical_sequence_data: Generates hierarchical sequence data for training.

Command-line Arguments:
    - input_folder: Path to the directory containing input data.
    - output_folder: Path to the directory where the output will be saved.
    - domain_table_list: List of domain tables to process.
    - date_filter: Optional filter for processing the data based on date.
    - max_num_of_visits_per_person: Maximum number of visits per patient to include.
    - min_observation_period: Minimum observation period in days for patients to be included.
    - include_concept_list: Whether to apply a filter to retain certain concepts.
    - include_incomplete_visit: Whether to include incomplete visit records in the training data.
"""

import datetime
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from ..config.output_names import PARQUET_DATA_PATH, QUALIFIED_CONCEPT_LIST_PATH
from ..const.common import (
    MEASUREMENT,
    OBSERVATION_PERIOD,
    PERSON,
    REQUIRED_MEASUREMENT,
    VISIT_OCCURRENCE,
)
from ..utils.spark_utils import (
    create_hierarchical_sequence_data,
    join_domain_tables,
    preprocess_domain_table,
    process_measurement,
    validate_table_names,
)


def main(
    input_folder,
    output_folder,
    domain_table_list,
    date_filter,
    max_num_of_visits_per_person,
    min_observation_period: int = 360,
    include_concept_list: bool = True,
    include_incomplete_visit: bool = True,
):
    """
    Main function to generate hierarchical BERT training data from domain tables.

    Args:
        input_folder (str): The path to the input folder containing raw data.
        output_folder (str): The path to the output folder for storing the training data.
        domain_table_list (list): A list of domain tables to process (e.g., condition_occurrence).
        date_filter (str): Date filter for processing data, default is '2018-01-01'.
        max_num_of_visits_per_person (int): The maximum number of visits to include per person.
        min_observation_period (int, optional): Minimum observation period in days. Default is 360.
        include_concept_list (bool, optional): Whether to filter by concept list. Default is True.
        include_incomplete_visit (bool, optional): Whether to include incomplete visits. Default is
        True.

    This function preprocesses domain tables, filters and processes measurement data,
    and generates hierarchical sequence data for training BERT models on EHR records.
    """
    spark = SparkSession.builder.appName("Generate Hierarchical Bert Training Data").getOrCreate()

    logger = logging.getLogger(__name__)
    logger.info(
        "input_folder: %s\n"
        "output_folder: %s\n"
        "domain_table_list: %s\n"
        "date_filter: %s\n"
        "max_num_of_visits_per_person: %s\n"
        "min_observation_period: %s\n"
        "include_concept_list: %s\n"
        "include_incomplete_visit: %s",
        input_folder,
        output_folder,
        domain_table_list,
        date_filter,
        max_num_of_visits_per_person,
        min_observation_period,
        include_concept_list,
        include_incomplete_visit,
    )

    domain_tables = []
    # Exclude measurement from domain_table_list if exists because we need to process measurement
    # in a different way
    for domain_table_name in domain_table_list:
        if domain_table_name != MEASUREMENT:
            domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    observation_period = (
        preprocess_domain_table(spark, input_folder, OBSERVATION_PERIOD)
        .withColumn(
            "observation_period_start_date",
            F.col("observation_period_start_date").cast("date"),
        )
        .withColumn(
            "observation_period_end_date",
            F.col("observation_period_end_date").cast("date"),
        )
        .withColumn(
            "period",
            F.datediff("observation_period_end_date", "observation_period_start_date"),
        )
        .where(F.col("period") >= min_observation_period)
        .select("person_id")
    )

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)

    # Filter for the persons that have enough observation period
    person = person.join(observation_period, "person_id").select(
        [person[f] for f in person.schema.fieldNames()]
    )

    # Union all domain table records
    patient_events = join_domain_tables(domain_tables)

    column_names = patient_events.schema.fieldNames()

    if include_concept_list and patient_events:
        # Filter out concepts
        qualified_concepts = F.broadcast(
            preprocess_domain_table(spark, input_folder, QUALIFIED_CONCEPT_LIST_PATH)
        )
        # The select is necessary to make sure the order of the columns is the same as the
        # original dataframe
        patient_events = patient_events.join(qualified_concepts, "standard_concept_id").select(
            column_names
        )

    # Process the measurement table if exists
    if MEASUREMENT in domain_table_list:
        measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
        required_measurement = preprocess_domain_table(spark, input_folder, REQUIRED_MEASUREMENT)
        # The select is necessary to make sure the order of the columns is the same as the
        # original dataframe, otherwise the union might use the wrong columns
        scaled_measurement = process_measurement(spark, measurement, required_measurement).select(
            column_names
        )

        if patient_events:
            # Union all measurement records together with other domain records
            patient_events = patient_events.union(scaled_measurement)
        else:
            patient_events = scaled_measurement

    # cohort_member_id is the same as the person_id
    patient_events = patient_events.withColumn("cohort_member_id", F.col("person_id"))

    sequence_data = create_hierarchical_sequence_data(
        person,
        visit_occurrence,
        patient_events,
        date_filter=date_filter,
        max_num_of_visits_per_person=max_num_of_visits_per_person,
        include_incomplete_visit=include_incomplete_visit,
    )

    sequence_data.write.mode("overwrite").parquet(os.path.join(output_folder, PARQUET_DATA_PATH))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Arguments for generate training data for Hierarchical Bert"
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the raw data is",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "-tc",
        "--domain_table_list",
        dest="domain_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to download",
        type=validate_table_names,
        required=True,
    )
    parser.add_argument(
        "-d",
        "--date_filter",
        dest="date_filter",
        type=lambda s: datetime.datetime.strptime(s, "%Y-%m-%d"),
        action="store",
        required=False,
        default="2018-01-01",
    )
    parser.add_argument(
        "--max_num_of_visits",
        dest="max_num_of_visits",
        action="store",
        type=int,
        default=200,
        help="Max no.of visits per patient to be included",
        required=False,
    )
    parser.add_argument(
        "--min_observation_period",
        dest="min_observation_period",
        action="store",
        type=int,
        default=1,
        help="Minimum observation period in days",
        required=False,
    )
    parser.add_argument("--include_concept_list", dest="include_concept_list", action="store_true")
    parser.add_argument(
        "--include_incomplete_visit",
        dest="include_incomplete_visit",
        action="store_true",
    )

    ARGS = parser.parse_args()

    main(
        input_folder=ARGS.input_folder,
        output_folder=ARGS.output_folder,
        domain_table_list=ARGS.domain_table_list,
        date_filter=ARGS.date_filter,
        max_num_of_visits_per_person=ARGS.max_num_of_visits,
        min_observation_period=ARGS.min_observation_period,
        include_concept_list=ARGS.include_concept_list,
        include_incomplete_visit=ARGS.include_incomplete_visit,
    )
