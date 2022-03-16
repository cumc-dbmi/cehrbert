import argparse
import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast

import config.parameters
from const.common import REQUIRED_MEASUREMENT
from utils.spark_utils import *
from utils.spark_utils import create_hierarchical_sequence_data
from spark_apps.sql_templates import measurement_unit_stats_query


def process_measurement(spark, measurement, required_measurement):
    # Register the tables in spark context
    measurement.createOrReplaceTempView(MEASUREMENT)
    required_measurement.createOrReplaceTempView(REQUIRED_MEASUREMENT)
    measurement_unit_stats_df = spark.sql(
        measurement_unit_stats_query
    )
    # Cache the stats in memory
    measurement_unit_stats_df.cache()
    # Broadcast df to local executors
    broadcast(measurement_unit_stats_df)
    # Create the temp view for this dataframe
    measurement_unit_stats_df.createOrReplaceTempView('measurement_unit_stats')

    scaled_numeric_lab = spark.sql('''
        SELECT
            m.person_id,
            m.measurement_concept_id AS standard_concept_id,
            CAST(m.measurement_date AS DATE) AS date,
            m.visit_occurrence_id,
            'measurement' AS domain,
            MEAN((m.value_as_number - s.value_mean) / value_stddev) AS concept_value,
            m.person_id AS cohort_member_id
        FROM measurement AS m
        JOIN measurement_unit_stats AS s
            ON s.measurement_concept_id = m.measurement_concept_id 
                AND s.unit_concept_id = m.unit_concept_id
        WHERE m.visit_occurrence_id IS NOT NULL
            AND m.value_as_number BETWEEN s.lower_bound AND s.upper_bound
        GROUP BY m.person_id, m.visit_occurrence_id, m.measurement_concept_id, m.measurement_date
    ''')

    return scaled_numeric_lab


def main(input_folder, output_folder, domain_table_list, date_filter, max_num_of_visits_per_person):
    spark = SparkSession.builder.appName('Generate Hierarchical Bert Training Data').getOrCreate()

    domain_tables = []
    # Exclude measurement from domain_table_list if exists because we need to process measurement
    # in a different way
    for domain_table_name in domain_table_list:
        if domain_table_name != MEASUREMENT:
            domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)

    # Union all domain table records
    patient_events = join_domain_tables(domain_tables) \
        .withColumn('cohort_member_id', F.col('person_id'))

    # Process the measurement table if exists
    if MEASUREMENT in domain_table_list:
        measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
        required_measurement = preprocess_domain_table(spark, input_folder, REQUIRED_MEASUREMENT)
        scaled_measurement = process_measurement(
            spark,
            measurement,
            required_measurement
        )

        if patient_events:
            # Union all measurement records together with other domain records
            patient_events = patient_events.union(
                scaled_measurement
            )
        else:
            patient_events = scaled_measurement

    sequence_data = create_hierarchical_sequence_data(
        person, visit_occurrence, patient_events,
        date_filter=date_filter,
        max_num_of_visits_per_person=max_num_of_visits_per_person
    )

    sequence_data.write.mode('overwrite').parquet(
        os.path.join(
            output_folder,
            config.parameters.parquet_data_path
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generate training '
                                                 'data for Hierarchical Bert')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the raw data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)
    parser.add_argument('-tc',
                        '--domain_table_list',
                        dest='domain_table_list',
                        nargs='+',
                        action='store',
                        help='The list of domain tables you want to download',
                        required=True)
    parser.add_argument('-d',
                        '--date_filter',
                        dest='date_filter',
                        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
                        action='store',
                        required=False,
                        default='2018-01-01')
    parser.add_argument('--max_num_of_visits',
                        dest='max_num_of_visits',
                        action='store',
                        type=int,
                        default=200,
                        help='Max no.of visits per patient to be included',
                        required=False)

    ARGS = parser.parse_args()

    main(ARGS.input_folder,
         ARGS.output_folder,
         ARGS.domain_table_list,
         ARGS.date_filter,
         ARGS.max_num_of_visits)
