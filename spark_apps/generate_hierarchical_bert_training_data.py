import argparse
import datetime
import os

from pyspark.sql import SparkSession

import config.parameters
from utils.spark_utils import *
from utils.spark_utils import create_hierarchical_sequence_data
from spark_apps.sql_templates import measurement_unit_stats_query

MEASUREMENT = 'measurement'
REQUIRED_MEASUREMENT = 'required_measurement'
VISIT_OCCURRENCE = 'visit_occurrence'
PERSON = 'person'


def process_measurement(spark, measurement, required_measurement, min_frequency=100):
    # Register the tables in spark context
    measurement.createOrReplaceTempView('measurement')
    required_measurement.createOrReplaceTempView('required_measurement')
    spark.sql(
        measurement_unit_stats_query
    ).createOrReplaceTempView('measurement_unit_stats')

    scaled_numeric_lab = spark.sql('''
        SELECT
            m.person_id,
            m.measurement_concept_id AS standard_concept_id,
            CAST(m.measurement_date AS DATE) AS date,
            m.visit_occurrence_id,
            'measurement' AS domain,
            MEAN((m.value_as_number - s.value_mean) / value_stddev) AS scaled_value
        FROM measurement_unit_stats AS s
        JOIN measurement AS m
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
    for domain_table_name in domain_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)

    # Remove measurement from the domain tables if exists because we need to process measurement
    # in a different way
    patient_discrete_events = join_domain_tables(
        [d for d in domain_tables if d != MEASUREMENT]
    ).withColumn('cohort_member_id', F.col('person_id'))

    # Process the measurement table if exists
    if MEASUREMENT in domain_tables:
        measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
        required_measurement = preprocess_domain_table(spark, input_folder, REQUIRED_MEASUREMENT)

    sequence_data = create_hierarchical_sequence_data(
        person, visit_occurrence, patient_discrete_events,
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
