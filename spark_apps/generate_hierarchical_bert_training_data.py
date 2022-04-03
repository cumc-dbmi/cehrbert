import argparse
import datetime
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct

import config.parameters
from utils.spark_utils import *
from utils.spark_utils import create_hierarchical_sequence_data, process_measurement


def main(input_folder,
         output_folder,
         domain_table_list,
         date_filter,
         max_num_of_visits_per_person,
         min_num_of_patients
         ):
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
    patient_events = join_domain_tables(domain_tables)

    # Filter out concepts that are linked to less than 100 patients
    qualified_concepts = patient_events.groupBy('standard_concept_id') \
        .agg(countDistinct('person_id').alias('freq')) \
        .where(F.col('freq') >= min_num_of_patients) \
        .select('standard_concept_id')

    qualified_concepts.cache()
    broadcast(qualified_concepts)

    patient_events = patient_events.join(
        qualified_concepts,
        'standard_concept_id'
    )

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

    # cohort_member_id is the same as the person_id
    patient_events = patient_events.withColumn('cohort_member_id', F.col('person_id'))

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
    parser.add_argument('--min_num_of_patients',
                        dest='min_num_of_patients',
                        action='store',
                        type=int,
                        default=0,
                        help='Min no.of patients linked to concepts to be included',
                        required=False)

    ARGS = parser.parse_args()

    main(ARGS.input_folder,
         ARGS.output_folder,
         ARGS.domain_table_list,
         ARGS.date_filter,
         ARGS.max_num_of_visits,
         ARGS.min_num_of_patients)
