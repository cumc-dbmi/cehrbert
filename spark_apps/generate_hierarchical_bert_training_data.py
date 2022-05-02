import argparse
import datetime
import os

from pyspark.sql import SparkSession

import config.parameters
from utils.spark_utils import *
from const.common import CDM_TABLES


def main(
        input_folder,
        output_folder,
        domain_table_list,
        date_filter,
        mlm_skip_table_list,
        max_num_of_visits_per_person,
        include_concept_list: bool = True
):
    spark = SparkSession.builder.appName('Generate Hierarchical Bert Training Data').getOrCreate()

    # Translate the cdm tables to domain names
    mlm_skip_domains = get_mlm_skip_domains(
        spark=spark,
        input_folder=input_folder,
        mlm_skip_table_list=mlm_skip_table_list
    )

    logger = logging.getLogger(__name__)
    logger.info(
        f'input_folder: {input_folder}\n'
        f'output_folder: {output_folder}\n'
        f'domain_table_list: {domain_table_list}\n'
        f'date_filter: {date_filter}\n'
        f'mlm_skip_table_list: {mlm_skip_table_list}\n'
        f'mlm_skip_domains: {mlm_skip_domains}\n'
        f'max_num_of_visits_per_person: {max_num_of_visits_per_person}\n'
        f'include_concept_list: {include_concept_list}'
    )

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

    column_names = patient_events.schema.fieldNames()

    if include_concept_list and patient_events:
        # Filter out concepts
        qualified_concepts = broadcast(
            preprocess_domain_table(
                spark,
                input_folder,
                config.parameters.qualified_concept_list_path
            )
        )
        # The select is necessary to make sure the order of the columns is the same as the
        # original dataframe
        patient_events = patient_events.join(
            qualified_concepts,
            'standard_concept_id'
        ).select(column_names)

    # Process the measurement table if exists
    if MEASUREMENT in domain_table_list:
        measurement = preprocess_domain_table(spark, input_folder, MEASUREMENT)
        required_measurement = preprocess_domain_table(spark, input_folder, REQUIRED_MEASUREMENT)
        # The select is necessary to make sure the order of the columns is the same as the
        # original dataframe, otherwise the union might use the wrong columns
        scaled_measurement = process_measurement(
            spark,
            measurement,
            required_measurement
        ).select(column_names)

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
        mlm_skip_domains=mlm_skip_domains,
        max_num_of_visits_per_person=max_num_of_visits_per_person
    )

    sequence_data.write.mode('overwrite').parquet(
        os.path.join(
            output_folder,
            config.parameters.parquet_data_path
        )
    )


def get_mlm_skip_domains(spark, input_folder, mlm_skip_table_list):
    """
    Translate the domain_table_name to the domain name

    :param spark:
    :param input_folder:
    :param mlm_skip_table_list:
    :return:
    """
    domain_tables = [
        preprocess_domain_table(spark, input_folder, domain_table_name)
        for domain_table_name in mlm_skip_table_list
    ]

    return list(map(get_domain_field, domain_tables))


def validate_table_names(domain_names):
    for domain_name in domain_names.split(' '):
        if domain_name not in CDM_TABLES:
            raise argparse.ArgumentTypeError(f'{domain_name} is an invalid CDM table name')
    return domain_names


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
                        type=validate_table_names,
                        required=True)
    parser.add_argument('--mlm_skip_table_list',
                        dest='mlm_skip_table_list',
                        nargs='+',
                        action='store',
                        help='The list of domains that will be skipped in MLM',
                        required=False,
                        type=validate_table_names,
                        default=[])
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
    parser.add_argument('--include_concept_list',
                        dest='include_concept_list',
                        action='store_true')

    ARGS = parser.parse_args()

    main(
        input_folder=ARGS.input_folder,
        output_folder=ARGS.output_folder,
        domain_table_list=ARGS.domain_table_list,
        date_filter=ARGS.date_filter,
        mlm_skip_table_list=ARGS.mlm_skip_table_list,
        max_num_of_visits_per_person=ARGS.max_num_of_visits,
        include_concept_list=ARGS.include_concept_list
    )
