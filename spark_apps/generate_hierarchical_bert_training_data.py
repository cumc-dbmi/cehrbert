import datetime
import os

from pyspark.sql import SparkSession

import config.parameters
from utils.spark_utils import *
from const.common import OBSERVATION_PERIOD, VISIT_OCCURRENCE, PERSON
from utils.spark_utils import get_mlm_skip_domains, validate_table_names


def main(
        input_folder,
        output_folder,
        domain_table_list,
        date_filter,
        max_num_of_visits_per_person,
        min_observation_period: int = 360,
        include_concept_list: bool = True,
        include_incomplete_visit: bool = True
):
    spark = SparkSession.builder.appName('Generate Hierarchical Bert Training Data').getOrCreate()

    logger = logging.getLogger(__name__)
    logger.info(
        f'input_folder: {input_folder}\n'
        f'output_folder: {output_folder}\n'
        f'domain_table_list: {domain_table_list}\n'
        f'date_filter: {date_filter}\n'
        f'max_num_of_visits_per_person: {max_num_of_visits_per_person}\n'
        f'min_observation_period: {min_observation_period}\n'
        f'include_concept_list: {include_concept_list}\n'
        f'include_incomplete_visit: {include_incomplete_visit}'
    )

    domain_tables = []
    # Exclude measurement from domain_table_list if exists because we need to process measurement
    # in a different way
    for domain_table_name in domain_table_list:
        if domain_table_name != MEASUREMENT:
            domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    observation_period = (
        preprocess_domain_table(spark, input_folder, OBSERVATION_PERIOD).withColumn(
            'observation_period_start_date',
            F.col('observation_period_start_date').cast('date')
        ).withColumn(
            'observation_period_end_date',
            F.col('observation_period_end_date').cast('date')
        ).withColumn(
            'period',
            F.datediff('observation_period_end_date', 'observation_period_start_date')
        ).where(F.col('period') >= min_observation_period).select('person_id')
    )

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)

    # Filter for the persons that have enough observation period
    person = person.join(
        observation_period,
        'person_id'
    ).select([person[f] for f in person.schema.fieldNames()])

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
        max_num_of_visits_per_person=max_num_of_visits_per_person,
        include_incomplete_visit=include_incomplete_visit
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
    parser.add_argument(
        '-i',
        '--input_folder',
        dest='input_folder',
        action='store',
        help='The path for your input_folder where the raw data is',
        required=True
    )
    parser.add_argument(
        '-o',
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The path for your output_folder',
        required=True
    )
    parser.add_argument(
        '-tc',
        '--domain_table_list',
        dest='domain_table_list',
        nargs='+',
        action='store',
        help='The list of domain tables you want to download',
        type=validate_table_names,
        required=True
    )
    parser.add_argument(
        '-d',
        '--date_filter',
        dest='date_filter',
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'),
        action='store',
        required=False,
        default='2018-01-01'
    )
    parser.add_argument(
        '--max_num_of_visits',
        dest='max_num_of_visits',
        action='store',
        type=int,
        default=200,
        help='Max no.of visits per patient to be included',
        required=False
    )
    parser.add_argument(
        '--min_observation_period',
        dest='min_observation_period',
        action='store',
        type=int,
        default=1,
        help='Minimum observation period in days',
        required=False
    )
    parser.add_argument(
        '--include_concept_list',
        dest='include_concept_list',
        action='store_true'
    )
    parser.add_argument(
        '--include_incomplete_visit',
        dest='include_incomplete_visit',
        action='store_true'
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
        include_incomplete_visit=ARGS.include_incomplete_visit
    )
