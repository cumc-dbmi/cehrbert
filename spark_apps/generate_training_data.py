import os
import argparse
import datetime

from pyspark.sql import SparkSession

from utils.spark_utils import *
import spark_apps.parameters as p

VISIT_OCCURRENCE = 'visit_occurrence'
PERSON = 'person'


def main(input_folder, output_folder, domain_table_list, date_filter,
         include_visit_type, is_new_patient_representation, exclude_visit_tokens, is_classic_bert):
    spark = SparkSession.builder.appName('Generate Bert Training Data').getOrCreate()
    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    visit_occurrence = preprocess_domain_table(spark, input_folder, VISIT_OCCURRENCE)
    person = preprocess_domain_table(spark, input_folder, PERSON)
    patient_event = join_domain_tables(domain_tables)
    patient_event = patient_event.where('visit_occurrence_id IS NOT NULL').distinct()
    patient_event = patient_event.join(visit_occurrence, 'visit_occurrence_id') \
        .join(person, 'person_id') \
        .select(patient_event['person_id'], patient_event['standard_concept_id'],
                patient_event['date'], patient_event['visit_occurrence_id'],
                patient_event['domain'], visit_occurrence['visit_concept_id'],
                person['birth_datetime']) \
        .withColumn('cohort_member_id', F.col('person_id')) \
        .withColumn('age', F.months_between(F.col('date'),F.col("birth_datetime"))/F.lit(12))

    if is_new_patient_representation:
        sequence_data = create_sequence_data_time_delta_embedded(patient_event,
                                                                 date_filter=date_filter,
                                                                 exclude_visit_tokens=exclude_visit_tokens,
                                                                 include_visit_type=include_visit_type)
    else:
        sequence_data = create_sequence_data(patient_event, date_filter=date_filter,
                                             include_visit_type=include_visit_type,
                                             classic_bert_seq=is_classic_bert)

    sequence_data.write.mode('overwrite').parquet(os.path.join(output_folder, p.parquet_data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generate training data for Bert')
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

    parser.add_argument('-iv',
                        '--include_visit_type',
                        dest='include_visit_type',
                        action='store_true',
                        help='Specify whether to include visit types for '
                             'generating the training data')

    parser.add_argument('-ip',
                        '--is_new_patient_representation',
                        dest='is_new_patient_representation',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the new patient representation')
    parser.add_argument('-ib',
                        '--is_classic_bert_sequence',
                        dest='is_classic_bert_sequence',
                        action='store_true',
                        help='Specify whether to generate the sequence of '
                             'EHR records using the classic BERT sequence')
    parser.add_argument('-ev',
                        '--exclude_visit_tokens',
                        dest='exclude_visit_tokens',
                        action='store_true',
                        help='Specify whether or not to exclude the VS and VE tokens')
    ARGS = parser.parse_args()

    main(ARGS.input_folder, ARGS.output_folder, ARGS.domain_table_list, ARGS.date_filter,
         ARGS.include_visit_type, ARGS.is_new_patient_representation, ARGS.exclude_visit_tokens,
         ARGS.is_classic_bert_sequence)
