import os
import argparse
import datetime

from pyspark.sql import SparkSession

from utils.common import *
import spark_apps.parameters as p


def main(input_folder, output_folder, domain_table_list, date_filter):
    spark = SparkSession.builder.appName('Generate Bert Training Data').getOrCreate()
    domain_tables = []
    for domain_table_name in domain_table_list:
        domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    patient_event = join_domain_tables(domain_tables)
    patient_event = patient_event.where('visit_occurrence_id IS NOT NULL').distinct()
    sequence_data = create_sequence_data(patient_event, date_filter)
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
    ARGS = parser.parse_args()

    main(ARGS.input_folder, ARGS.output_folder, ARGS.domain_table_list, ARGS.date_filter)
