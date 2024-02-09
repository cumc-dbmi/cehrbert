import argparse
import os

from pyspark.sql import SparkSession
from pyspark.sql.functions import countDistinct

import config.output_names
from utils.spark_utils import *

DOMAIN_TABLE_LIST = ['condition_occurrence', 'procedure_occurrence', 'drug_exposure']


def main(input_folder,
         output_folder,
         min_num_of_patients
         ):
    spark = SparkSession.builder.appName('Generate concept list').getOrCreate()

    domain_tables = []
    # Exclude measurement from domain_table_list if exists because we need to process measurement
    # in a different way
    for domain_table_name in DOMAIN_TABLE_LIST:
        if domain_table_name != MEASUREMENT:
            domain_tables.append(preprocess_domain_table(spark, input_folder, domain_table_name))

    # Union all domain table records
    patient_events = join_domain_tables(domain_tables)

    # Filter out concepts that are linked to less than 100 patients
    qualified_concepts = patient_events.where('visit_occurrence_id IS NOT NULL')  \
        .groupBy('standard_concept_id') \
        .agg(countDistinct('person_id').alias('freq')) \
        .where(F.col('freq') >= min_num_of_patients)

    qualified_concepts.write.mode('overwrite').parquet(
        os.path.join(
            output_folder,
            config.parameters.qualified_concept_list_path
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generate '
                                                 'concept list to be included')
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
         ARGS.min_num_of_patients)
