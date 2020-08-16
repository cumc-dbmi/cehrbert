import os
import argparse

import spark_apps.parameters as p

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import Window as W

from utils.common import *


def main(spark, input_folder, output_folder):
    sequence_data = spark.read.parquet(os.path.join(input_folder, p.parquet_data_path))
    death = preprocess_domain_table(spark, input_folder, 'death')
    death = death.groupby('person_id').agg(F.max(F.col('death_date')).alias('death_date'))

    sequence_data.join(death, 'person_id', 'left') \
        .where('death_date IS NULL OR death_date >= max_event_date') \
        .withColumn('mortality', F.col('death_date').isNotNull().cast('int')) \
        .select('person_id', 'mortality').write.mode('overwrite') \
        .parquet(os.path.join(output_folder, p.mortality_data_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for generating mortality labels')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the sequence data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)

    ARGS = parser.parse_args()

    spark = SparkSession.builder.appName('Generate Mortality labels').getOrCreate()
    main(spark, ARGS.input_folder, ARGS.output_folder)
