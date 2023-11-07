from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from analyses.gpt.compare_cooccurrence import preprocess_coocurrence


def main(
        args
):
    spark = SparkSession \
        .builder \
        .appName('Generate Conditional Probability') \
        .getOrCreate()

    cooccurrence = spark.read.parquet(args.cooccurrence_path)
    cooccurrence = preprocess_coocurrence(cooccurrence)
    concept_count = cooccurrence.groupBy('concept_id_1').agg(f.sum('count').alias('concept_id_1_count'))

    cooccurrence.join(concept_count, 'concept_id_1') \
        .withColumn('conditional_prob', f.col('count') / f.col('concept_id_1_count')) \
        .write.mode('overwrite').parquet(args.output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Arguments for generating the conditional probability table')

    parser.add_argument(
        '--cooccurrence_path',
        dest='cooccurrence_path',
        action='store',
        help='The path for reference data co-occurrence',
        required=True
    )
    parser.add_argument(
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The output folder for storing the conditional probability table ',
        required=True
    )

    main(parser.parse_args())
