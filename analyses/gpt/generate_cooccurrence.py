from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from itertools import combinations


@udf(ArrayType(ArrayType(StringType())))
def concept_pair(sequence):
    all_combinations = combinations(set([concept for concept in sequence if concept.isnumeric()]), 2)
    return list(all_combinations)


def generate_cooccurrence(
        dataframe
):
    total_size = dataframe.count()
    concept_pair_dataframe = dataframe.withColumn(
        'concept_pair',
        concept_pair('concept_ids')
    ).select(f.explode('concept_pair').alias('concept_pair')) \
        .select(f.sort_array('concept_pair').alias('concept_pair')) \
        .withColumn('concept_id_1', f.col('concept_pair').getItem(0)) \
        .withColumn('concept_id_2', f.col('concept_pair').getItem(1)) \
        .drop('concept_pair')
    dataframe_cooccurrence = concept_pair_dataframe \
        .groupBy('concept_id_1', 'concept_id_2').count() \
        .withColumn('prevalence', f.col('count') / f.lit(total_size))
    total = concept_pair_dataframe.count()
    dataframe_cooccurrence = dataframe_cooccurrence.withColumn('prob', f.col('count') / f.lit(total))
    return dataframe_cooccurrence


def get_domain(
        coocurrence_dataframe,
        concept_dataframe
):
    c = concept_dataframe.select('concept_id', 'domain_id')

    coocurrence_dataframe = coocurrence_dataframe \
        .join(c, coocurrence_dataframe.concept_id_1 == c.concept_id) \
        .withColumnRenamed('domain_id', 'concept_id_1_domain_id') \
        .drop('concept_id')

    coocurrence_dataframe = coocurrence_dataframe \
        .join(c, coocurrence_dataframe.concept_id_2 == c.concept_id) \
        .withColumnRenamed('domain_id', 'concept_id_2_domain_id') \
        .drop('concept_id')

    return coocurrence_dataframe


def main(
        args
):
    spark = SparkSession \
        .builder \
        .appName('Generate cooccurrence matrices') \
        .getOrCreate()

    source_data = spark.read.parquet(args.sequence_data_path)
    source_data = source_data.where(f'num_of_concepts <= {args.num_of_concepts}')
    if args.use_sample:
        source_data = source_data.sample(args.sample_frac)

    concept = spark.read.parquet(args.concept_path)

    source_data_cooccurrence = generate_cooccurrence(
        source_data
    )
    source_data_cooccurrence = get_domain(
        source_data_cooccurrence,
        concept
    )
    source_data_cooccurrence.write.mode('overwrite').parquet(args.data_cooccurrence_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Arguments for generating the co-occurrence matrix')

    parser.add_argument(
        '--sequence_data_path',
        dest='sequence_data_path',
        action='store',
        help='The path for patient sequence data',
        required=True
    )
    parser.add_argument(
        '--concept_path',
        dest='concept_path',
        action='store',
        help='The path for concept data',
        required=True
    )
    parser.add_argument(
        '--num_of_concepts',
        dest='num_of_concepts',
        action='store',
        type=int,
        default=512,
        required=False
    )
    parser.add_argument(
        '--use_sample',
        dest='use_sample',
        action='store_true'
    )
    parser.add_argument(
        '--sample_frac',
        dest='sample_frac',
        action='store',
        type=float,
        default=0.1,
        required=False
    )
    parser.add_argument(
        '--data_cooccurrence_path',
        dest='data_cooccurrence_path',
        action='store',
        required=True
    )
    main(parser.parse_args())
