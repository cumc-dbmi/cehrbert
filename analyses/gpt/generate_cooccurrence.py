from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, StringType
from pyspark.sql.window import Window
from itertools import combinations


@udf(ArrayType(ArrayType(StringType())))
def concept_pair(sequence):
    all_combinations = combinations(set([concept for concept in sequence if concept.isnumeric()]), 2)
    return list(all_combinations)


@udf(ArrayType(StringType()))
def unique_concepts(sequence):
    return list(set([concept for concept in sequence if concept.isnumeric()]))


def compute_marginal(
        dataframe,
        num_of_partitions
):
    all_concepts_dataframe = dataframe.withColumn(
        'unique_concepts',
        unique_concepts('concept_ids')
    ).select(f.explode('unique_concepts').alias('concept_id')) \
        .drop('unique_concepts')
    marginal_dist = all_concepts_dataframe.groupBy('concept_id').count()
    data_size = all_concepts_dataframe.count()
    marginal_dist = marginal_dist \
        .withColumn('prob', f.col('count') / f.lit(data_size)) \
        .withColumn('concept_order', f.row_number().over(Window.orderBy(f.desc('prob'))))
    num_of_concepts = marginal_dist.count()
    partition_size = num_of_concepts // num_of_partitions
    marginal_dist = marginal_dist \
        .withColumn('concept_partition',
                    f.floor(f.col('concept_order') / f.lit(partition_size)).cast('int') + f.lit(1)) \
        .withColumn('concept_partition',
                    f.when(f.col('concept_partition') > num_of_partitions, num_of_partitions).otherwise(
                        f.col('concept_partition'))).drop('concept_order')
    return marginal_dist


def generate_cooccurrence(
        dataframe,
        stratify_by_frequency=False,
        num_of_partitions=3
):
    num_of_patients = dataframe.count()
    concept_pair_dataframe = dataframe.withColumn(
        'concept_pair',
        concept_pair('concept_ids')
    ).select(f.explode('concept_pair').alias('concept_pair')) \
        .select(f.sort_array('concept_pair').alias('concept_pair')) \
        .withColumn('concept_id_1', f.col('concept_pair').getItem(0)) \
        .withColumn('concept_id_2', f.col('concept_pair').getItem(1)) \
        .drop('concept_pair')

    # Compute the co-occurrence matrix and calculate prevalence (denominator: total num of patients)
    # and prob (denominator: total num of pairs)
    cooccurrence = concept_pair_dataframe \
        .groupBy('concept_id_1', 'concept_id_2').count() \
        .withColumn('prevalence', f.col('count') / f.lit(num_of_patients))
    num_of_concept_pairs = concept_pair_dataframe.count()
    cooccurrence = cooccurrence \
        .withColumn('prob', f.col('count') / f.lit(num_of_concept_pairs))

    if stratify_by_frequency:
        # marginal_dist = compute_marginal(dataframe, num_of_partitions).select('concept_id', 'concept_partition')
        unique_num_of_concept_pairs = cooccurrence.count()
        partition_size = unique_num_of_concept_pairs // num_of_partitions
        cooccurrence = cooccurrence \
            .withColumn('concept_order', f.row_number().over(Window.orderBy(f.desc('count')))) \
            .withColumn('concept_partition',
                        f.floor(f.col('concept_order') / f.lit(partition_size)).cast('int') + f.lit(1)) \
            .withColumn('concept_partition',
                        f.when(f.col('concept_partition') > num_of_partitions, f.lit(num_of_partitions)).otherwise(
                            f.col('concept_partition'))).drop('concept_order')
    return cooccurrence


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
    if 'num_of_concepts' in source_data.columns:
        source_data = source_data.where(f'num_of_concepts <= {args.num_of_concepts}')
    if args.use_sample:
        source_data = source_data.sample(args.sample_frac)

    concept = spark.read.parquet(args.concept_path)

    source_data_cooccurrence = generate_cooccurrence(
        source_data,
        args.stratify_by_frequency,
        args.num_of_partitions
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
    parser.add_argument(
        '--stratify_by_frequency',
        dest='stratify_by_frequency',
        action='store_true'
    )
    parser.add_argument(
        '--num_of_partitions',
        dest='num_of_partitions',
        action='store',
        type=int,
        default=3,
        required=False
    )
    main(parser.parse_args())
