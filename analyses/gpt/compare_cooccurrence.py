from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.functions import udf
import numpy as np


def preprocess_coocurrence(
        cooccurrence
):
    cooccurrence = cooccurrence.where('concept_id_1 <> 0 AND concept_id_2 <> 0')
    cooccurrence = cooccurrence.where(
        'concept_id_1_domain_id NOT IN ("Metadata", "Gender", "Visit") '
        'AND concept_id_2_domain_id NOT IN ("Metadata", "Gender", "Visit")'
    )
    total = cooccurrence.select(f.sum('count').alias('total'))
    # Rescale probability distribution after removing certain concepts
    cooccurrence = cooccurrence.crossJoin(total) \
        .withColumn('prob', f.col('count') / f.col('total')) \
        .drop('total')
    return cooccurrence


@udf(t.FloatType())
def kl_divergence_udf(prob1, prob2):
    return prob1 * np.log(prob1 / (prob2 + 1e-10))


def main(args):
    spark = SparkSession \
        .builder \
        .appName('Generate KL divergence') \
        .getOrCreate()

    reference_cooccurrence = spark.read.parquet(args.reference_cooccurrence_path)
    comparison_cooccurrence = spark.read.parquet(args.comparison_cooccurrence_path)

    reference_cooccurrence = preprocess_coocurrence(reference_cooccurrence)
    comparison_cooccurrence = preprocess_coocurrence(comparison_cooccurrence)

    join_conditions = (
            (comparison_cooccurrence.concept_id_1 == reference_cooccurrence.concept_id_1)
            & (comparison_cooccurrence.concept_id_2 == reference_cooccurrence.concept_id_2)
    )
    columns = [
        reference_cooccurrence.prob.alias('reference_prob'),
        f.coalesce(comparison_cooccurrence.prob, f.lit(1e-10)).alias('prob')
    ]
    if args.stratify_by_partition:
        columns += [reference_cooccurrence.concept_partition]

    joined_results = reference_cooccurrence.join(
        comparison_cooccurrence,
        join_conditions,
        'left_outer'
    ).select(
        columns
    ).withColumn(
        'kl',
        f.col('reference_prob') * f.log(f.col('reference_prob') / f.col('prob'))
    )

    if args.stratify_by_partition:
        joined_results \
            .groupby('concept_partition') \
            .agg(f.bround(f.sum('kl'), 4).alias('kl')) \
            .orderBy('concept_partition').show()
    else:
        joined_results.select(f.bround(f.sum('kl'), 4)).show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Arguments for calculating the KL divergent co-occurrence matrix')

    parser.add_argument(
        '--reference_cooccurrence_path',
        dest='reference_cooccurrence_path',
        action='store',
        help='The path for reference data coocurrence',
        required=True
    )
    parser.add_argument(
        '--comparison_cooccurrence_path',
        dest='comparison_cooccurrence_path',
        action='store',
        help='The path for comparison data coocurrence',
        required=True
    )
    parser.add_argument(
        '--stratify_by_partition',
        dest='stratify_by_partition',
        action='store_true'
    )
    main(parser.parse_args())
