from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.functions import udf
import numpy as np


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

    comparison_cooccurrence.join(
        reference_cooccurrence,
        (comparison_cooccurrence.concept_id_1 == reference_cooccurrence.concept_id_1) &
        (comparison_cooccurrence.concept_id_2 == reference_cooccurrence.concept_id_2),
        'left_outer'
    ).select(
        reference_cooccurrence.prob.alias('reference_prob'),
        f.coalesce(comparison_cooccurrence.prob, f.lit(1e-10)).alias('prob'),
    ).withColumn('kl', f.col('reference_prob') * f.log(f.col('reference_prob') / (f.col('prob') + f.lit(1e-10)))) \
        .select(f.sum('kl')).show()


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
    main(parser.parse_args())
