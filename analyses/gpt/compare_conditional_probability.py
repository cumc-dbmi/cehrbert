from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from analyses.gpt.compare_cooccurrence import preprocess_coocurrence


def generate_conditional_probability(
        cooccurrence
):
    cooccurrence = preprocess_coocurrence(cooccurrence)
    concept_count = cooccurrence.groupBy('concept_id_1').agg(f.sum('count').alias('concept_id_1_count'))
    cooccurrence = cooccurrence.join(concept_count, 'concept_id_1') \
        .withColumn('conditional_prob', f.col('count') / f.col('concept_id_1_count'))
    return cooccurrence


def main(args):
    spark = SparkSession \
        .builder \
        .appName('Generate KL divergence for conditional probability distributions') \
        .getOrCreate()

    reference_cooccurrence = spark.read.parquet(args.reference_cooccurrence_path)
    comparison_cooccurrence = spark.read.parquet(args.comparison_cooccurrence_path)

    reference_conditional_prob = generate_conditional_probability(reference_cooccurrence)
    comparison_conditional_prob = generate_conditional_probability(comparison_cooccurrence)

    join_conditions = (
            (comparison_conditional_prob.concept_id_1 == reference_conditional_prob.concept_id_1)
            & (comparison_conditional_prob.concept_id_2 == reference_conditional_prob.concept_id_2)
    )

    columns = [
        reference_conditional_prob.concept_id_1,
        reference_conditional_prob.concept_id_2,
        reference_conditional_prob.concept_id_1_domain_id,
        reference_conditional_prob.concept_id_2_domain_id,
        f.coalesce(reference_conditional_prob.prob, f.lit(1e-10)).alias('reference_prob'),
        f.coalesce(comparison_conditional_prob.prob, f.lit(1e-10)).alias('prob')
    ]

    reference_conditional_prob.join(
        comparison_conditional_prob,
        join_conditions,
        'left_outer'
    ).select(
        columns
    ).withColumn(
        'kl',
        f.col('reference_prob') * f.log(f.col('reference_prob') / f.col('prob'))
    ).groupBy('concept_id_1').agg(f.sum('kl').alias('kl')) \
        .write.mode('overwrite').parquet(args.output_folder)


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
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The output folder for storing the conditional probability table ',
        required=True
    )
    main(parser.parse_args())
