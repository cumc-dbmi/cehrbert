from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import Window
from analyses.gpt.compare_cooccurrence import preprocess_coocurrence


def main(args):
    spark = SparkSession \
        .builder \
        .appName('Join the co-occurrence matrices for visualization') \
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
        f.coalesce(
            reference_cooccurrence.concept_id_1,
            comparison_cooccurrence.concept_id_1
        ).alias('concept_id_1'),
        f.coalesce(
            reference_cooccurrence.concept_id_2,
            comparison_cooccurrence.concept_id_2
        ).alias('concept_id_2'),
        f.coalesce(
            reference_cooccurrence.concept_id_1_domain_id,
            comparison_cooccurrence.concept_id_1_domain_id
        ).alias('concept_id_1_domain_id'),
        f.coalesce(
            reference_cooccurrence.concept_id_2_domain_id,
            comparison_cooccurrence.concept_id_2_domain_id
        ).alias('concept_id_2_domain_id'),
        f.coalesce(reference_cooccurrence.prob, f.lit(1e-10)).alias('reference_prob'),
        f.coalesce(comparison_cooccurrence.prob, f.lit(1e-10)).alias('prob'),
        f.coalesce(reference_cooccurrence.prevalence, f.lit(1e-10)).alias('reference_prevalence'),
        f.coalesce(comparison_cooccurrence.prevalence, f.lit(1e-10)).alias('prevalence')
        # f.coalesce(reference_cooccurrence.count.cast('int'), f.lit(1)).alias('reference_count'),
        # f.coalesce(comparison_cooccurrence.count.cast('int'), f.lit(1)).alias('count')
    ]

    reference_cooccurrence.join(
        comparison_cooccurrence,
        join_conditions,
        'full_outer'
    ).select(
        columns
    ).withColumn('group', f.concat(f.col('concept_id_1_domain_id'), f.lit('_'), f.col('concept_id_2_domain_id'))) \
        .withColumn('group_rank', f.row_number().over(Window.partitionBy('group').orderBy(f.desc('reference_prob')))) \
        .write.mode('overwrite').parquet(args.output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Arg parser for join the co-occurrence matrices for visualization')

    parser.add_argument(
        '--reference_cooccurrence_path',
        dest='reference_cooccurrence_path',
        action='store',
        help='The path for reference data cooccurrence',
        required=True
    )
    parser.add_argument(
        '--comparison_cooccurrence_path',
        dest='comparison_cooccurrence_path',
        action='store',
        help='The path for comparison data cooccurrence',
        required=True
    )
    parser.add_argument(
        '--output_folder',
        dest='output_folder'
    )
    main(parser.parse_args())
