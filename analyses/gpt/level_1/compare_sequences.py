import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as f
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compare_sequence_prevalence")


def calculate_prevalence(dataframe: DataFrame) -> DataFrame:
    row_count = dataframe.count()
    concept_dataframe = dataframe \
        .select(f.explode('concept_ids').alias('concept_id'))
    return concept_dataframe.groupby('concept_id').count() \
        .withColumn('prevalence', f.col('count') / f.lit(row_count)) \
        .drop('count')


def main(args):
    spark = SparkSession \
        .builder \
        .appName('Compare Sequence Prevalence') \
        .getOrCreate()

    logger.info(f'Loading the reference sequence at {args.reference_sequence_path}')
    reference_sequences = spark.read.parquet(args.reference_sequence_path)
    logger.info(f'Loading the synthetic sequence at {args.synthetic_sequence_path}')
    synthetic_sequences = spark.read.parquet(args.synthetic_sequence_path)
    ref_count = reference_sequences.count()
    syn_count = synthetic_sequences.count()
    mse_metrics = []
    l2_metrics = []
    for _ in range(args.n_iterations):
        reference_prevalence = calculate_prevalence(
            reference_sequences.sample(args.sample_size / ref_count)
        ).withColumnRenamed('prevalence', 'ref_prevalence')
        synthetic_prevalence = calculate_prevalence(
            synthetic_sequences.sample(args.sample_size / syn_count)
        ).withColumnRenamed('prevalence', 'syn_prevalence')

        merge_dataframe = reference_prevalence.join(
            synthetic_prevalence,
            on='concept_id',
            how='full_outer'
        ).withColumn('squared_error', f.pow(f.col('ref_prevalence') - f.col('syn_prevalence'), 2))

        results = merge_dataframe.select(
            f.sqrt(f.sum('squared_error')).alias('l2_norm'),
            f.mean('squared_error').alias('mse')
        ).head(1)[0]

        print(f'Iteration {_}. Mean Squared Error: {results.mse}; L2 norm {results.l2_norm}')
        mse_metrics.append(results.mse)
        l2_metrics.append(results.l2_norm)

    print(f'Average Mean Squared Error: {np.mean(mse_metrics)}; Average L2 norm {np.mean(l2_metrics)}')


def create_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare L2 norm of the prevalence of two sequences"
    )
    parser.add_argument(
        "--reference_sequence_path",
        dest="reference_sequence_path",
        action="store",
        required=True
    )
    parser.add_argument(
        "--synthetic_sequence_path",
        dest="synthetic_sequence_path",
        action="store",
        required=True
    )
    parser.add_argument(
        "--n_iterations",
        dest="n_iterations",
        action="store",
        type=int,
        default=10,
        required=False
    )
    parser.add_argument(
        "--sample_size",
        dest="sample_size",
        action="store",
        type=int,
        default=10000,
        required=False
    )
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
