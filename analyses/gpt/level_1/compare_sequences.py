import logging

import numpy as np
import pandas as pd
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("compare_sequence_prevalence")


def main(args):
    logger.info(f'Loading the reference sequence at {args.reference_sequence_path}')
    reference_sequences = pd.read_parquet(args.reference_sequence_path)
    logger.info(f'Loading the synthetic sequence at {args.synthetic_sequence_path}')
    synthetic_sequences = pd.read_parquet(args.synthetic_sequence_path)
    mse_metrics = []
    l2_metrics = []
    for _ in range(args.n_iterations):
        reference_samples = reference_sequences.sample(args.sample_size)
        synthetic_samples = synthetic_sequences.sample(args.sample_size)

        reference_prevalence = (
                reference_samples.concept_ids.explode().value_counts() / len(reference_samples)
        ).reset_index().rename(
            columns={'concept_ids': 'concept_id', 'count': 'ref_prevalence'}
        )

        synthetic_prevalence = (
                synthetic_samples.concept_ids.explode().value_counts() / len(reference_samples)
        ).reset_index().rename(
            columns={'concept_ids': 'concept_id', 'count': 'syn_prevalence'}
        )

        merge_pd = reference_prevalence.merge(
            synthetic_prevalence,
            how='outer',
            on='concept_id'
        ).fillna(0.0)

        diff = (merge_pd['ref_prevalence'] - merge_pd['syn_prevalence']) ** 2

        print(f'Iteration {_}. Mean Squared Error: {diff.mean()}; L2 norm {math.sqrt(diff.sum())}')
        mse_metrics.append(diff.mean())
        l2_metrics.append(math.sqrt(diff.sum()))

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
