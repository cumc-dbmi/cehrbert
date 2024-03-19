import os
import pickle
import logging
import random
from datetime import datetime
import numpy as np
import pandas as pd

from analyses.gpt.privacy.utils import create_race_encoder, create_gender_encoder, scale_age, create_demographics, \
    create_vector_representations, find_match, find_match_self, RANDOM_SEE

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("NNAA")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(args):
    LOG.info(f'Started loading tokenizer at {args.concept_tokenizer_path}')
    with open(args.concept_tokenizer_path, 'rb') as f:
        concept_tokenizer = pickle.load(f)

    LOG.info(f'Started loading training data at {args.training_data_folder}')
    train_data = pd.read_parquet(args.training_data_folder)

    LOG.info(f'Started loading evaluation data at {args.evaluation_data_folder}')
    evaluation_data = pd.read_parquet(args.evaluation_data_folder)

    LOG.info(f'Started loading synthetic_data at {args.synthetic_data_folder}')
    synthetic_data = pd.read_parquet(args.synthetic_data_folder)

    LOG.info('Started extracting the demographic information from the patient sequences')
    train_data = create_demographics(train_data)
    evaluation_data = create_demographics(evaluation_data)
    synthetic_data = create_demographics(synthetic_data)

    LOG.info('Started rescaling age columns')
    train_data = scale_age(train_data)
    evaluation_data = scale_age(evaluation_data)
    synthetic_data = scale_age(synthetic_data)

    LOG.info('Started encoding gender')
    gender_encoder = create_gender_encoder(
        train_data,
        evaluation_data,
        synthetic_data
    )
    LOG.info('Completed encoding gender')

    LOG.info('Started encoding race')
    race_encoder = create_race_encoder(
        train_data,
        evaluation_data,
        synthetic_data
    )
    LOG.info('Completed encoding race')

    random.seed(RANDOM_SEE)
    metrics = []

    for i in range(args.n_iterations):
        LOG.info(f'Iteration {i} Started')
        train_data_sample = train_data.sample(args.num_of_samples)
        evaluation_data_sample = evaluation_data.sample(args.num_of_samples)
        synthetic_data_sample = synthetic_data.sample(args.num_of_samples)
        LOG.info(f'Iteration {i}: Started creating train vectors')
        train_vectors = create_vector_representations(
            train_data_sample,
            concept_tokenizer,
            gender_encoder,
            race_encoder
        )
        LOG.info(f'Iteration {i}: Started creating evaluation vectors')
        evaluation_vectors = create_vector_representations(
            evaluation_data_sample,
            concept_tokenizer,
            gender_encoder,
            race_encoder
        )
        LOG.info(f'Iteration {i}: Started creating synthetic vectors')
        synthetic_vectors = create_vector_representations(
            synthetic_data_sample,
            concept_tokenizer,
            gender_encoder,
            race_encoder
        )
        LOG.info(f'Iteration {i}: Started calculating the distances between synthetic and training vectors')
        distance_train_TS = find_match(train_vectors, synthetic_vectors)
        distance_train_ST = find_match(synthetic_vectors, train_vectors)
        distance_train_TT = find_match_self(train_vectors, train_vectors)
        distance_train_SS = find_match_self(synthetic_vectors, synthetic_vectors)

        aa_train = (np.sum(distance_train_TS > distance_train_TT) + np.sum(
            distance_train_ST > distance_train_SS)) / args.num_of_samples / 2

        LOG.info(f'Iteration {i}: Started calculating the distances between synthetic and evaluation vectors')
        distance_test_TS = find_match(evaluation_vectors, synthetic_vectors)
        distance_test_ST = find_match(synthetic_vectors, evaluation_vectors)
        distance_test_TT = find_match_self(evaluation_vectors, evaluation_vectors)
        distance_test_SS = find_match_self(synthetic_vectors, synthetic_vectors)

        aa_test = (np.sum(distance_test_TS > distance_test_TT) + np.sum(
            distance_test_ST > distance_test_SS)) / args.num_of_samples / 2

        privacy_loss = aa_test - aa_train
        metrics.append(privacy_loss)
        LOG.info(f'Iteration {i}: Privacy loss {privacy_loss}')

    results = {
        "NNAAE": metrics
    }

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    pd.DataFrame([results], columns=['NNAAE']).to_parquet(
        os.path.join(args.metrics_folder, f"nearest_neighbor_inference_{current_time}.parquet")
    )


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Nearest Neighbor Inference Analysis Arguments using the GPT model'
    )
    parser.add_argument(
        '--training_data_folder',
        dest='training_data_folder',
        action='store',
        help='The path for where the training data folder',
        required=True
    )
    parser.add_argument(
        '--evaluation_data_folder',
        dest='evaluation_data_folder',
        action='store',
        help='The path for where the evaluation data folder',
        required=True
    )
    parser.add_argument(
        '--synthetic_data_folder',
        dest='synthetic_data_folder',
        action='store',
        help='The path for where the synthetic data folder',
        required=True
    )
    parser.add_argument(
        '--concept_tokenizer_path',
        dest='concept_tokenizer_path',
        action='store',
        help='The path for where the concept tokenizer is located',
        required=True
    )
    parser.add_argument(
        '--num_of_samples',
        dest='num_of_samples',
        action='store',
        type=int,
        required=False,
        default=5000
    )
    parser.add_argument(
        '--n_iterations',
        dest='n_iterations',
        action='store',
        type=int,
        required=False,
        default=1
    )
    parser.add_argument(
        '--metrics_folder',
        dest='metrics_folder',
        action='store',
        help='The folder that stores the metrics',
        required=False
    )
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
