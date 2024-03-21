import os
import pickle
import pandas as pd
import logging
import random
from sklearn import metrics
from datetime import datetime
from utils.model_utils import tokenize_one_field
import yaml
from typing import Union
import numpy as np

from analyses.gpt.privacy.utils import (create_race_encoder, create_gender_encoder, scale_age, create_demographics \
    , create_vector_representations_for_attribute, batched_pairwise_euclidean_distance_indices, RANDOM_SEE)

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger('attribute_inference')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main(
        args
):
    try:
        with open(args.attribute_config, 'r') as file:
            data = yaml.safe_load(file)
        if 'common_attributes' in data:
            common_attributes = data['common_attributes']
        if 'sensitive_attributes' in data:
            sensitive_attributes = data['sensitive_attributes']
    except Union[FileNotFoundError, PermissionError, OSError] as e:
        raise e

    LOG.info(f'Started loading tokenizer at {args.tokenizer_path}')
    with open(args.tokenizer_path, 'rb') as f:
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
    all_results = []
    for i in range(1, args.n_iterations + 1):
        dist_metrics = []
        LOG.info(f'Iteration {i} Started')
        LOG.info(f'Iteration {i}: Started creating train vectors')
        train_common_vectors, train_sensitive_vectors = create_vector_representations_for_attribute(
            train_data,
            concept_tokenizer,
            gender_encoder,
            race_encoder,
            common_attributes=common_attributes,
            sensitive_attributes=sensitive_attributes
        )

        LOG.info(f'Iteration {i}: Started creating synthetic vectors')
        synthetic_common_vectors, synthetic_sensitive_vectors = create_vector_representations_for_attribute(
            synthetic_data,
            concept_tokenizer,
            gender_encoder,
            race_encoder,
            common_attributes=common_attributes,
            sensitive_attributes=sensitive_attributes
        )

        LOG.info(f'Iteration {i}: Started calculating the distances between synthetic and training vectors')
        synthetic_train_index = batched_pairwise_euclidean_distance_indices(train_common_vectors,
                                                                            synthetic_common_vectors,
                                                                            batch_size=args.matching_batch_size)
        train_train_index = batched_pairwise_euclidean_distance_indices(train_common_vectors, train_common_vectors,
                                                                        batch_size=args.matching_batch_size,
                                                                        self_exclude=True)

        def cal_f1_score(vector_a, vector_b, index_matrix):
            shared_vector = np.logical_and(vector_a[index_matrix], vector_b[:index_matrix]).astype(int)
            shared_vector_sum = np.sum(shared_vector, axis=1)
            precision = shared_vector_sum / np.sum(vector_a, axis=1)
            recall = shared_vector_sum / np.sum(vector_b, axis=1)
            if recall > 0 and precision > 0:
                f1 = 2 * recall * precision / (recall + precision)
            else:
                f1 = None
            return f1, precision, recall

        f1_syn_train, precision_syn_train, recall_syn_train = cal_f1_score(train_sensitive_vectors, synthetic_sensitive_vectors, synthetic_train_index)
        f1_train_train, precision_train_train, recall_train_train = cal_f1_score(train_sensitive_vectors, train_sensitive_vectors, train_train_index)

        results = {
            "Iteration": i,
            "Precision Synthetic Train": precision_syn_train,
            "Recall Synthetic Train": recall_syn_train,
            "F1 Synthetic Train": f1_syn_train,
            "Precision Train Train": precision_train_train,
            "Recall Train Train": recall_train_train,
            "F1 Train Train": f1_train_train
        }
        all_results.append(results)
        LOG.info(f'Iteration {i}: Attribute Inference Risk {results}')

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    pd.DataFrame(
        all_results,
        columns=["Iteration", "Precision Synthetic Train", "Recall Synthetic Train", "F1 Synthetic Train",
                 "Precision Train Train", "Recall Train Train",  "F1 Train Train"]
    ).to_parquet(
        os.path.join(args.output_folder, f"attribute_inference_{current_time}.parquet")
    )


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Attribute Inference Analysis Arguments'
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
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The output folder that stores the metrics',
        required=True
    )
    parser.add_argument(
        '--tokenizer_path',
        dest='tokenizer_path',
        action='store',
        help='The path to ConceptTokenizer',
        required=True
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
        '--attribute_config',
        dest='attribute_config',
        action='store',
        help='The configuration yaml file for common and sensitive attributes',
        required=True
    )
    parser.add_argument(
        '--matching_batch_size',
        dest='matching_batch_size',
        action='store',
        default=10000,
        help='The batch size of the matching algorithm',
        required=False
    )
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
