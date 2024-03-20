import os
import pickle
import pandas as pd
import logging
import random
from sklearn import metrics
from datetime import datetime

from analyses.gpt.privacy.utils import create_race_encoder, create_gender_encoder, scale_age, create_demographics, \
    create_vector_representations, find_match, RANDOM_SEE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOG = logging.getLogger('member_inference')


def main(
        args
):
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
        synthetic_train_dist = find_match(synthetic_vectors, train_vectors)
        synthetic_evaluation_dist = find_match(synthetic_vectors, evaluation_vectors)

        dist_metrics.extend([(_, 1) for _ in synthetic_train_dist])
        dist_metrics.extend([(_, 0) for _ in synthetic_evaluation_dist])

        metrics_pd = pd.DataFrame(dist_metrics, columns=['dist', 'label'])
        threshold = metrics_pd.dist.median()
        metrics_pd['pred'] = (metrics_pd.dist < threshold).astype(int)

        results = {
            "Iteration": i,
            "Accuracy": metrics.accuracy_score(metrics_pd.label, metrics_pd.pred),
            "Precision": metrics.precision_score(metrics_pd.label, metrics_pd.pred),
            "Recall": metrics.recall_score(metrics_pd.label, metrics_pd.pred),
            "F1": metrics.f1_score(metrics_pd.label, metrics_pd.pred)
        }
        all_results.append(results)
        LOG.info(f'Iteration {i}: Privacy loss {results}')

    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    pd.DataFrame(
        all_results,
        columns=["Iteration", "Accuracy", "Precision", "Recall", "F1"]
    ).to_parquet(
        os.path.join(args.output_folder, f"membership_inference_{current_time}.parquet")
    )


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Membership Inference Analysis Arguments'
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
    return parser


if __name__ == "__main__":
    main(create_argparser().parse_args())
