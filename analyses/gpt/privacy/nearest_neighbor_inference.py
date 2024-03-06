import os
import pickle
import logging
import random
from datetime import datetime
import numpy as np
import pandas as pd
from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger("NNAA")
validate_demographics = PatientDataIndex.validate_demographics
get_demographics = PatientDataIndex.get_demographics

RANDOM_SEE = 42
NUM_OF_GENDERS = 3
NUM_OF_RACES = 21


def create_race_encoder(train_data_sample, evaluation_data_sample, synthetic_data_sample):
    race_encoder = OneHotEncoder()
    all_unique_races = np.unique(np.concatenate([
        train_data_sample.race.unique(),
        evaluation_data_sample.race.unique(),
        synthetic_data_sample.race.unique()], axis=0)
    )
    race_encoder.fit(all_unique_races[:, np.newaxis])
    return race_encoder


def create_gender_encoder(train_data_sample, evaluation_data_sample, synthetic_data_sample):
    gender_encoder = OneHotEncoder()
    all_unique_genders = np.unique(np.concatenate([
        train_data_sample.gender.unique(),
        evaluation_data_sample.gender.unique(),
        synthetic_data_sample.gender.unique()], axis=0)
    )
    gender_encoder.fit(all_unique_genders[:, np.newaxis])
    return gender_encoder


def transform_concepts(dataset, concept_tokenizer):
    def extract_medical_concepts(concept_ids):
        concept_ids = [_ for _ in concept_ids[4:] if str.isnumeric(_)]
        return list(set(concept_ids))

    def create_binary_format(concept_ids):
        indices = np.array(concept_tokenizer.encode(concept_ids)).flatten().astype(int)
        embeddings = np.zeros(concept_tokenizer.get_vocab_size())
        embeddings.put(indices, 1)
        return embeddings

    embedding_list = []
    for _, pat_seq in dataset.concept_ids.items():
        embedding_list.append(create_binary_format(extract_medical_concepts(pat_seq)))

    return np.asarray(embedding_list)


def scale_age(dataset):
    # The first 4 elements have a value of -1 because the corresponding positions are demographic tokens
    ages = dataset.ages.apply(lambda age_list: age_list[4])
    assert (ages >= 0).all() > 0
    max_age = ages.max()
    dataset['scaled_age'] = ages / max_age
    return dataset


def create_demographics(dataset):
    genders = dataset.concept_ids.apply(lambda concept_list: get_demographics(concept_list)[2])
    races = dataset.concept_ids.apply(lambda concept_list: get_demographics(concept_list)[3])
    dataset['gender'] = genders
    dataset['race'] = races
    return dataset


def create_vector_representations(
        dataset,
        concept_tokenizer,
        gender_encoder,
        race_encoder
):
    concept_vectors = transform_concepts(dataset, concept_tokenizer)
    gender_vectors = gender_encoder.transform(dataset.gender.to_numpy()[:, np.newaxis]).todense()
    race_vectors = race_encoder.transform(dataset.race.to_numpy()[:, np.newaxis]).todense()
    age_vectors = dataset.scaled_age.to_numpy()[:, np.newaxis]

    pat_vectors = np.concatenate([
        age_vectors,
        gender_vectors,
        race_vectors,
        concept_vectors
    ], axis=-1)

    return np.asarray(pat_vectors)


def find_replicant(source, target):
    a = np.sum(target ** 2, axis=1).reshape(target.shape[0], 1) + np.sum(source.T ** 2, axis=0)
    b = np.dot(target, source.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)


def find_replicant_self(source, target):
    a = np.sum(target ** 2, axis=1).reshape(target.shape[0], 1) + np.sum(source.T ** 2, axis=0)
    b = np.dot(target, source.T) * 2
    distance_matrix = a - b
    n_col = np.shape(distance_matrix)[1]
    min_distance = np.zeros(n_col)
    for i in range(n_col):
        sorted_column = np.sort(distance_matrix[:, i])
        min_distance[i] = sorted_column[1]
    return min_distance


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
        distance_train_TS = find_replicant(train_vectors, synthetic_vectors)
        distance_train_ST = find_replicant(synthetic_vectors, train_vectors)
        distance_train_TT = find_replicant_self(train_vectors, train_vectors)
        distance_train_SS = find_replicant_self(synthetic_vectors, synthetic_vectors)

        aa_train = (np.sum(distance_train_TS > distance_train_TT) + np.sum(
            distance_train_ST > distance_train_SS)) / args.num_of_samples / 2

        LOG.info(f'Iteration {i}: Started calculating the distances between synthetic and evaluation vectors')
        distance_test_TS = find_replicant(evaluation_vectors, synthetic_vectors)
        distance_test_ST = find_replicant(synthetic_vectors, evaluation_vectors)
        distance_test_TT = find_replicant_self(evaluation_vectors, evaluation_vectors)
        distance_test_SS = find_replicant_self(synthetic_vectors, synthetic_vectors)

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
