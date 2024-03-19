import numpy as np
from sklearn.preprocessing import OneHotEncoder

from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex

RANDOM_SEE = 42
NUM_OF_GENDERS = 3
NUM_OF_RACES = 21

get_demographics = PatientDataIndex.get_demographics


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


def find_match(source, target):
    a = np.sum(target ** 2, axis=1).reshape(target.shape[0], 1) + np.sum(source.T ** 2, axis=0)
    b = np.dot(target, source.T) * 2
    distance_matrix = a - b
    return np.min(distance_matrix, axis=0)


def find_match_self(source, target):
    a = np.sum(target ** 2, axis=1).reshape(target.shape[0], 1) + np.sum(source.T ** 2, axis=0)
    b = np.dot(target, source.T) * 2
    distance_matrix = a - b
    n_col = np.shape(distance_matrix)[1]
    min_distance = np.zeros(n_col)
    for i in range(n_col):
        sorted_column = np.sort(distance_matrix[:, i])
        min_distance[i] = sorted_column[1]
    return min_distance
