import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import dask.dataframe as dd
from multiprocessing import Pool

from analyses.gpt.privacy.patient_index.base_indexer import PatientDataIndex


def calculate_hamming_distance(
        ehr_source,
        synthetic_match
):
    dist = 0
    dist += abs(ehr_source['year'] - synthetic_match['year'])
    dist += abs(ehr_source['age'] - synthetic_match['age'])
    dist += abs(ehr_source['num_of_visits'] - synthetic_match['num_of_visits'])

    num_of_overlap = sum([1 for _ in synthetic_match['concept_ids'] if _ in ehr_source['concept_ids']])
    num_of_overlap += sum([1 for _ in ehr_source['concept_ids'] if _ in synthetic_match['concept_ids']])
    dist += len(synthetic_match['concept_ids']) + len(ehr_source['concept_ids']) - num_of_overlap
    return dist


def match_patients(
        data_partition,
        index_folder,
        output_folder,
        tokenizer_path,
        set_unique_concepts
):
    patient_indexer = PatientDataIndex(
        index_folder=index_folder,
        tokenizer_path=tokenizer_path,
        set_unique_concepts=set_unique_concepts
    )

    labels = []
    dists = []
    for t in tqdm(data_partition.itertuples(), total=len(data_partition)):

        if not patient_indexer.validate_demographics(t.concept_ids):
            continue

        year, age, gender, race = patient_indexer.get_demographics(t.concept_ids)
        concept_ids = patient_indexer.extract_medical_concepts(t.concept_ids)
        num_of_visits = t.num_of_visits
        num_of_concepts = t.num_of_concepts

        ehr_source = {
            'year': year,
            'age': age,
            'gender': gender,
            'race': race,
            'concept_ids': concept_ids,
            'num_of_visits': num_of_visits,
            'num_of_concepts': num_of_concepts
        }
        synthetic_match = patient_indexer.search(t.concept_ids)

        if synthetic_match:
            dist = calculate_hamming_distance(ehr_source, synthetic_match)
            labels.append(t.label)
            dists.append(dist)

    results_df = pd.DataFrame(zip(dists, labels), columns=['dist', 'label'])
    current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    results_df.to_parquet(os.path.join(output_folder, f'{current_time}.parquet'))


def main_parallel(
        args
):
    dataset = dd.read_parquet(args.attack_data_folder)
    dataset = dataset.repartition(args.num_of_cores)

    pool_tuples = []
    for i in range(0, args.num_of_cores):
        pool_tuples.append(
            (
                dataset.get_partition(i).compute(), args.index_folder, args.output_folder,
                args.tokenizer_path, args.set_unique_concepts
            )
        )
    with Pool(processes=args.num_of_cores) as p:
        p.starmap(match_patients, pool_tuples)
        p.close()
        p.join()
    print('Done')


def create_argparser():
    import argparse
    parser = argparse.ArgumentParser(
        description='Membership Inference Analysis Arguments'
    )
    parser.add_argument(
        '--attack_data_folder',
        dest='attack_data_folder',
        action='store',
        help='The path for where the attack data folder',
        required=True
    )
    parser.add_argument(
        '--index_folder',
        dest='index_folder',
        action='store',
        help='The index folder',
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
        '--num_of_cores',
        dest='num_of_cores',
        type=int,
        action='store',
        required=True
    )
    parser.add_argument(
        '--set_unique_concepts',
        dest='set_unique_concepts',
        action='store_true',
        help='Indicate whether to use the unique set of concepts for each patient'
    )
    return parser


if __name__ == "__main__":
    main_parallel(create_argparser().parse_args())
