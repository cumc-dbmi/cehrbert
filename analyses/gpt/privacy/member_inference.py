import os
import pickle
import sys
import traceback
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import dask.dataframe as dd
from multiprocessing import Pool
import logging

from analyses.gpt.privacy.patient_index import (
    index_options, PatientDataWeaviateDocumentIndex, PatientDataHnswDocumentIndex
)

logger = logging.getLogger('member_inference')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        patient_data_index_cls,
        output_folder,
        tokenizer_path,
        set_unique_concepts,
        batch_size,
        cls_args
):
    try:
        concept_tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    except (AttributeError, EOFError, ImportError, IndexError, OSError) as e:
        sys.exit(traceback.format_exc(e))
    except Exception as e:
        # everything else, possibly fatal
        sys.exit(traceback.format_exc(e))

    patient_indexer = patient_data_index_cls(
        concept_tokenizer=concept_tokenizer,
        set_unique_concepts=set_unique_concepts,
        **cls_args
    )

    labels = []
    dists = []
    attack_person_ids = []
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

        if synthetic_match and len(synthetic_match) > 0:
            if synthetic_match[0]['concept_ids']:
                dist = calculate_hamming_distance(ehr_source, synthetic_match[0])
                labels.append(t.label)
                dists.append(dist)
                attack_person_ids.append(t.person_id)

        if len(labels) > 0 and len(labels) % batch_size == 0:
            results_df = pd.DataFrame(zip(attack_person_ids, dists, labels), columns=['person_id', 'dist', 'label'])
            current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            results_df.to_parquet(os.path.join(output_folder, f'{current_time}.parquet'))

            # Clear the lists for the next batch
            attack_person_ids.clear()
            dists.clear()
            labels.clear()

    # Final flush to the disk in case of any leftover
    if len(labels) > 0:
        results_df = pd.DataFrame(zip(attack_person_ids, dists, labels), columns=['person_id', 'dist', 'label'])
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        results_df.to_parquet(os.path.join(output_folder, f'{current_time}.parquet'))


def remove_processed_records(dataset, output_folder):
    try:
        existing_results = dd.read_parquet(output_folder)
        existing_person_ids = existing_results.person_id.compute().tolist()
        return dataset[~dataset.person_id.isin(existing_person_ids)]
    except Exception as e:
        logger.warning(e)
    return dataset


def main_parallel(
        args
):
    dataset = dd.read_parquet(args.attack_data_folder)
    dataset = dataset.repartition(args.num_of_cores)

    dataset = remove_processed_records(dataset, args.output_folder)

    patient_data_index_class = index_options[args.index_option]
    if patient_data_index_class == PatientDataHnswDocumentIndex:
        cls_args = {
            'index_folder': args.index_folder
        }
    elif patient_data_index_class == PatientDataWeaviateDocumentIndex:
        cls_args = {
            'index_name': args.index_name,
            'server_name': args.server_name
        }
    else:
        raise RuntimeError(f'{args.index_option} is an invalid PatientDataIndex')

    pool_tuples = []
    for i in range(0, args.num_of_cores):
        pool_tuples.append(
            (
                dataset.get_partition(i).compute(), patient_data_index_class, args.output_folder,
                args.tokenizer_path, args.set_unique_concepts, args.batch_size, cls_args
            )
        )
    with Pool(processes=args.num_of_cores) as p:
        p.starmap(match_patients, pool_tuples)
        p.close()
        p.join()
    print('Done')


def create_argparser():
    import argparse
    from sys import argv
    weaviate_index_required = 'PatientDataWeaviateDocumentIndex' in argv
    hnsw_index_required = 'PatientDataHnswDocumentIndex' in argv
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
        '--index_option',
        dest='index_option',
        action='store',
        choices=index_options.keys(),
        required=True
    )
    parser.add_argument(
        '--index_folder',
        dest='index_folder',
        action='store',
        help='The index folder',
        required=hnsw_index_required
    )
    parser.add_argument(
        '--server_name',
        dest='server_name',
        action='store',
        help='The index folder',
        required=weaviate_index_required
    )
    parser.add_argument(
        '--index_name',
        dest='index_name',
        action='store',
        help='The index folder',
        required=weaviate_index_required
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
        '--batch_size',
        dest='batch_size',
        type=int,
        action='store',
        required=False,
        default=1024
    )
    parser.add_argument(
        '--num_of_cores',
        dest='num_of_cores',
        type=int,
        action='store',
        required=False,
        default=1
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
