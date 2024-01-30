import os
import pickle
import sys
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from multiprocessing import Pool
import logging
from typing import Union
import yaml
import uuid
from pyspark.sql.types import DoubleType

from analyses.gpt.privacy.patient_index import (
    index_options, PatientDataWeaviateDocumentIndex, PatientDataHnswDocumentIndex
)

logger = logging.getLogger('attribute_inference')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def match_patients(
        data_partition,
        output_folder,
        tokenizer_path,
        set_unique_concepts,
        batch_size,
        year_std,
        age_std,
        common_attributes,
        sensitive_attributes,
        cls_args
):
    try:
        concept_tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    except (AttributeError, EOFError, ImportError, IndexError, OSError) as e:
        sys.exit(traceback.format_exc(e))
    except Exception as e:
        # everything else, possibly fatal
        sys.exit(traceback.format_exc(e))

    patient_indexer = PatientDataWeaviateDocumentIndex(
        concept_tokenizer=concept_tokenizer,
        set_unique_concepts=set_unique_concepts,
        common_attributes=common_attributes,
        sensitive_attributes=sensitive_attributes,
        **cls_args
    )
    attack_person_ids = []
    syn_sensitive_concept_ids = []
    ehr_source_sensitive_concept_ids = []
    precision_list = []
    recall_list = []
    f1_list = []
    for t in tqdm(data_partition.itertuples(), total=len(data_partition)):

        if not patient_indexer.validate_demographics(t.concept_ids):
            continue

        year, age, gender, race = patient_indexer.get_demographics(t.concept_ids)
        concept_ids = patient_indexer.extract_medical_concepts(t.concept_ids)
        common_concept_ids = patient_indexer.extract_common_medical_concepts(t.concept_ids)
        sensitive_concept_ids = patient_indexer.extract_sensitive_medical_concepts(t.concept_ids)
        num_of_visits = t.num_of_visits
        num_of_concepts = t.num_of_concepts

        ehr_source = {
            'year': year,
            'age': age,
            'gender': gender,
            'race': race,
            'concept_ids': concept_ids,
            'common_concept_ids': common_concept_ids,
            'sensitive_concept_ids': sensitive_concept_ids,
            'num_of_visits': num_of_visits,
            'num_of_concepts': num_of_concepts
        }
        if common_concept_ids:
            synthetic_match = patient_indexer.search(
                t.concept_ids,
                age_std=age_std,
                year_std=year_std
            )

            if synthetic_match and len(synthetic_match) > 0:
                common_concept_ids, sensitive_concept_ids = synthetic_match[0]['concept_ids'],synthetic_match[0]['sensitive_attributes']
#                 print('matched concepts are: ', common_concept_ids)
#                 print('matched sensitive concepts are: ', sensitive_concept_ids)
#                 print('real sensitive concepts are: ', ehr_source['sensitive_concept_ids'])
                if len(common_concept_ids) > 0 and len(sensitive_concept_ids) > 0 and len(ehr_source['sensitive_concept_ids']) > 0:
                    attack_person_ids.append(t.person_id)
                    syn_sensitive_concept_ids.append(sensitive_concept_ids)
                    ehr_source_sensitive_concept_ids.append(ehr_source['sensitive_concept_ids'])

                    shared_concept_ids = set(sensitive_concept_ids).intersection(set(ehr_source['sensitive_concept_ids']))
                    recall = len(shared_concept_ids)/len(ehr_source['sensitive_concept_ids'])
                    precision = len(shared_concept_ids)/len(sensitive_concept_ids)
                    if recall > 0 and precision > 0:
                        f1 = 2*recall*precision/(recall+precision)
                    else:
                        f1 = None
                    recall_list.append(recall)
                    precision_list.append(precision)
                    f1_list.append(f1)
        if len(attack_person_ids) > 0 and len(attack_person_ids) % batch_size == 0:
            results_df = pd.DataFrame(zip(attack_person_ids, 
                                          syn_sensitive_concept_ids, 
                                          ehr_source_sensitive_concept_ids, 
                                          recall_list, 
                                          precision_list, 
                                          f1_list), 
                                      columns=['person_id', 'syn_sensitive_concept_ids', 'ehr_source_sensitive_concept_id', 'recall', 'precision', 'f1'])
            current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            results_df['f1'] = results_df['f1'].astype(float)
            results_df.to_parquet(os.path.join(output_folder, f'{uuid.uuid4()}.parquet'))

            # Clear the lists for the next batch
            attack_person_ids.clear()
            syn_sensitive_concept_ids.clear()
            ehr_source_sensitive_concept_ids.clear()
            recall_list.clear()
            precision_list.clear()
            f1_list.clear()
            
    # Final flush to the disk in case of any leftover
    if len(attack_person_ids) > 0:
        results_df = pd.DataFrame(zip(attack_person_ids, 
                                      syn_sensitive_concept_ids, 
                                      ehr_source_sensitive_concept_ids,
                                      recall_list,
                                      precision_list,
                                      f1_list), 
                                  columns=['person_id', 'syn_sensitive_concept_ids', 'ehr_source_sensitive_concept_id', 'recall', 'precision', 'f1'])
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        results_df['f1'] = results_df['f1'].astype(float)
        results_df.to_parquet(os.path.join(output_folder, f'{uuid.uuid4()}.parquet'))


def remove_processed_records(dataset, output_folder):
    try:
        existing_results = pd.read_parquet(output_folder)
        existing_person_ids = existing_results.person_id.tolist()
        return dataset[~dataset.person_id.isin(existing_person_ids)]
    except Exception as e:
        logger.warning(e)
    return dataset


def main_parallel(
        args
):
    dataset = pd.read_parquet(args.attack_data_folder)
    dataset = remove_processed_records(dataset, args.output_folder)
    dataset_parts = np.array_split(dataset, args.num_of_cores)
    
    cls_args = {
        'index_name': args.index_name,
        'server_name': args.server_name
    }
    if args.attribute_config:
        try:
            with open(args.attribute_config, 'r') as file:
                data = yaml.safe_load(file)
            if 'common_attributes' in data:
                common_attributes = data['common_attributes']
            if 'sensitive_attributes' in data:
                sensitive_attributes = data['sensitive_attributes']
        except FileNotFoundError:
            print(f"The file {args.attribute_config} was not found")
        except Union[PermissionError, OSError] as e:
            print(e)

    pool_tuples = []
    for i in range(0, args.num_of_cores):
        pool_tuples.append(
            (
                dataset_parts[i], args.output_folder,
                args.tokenizer_path, args.set_unique_concepts, args.batch_size,
                args.year_std, args.age_std, common_attributes, sensitive_attributes, cls_args
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
    parser = argparse.ArgumentParser(
        description='Attribute Inference Analysis Arguments'
    )
    parser.add_argument(
        '--attack_data_folder',
        dest='attack_data_folder',
        action='store',
        help='The path for where the attack data folder',
        required=True
    )
    parser.add_argument(
        '--server_name',
        dest='server_name',
        action='store',
        help='The index folder',
        required=True
    )
    parser.add_argument(
        '--index_name',
        dest='index_name',
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
        '--batch_size',
        dest='batch_size',
        type=int,
        action='store',
        required=False,
        default=1024
    )
    parser.add_argument(
        '--age_std',
        dest='age_std',
        type=int,
        action='store',
        required=False,
        default=1
    )
    parser.add_argument(
        '--year_std',
        dest='year_std',
        type=int,
        action='store',
        required=False,
        default=10
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
    parser.add_argument(
        '--attribute_config',
        dest='attribute_config',
        action='store',
        help='The configuration yaml file for common and sensitive attributes',
        required=False
    )
    return parser


if __name__ == "__main__":
    main_parallel(create_argparser().parse_args())
