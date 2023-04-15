import numpy as np
from omop_entity import Person, VisitOccurrence, ConditionOccurrence, ProcedureOccurrence, DrugExposure
from datetime import date, timedelta
from models.gpt_model import generate_artificial_time_tokens
from tqdm import tqdm
import pandas as pd
import os
import pickle
import argparse
import uuid
from pathlib import Path
from multiprocessing import Pool

cores = 5

CURRENT_PATH = Path(__file__).parent
start_token_size = 4
ATT_TIME_TOKENS = generate_artificial_time_tokens()


def detokenize_concept_ids(
        number,
        tokenizer):
    """
    Detokenize and return the concept_id
    :param number: tokenized concept_id
    :param tokenizer:
    :return: concept_id
    """
    concept_id = tokenizer.decode([[number]])[0]
    return concept_id


def generate_omop_concept_domain(concept_parquet):
    """
    Generate a dictionary of concept_id to domain_id
    :param concept_parquet: concept dataframe read from parquet file
    :return: dictionary of concept_id to domain_id
    """
    domain_dict = {}
    for i in concept_parquet.itertuples():
        domain_dict[i.concept_id] = i.domain_id
    return domain_dict


def append_to_dict(export_dict, omop_entity):
    if omop_entity.get_table_name() not in export_dict:
        export_dict[omop_entity.get_table_name()] = []
    export_dict[omop_entity.get_table_name()].append(omop_entity)
    return export_dict


def export_and_clear_csv(output_folder, export_dict, buffer_size):
    for table_name in export_dict.keys():
        if len(export_dict[table_name]) >= buffer_size:
            records_to_export = export_dict[table_name]
            records_in_json = [record.export_as_json() for record in export_dict[table_name]]
            schema = records_to_export[0].get_schema()
            file_name = table_name + '.csv'
            output_folder_path = Path(output_folder)
            file_path = output_folder_path / file_name
            if not os.path.isfile(file_path):
                pd.DataFrame(
                    records_in_json,
                    columns=schema
                ).to_csv(file_path, header=schema, index=False)
            else:
                pd.DataFrame(
                    records_in_json,
                    columns=schema
                ).to_csv(file_path, mode='a', header=False, index=False)
            export_dict[table_name].clear()
    return export_dict


def export_and_clear_parquet(output_folder, export_dict):
    for table_name, records_to_export in export_dict.items():
        records_in_json = [record.export_as_json() for record in export_dict[table_name]]
        schema = records_to_export[0].get_schema()
        output_folder_path = Path(output_folder)
        file_path = output_folder_path / f'{table_name}.{uuid.uuid4()}.parquet'
        table_df = pd.DataFrame(records_in_json, columns=schema)
        table_df.to_parquet(file_path)
        export_dict[table_name].clear()
    return export_dict


def gpt_to_omop_converter_serial(const, pat_seq_split, domain_map, output_folder, buffer_size):
    person_id: int = const+1
    visit_occurrence_id: int = const+1
    condition_occurrence_id: int = const+1
    procedure_occurrence_id: int = const+1
    drug_exposure_id: int = const+1
    omop_export_dict = {}
    pat_seq_len = pat_seq_split.shape[0]
    for index, row in tqdm(pat_seq_split.iterrows(), total=pat_seq_len):
        # ignore start token
        if 'start' in row[0].lower():
            row = row[1:]
        tokens_generated = row[start_token_size:]
        #TODO:Need to decode if the input is tokenized
        start_tokens = row[0:start_token_size]
        [start_year, start_age, start_gender, start_race] = [_ for _ in start_tokens]
        if 'year' not in start_year.lower():
            continue
        start_year = start_year.split(':')[1]
        start_age = start_age.split(':')[1]
        birth_year = int(start_year) - int(start_age)
        p = Person(person_id, start_gender, birth_year, start_race)
        person_id += 1
        append_to_dict(omop_export_dict, p)
        VS_DATE = date(int(start_year), 1, 1)
        ATT_DATE_DELTA = 0

        vo = None
        for idx, x in enumerate(tokens_generated, 0):
            if x == 'VS':
                visit_concept_id = int(tokens_generated[idx + 1])
                VS_DATE = VS_DATE + timedelta(days=ATT_DATE_DELTA)
                vo = VisitOccurrence(visit_occurrence_id, visit_concept_id, VS_DATE, p)
                append_to_dict(omop_export_dict, vo)
                visit_occurrence_id += 1
            elif x in ATT_TIME_TOKENS:
                if x[0] == 'W':
                    ATT_DATE_DELTA = int(x[1:]) * 7
                elif x[0] == 'M':
                    ATT_DATE_DELTA = int(x[1:]) * 30
                elif x == 'LT':
                    ATT_DATE_DELTA = 365
            elif x == 'VE':
                # If it's a VE token, nothing needs to be updated because it just means the visit ended
                pass
            elif x in ['START', start_year, start_age, start_gender, start_race]:
                # If it's a start token, skip it
                pass
            else:
                if int(x) not in domain_map:
                    continue
                domain = domain_map[int(x)]
                if domain == 'Condition':
                    co = ConditionOccurrence(condition_occurrence_id, x, vo)
                    append_to_dict(omop_export_dict, co)
                    condition_occurrence_id += 1
                elif domain == 'Procedure':
                    po = ProcedureOccurrence(procedure_occurrence_id, x, vo)
                    append_to_dict(omop_export_dict, po)
                    procedure_occurrence_id += 1
                elif domain == 'Drug':
                    de = DrugExposure(drug_exposure_id, x, vo)
                    append_to_dict(omop_export_dict, de)
                    drug_exposure_id += 1
        if index % buffer_size == 0 or index == pat_seq_len:
            omop_export_dict = export_and_clear_parquet(output_folder, omop_export_dict)


def gpt_to_omop_converter_parallel(output_folder, concept_parquet_file, patient_sequences_concept_ids, buffer_size):
    patient_sequences = patient_sequences_concept_ids['concept_ids']
    domain_map = generate_omop_concept_domain(concept_parquet_file)
    pool_tuples = []
    #TODO: Need to make this dynamic
    const = 10000000
    patient_sequences_list = np.array_split(patient_sequences, cores)
    for i in range(1, cores+1):
        pool_tuples.append((const*i, patient_sequences_list[i-1], domain_map, output_folder, buffer_size))

    with Pool(processes=cores) as p:
        results = p.starmap(gpt_to_omop_converter_serial, pool_tuples)
        p.close()
        p.join()

    return print('Done')


def main(args):
    #tokenizer_path = os.path.join(args.model_folder, 'tokenizer.pickle')
    #tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    concept_parquet_file = pd.read_parquet(os.path.join(args.concept_path))
    patient_sequences_conept_ids = pd.read_parquet(os.path.join(args.patient_sequence_path), columns=['concept_ids'])
    gpt_to_omop_converter_parallel(args.output_folder, concept_parquet_file, patient_sequences_conept_ids, args.buffer_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for converting patient sequences to OMOP')

    parser.add_argument(
        '--model_folder',
        dest='model_folder',
        action='store',
        help='The path for your model_folder',
        required=True
    )

    parser.add_argument(
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The path for the output_folder',
        required=True
    )

    parser.add_argument(
        '--concept_path',
        dest='concept_path',
        action='store',
        help='The path for your concept_path',
        required=True
    )
    parser.add_argument(
        '--buffer_size',
        dest='buffer_size',
        action='store',
        type=int,
        help='The size of the batch',
        required=True
    )
    parser.add_argument(
        '--patient_sequence_path',
        dest='patient_sequence_path',
        action='store',
        help='The path for your patient sequence',
        required=False
    )

    main(parser.parse_args())
