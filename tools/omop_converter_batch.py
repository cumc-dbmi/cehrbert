import numpy as np
from omop_entity import Person, VisitOccurrence, ConditionOccurrence, ProcedureOccurrence, \
    DrugExposure
from datetime import date, timedelta
from models.gpt_model import generate_artificial_time_tokens
from tqdm import tqdm
import pandas as pd
import os
import argparse
import uuid
from pathlib import Path
from multiprocessing import Pool

CURRENT_PATH = Path(__file__).parent
start_token_size = 4
ATT_TIME_TOKENS = generate_artificial_time_tokens()
TABLE_LIST = ['person', 'visit_occurrence', 'condition_occurrence', 'procedure_occurrence',
              'drug_exposure']


def create_folder_if_not_exists(output_folder, table_name):
    if not os.path.isdir(Path(output_folder) / table_name):
        os.mkdir(Path(output_folder) / table_name)


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


def append_to_dict(export_dict, omop_entity, id):
    if omop_entity.get_table_name() not in list(export_dict.keys()):
        export_dict[omop_entity.get_table_name()] = {}
    export_dict[omop_entity.get_table_name()][id] = omop_entity
    return export_dict


def delete_bad_sequence(target_dict, id_mappings, person_id):
    for table_name, id_mapping in id_mappings.items():
        omop_id_mapping = np.array(list(id_mapping.keys()))
        person_id_mapping = np.array(list(id_mapping.values()))
        ids_to_delete = omop_id_mapping[np.where(person_id_mapping == person_id)]
        for id in ids_to_delete:
            target_dict[table_name].pop(id)
    return target_dict


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


def export_and_clear_parquet(
        output_folder,
        export_dict,
        export_error,
        id_mappings_dict,
        pt_seq_dict
):
    for table_name, records_to_export in export_dict.items():
        export_error[table_name] = []
        records_in_json = []
        omop_id_mapping = np.array(list(id_mappings_dict[table_name].keys()))
        person_id_mapping = np.array(list(id_mappings_dict[table_name].values()))

        # If there is no record, we skip it
        if len(export_dict[table_name]) == 0:
            continue

        for idx, record in export_dict[table_name].items():
            try:
                records_in_json.append(record.export_as_json())
            except AttributeError:
                # append patient sequence to export error list using pt_seq_dict.
                export_error[table_name].append(
                    pt_seq_dict[person_id_mapping[np.where(omop_id_mapping == idx)][0]])
                continue
        schema = next(iter(records_to_export.items()))[1].get_schema()
        output_folder_path = Path(output_folder)
        file_path = output_folder_path / table_name / f'{uuid.uuid4()}.parquet'
        table_df = pd.DataFrame(records_in_json, columns=schema)
        table_df.to_parquet(file_path)
        export_dict[table_name].clear()
    return export_dict, export_error


def gpt_to_omop_converter_serial(const, pat_seq_split, domain_map, output_folder, buffer_size,
                                 original_person_id):
    omop_export_dict = {}
    error_dict = {}
    export_error = {}
    id_mappings_dict = {}
    pt_seq_dict = {}

    for tb in TABLE_LIST:
        create_folder_if_not_exists(output_folder, tb)
        id_mappings_dict[tb] = {}
    pat_seq_len = pat_seq_split.shape[0]

    visit_occurrence_id: int = const + 1
    condition_occurrence_id: int = const + 1
    procedure_occurrence_id: int = const + 1
    drug_exposure_id: int = const + 1

    person_id: int = const + 1

    for index, row in tqdm(enumerate(pat_seq_split), total=pat_seq_len):
        bad_sequence = False
        # ignore start token
        if original_person_id:
            person_id = row[0]
            if 'start' in row[1].lower():
                row = row[2:]
            else:
                row = row[1:]
        else:
            if 'start' in row[0].lower():
                row = row[1:]
            else:
                row = row[0:]
        tokens_generated = row[start_token_size:]
        # TODO:Need to decode if the input is tokenized
        start_tokens = row[0:start_token_size]
        [start_year, start_age, start_gender, start_race] = [_ for _ in start_tokens]
        if 'year' not in start_year.lower():
            continue
        start_year = start_year.split(':')[1]
        start_age = start_age.split(':')[1]
        birth_year = int(start_year) - int(start_age)
        p = Person(person_id, start_gender, birth_year, start_race)
        omop_export_dict = append_to_dict(omop_export_dict, p, person_id)
        id_mappings_dict['person'][person_id] = person_id
        pt_seq_dict[person_id] = row
        VS_DATE = date(int(start_year), 1, 1)
        ATT_DATE_DELTA = 0
        vo = None
        for idx, x in enumerate(tokens_generated, 0):
            if x == 'VS':
                try:
                    visit_concept_id = int(tokens_generated[idx + 1])
                except (IndexError, ValueError):
                    error_dict[person_id] = {}
                    error_dict[person_id]['row'] = row
                    error_dict[person_id]['error'] = 'Wrong visit concept id'
                    bad_sequence = True
                    continue
                VS_DATE = VS_DATE + timedelta(days=ATT_DATE_DELTA)
                vo = VisitOccurrence(visit_occurrence_id, visit_concept_id, VS_DATE, p)
                omop_export_dict = append_to_dict(omop_export_dict, vo, visit_occurrence_id)
                id_mappings_dict['visit_occurrence'][visit_occurrence_id] = person_id
                visit_occurrence_id += 1
            elif x in ATT_TIME_TOKENS:
                if x[0] == 'D':
                    ATT_DATE_DELTA = int(x[1:])
                elif x[0] == 'W':
                    ATT_DATE_DELTA = int(x[1:]) * 7
                elif x[0] == 'M':
                    ATT_DATE_DELTA = int(x[1:]) * 30
                elif x == 'LT':
                    ATT_DATE_DELTA = 365 * 3
            elif x == 'VE':
                # If it's a VE token, nothing needs to be updated because it just means the visit ended
                pass
            elif x in ['START', start_year, start_age, start_gender, start_race]:
                # If it's a start token, skip it
                pass
            else:
                try:
                    concept_id = int(x)
                    if concept_id not in domain_map:
                        error_dict[person_id] = {}
                        error_dict[person_id]['row'] = row
                        error_dict[person_id]['error'] = 'No concept id found'
                        bad_sequence = True
                        continue
                    else:
                        domain = domain_map[concept_id]
                        if domain == 'Condition':
                            co = ConditionOccurrence(condition_occurrence_id, x, vo)
                            omop_export_dict = append_to_dict(omop_export_dict, co,
                                                              condition_occurrence_id)
                            id_mappings_dict['condition_occurrence'][
                                condition_occurrence_id] = person_id
                            condition_occurrence_id += 1
                        elif domain == 'Procedure':
                            po = ProcedureOccurrence(procedure_occurrence_id, x, vo)
                            omop_export_dict = append_to_dict(omop_export_dict, po,
                                                              procedure_occurrence_id)
                            id_mappings_dict['procedure_occurrence'][
                                procedure_occurrence_id] = person_id
                            procedure_occurrence_id += 1
                        elif domain == 'Drug':
                            de = DrugExposure(drug_exposure_id, x, vo)
                            omop_export_dict = append_to_dict(omop_export_dict, de,
                                                              drug_exposure_id)
                            id_mappings_dict['drug_exposure'][drug_exposure_id] = person_id
                            drug_exposure_id += 1
                except ValueError:
                    error_dict[person_id] = {}
                    error_dict[person_id]['row'] = row
                    error_dict[person_id]['error'] = 'Wrong concept id'
                    bad_sequence = True
                    continue
        if bad_sequence:
            omop_export_dict = delete_bad_sequence(omop_export_dict, id_mappings_dict, person_id)
        if not original_person_id:
            person_id += 1

        if index != 0 and index % buffer_size == 0:
            omop_export_dict, export_error = export_and_clear_parquet(
                output_folder,
                omop_export_dict,
                export_error,
                id_mappings_dict, pt_seq_dict
            )

    # Final flush to the disk if there are still records in the cache
    omop_export_dict, export_error = export_and_clear_parquet(
        output_folder,
        omop_export_dict,
        export_error,
        id_mappings_dict, pt_seq_dict
    )

    with open(Path(output_folder) / "concept_errors.txt", "a") as f:
        f.write(str(error_dict))
    with open(Path(output_folder) / "export_errors.txt", "a") as f:
        f.write(str(export_error))


def gpt_to_omop_converter_parallel(output_folder, concept_parquet_file,
                                   patient_sequences_concept_ids, buffer_size,
                                   cores, original_person_id):
    patient_sequences = patient_sequences_concept_ids['concept_ids']
    domain_map = generate_omop_concept_domain(concept_parquet_file)
    pool_tuples = []
    # TODO: Need to make this dynamic
    const = 10000000
    patient_sequences_list = np.array_split(patient_sequences.tolist(), cores)
    for i in range(1, cores + 1):
        pool_tuples.append(
            (const * i, patient_sequences_list[i - 1], domain_map, output_folder, buffer_size,
             original_person_id))

    with Pool(processes=cores) as p:
        results = p.starmap(gpt_to_omop_converter_serial, pool_tuples)
        p.close()
        p.join()

    return print('Done')


def main(args):
    # tokenizer_path = os.path.join(args.model_folder, 'tokenizer.pickle')
    # tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    concept_parquet_file = pd.read_parquet(os.path.join(args.concept_path))

    if args.original_person_id:
        patient_sequences_concept_ids = pd.read_parquet(os.path.join(args.patient_sequence_path),
                                                        columns=['person_id', 'concept_ids'])
        patient_sequences_concept_ids['concept_ids'] = patient_sequences_concept_ids. \
            apply(lambda row: np.append(row.person_id, row.concept_ids), axis=1)
        patient_sequences_concept_ids.drop(columns=['person_id'], inplace=True)
    else:
        patient_sequences_concept_ids = pd.read_parquet(os.path.join(args.patient_sequence_path),
                                                        columns=['concept_ids'])
    gpt_to_omop_converter_parallel(args.output_folder, concept_parquet_file,
                                   patient_sequences_concept_ids,
                                   args.buffer_size, args.cpu_cores, args.original_person_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Arguments for converting patient sequences to OMOP')
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
    parser.add_argument(
        '--cpu_cores',
        dest='cpu_cores',
        type=int,
        action='store',
        help='The number of cpu cores to use for multiprocessing',
        required=False
    )
    parser.add_argument(
        '--original_person_id',
        dest='original_person_id',
        action='store_true',
        help='Whether or not to use the original person id'
    )

    main(parser.parse_args())
