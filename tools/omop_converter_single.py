import sys
from pathlib import Path
import math
from datetime import datetime, date, timedelta

import csv
# from pyspark.sql import SparkSession
from models.gpt_model import generate_artificial_time_tokens

ATT_TIME_TOKENS = generate_artificial_time_tokens()
OUTPUT_PATH = Path('__file__').parent.parent / 'gpt_omop_data'


# spark = SparkSession.builder.appName('Convert GPT patient representations to OMOP').getOrCreate()


def detokenize_concept_ids(
        number,
        tokenizer):
    # TODO: Add batch processing
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


def create_omop_visit_record(person_id, visit_occurrence_id, visit_date):
    visit_occurrence_record = {'visit_occurrence_id': visit_occurrence_id,
                               'person_id': person_id,
                               'visit_concept_id': 9201,
                               'visit_start_date': visit_date,
                               'visit_start_datetime': datetime.combine(visit_date, datetime.min.time()),
                               'visit_end_date': visit_date,
                               'visit_end_datetime': datetime.combine(visit_date, datetime.max.time()),
                               'visit_type_concept_id': 44818702,
                               'provider_id': 0,
                               'care_site_id': 0,
                               'visit_source_value': '',
                               'visit_source_concept_id': 0,
                               'admitting_source_concept_id': 0,
                               'admitting_source_value': '',
                               'discharge_to_concept_id': 0,
                               'discharge_to_source_value': '',
                               'preceding_visit_occurrence_id': 0
                               }
    return visit_occurrence_record


def save_to_csv(event_list, output_path):
    """
    Save a dictionary to a pyspark parquet file
    :param event_list: list of dictionary for each OMOP event record
    :param output_path: output path
    :return: None
    """
    if len(event_list) == 0:
        return print('No events to save')
    with open(output_path, 'a', newline='') as file:
        fieldnames = list(event_list[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        file.seek(0, 2)
        if file.tell() == 0:
            writer.writeheader()
        for row in event_list:
            writer.writerow(row)
        return print('Saved csv file to: {}'.format(output_path))


def create_omop_event_record(person_id,
                             visit_occurrence_id,
                             visit_date,
                             domain,
                             concept_id,
                             condition_occurrence_id,
                             procedure_occurrence_id,
                             drug_exposure_id):
    if domain == 'Condition':
        event_occurrence_record = {'condition_occurrence_id': condition_occurrence_id,
                                   'person_id': person_id,
                                   'condition_concept_id': concept_id,
                                   'condition_start_date': visit_date,
                                   'condition_start_datetime': datetime.combine(visit_date, datetime.min.time()),
                                   'condition_end_date': visit_date,
                                   'condition_end_datetime': datetime.combine(visit_date, datetime.max.time()),
                                   'condition_type_concept_id': 32817,
                                   'condition_status_concept_id': 0,
                                   'stop_reason': '',
                                   'provider_id': 0,
                                   'visit_occurrence_id': visit_occurrence_id,
                                   'visit_detail_id': 0,
                                   'condition_source_value': '',
                                   'condition_source_concept_id': 0,
                                   'condition_status_source_value': ''
                                   }
        condition_occurrence_id += 1
    elif domain == 'Procedure':
        event_occurrence_record = {'procedure_occurrence_id': procedure_occurrence_id,
                                   'person_id': person_id,
                                   'procedure_concept_id': concept_id,
                                   'procedure_date': visit_date,
                                   'procedure_datetime': datetime.combine(visit_date, datetime.min.time()),
                                   'procedure_type_concept_id': 32817,
                                   'modifier_concept_id': 0,
                                   'quantity': 0,
                                   'provider_id': 0,
                                   'visit_occurrence_id': visit_occurrence_id,
                                   'visit_detail_id': 0,
                                   'procedure_source_value': '',
                                   'procedure_source_concept_id': 0,
                                   'modifier_source_value': ''
                                   }
        procedure_occurrence_id += 1
    elif domain == 'Drug':
        event_occurrence_record = {'drug_exposure_id': drug_exposure_id,
                                   'person_id': person_id,
                                   'drug_concept_id': concept_id,
                                   'drug_exposure_start_date': visit_date,
                                   'drug_exposure_start_datetime': datetime.combine(visit_date, datetime.min.time()),
                                   'drug_exposure_end_date': visit_date,
                                   'drug_exposure_end_datetime': datetime.combine(visit_date, datetime.max.time()),
                                   'verbatim_end_date': '',
                                   'drug_type_concept_id': 32817,
                                   'stop_reason': '',
                                   'refills': None,
                                   'quantity': None,
                                   'days_supply': None,
                                   'sig': '',
                                   'route_concept_id': 0,
                                   'lot_number': '',
                                   'provider_id': 0,
                                   'visit_occurrence_id': visit_occurrence_id,
                                   'visit_detail_id': 0,
                                   'drug_source_value': '',
                                   'drug_source_concept_id': 0,
                                   'route_source_value': '',
                                   'dose_unit_source_value': ''
                                   }
        drug_exposure_id += 1
    else:
        pass
    return [condition_occurrence_id, procedure_occurrence_id, drug_exposure_id, event_occurrence_record]


def gpt_to_omop_converter(concept_parquet, person_id, tokenizer, start_tokens, generated_tokens, visit_occurrence_id,
                          condition_occurrence_id, procedure_occurrence_id, drug_exposure_id):
    domain_map = generate_omop_concept_domain(concept_parquet)
    detokenized_sequence = [detokenize_concept_ids(_, tokenizer) for _ in generated_tokens]
    #TODO: deduplicate detokenized_sequence because one concept can repeat multiple times in a visit
    # create person record
    [start_year, start_age, start_gender, start_race] = [detokenize_concept_ids(_, tokenizer) for _ in
                                                         start_tokens[1:5]]  # Exclude [START] token
    start_year = start_year.split(':')[1]
    start_age = start_age.split(':')[1]
    person_record = [{'person_id': person_id,
                      'gender_concept_id': start_gender,
                      'year_of_birth': math.ceil(int(start_year) - int(start_age)),
                      'month_of_birth': 1,
                      'day_of_birth': 1,
                      'birth_datetime': datetime.strptime(str(math.ceil(int(start_year) - int(start_age)))
                                                          + '-01' + '-01', '%Y-%m-%d'),
                      'race_concept_id': start_race,
                      'ethnicity_concept_id': 0,
                      'location_id': 0,
                      'provider_id': 0,
                      'care_site_id': 0,
                      'person_source_value': '',
                      'gender_source_value': '',
                      'gender_souce_concept_id': 0,
                      'race_source_value': '',
                      'race_source_concept_id': 0,
                      'ethnicity_source_value': '',
                      'ethnicity_source_concept_id': 0}]
    # create visit_occurrence records
    visit_occurrence_records = []
    condition_records = []
    procedure_records = []
    drug_records = []
    # separate visit sequences and att tokens
    VS_DATE = date(int(start_year), 1, 1)
    ATT_DATE_DELTA = 0
    for idx, x in enumerate(detokenized_sequence, 1):
        if x == 'VS':
            VS_DATE = VS_DATE + timedelta(days=ATT_DATE_DELTA)
            visit_occurrence_records.append(create_omop_visit_record(person_id, visit_occurrence_id, VS_DATE))
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
            domain = domain_map[int(x)]
            [condition_occurrence_id, procedure_occurrence_id, drug_exposure_id,
             event_record] = create_omop_event_record(person_id, visit_occurrence_id, VS_DATE, domain, int(x),
                                                      condition_occurrence_id, procedure_occurrence_id,
                                                      drug_exposure_id)
            if domain == 'Condition':
                condition_records.append(event_record)
            elif domain == 'Procedure':
                procedure_records.append(event_record)
            elif domain == 'Drug':
                drug_records.append(event_record)
    for event_list, output_path in zip(
            [person_record, visit_occurrence_records, condition_records, procedure_records, drug_records],
            [OUTPUT_PATH / 'person.csv',
             OUTPUT_PATH / 'visit_occurrence.csv',
             OUTPUT_PATH / 'condition_occurrence.csv',
             OUTPUT_PATH / 'procedure_occurrence.csv',
             OUTPUT_PATH / 'drug_exposure.csv']):
        save_to_csv(event_list=event_list, output_path=output_path)
    return [visit_occurrence_id, condition_occurrence_id, procedure_occurrence_id, drug_exposure_id]


if __name__ == '__main__':
    concept_parquet = sys.argv[1]
    person_id = sys.argv[2]
    tokenizer = sys.argv[3]
    start_tokens = sys.argv[4]
    generated_tokens = sys.argv[5]
    visit_occurrence_id = sys.argv[6]
    condition_occurrence_id = sys.argv[7]
    procedure_occurrence_id = sys.argv[8]
    drug_exposure_id = sys.argv[9]
    gpt_to_omop_converter(concept_parquet, person_id, tokenizer, start_tokens, generated_tokens, visit_occurrence_id,
                          condition_occurrence_id, procedure_occurrence_id, drug_exposure_id)
