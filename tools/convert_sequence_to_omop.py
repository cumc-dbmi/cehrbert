from omop_converter_sequence import OmopEntity, Person, VisitOccurrence, ConditionOccurrence
from pyspark.sql import SparkSession
from typing import List
from datetime import datetime, date, timedelta
from models.gpt_model import generate_artificial_time_tokens


spark = SparkSession.builder.appName('Convert GPT patient sequenct to OMOP').getOrCreate()
batch_size = 20
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


def gpt_to_omop_converter(concept_parquet_file, patient_sequence, start_token_size, tokenizer):
    person_id: int = 1
    visit_occurrence_id: int = 1
    condition_occurrence_id: int = 1
    procedure_occurrence_id: int = 1
    drug_exposure_id: int = 1
    domain_map = generate_omop_concept_domain(concept_parquet_file)
    for i in patient_sequence:
        row = i[0]
        tokens_generated = row[start_token_size:]
        start_tokens = row[0:start_token_size]
        [start_year, start_age, start_gender, start_race] = [detokenize_concept_ids(_, tokenizer) for _ in
                                                             start_tokens]
        start_year = start_year.split(':')[1]
        start_age = start_age.split(':')[1]
        birth_year = int(start_year) - int(start_age)
        p = Person(person_id, start_gender, birth_year, start_race)

        VS_DATE = date(int(start_year), 1, 1)
        ATT_DATE_DELTA = 0

        for idx, x in enumerate(tokens_generated, 0):
            if x == 'VS':
                VS_DATE = VS_DATE + timedelta(days=ATT_DATE_DELTA)
                vo = VisitOccurrence(visit_occurrence_id, VS_DATE, p)
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
                if domain == 'Condition':
                    co = ConditionOccurrence(condition_occurrence_id, vo)
                elif domain == 'Procedure':
                    procedure_records.append(event_record)
                elif domain == 'Drug':
                    drug_records.append(event_record)







p = Person(1)
vo = VisitOccurrence(1, 212, p)
omop_entities = [
    vo,
    ConditionOccurrence(
        1,
        vo
    ),
    ConditionOccurrence(
        2,
        vo
    )
]



export_dict: [str, List[OmopEntity]] = dict()

for omop_entity in omop_entities:
    if omop_entity.get_table_name() not in export_dict:
        export_dict[omop_entity.get_table_name()] = []
    export_dict[omop_entity.get_table_name()].append(omop_entity)




from typing import List
import pandas as pd

batch_size = 100
export_dict: [str, List[OmopEntity]] = dict()

for omop_entity in omop_entities:
    if omop_entity.get_table_name() not in export_dict:
        export_dict[omop_entity.get_table_name()] = []
    export_dict[omop_entity.get_table_name()].append(omop_entity)

for table_name in export_dict.keys():
    if len(export_dict[table_name]) >= batch_size:
        records_to_export = export_dict[table_name]
        records_in_json = [record.export_as_json() for record in export_dict[table_name]]
        table_name = records_to_export[0].get_table_name()
        schema = records_to_export[0].get_schema()
        pd.DataFrame(
            records_in_json,
            mode='a',
            columns=schema
        ).to_csv(table_name)


# Read the patient sequence
df = spark.read.parquet('../gpt/real_patient_sequence')
patient_sequence = df.select('concept_ids').collect()