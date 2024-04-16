from typing import Union
from datasets import Dataset, DatasetDict
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from data_generators.hf_data_generator.hf_dataset_mapping import (
    MedToCehrBertDatasetMapping,
    SortPatientSequenceMapping,
    GenerateStartEndIndexMapping,
    HFMaskedLanguageModellingMapping,
    HFFineTuningMapping,
    TruncationType
)
from spark_apps.decorators.patient_event_decorator import time_token_func, time_day_token

CEHRBERT_COLUMNS = [
    'concept_ids', 'ages', 'dates', 'visit_segments',
    'visit_concept_orders', 'concept_values', 'concept_value_masks'
]

TRANSFORMER_COLUMNS = ['input_ids', 'labels']

FINETUNING_COLUMNS = ['age_at_index', 'classifier_label']


def create_cehrbert_pretraining_dataset(
        dataset: Union[Dataset, DatasetDict],
        concept_tokenizer: CehrBertTokenizer,
        max_sequence_length: int,
        is_pretraining: bool = True,
        num_proc: int = 4,
        is_data_in_med: bool = False,
        include_inpatient_att_token: bool = False
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS
    mapping_functions = [
        SortPatientSequenceMapping(),
        GenerateStartEndIndexMapping(max_sequence_length),
        HFMaskedLanguageModellingMapping(concept_tokenizer, is_pretraining)
    ]

    if is_data_in_med:
        med_to_cehrbert_mapping = MedToCehrBertDatasetMapping(
            time_token_function=time_token_func,
            include_inpatient_att=include_inpatient_att_token,
            inpatient_time_token_function=time_day_token
        )
        mapping_functions.insert(0, med_to_cehrbert_mapping)

    for mapping_function in mapping_functions:
        dataset = dataset.map(mapping_function.transform, num_proc=num_proc)

    if isinstance(dataset, DatasetDict):
        all_columns = dataset['train'].column_names
    else:
        all_columns = dataset.column_names

    columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def create_cehrbert_finetuning_dataset(
        dataset: Union[Dataset, DatasetDict],
        concept_tokenizer: CehrBertTokenizer,
        max_sequence_length: int,
        num_proc: int = 4,
        is_data_in_med: bool = False,
        include_inpatient_att_token: bool = False
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS + FINETUNING_COLUMNS
    mapping_functions = [
        SortPatientSequenceMapping(),
        GenerateStartEndIndexMapping(max_sequence_length, truncate_type=TruncationType.TAIL),
        HFMaskedLanguageModellingMapping(concept_tokenizer, False),
        HFFineTuningMapping()
    ]

    if is_data_in_med:
        med_to_cehrbert_mapping = MedToCehrBertDatasetMapping(
            time_token_function=time_token_func,
            include_inpatient_att=include_inpatient_att_token,
            inpatient_time_token_function=time_day_token
        )
        mapping_functions.insert(0, med_to_cehrbert_mapping)

    for mapping_function in mapping_functions:
        dataset = dataset.map(mapping_function.transform, num_proc=num_proc)

    if isinstance(dataset, DatasetDict):
        all_columns = dataset['train'].column_names
    else:
        all_columns = dataset.column_names

    columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset
