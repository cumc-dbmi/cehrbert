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
from runner.hf_runner_argument_dataclass import DataTrainingArguments

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
        data_args: DataTrainingArguments
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS
    mapping_functions = [
        SortPatientSequenceMapping(),
        GenerateStartEndIndexMapping(max_sequence_length),
        HFMaskedLanguageModellingMapping(concept_tokenizer, True)
    ]

    if data_args.is_data_in_med:
        med_to_cehrbert_mapping = MedToCehrBertDatasetMapping(
            data_args
        )
        mapping_functions.insert(0, med_to_cehrbert_mapping)

    for mapping_function in mapping_functions:
        dataset = dataset.map(mapping_function.transform, num_proc=data_args.preprocessing_num_workers)

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
        data_args: DataTrainingArguments
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS + FINETUNING_COLUMNS
    mapping_functions = [
        SortPatientSequenceMapping(),
        GenerateStartEndIndexMapping(max_sequence_length, truncate_type=TruncationType.TAIL),
        HFMaskedLanguageModellingMapping(concept_tokenizer, False),
        HFFineTuningMapping()
    ]

    if data_args.is_data_in_med:
        med_to_cehrbert_mapping = MedToCehrBertDatasetMapping(
            data_args
        )
        mapping_functions.insert(0, med_to_cehrbert_mapping)

    for mapping_function in mapping_functions:
        dataset = dataset.map(mapping_function.transform, num_proc=data_args.preprocessing_num_workers)

    if isinstance(dataset, DatasetDict):
        all_columns = dataset['train'].column_names
    else:
        all_columns = dataset.column_names

    columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset
