from typing import Union
from datasets import Dataset, DatasetDict, IterableDatasetDict
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from data_generators.hf_data_generator.hf_dataset_mapping import (
    MedToCehrBertDatasetMapping,
    SortPatientSequenceMapping,
    HFTokenizationMapping,
    HFFineTuningMapping
)
from runner.hf_runner_argument_dataclass import DataTrainingArguments

CEHRBERT_COLUMNS = [
    'concept_ids', 'ages', 'dates', 'visit_segments',
    'visit_concept_orders', 'concept_values', 'concept_value_masks',
    'mlm_skip_values'
]

TRANSFORMER_COLUMNS = ['input_ids', 'labels']

FINETUNING_COLUMNS = ['age_at_index', 'classifier_label', 'index_date', 'person_id']


def create_cehrbert_pretraining_dataset(
        dataset: Union[Dataset, DatasetDict],
        concept_tokenizer: CehrBertTokenizer,
        data_args: DataTrainingArguments
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS
    # If the data is already in meds, we don't need to sort the sequence anymore
    if data_args.is_data_in_med:
        mapping_functions = [
            MedToCehrBertDatasetMapping(
                data_args,
                True
            ),
            HFTokenizationMapping(concept_tokenizer, True)
        ]
    else:
        mapping_functions = [
            SortPatientSequenceMapping(),
            HFTokenizationMapping(concept_tokenizer, True)
        ]

    for mapping_function in mapping_functions:
        dataset = _apply_mapping(data_args, dataset, mapping_function)

    if not data_args.streaming:
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
        data_args: DataTrainingArguments
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS + FINETUNING_COLUMNS

    if data_args.is_data_in_med:
        mapping_functions = [
            MedToCehrBertDatasetMapping(
                data_args,
                False
            ),
            HFTokenizationMapping(concept_tokenizer, False),
            HFFineTuningMapping()
        ]
    else:
        mapping_functions = [
            SortPatientSequenceMapping(),
            HFTokenizationMapping(concept_tokenizer, False),
            HFFineTuningMapping()
        ]

    for mapping_function in mapping_functions:
        dataset = _apply_mapping(data_args, dataset, mapping_function)

    if not data_args.streaming:
        if isinstance(dataset, DatasetDict):
            all_columns = dataset['train'].column_names
        else:
            all_columns = dataset.column_names
        columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
        dataset = dataset.remove_columns(columns_to_remove)
    return dataset


def _apply_mapping(data_args, dataset, mapping_function):
    if data_args.streaming:
        if isinstance(dataset, DatasetDict):
            for dataset_name in dataset.keys():
                dataset[dataset_name] = (
                    dataset[dataset_name].map(
                        mapping_function.batch_transform,
                        batched=True,
                        batch_size=data_args.preprocessing_batch_size
                    )
                )
                dataset[dataset_name] = dataset[dataset_name].remove_columns(mapping_function.remove_columns())
        else:
            dataset = dataset.map(
                mapping_function.batch_transform,
                batched=True,
                batch_size=data_args.preprocessing_batch_size
            )
            dataset = dataset.remove_columns(mapping_function.remove_columns())
    else:
        dataset = dataset.map(
            mapping_function.batch_transform,
            num_proc=data_args.preprocessing_num_workers,
            batched=True,
            batch_size=data_args.preprocessing_batch_size
        )
        dataset = dataset.remove_columns(mapping_function.remove_columns())
    return dataset
