from typing import Union
from datasets import Dataset, DatasetDict
from models.hf_models.tokenization_hf_cehrgpt import CehrGptTokenizer
from data_generators.hf_data_generator.hf_dataset_mapping import (
    SortPatientSequenceMapping, HFCehrGptTokenizationMapping
)
from data_generators.hf_data_generator.hf_dataset import (
    FINETUNING_COLUMNS, HFFineTuningMapping, MedToCehrBertDatasetMapping
)
from runner.hf_runner_argument_dataclass import DataTrainingArguments

CEHRGPT_COLUMNS = [
    'concept_ids', 'concept_values', 'concept_value_masks',
    'mlm_skip_values', 'num_of_concepts', 'num_of_visits',
    'orders', 'dates', 'record_ranks'
]

TRANSFORMER_COLUMNS = ['input_ids', 'concept_ids']


def create_cehrgpt_pretraining_dataset(
        dataset: Union[Dataset, DatasetDict],
        cehrgpt_tokenizer: CehrGptTokenizer,
        data_args: DataTrainingArguments
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRGPT_COLUMNS
    mapping_functions = [
        SortPatientSequenceMapping(),
        HFCehrGptTokenizationMapping(cehrgpt_tokenizer)
    ]
    for mapping_function in mapping_functions:
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
            else:
                dataset = dataset.map(
                    mapping_function.batch_transform,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size
                )
        else:
            dataset = dataset.map(
                mapping_function.batch_transform,
                num_proc=data_args.preprocessing_num_workers,
                batched=True,
                batch_size=data_args.preprocessing_batch_size
            )

    if isinstance(dataset, DatasetDict):
        all_columns = dataset['train'].column_names
    else:
        all_columns = dataset.column_names

    if not data_args.streaming:
        columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
        dataset = dataset.remove_columns(columns_to_remove)

    return dataset


def create_cehrgpt_finetuning_dataset(
        dataset: Union[Dataset, DatasetDict],
        cehrgpt_tokenizer: CehrGptTokenizer,
        data_args: DataTrainingArguments
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRGPT_COLUMNS + FINETUNING_COLUMNS

    if data_args.is_data_in_med:
        mapping_functions = [
            HFCehrGptTokenizationMapping(cehrgpt_tokenizer),
            HFFineTuningMapping()
        ]
    else:
        mapping_functions = [
            SortPatientSequenceMapping(),
            HFCehrGptTokenizationMapping(cehrgpt_tokenizer),
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
