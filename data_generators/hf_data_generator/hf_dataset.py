from datasets import Dataset, DatasetDict
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from data_generators.hf_data_generator.hf_dataset_mapping import (
    SortPatientSequenceMapping, GenerateStartEndIndexMapping, HFMaskedLanguageModellingMapping
)

CEHRBERT_COLUMNS = [
    'concept_ids', 'ages', 'dates', 'visit_segments',
    'visit_concept_orders', 'concept_values', 'concept_value_masks'
]

TRANSFORMER_COLUMNS = ['input_ids', 'labels']


def create_cehrbert_dataset(
        dataset: Dataset,
        concept_tokenizer: CehrBertTokenizer,
        max_sequence_length: int,
        is_pretraining: bool = True,
        num_proc: int = 4
) -> Dataset:
    required_columns = TRANSFORMER_COLUMNS + CEHRBERT_COLUMNS
    mapping_functions = [
        SortPatientSequenceMapping(),
        GenerateStartEndIndexMapping(max_sequence_length),
        HFMaskedLanguageModellingMapping(concept_tokenizer, is_pretraining)
    ]

    for mapping_function in mapping_functions:
        dataset = dataset.map(mapping_function.transform, num_proc=num_proc)

    if isinstance(dataset, DatasetDict):
        all_columns = dataset['train'].column_names
    else:
        all_columns = dataset.column_names

    columns_to_remove = [_ for _ in all_columns if _ not in required_columns]
    dataset = dataset.remove_columns(columns_to_remove)
    return dataset
