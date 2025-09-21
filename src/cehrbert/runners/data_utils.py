import os
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl
from datasets import DatasetDict, load_from_disk
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import ExtractTokenizedSequenceDataMapping
from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, DataTrainingArguments

LOG = logging.get_logger("transformers")


def extract_cohort_sequences(
    data_args: DataTrainingArguments,
    cehrbert_args: CehrBertArguments,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> DatasetDict:
    """
    Extracts and processes cohort-specific tokenized sequences from a pre-tokenized dataset,.

    based on the provided cohort Parquet files and observation window constraints.

    This function performs the following steps:
    1. Loads cohort definitions from Parquet files located in `data_args.cohort_folder`.
    2. Renames relevant columns if the data originates from a Meds format.
    3. Filters a pre-tokenized dataset (loaded from `cehrgpt_args.tokenized_full_dataset_path`)
       to include only patients present in the cohort.
    4. Aggregates each person's index date and label into a mapping.
    5. Checks for consistency to ensure all cohort person_ids are present in the tokenized dataset.
    6. Applies a transformation (`ExtractTokenizedSequenceDataMapping`) to generate
       observation-window-constrained patient sequences.
    7. Caches both the filtered and processed datasets using the provided `cache_file_collector`.

    Args:
        data_args (DataTrainingArguments): Configuration parameters for data processing,
            including cohort folder, observation window, batch size, and parallelism.
        cehrbert_args (CehrGPTArguments): Contains paths to pre-tokenized datasets and CEHR-GPT-specific arguments.
        cache_file_collector (CacheFileCollector): Utility to register and manage dataset cache files.

    Returns:
        DatasetDict: A Hugging Face `DatasetDict` containing the processed datasets (e.g., train/validation/test),
                     where each entry includes sequences filtered and truncated by the observation window.

    Raises:
        RuntimeError: If any `person_id` in the cohort is missing from the tokenized dataset.
    """

    cohort = pl.read_parquet(os.path.join(data_args.cohort_folder, "*.parquet"))
    if data_args.is_data_in_meds:
        cohort = cohort.rename(
            mapping={
                "prediction_time": "index_date",
                "subject_id": "person_id",
                "boolean_value": "label",
            }
        )
    all_person_ids = cohort["person_id"].unique().to_list()
    # In case the label column does not exist, we add a fake column to the dataframe so subsequent process can work
    if "label" not in cohort.columns:
        cohort = cohort.with_columns(pl.Series(name="label", values=np.zeros_like(cohort["person_id"].to_numpy())))

    # data_args.observation_window
    tokenized_dataset = load_from_disk(cehrbert_args.tokenized_full_dataset_path)
    filtered_tokenized_dataset = tokenized_dataset.filter(
        lambda batch: [person_id in all_person_ids for person_id in batch["person_id"]],
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        num_proc=data_args.preprocessing_num_workers,
    )
    person_index_date_agg = cohort.group_by("person_id").agg(pl.struct("index_date", "label").alias("index_date_label"))
    # Convert to dictionary
    person_index_date_map: Dict[int, List[Dict[str, Any]]] = dict(
        zip(
            person_index_date_agg["person_id"].to_list(),
            person_index_date_agg["index_date_label"].to_list(),
        )
    )
    LOG.info(f"person_index_date_agg: {person_index_date_agg}")
    tokenized_person_ids = []
    for _, dataset in filtered_tokenized_dataset.items():
        tokenized_person_ids.extend(dataset["person_id"])
    missing_person_ids = [
        person_id for person_id in person_index_date_map.keys() if person_id not in tokenized_person_ids
    ]
    if missing_person_ids:
        raise RuntimeError(
            f"There are {len(missing_person_ids)} missing in the tokenized dataset. "
            f"The list contains: {missing_person_ids}"
        )
    processed_dataset = filtered_tokenized_dataset.map(
        ExtractTokenizedSequenceDataMapping(person_index_date_map, data_args.observation_window).batch_transform,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=filtered_tokenized_dataset["train"].column_names,
    )
    return processed_dataset
