import glob
import os
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import TrainingArguments
from transformers.utils import is_flash_attn_2_available, logging

from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import (
    CehrBertDataCollator,
    SamplePackingCehrBertDataCollator,
)
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import MedToCehrBertDatasetMapping
from cehrbert.data_generators.hf_data_generator.meds_utils import CacheFileCollector, create_dataset_from_meds_reader
from cehrbert.data_generators.hf_data_generator.sample_packing_sampler import SamplePackingBatchSampler
from cehrbert.models.hf_models.hf_cehrbert import CehrBertForPreTraining
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments
from cehrbert.runners.runner_util import (
    convert_dataset_to_iterable_dataset,
    generate_prepared_ds_path,
    get_meds_extension_path,
    load_parquet_as_dataset,
    parse_runner_args,
)

LOG = logging.get_logger("transformers")


def extract_averaged_embeddings_from_packed_sequence(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:

    # Step 1: Find boundaries (where padding is 0)
    mask = attention_mask[0]  # remove batch dimension for easier processing
    boundary_indices = (mask == 0).nonzero(as_tuple=False).flatten()

    # Add start and end manually
    start_indices = torch.cat([torch.tensor([-1]), boundary_indices])
    end_indices = torch.cat([boundary_indices, torch.tensor([mask.size(0)])])

    # Step 2: Extract embeddings between boundaries and average
    sample_embeddings = []
    for start, end in zip(start_indices, end_indices):
        # Select embeddings between (start, end)
        # Skip if no valid tokens
        if end - start > 1:
            sample = hidden_states[0, start + 1 : end, :]  # slice (start+1) to (end-1)
            avg_embedding = sample.mean(dim=0)  # average over sequence length
            sample_embeddings.append(avg_embedding)
    # Stack results
    sample_embeddings = torch.stack(sample_embeddings, dim=0)
    return sample_embeddings


def prepare_finetune_dataset(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    cache_file_collector: CacheFileCollector,
):
    # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
    if data_args.is_data_in_meds:
        meds_extension_path = get_meds_extension_path(
            data_folder=os.path.expanduser(data_args.cohort_folder),
            dataset_prepared_path=os.path.expanduser(data_args.dataset_prepared_path),
        )
        try:
            LOG.info(f"Trying to load the MEDS extension from disk at {meds_extension_path}...")
            dataset = load_from_disk(meds_extension_path)
            if data_args.streaming:
                dataset = convert_dataset_to_iterable_dataset(dataset, num_shards=training_args.dataloader_num_workers)
        except Exception as e:
            LOG.exception(e)
            dataset = create_dataset_from_meds_reader(
                data_args,
                dataset_mappings=[MedToCehrBertDatasetMapping(data_args=data_args, is_pretraining=False)],
                cache_file_collector=cache_file_collector,
            )
            if not data_args.streaming:
                dataset.save_to_disk(str(meds_extension_path))
                stats = dataset.cleanup_cache_files()
                LOG.info(
                    "Clean up the cached files for the cehrbert dataset transformed from the MEDS: %s",
                    stats,
                )
                # Clean up the files created from the data generator
                cache_file_collector.remove_cache_files()
                dataset = load_from_disk(str(meds_extension_path))

        train_set = dataset["train"]
        validation_set = dataset["validation"]
        test_set = dataset["test"]
    else:
        dataset = load_parquet_as_dataset(os.path.expanduser(data_args.data_folder))
        test_set = None
        if data_args.test_data_folder:
            test_set = load_parquet_as_dataset(data_args.test_data_folder)
        # Split the dataset into train/val
        train_val = dataset.train_test_split(
            test_size=data_args.validation_split_percentage,
            seed=training_args.seed,
        )
        train_set = train_val["train"]
        validation_set = train_val["test"]
        if not test_set:
            test_valid = validation_set.train_test_split(test_size=data_args.test_eval_ratio, seed=training_args.seed)
            validation_set = test_valid["train"]
            test_set = test_valid["test"]

    # Organize them into a single DatasetDict
    return DatasetDict({"train": train_set, "validation": validation_set, "test": test_set})


def main():
    cehrbert_args, data_args, model_args, training_args = parse_runner_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    cehrgpt_tokenizer = CehrBertTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    cehrbert_model = (
        CehrBertForPreTraining.from_pretrained(
            model_args.model_name_or_path,
            attn_implementation=("flash_attention_2" if is_flash_attn_2_available() else "eager"),
        )
        .eval()
        .to(device)
    )
    prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)
    cache_file_collector = CacheFileCollector()
    processed_dataset = None
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")

    if processed_dataset is None:
        # Organize them into a single DatasetDict
        final_splits = prepare_finetune_dataset(data_args, training_args, cache_file_collector)

        # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
        if not data_args.streaming:
            all_columns = final_splits["train"].column_names
            if "visit_concept_ids" in all_columns:
                final_splits = final_splits.remove_columns(["visit_concept_ids"])

        processed_dataset = create_cehrbert_finetuning_dataset(
            dataset=final_splits,
            concept_tokenizer=cehrgpt_tokenizer,
            data_args=data_args,
            cache_file_collector=cache_file_collector,
        )
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)
            processed_dataset.cleanup_cache_files()

        # Remove all the cached files if processed_dataset.cleanup_cache_files() did not remove them already
        cache_file_collector.remove_cache_files()

    # Getting the existing features
    feature_folders = glob.glob(os.path.join(training_args.output_dir, "*", "features", "*.parquet"))
    if feature_folders:
        existing_features = pd.concat(
            [pd.read_parquet(f, columns=["subject_id", "prediction_time_posix"]) for f in feature_folders],
            ignore_index=True,
        )
        subject_prediction_tuples = set(
            existing_features.apply(
                lambda row: f"{int(row['subject_id'])}-{int(row['prediction_time_posix'])}",
                axis=1,
            ).tolist()
        )
        processed_dataset = processed_dataset.filter(
            lambda _batch: [
                f"{int(subject)}-{int(time)}" not in subject_prediction_tuples
                for subject, time in zip(_batch["person_id"], _batch["index_date"])
            ],
            num_proc=data_args.preprocessing_num_workers,
            batch_size=data_args.preprocessing_batch_size,
            batched=True,
        )
        LOG.info(
            "The datasets after filtering (train: %s, validation: %s, test: %s)",
            len(processed_dataset["train"]),
            len(processed_dataset["validation"]),
            len(processed_dataset["test"]),
        )

    train_set = concatenate_datasets([processed_dataset["train"], processed_dataset["validation"]])

    if cehrbert_args.sample_packing:
        per_device_eval_batch_size = 1
        data_collator_fn = partial(
            SamplePackingCehrBertDataCollator,
            cehrbert_args.max_tokens_per_batch,
            cehrbert_model.config.max_position_embeddings,
        )
        train_batch_sampler = SamplePackingBatchSampler(
            lengths=train_set["num_of_concepts"],
            max_tokens_per_batch=cehrbert_args.max_tokens_per_batch,
            max_position_embeddings=cehrbert_model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
        test_batch_sampler = SamplePackingBatchSampler(
            lengths=processed_dataset["test"]["num_of_concepts"],
            max_tokens_per_batch=cehrbert_args.max_tokens_per_batch,
            max_position_embeddings=cehrbert_model.config.max_position_embeddings,
            drop_last=training_args.dataloader_drop_last,
            seed=training_args.seed,
        )
    else:
        data_collator_fn = CehrBertDataCollator
        train_batch_sampler = None
        test_batch_sampler = None
        per_device_eval_batch_size = training_args.per_device_eval_batch_size

    # We suppress the additional learning objectives in fine-tuning
    data_collator = data_collator_fn(
        tokenizer=cehrgpt_tokenizer,
        max_length=(
            cehrbert_args.max_tokens_per_batch
            if cehrbert_args.sample_packing
            else cehrbert_model.config.max_position_embeddings
        ),
        is_pretraining=False,
    )

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=train_batch_sampler,
    )

    test_dataloader = DataLoader(
        dataset=processed_dataset["test"],
        batch_size=per_device_eval_batch_size,
        num_workers=training_args.dataloader_num_workers,
        collate_fn=data_collator,
        pin_memory=training_args.dataloader_pin_memory,
        batch_sampler=test_batch_sampler,
    )

    # Loading demographics
    print("Loading demographics as a dictionary")
    demographics_df = pd.concat(
        [
            pd.read_parquet(
                data_dir,
                columns=[
                    "person_id",
                    "index_date",
                    "gender_concept_id",
                    "race_concept_id",
                ],
            )
            for data_dir in [data_args.data_folder, data_args.test_data_folder]
        ]
    )
    demographics_df["index_date"] = demographics_df.index_date.dt.date
    demographics_dict = {
        (row["person_id"], row["index_date"]): {
            "gender_concept_id": row["gender_concept_id"],
            "race_concept_id": row["race_concept_id"],
        }
        for _, row in demographics_df.iterrows()
    }

    data_loaders = [("train", train_loader), ("test", test_dataloader)]

    for split, data_loader in data_loaders:

        # Ensure prediction folder exists
        feature_output_folder = Path(training_args.output_dir) / split / "features"
        feature_output_folder.mkdir(parents=True, exist_ok=True)

        LOG.info("Generating features for %s set at %s", split, feature_output_folder)

        with torch.no_grad():
            for index, batch in enumerate(tqdm(data_loader, desc="Generating features")):
                prediction_time_ages = batch.pop("age_at_index").numpy().astype(float).squeeze()
                if prediction_time_ages.ndim == 0:
                    prediction_time_ages = np.asarray([prediction_time_ages])

                person_ids = batch.pop("person_id").numpy().astype(int).squeeze()
                if person_ids.ndim == 0:
                    person_ids = np.asarray([person_ids])
                prediction_time_posix = batch.pop("index_date").numpy().squeeze()
                if prediction_time_posix.ndim == 0:
                    prediction_time_posix = np.asarray([prediction_time_posix])
                prediction_time = list(map(datetime.fromtimestamp, prediction_time_posix))
                labels = batch.pop("classifier_label").float().cpu().numpy().astype(bool).squeeze()
                if labels.ndim == 0:
                    labels = np.asarray([labels])

                batch = {k: v.to(device) for k, v in batch.items()}
                # Forward pass
                cehrbert_output = cehrbert_model(**batch, output_attentions=False, output_hidden_states=False)

                cls_token_indices = batch["input_ids"] == cehrgpt_tokenizer.cls_token_index
                if cehrbert_args.sample_packing:
                    if cehrbert_args.average_over_sequence:
                        features = extract_averaged_embeddings_from_packed_sequence(
                            cehrbert_output.last_hidden_state, batch["attention_mask"]
                        )
                    else:
                        features = cehrbert_output.last_hidden_state[cls_token_indices]
                    features = features.cpu().float().detach().numpy()
                else:
                    if cehrbert_args.average_over_sequence:
                        features = torch.where(
                            batch["attention_mask"].unsqueeze(dim=-1).to(torch.bool),
                            cehrbert_output.last_hidden_state,
                            0,
                        )
                        # Average across the sequence
                        features = features.mean(dim=1)
                    else:
                        cls_token_index = torch.argmax((cls_token_indices).to(torch.int), dim=-1)
                        features = (
                            cehrbert_output.last_hidden_state[..., cls_token_index, :].cpu().float().detach().numpy()
                        )
                    features = features.cpu().float().detach().numpy()
                assert len(features) == len(labels), "the number of features must match the number of labels"
                # Flatten features or handle them as a list of arrays (one array per row)
                features_list = [feature for feature in features]
                race_concept_ids = []
                gender_concept_ids = []
                for person_id, index_date in zip(person_ids, prediction_time):
                    key = (person_id, index_date.date())
                    if key in demographics_dict:
                        demographics = demographics_dict[key]
                        gender_concept_ids.append(demographics["gender_concept_id"])
                        race_concept_ids.append(demographics["race_concept_id"])
                    else:
                        gender_concept_ids.append(0)
                        race_concept_ids.append(0)

                features_pd = pd.DataFrame(
                    {
                        "subject_id": person_ids,
                        "prediction_time": prediction_time,
                        "prediction_time_posix": prediction_time_posix,
                        "boolean_value": labels,
                        "age_at_index": prediction_time_ages,
                    }
                )
                # Adding features as a separate column where each row contains a feature array
                features_pd["features"] = features_list
                features_pd["race_concept_id"] = race_concept_ids
                features_pd["gender_concept_id"] = gender_concept_ids
                features_pd.to_parquet(feature_output_folder / f"{uuid.uuid4()}.parquet")


if __name__ == "__main__":
    main()
