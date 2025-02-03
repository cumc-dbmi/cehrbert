import json
import os
from typing import Optional, Union

from datasets import Dataset, DatasetDict, IterableDatasetDict, load_from_disk
from transformers import AutoConfig, Trainer, set_seed
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_pretraining_dataset
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.data_generators.hf_data_generator.meds_utils import create_dataset_from_meds_reader
from cehrbert.models.hf_models.config import CehrBertConfig
from cehrbert.models.hf_models.hf_cehrbert import CehrBertForPreTraining
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments, ModelArguments
from cehrbert.runners.runner_util import (
    convert_dataset_to_iterable_dataset,
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
    parse_runner_args,
)

LOG = logging.get_logger("transformers")


def load_and_create_tokenizer(
    data_args: DataTrainingArguments,
    model_args: ModelArguments,
    dataset: Optional[Union[Dataset, DatasetDict]] = None,
) -> CehrBertTokenizer:
    """
    Loads a pretrained tokenizer or creates a new one if it cannot be loaded.

    Args:
        data_args (DataTrainingArguments): Data-related arguments used for training the tokenizer.
        model_args (ModelArguments): Model-related arguments including the tokenizer's path or name.
        dataset (Optional[Union[Dataset, DatasetDict]]): A dataset used to train the tokenizer if it cannot be loaded.

    Returns:
        CehrBertTokenizer: The loaded or newly created and trained tokenizer.

    Raises:
        RuntimeError: If the tokenizer cannot be loaded and no dataset is provided to create a new tokenizer.

    Behavior:
        - Attempts to load the tokenizer from the specified path in `model_args.tokenizer_name_or_path`.
        - If loading fails and no dataset is provided, it raises the original exception.
        - If a dataset is provided, it trains a new tokenizer on the dataset using the `concept_ids` feature.
        - Saves the newly created tokenizer at the specified path.

    Example:
        tokenizer = load_and_create_tokenizer(data_args, model_args, dataset)
    """
    # Try to load the pretrained tokenizer
    tokenizer_name_or_path = os.path.expanduser(model_args.tokenizer_name_or_path)
    try:
        tokenizer = CehrBertTokenizer.from_pretrained(tokenizer_name_or_path)
    except (OSError, RuntimeError, FileNotFoundError, json.JSONDecodeError) as e:
        LOG.warning(
            "Failed to load the tokenizer from %s with the error "
            "\n%s\nTried to create the tokenizer, however the dataset is not provided.",
            tokenizer_name_or_path,
            e,
        )
        if dataset is None:
            raise e
        tokenizer = CehrBertTokenizer.train_tokenizer(
            dataset, feature_names=["concept_ids"], concept_name_mapping={}, data_args=data_args
        )
        tokenizer.save_pretrained(tokenizer_name_or_path)

    return tokenizer


def load_and_create_model(model_args: ModelArguments, tokenizer: CehrBertTokenizer) -> CehrBertForPreTraining:
    """
    Loads a pretrained model or creates a new model configuration if the pretrained model cannot be loaded.

    Args:
        model_args (ModelArguments): Model-related arguments including the model's path or configuration details.
        tokenizer (CehrBertTokenizer): The tokenizer to be used with the model, providing vocab and token information.

    Returns:
        CehrBertForPreTraining: The loaded or newly configured model for pretraining.

    Behavior:
        - Attempts to load the model's configuration from the specified path in `model_args.model_name_or_path`.
        - If loading fails, it logs the error and creates a new model configuration using the tokenizer's vocab size
          and lab token IDs.
        - Returns a `CehrBertForPreTraining` model initialized with the loaded or newly created configuration.

    Example:
        model = load_and_create_model(model_args, tokenizer)
    """
    try:
        model_config = AutoConfig.from_pretrained(os.path.expanduser(model_args.model_name_or_path))
    except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
        LOG.warning(e)
        model_config = CehrBertConfig(
            vocab_size=tokenizer.vocab_size,
            lab_token_ids=tokenizer.lab_token_ids,
            **model_args.as_dict(),
        )
    return CehrBertForPreTraining(model_config)


def main():
    """
    Main function for preparing, loading, and training a CEHR-BERT model for pretraining.

    This function handles:
    - Parsing input arguments for data, model, and training configurations.
    - Loading or creating a dataset, either from a previously saved state or raw data (e.g., MEDS).
    - Creating or loading a CEHR-BERT tokenizer, depending on whether a tokenizer exists.
    - Creating and configuring the CEHR-BERT model for pretraining.
    - Setting up a data collator and trainer for pretraining using Hugging Face's `Trainer` class.
    - Handling dataset splitting for training and validation.
    - Optionally resuming training from the last checkpoint.

    Key Steps:
    1. Check for streaming data support and adjust settings accordingly.
    2. Load the dataset from disk if available, or create it from raw data.
    3. Tokenize the dataset using the CEHR-BERT tokenizer.
    4. Train the model, resume from a checkpoint if specified, and save the final model and metrics.

    Raises:
        RuntimeError: Raised if required arguments (e.g., validation split details) are missing.

    Example Usage:
        Run this function in a script with appropriate arguments:
        ```
        python hf_cehrbert_pretrain_runner.py --data_args <data_args> --model_args <model_args> \
        --training_args <training_args>
        ```

    Dependencies:
        - Hugging Face Transformers (Trainer, Dataset, DatasetDict, etc.)
        - CEHR-BERT modules such as `CehrBertTokenizer`, `CehrBertForPreTraining`,
        and `CehrBertDataCollator`.

    Notes:
    - Assumes the data is in the CEHR-BERT format or needs conversion from the MEDS format.
    - Supports both disk-based and streaming datasets, depending on the argument configuration.
    - The tokenizer and model are saved to disk after the training process completes.
    """
    data_args, model_args, training_args = parse_runner_args()

    if data_args.streaming:
        # This happens only when streaming is enabled. This is for disabling the warning message
        # https://github.com/huggingface/transformers/issues/5486
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to
        # be set to 0. Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)

    if any(prepared_ds_path.glob("*")):
        LOG.info("Loading prepared dataset from disk at %s...", prepared_ds_path)
        processed_dataset = load_from_disk(str(prepared_ds_path))
        if data_args.streaming:
            processed_dataset = convert_dataset_to_iterable_dataset(
                processed_dataset, num_shards=training_args.dataloader_num_workers
            )
        LOG.info("Prepared dataset loaded from disk...")
        # If the data has been processed in the past, it's assume the tokenizer has been created
        # before. We load the CEHR-BERT tokenizer from the output folder.
        tokenizer = load_and_create_tokenizer(data_args=data_args, model_args=model_args, dataset=processed_dataset)
    else:
        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_meds:
            meds_extension_path = get_meds_extension_path(
                data_folder=os.path.expanduser(data_args.data_folder),
                dataset_prepared_path=os.path.expanduser(data_args.dataset_prepared_path),
            )
            try:
                LOG.info(
                    "Trying to load the MEDS extension from disk at %s...",
                    meds_extension_path,
                )
                dataset = load_from_disk(meds_extension_path)
                if data_args.streaming:
                    dataset = convert_dataset_to_iterable_dataset(
                        dataset, num_shards=training_args.dataloader_num_workers
                    )
            except FileNotFoundError as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(data_args, is_pretraining=True)
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
        else:
            # Load the dataset from the parquet files
            dataset = load_parquet_as_dataset(
                os.path.expanduser(data_args.data_folder), split="train", streaming=data_args.streaming
            )
            # If streaming is enabled, we need to manually split the data into train/val
            if data_args.streaming and data_args.validation_split_num:
                dataset = dataset.shuffle(buffer_size=10_000, seed=training_args.seed)
                train_set = dataset.skip(data_args.validation_split_num)
                val_set = dataset.take(data_args.validation_split_num)
                dataset = DatasetDict({"train": train_set, "validation": val_set})
            elif data_args.validation_split_percentage:
                dataset = dataset.train_test_split(
                    test_size=data_args.validation_split_percentage,
                    seed=training_args.seed,
                )
                dataset = DatasetDict({"train": dataset["train"], "validation": dataset["test"]})
            else:
                raise RuntimeError(
                    f"Can not split the data. If streaming is enabled, validation_split_num needs  "
                    f"to be defined, otherwise validation_split_percentage needs to be provided. "
                    f"The current values are:\n"
                    f"validation_split_percentage: {data_args.validation_split_percentage}\n"
                    f"validation_split_num: {data_args.validation_split_num}\n"
                    f"streaming: {data_args.streaming}"
                )

        # Create the CEHR-BERT tokenizer if it's not available in the output folder
        tokenizer = load_and_create_tokenizer(data_args=data_args, model_args=model_args, dataset=dataset)
        # sort the patient features chronologically and tokenize the data
        processed_dataset = create_cehrbert_pretraining_dataset(
            dataset=dataset, concept_tokenizer=tokenizer, data_args=data_args
        )
        # only save the data to the disk if it is not streaming
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    def filter_func(examples):
        return [_ >= data_args.min_num_tokens for _ in examples["num_of_concepts"]]

    # Create the args for batched filtering
    filter_args = {"batched": True, "batch_size": data_args.preprocessing_batch_size}
    # If the dataset is not in a streaming mode, we could add num_proc to enable parallelization
    if not data_args.streaming:
        filter_args["num_proc"] = data_args.preprocessing_num_workers

    # The filter can't be applied to a DatasetDict of IterableDataset (in case of streaming)
    # we need to iterate through all the datasets and apply the filter separately
    if isinstance(processed_dataset, DatasetDict) or isinstance(processed_dataset, IterableDatasetDict):
        for key in processed_dataset.keys():
            processed_dataset[key] = processed_dataset[key].filter(filter_func, **filter_args)
    else:
        processed_dataset = processed_dataset.filter(filter_func, **filter_args)

    model = load_and_create_model(model_args, tokenizer)

    collator = CehrBertDataCollator(
        tokenizer=tokenizer,
        max_length=model_args.max_position_embeddings,
        is_pretraining=True,
        mlm_probability=model.config.mlm_probability,
    )

    # Detecting last checkpoint.
    last_checkpoint = get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
        processed_dataset.set_format("pt")

    trainer = Trainer(
        model=model,
        data_collator=collator,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        args=training_args,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
