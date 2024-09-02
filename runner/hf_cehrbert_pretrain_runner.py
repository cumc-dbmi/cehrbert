import os

from typing import Union, Optional

from datasets import DatasetDict, IterableDatasetDict, Dataset, load_dataset
from transformers.utils import logging
from transformers import AutoConfig, Trainer, set_seed

from data_generators.hf_data_generator.meds_utils import create_dataset_from_meds_reader
from data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from data_generators.hf_data_generator.hf_dataset import (
    create_cehrbert_pretraining_dataset
)
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from models.hf_models.config import CehrBertConfig
from models.hf_models.hf_cehrbert import CehrBertForPreTraining
from runner.runner_util import generate_prepared_ds_path, load_parquet_as_dataset, get_last_hf_checkpoint, \
    parse_runner_args, get_meds_extension_path
from runner.hf_runner_argument_dataclass import DataTrainingArguments, ModelArguments

LOG = logging.get_logger("transformers")


def load_and_create_tokenizer(
        data_args: DataTrainingArguments,
        model_args: ModelArguments,
        dataset: Optional[Union[Dataset, DatasetDict]] = None
) -> CehrBertTokenizer:
    # Try to load the pretrained tokenizer
    tokenizer_abspath = os.path.abspath(model_args.tokenizer_name_or_path)
    try:
        tokenizer = CehrBertTokenizer.from_pretrained(tokenizer_abspath)
    except Exception as e:
        LOG.warning(e)
        if dataset is None:
            raise RuntimeError(
                f"Failed to load the tokenizer from {tokenizer_abspath} with the error \n{e}\n"
                f"Tried to create the tokenizer, however the dataset is not provided."
            )

        tokenizer = CehrBertTokenizer.train_tokenizer(
            dataset, ['concept_ids'], {}, data_args
        )
        tokenizer.save_pretrained(tokenizer_abspath)

    return tokenizer


def load_and_create_model(
        model_args: ModelArguments,
        tokenizer: CehrBertTokenizer
) -> CehrBertForPreTraining:
    try:
        model_abspath = os.path.abspath(model_args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(model_abspath)
    except Exception as e:
        LOG.warning(e)
        model_config = CehrBertConfig(
            vocab_size=tokenizer.vocab_size,
            lab_token_ids=tokenizer.lab_token_ids,
            **model_args.as_dict()
        )

    return CehrBertForPreTraining(model_config)


def main():
    data_args, model_args, training_args = parse_runner_args()

    if data_args.streaming:
        # This is for disabling the warning message https://github.com/huggingface/transformers/issues/5486
        # This happens only when streaming is enabled
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to be set to 0
        # Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)

    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_dataset(str(prepared_ds_path), streaming=data_args.streaming)
        LOG.info("Prepared dataset loaded from disk...")
        # If the data has been processed in the past, it's assume the tokenizer has been created before.
        # we load the CEHR-BERT tokenizer from the output folder.
        tokenizer = load_and_create_tokenizer(
            data_args=data_args,
            model_args=model_args,
            dataset=processed_dataset
        )
    else:
        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_med:
            meds_extension_path = get_meds_extension_path(
                data_folder=data_args.data_folder,
                dataset_prepared_path=data_args.dataset_prepared_path
            )
            try:
                LOG.info(f"Trying to load the MEDS extension from disk at {meds_extension_path}...")
                dataset = load_dataset(meds_extension_path, streaming=data_args.streaming)
            except Exception as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(data_args, is_pretraining=True)
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
        else:
            # Load the dataset from the parquet files
            dataset = load_parquet_as_dataset(data_args.data_folder, split='train', streaming=data_args.streaming)
            # If streaming is enabled, we need to manually split the data into train/val
            if data_args.streaming and data_args.validation_split_num:
                dataset = dataset.shuffle(buffer_size=10_000, seed=training_args.seed)
                train_set = dataset.skip(data_args.validation_split_num)
                val_set = dataset.take(data_args.validation_split_num)
                dataset = DatasetDict({
                    'train': train_set,
                    'validation': val_set
                })
            elif data_args.validation_split_percentage:
                dataset = dataset.train_test_split(test_size=data_args.validation_split_percentage,
                                                   seed=training_args.seed)
            else:
                raise RuntimeError(
                    f"Can not split the data. If streaming is enabled, validation_split_num needs to be "
                    f"defined, otherwise validation_split_percentage needs to be provided. "
                    f"The current values are:\n"
                    f"validation_split_percentage: {data_args.validation_split_percentage}\n"
                    f"validation_split_num: {data_args.validation_split_num}\n"
                    f"streaming: {data_args.streaming}"
                )

        # Create the CEHR-BERT tokenizer if it's not available in the output folder
        tokenizer = load_and_create_tokenizer(
            data_args=data_args,
            model_args=model_args,
            dataset=dataset
        )
        # sort the patient features chronologically and tokenize the data
        processed_dataset = create_cehrbert_pretraining_dataset(
            dataset=dataset,
            concept_tokenizer=tokenizer,
            data_args=data_args
        )
        # only save the data to the disk if it is not streaming
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    model = load_and_create_model(model_args, tokenizer)

    collator = CehrBertDataCollator(
        tokenizer=tokenizer,
        max_length=model_args.max_position_embeddings,
        is_pretraining=True,
        mlm_probability=model.config.mlm_probability
    )

    # Detecting last checkpoint.
    last_checkpoint = get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
        processed_dataset.set_format('pt')

    eval_dataset = None
    if isinstance(processed_dataset, DatasetDict) or isinstance(processed_dataset, IterableDatasetDict):
        train_dataset = processed_dataset['train']
        if 'validation' in processed_dataset:
            eval_dataset = processed_dataset['validation']
    else:
        train_dataset = processed_dataset

    trainer = Trainer(
        model=model,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # compute_metrics=compute_metrics,
        args=training_args
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
