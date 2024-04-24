import os
import glob

from typing import Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers.utils import logging
from transformers import AutoConfig, Trainer, set_seed
from transformers import EvalPrediction

from data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from data_generators.hf_data_generator.hf_dataset import create_cehrbert_pretraining_dataset
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from models.hf_models.config import CehrBertConfig
from models.hf_models.hf_cehrbert import CehrBertForPreTraining
from runner.runner_util import generate_prepared_ds_path, load_parquet_as_dataset, get_last_hf_checkpoint, \
    parse_runner_args

LOG = logging.get_logger("transformers")


def compute_metrics(eval_pred: EvalPrediction):
    outputs, labels = eval_pred
    # Transformers Trainer will remove the loss from the model output
    # We need to take the first entry of the model output, which is logits
    logits = outputs[0]
    # Exclude entries where labels == -100
    mask = labels != -100
    valid_logits = logits[mask]
    valid_labels = labels[mask]

    # Convert logits to probabilities using the numerically stable softmax
    probabilities = F.softmax(valid_logits, dim=1)

    # Prepare labels for valid (non-masked) entries
    # Note: PyTorch can calculate cross-entropy directly from logits,
    # so converting logits to probabilities is unnecessary for loss calculation.
    # However, we will calculate manually to follow the specified steps.

    # Convert labels to one-hot encoding
    labels_one_hot = F.one_hot(valid_labels, num_classes=probabilities.shape[1]).float()

    # Compute log probabilities (log softmax is more numerically stable than log(softmax))
    log_probs = F.log_softmax(valid_logits, dim=1)

    # Compute cross-entropy loss for valid entries
    cross_entropy_loss = -torch.sum(labels_one_hot * log_probs, dim=1)

    # Calculate perplexity
    perplexity = torch.exp(torch.mean(cross_entropy_loss))

    return {"perplexity": perplexity.item()}  # Use .item() to extract the scalar value from the tensor


def load_model_and_tokenizer(data_args, model_args) -> Tuple[CehrBertForPreTraining, CehrBertTokenizer]:
    # Try to load the pretrained tokenizer
    try:
        tokenizer_abspath = os.path.abspath(model_args.tokenizer_name_or_path)
        tokenizer = CehrBertTokenizer.from_pretrained(tokenizer_abspath)
    except Exception as e:
        LOG.warning(e)
        dataset = load_parquet_as_dataset(data_args.data_folder)
        tokenizer = CehrBertTokenizer.train_tokenizer(
            dataset, ['concept_ids'], {},
            num_proc=data_args.preprocessing_num_workers
        )
        tokenizer.save_pretrained(os.path.abspath(model_args.tokenizer_name_or_path))

    # Try to load the pretrained model
    try:
        model_abspath = os.path.abspath(model_args.model_name_or_path)
        model_config = AutoConfig.from_pretrained(model_abspath)
    except Exception as e:
        LOG.warning(e)
        model_config = CehrBertConfig(vocab_size=tokenizer.vocab_size, **model_args.as_dict())

    return CehrBertForPreTraining(model_config), tokenizer


def main():
    data_args, model_args, training_args = parse_runner_args()

    model, tokenizer = load_model_and_tokenizer(data_args, model_args)

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)

    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
    else:
        data_abspath = os.path.abspath(data_args.data_folder)
        data_files = glob.glob(os.path.join(data_abspath, "*.parquet"))
        dataset = load_dataset('parquet', data_files=data_files, split='train', streaming=data_args.streaming)

        if data_args.streaming and data_args.validation_split_num:
            dataset = dataset.shuffle(buffer_size=10_000)
            train_set = dataset.skip(data_args.validation_split_num)
            val_set = dataset.take(data_args.validation_split_num)
            dataset = DatasetDict({
                'train': train_set,
                'test': val_set
            })
        elif data_args.validation_split_percentage:
            dataset = dataset.train_test_split(test_size=data_args.validation_split_percentage, seed=training_args.seed)

        processed_dataset = create_cehrbert_pretraining_dataset(
            dataset=dataset,
            concept_tokenizer=tokenizer,
            data_args=data_args
        )
        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    collator = CehrBertDataCollator(
        tokenizer=tokenizer,
        max_length=model_args.max_position_embeddings,
        is_pretraining=True
    )

    # Detecting last checkpoint.
    last_checkpoint = get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
        processed_dataset.set_format('pt')

    eval_dataset = None
    if isinstance(processed_dataset, DatasetDict):
        train_dataset = processed_dataset['train']
        if 'test' in processed_dataset:
            eval_dataset = processed_dataset['test']
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
