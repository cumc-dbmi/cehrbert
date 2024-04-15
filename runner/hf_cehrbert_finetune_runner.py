import os
import sys
import glob
import json

from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from scipy.special import expit as sigmoid

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers.utils import logging
from transformers import Trainer, set_seed
from transformers import HfArgumentParser, TrainingArguments
from transformers import EarlyStoppingCallback

from runner.hf_runner_argument_dataclass import DataTrainingArguments, ModelArguments
from data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from models.hf_models.config import CehrBertConfig
from models.hf_models.hf_cehrbert import CehrBertPreTrainedModel, CehrBertForClassification
from runner.runner_util import get_last_hf_checkpoint, load_parquet_as_dataset, generate_prepared_ds_path

LOG = logging.get_logger("transformers")


def compute_metrics(eval_pred):
    outputs, labels = eval_pred
    logits = outputs[0]

    # Convert logits to probabilities using sigmoid
    probabilities = sigmoid(logits)

    if probabilities.shape[1] == 2:
        positive_probs = probabilities[:, 1]
    else:
        positive_probs = probabilities.squeeze()  # Ensure it's a 1D array

    # Calculate predictions based on probability threshold of 0.5
    predictions = (positive_probs > 0.5).astype(np.int32)

    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(labels, positive_probs)

    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(labels, positive_probs)
    pr_auc = auc(recall, precision)

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def load_pretrained_model_and_tokenizer(data_args, model_args) -> Tuple[CehrBertPreTrainedModel, CehrBertTokenizer]:
    # Try to load the pretrained tokenizer
    try:
        tokenizer_abspath = os.path.abspath(model_args.tokenizer_name_or_path)
        tokenizer = CehrBertTokenizer.from_pretrained(tokenizer_abspath)
    except Exception as e:
        LOG.warning(e)
        data_folder_abspath = os.path.abspath(data_args.data_folder)
        data_files = glob.glob(os.path.join(data_folder_abspath, "*.parquet"))
        dataset = load_dataset('parquet', data_files=data_files, split='train')
        tokenizer = CehrBertTokenizer.train_tokenizer(dataset, ['concept_ids'], {})
        tokenizer.save_pretrained(os.path.abspath(model_args.tokenizer_name_or_path))

    # Try to load the pretrained model
    try:
        model_abspath = os.path.abspath(model_args.model_name_or_path)
        model = CehrBertForClassification.from_pretrained(model_abspath)
    except Exception as e:
        LOG.warning(e)
        model_config = CehrBertConfig(vocab_size=tokenizer.vocab_size, **model_args.as_dict())
        model = CehrBertForClassification(model_config)

    return model, tokenizer


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, tokenizer = load_pretrained_model_and_tokenizer(data_args, model_args)

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)

    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        LOG.info("Prepared dataset loaded from disk...")
    else:
        dataset = load_parquet_as_dataset(data_args.data_folder)
        test_set = None
        if data_args.test_data_folder:
            test_set = load_parquet_as_dataset(data_args.test_data_folder)

        # Split the dataset into train/val
        train_val = dataset.train_test_split(test_size=data_args.validation_split_percentage, seed=training_args.seed)

        # If the test set is not provided, split the validation further into val/test sets.
        if not test_set:
            test_valid = train_val['test'].train_test_split(
                test_size=data_args.test_eval_ratio, seed=training_args.seed
            )
            # Organize them into a single DatasetDict
            final_splits = DatasetDict({
                'train': train_val['train'],
                'validation': test_valid['train'],
                'test': test_valid['test']
            })
        else:
            final_splits = DatasetDict({
                'train': train_val['train'],
                'validation': train_val['test'],
                'test': test_set
            })

        processed_dataset = create_cehrbert_finetuning_dataset(
            dataset=final_splits,
            concept_tokenizer=tokenizer,
            max_sequence_length=model_args.max_position_embeddings,
            num_proc=data_args.preprocessing_num_workers
        )
        processed_dataset.save_to_disk(prepared_ds_path)

    collator = CehrBertDataCollator(tokenizer, model_args.max_position_embeddings)

    # Detecting last checkpoint.
    get_last_hf_checkpoint(training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    processed_dataset.set_format('pt')

    trainer = Trainer(
        model=model,
        data_collator=collator,
        train_dataset=processed_dataset['train'],
        eval_dataset=processed_dataset['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback()],
        args=training_args
    )

    checkpoint = get_last_hf_checkpoint(training_args)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    test_results = trainer.evaluate(processed_dataset['test'])
    # Save results to JSON
    test_results_path = os.path.join(training_args.output_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=4)

    LOG.info(f'Test results: {test_results}')


if __name__ == "__main__":
    main()
