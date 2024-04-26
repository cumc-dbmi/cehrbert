import os
import glob
import json

from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from scipy.special import expit as sigmoid

from datasets import load_dataset, load_from_disk, DatasetDict
from transformers.utils import logging
from transformers import Trainer, set_seed
from transformers import EarlyStoppingCallback
from peft import LoraConfig, get_peft_model

from data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from models.hf_models.config import CehrBertConfig
from models.hf_models.hf_cehrbert import (
    CehrBertPreTrainedModel, CehrBertForClassification, CehrBertLstmForClassification
)
from runner.hf_runner_argument_dataclass import FineTuneModelType
from runner.runner_util import (
    get_last_hf_checkpoint, load_parquet_as_dataset, generate_prepared_ds_path, parse_runner_args
)

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
        tokenizer = CehrBertTokenizer.train_tokenizer(
            dataset,
            ['concept_ids'],
            {},
            num_proc=data_args.preprocessing_num_workers,
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.min_frequency
        )
        tokenizer.save_pretrained(os.path.abspath(model_args.tokenizer_name_or_path))

    if model_args.finetune_model_type == FineTuneModelType.POOLING.value:
        finetune_model_cls = CehrBertForClassification
    elif model_args.finetune_model_type == FineTuneModelType.LSTM.value:
        finetune_model_cls = CehrBertLstmForClassification
    else:
        raise ValueError(
            f'finetune_model_type can be one of the following types {[e.value for e in FineTuneModelType]}'
        )

    # Try to load the pretrained model
    try:
        model_abspath = os.path.abspath(model_args.model_name_or_path)
        model = finetune_model_cls.from_pretrained(model_abspath)
    except Exception as e:
        LOG.warning(e)
        model_config = CehrBertConfig(vocab_size=tokenizer.vocab_size, **model_args.as_dict())
        model = finetune_model_cls(model_config)

    return model, tokenizer


def main():
    data_args, model_args, training_args = parse_runner_args()

    model, tokenizer = load_pretrained_model_and_tokenizer(data_args, model_args)

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args)

    # If lora is enabled, we add LORA adapters to the model
    if model_args.use_lora:
        # When LORA is used, the trainer could not automatically find this label,
        # therefore we need to manually set label_names to "classifier_label" so the model
        # can compute the loss during the evaluation
        if training_args.label_names:
            training_args.label_names.append("classifier_label")
        else:
            training_args.label_names = ["classifier_label"]

        if model_args.finetune_model_type == FineTuneModelType.POOLING.value:
            config = LoraConfig(
                r=model_args.lora_rank,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.target_modules,
                lora_dropout=model_args.lora_dropout,
                bias="none",
                modules_to_save=["classifier", "age_batch_norm", "dense_layer"]
            )
            model = get_peft_model(model, config)
        else:
            raise ValueError(f'The LORA adapter is not supported for {model_args.finetune_model_type}')

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
            data_args=data_args
        )
        processed_dataset.save_to_disk(prepared_ds_path)

    collator = CehrBertDataCollator(tokenizer, model_args.max_position_embeddings, is_pretraining=False)

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
