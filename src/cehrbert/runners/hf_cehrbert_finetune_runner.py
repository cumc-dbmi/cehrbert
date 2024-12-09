import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import expit as sigmoid
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.data_generators.hf_data_generator.meds_utils import create_dataset_from_meds_reader
from cehrbert.models.hf_models.hf_cehrbert import (
    CehrBertForClassification,
    CehrBertLstmForClassification,
    CehrBertPreTrainedModel,
)
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import FineTuneModelType, ModelArguments
from cehrbert.runners.runner_util import (
    convert_dataset_to_iterable_dataset,
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
    parse_runner_args,
)

LOG = logging.get_logger("transformers")


def compute_metrics(references: Union[List[float], pd.Series], probs: Union[List[float], pd.Series]) -> Dict[str, Any]:
    """
    Computes evaluation metrics for binary classification, including ROC-AUC and PR-AUC, based on reference labels and model logits.

    Args:
        references (List[float]): Ground truth binary labels (0 or 1).
        logits (List[float]): Logits output from the model (raw prediction scores), which will be converted to probabilities using the sigmoid function.

    Returns:
        Dict[str, Any]: A dictionary containing:
            - 'roc_auc': The Area Under the Receiver Operating Characteristic Curve (ROC-AUC).
            - 'pr_auc': The Area Under the Precision-Recall Curve (PR-AUC).

    Notes:
        - The `sigmoid` function is used to convert the logits into probabilities.
        - ROC-AUC measures the model's ability to distinguish between classes, while PR-AUC focuses on performance when dealing with imbalanced data.
    """
    # # Calculate PR-AUC
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(references, probs)
    precision, recall, _ = precision_recall_curve(references, probs)
    pr_auc = auc(recall, precision)
    return {"roc_auc": roc_auc, "pr_auc": pr_auc}


def load_pretrained_tokenizer(
    model_args,
) -> CehrBertTokenizer:
    tokenizer_name_or_path = os.path.expanduser(model_args.tokenizer_name_or_path)
    try:
        return CehrBertTokenizer.from_pretrained(tokenizer_name_or_path)
    except Exception:
        raise ValueError(f"Can not load the pretrained tokenizer from {tokenizer_name_or_path}")


def load_finetuned_model(model_args: ModelArguments, model_name_or_path: str) -> CehrBertPreTrainedModel:
    if model_args.finetune_model_type == FineTuneModelType.POOLING.value:
        finetune_model_cls = CehrBertForClassification
    elif model_args.finetune_model_type == FineTuneModelType.LSTM.value:
        finetune_model_cls = CehrBertLstmForClassification
    else:
        raise ValueError(
            f"finetune_model_type can be one of the following types {[e.value for e in FineTuneModelType]}"
        )
    # Try to create a new model based on the base model
    model_name_or_path = os.path.expanduser(model_name_or_path)
    try:
        return finetune_model_cls.from_pretrained(model_name_or_path)
    except ValueError:
        raise ValueError(f"Can not load the finetuned model from {model_name_or_path}")


def main():
    data_args, model_args, training_args = parse_runner_args()

    tokenizer = load_pretrained_tokenizer(model_args)
    prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)

    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        if data_args.streaming:
            processed_dataset = convert_dataset_to_iterable_dataset(
                processed_dataset, num_shards=training_args.dataloader_num_workers
            )
        LOG.info("Prepared dataset loaded from disk...")
    else:
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
                    dataset = convert_dataset_to_iterable_dataset(
                        dataset, num_shards=training_args.dataloader_num_workers
                    )
            except Exception as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(data_args, is_pretraining=False)
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
            train_set = dataset["train"]
            validation_set = dataset["validation"]
            test_set = dataset["test"]
        else:
            dataset = load_parquet_as_dataset(os.path.expanduser(data_args.data_folder))
            test_set = None
            if data_args.test_data_folder:
                test_set = load_parquet_as_dataset(data_args.test_data_folder)

            if data_args.chronological_split:
                dataset = dataset.sort("index_date")
                # Determine the split index
                total_size = len(dataset)
                train_end = int((1 - data_args.validation_split_percentage) * total_size)
                # Perform the chronological split, use the historical data for training and future data
                # for validation/testing
                train_set = dataset.select(range(0, train_end))
                validation_set = dataset.select(range(train_end, total_size))
                if not test_set:
                    test_valid = validation_set.train_test_split(
                        test_size=data_args.test_eval_ratio, seed=training_args.seed
                    )
                    validation_set = test_valid["train"]
                    test_set = test_valid["test"]

            elif data_args.split_by_patient:
                LOG.info(f"Using the split_by_patient strategy")
                unique_patient_ids = np.unique(dataset["person_id"])
                LOG.info(f"There are {len(unique_patient_ids)} num of patients in total")
                np.random.seed(training_args.seed)
                np.random.shuffle(unique_patient_ids)

                train_end = int(len(unique_patient_ids) * (1 - data_args.validation_split_percentage))
                train_patient_ids = set(unique_patient_ids[:train_end])
                if not test_set:
                    # Calculate split indices
                    validation_end = (
                        int(len(unique_patient_ids) * data_args.validation_split_percentage * data_args.test_eval_ratio)
                        + train_end
                    )

                    # Split patient IDs
                    val_patient_ids = set(unique_patient_ids[train_end:validation_end])
                    test_patient_ids = set(unique_patient_ids[validation_end:])

                    def assign_split(example):
                        pid = example["person_id"]
                        if pid in train_patient_ids:
                            return "train"
                        elif pid in val_patient_ids:
                            return "validation"
                        elif pid in test_patient_ids:
                            return "test"
                        else:
                            raise ValueError(f"Unknown patient {pid}")

                    # Apply the function to assign splits
                    dataset = dataset.map(
                        lambda example: {"split": assign_split(example)},
                        num_proc=data_args.preprocessing_num_workers,
                    )
                    train_set = dataset.filter(
                        lambda example: example["split"] == "train",
                        num_proc=data_args.preprocessing_num_workers,
                    )
                    validation_set = dataset.filter(
                        lambda example: example["split"] == "validation",
                        num_proc=data_args.preprocessing_num_workers,
                    )
                    test_set = dataset.filter(
                        lambda example: example["split"] == "test",
                        num_proc=data_args.preprocessing_num_workers,
                    )
                else:
                    # Split patient IDs
                    val_patient_ids = set(unique_patient_ids[train_end:])

                    def assign_split(example):
                        pid = example["person_id"]
                        if pid in train_patient_ids:
                            return "train"
                        elif pid in val_patient_ids:
                            return "validation"
                        else:
                            raise ValueError(f"Unknown patient {pid}")

                    # Apply the function to assign splits
                    dataset = dataset.map(
                        lambda example: {"split": assign_split(example)},
                        num_proc=data_args.preprocessing_num_workers,
                    )
                    train_set = dataset.filter(
                        lambda example: example["split"] == "train",
                        num_proc=data_args.preprocessing_num_workers,
                    )
                    validation_set = dataset.filter(
                        lambda example: example["split"] == "validation",
                        num_proc=data_args.preprocessing_num_workers,
                    )
            else:
                # Split the dataset into train/val
                train_val = dataset.train_test_split(
                    test_size=data_args.validation_split_percentage,
                    seed=training_args.seed,
                )
                train_set = train_val["train"]
                validation_set = train_val["test"]
                if not test_set:
                    test_valid = validation_set.train_test_split(
                        test_size=data_args.test_eval_ratio, seed=training_args.seed
                    )
                    validation_set = test_valid["train"]
                    test_set = test_valid["test"]

        # Organize them into a single DatasetDict
        final_splits = DatasetDict({"train": train_set, "validation": validation_set, "test": test_set})

        processed_dataset = create_cehrbert_finetuning_dataset(
            dataset=final_splits, concept_tokenizer=tokenizer, data_args=data_args
        )

        if not data_args.streaming:
            processed_dataset.save_to_disk(prepared_ds_path)

    collator = CehrBertDataCollator(tokenizer, model_args.max_position_embeddings, is_pretraining=False)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming:
        processed_dataset.set_format("pt")

    if training_args.do_train:
        model = load_finetuned_model(model_args, model_args.model_name_or_path)
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
                    modules_to_save=["classifier", "age_batch_norm", "dense_layer"],
                )
                model = get_peft_model(model, config)
            else:
                raise ValueError(f"The LORA adapter is not supported for {model_args.finetune_model_type}")

        trainer = Trainer(
            model=model,
            data_collator=collator,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)],
            args=training_args,
        )

        checkpoint = get_last_hf_checkpoint(training_args)

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        test_dataloader = DataLoader(
            dataset=processed_dataset["test"],
            batch_size=training_args.per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=collator,
            pin_memory=training_args.dataloader_pin_memory,
        )
        do_predict(test_dataloader, model_args, training_args)


def do_predict(test_dataloader: DataLoader, model_args: ModelArguments, training_args: TrainingArguments):
    """
    Performs inference on the test dataset using a fine-tuned model, saves predictions and evaluation metrics.

    The reason we created this custom do_predict is that there is a memory leakage for transformers trainer.predict(),
    for large test sets, it will throw the CPU OOM error

    Args:
        test_dataloader (DataLoader): DataLoader containing the test dataset, with batches of input features and labels.
        model_args (ModelArguments): Arguments for configuring and loading the fine-tuned model.
        training_args (TrainingArguments): Arguments related to training, evaluation, and output directories.

    Returns:
        None. Results are saved to disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and LoRA adapters if applicable
    model = (
        load_finetuned_model(model_args, training_args.output_dir)
        if not model_args.use_lora
        else load_lora_model(model_args, training_args)
    )

    model = model.to(device).eval()

    # Ensure prediction folder exists
    test_prediction_folder = Path(training_args.output_dir) / "test_predictions"
    test_prediction_folder.mkdir(parents=True, exist_ok=True)

    LOG.info("Generating predictions for test set at %s", test_prediction_folder)

    test_losses = []
    with torch.no_grad():
        for index, batch in enumerate(tqdm(test_dataloader, desc="Predicting")):
            person_ids = batch.pop("person_id").numpy().squeeze().astype(int)
            index_dates = (
                map(datetime.fromtimestamp, batch.pop("index_date").numpy().squeeze().tolist())
                if "index_date" in batch
                else None
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            # Forward pass
            output = model(**batch, output_attentions=False, output_hidden_states=False)
            test_losses.append(output.loss.item())

            # Collect logits and labels for prediction
            logits = output.logits.cpu().numpy().squeeze()
            labels = batch["classifier_label"].cpu().numpy().squeeze().astype(bool)
            probabilities = sigmoid(logits)
            # Save predictions to parquet file
            test_prediction_pd = pd.DataFrame(
                {
                    "subject_id": person_ids,
                    "prediction_time": index_dates,
                    "predicted_boolean_probability": probabilities,
                    "predicted_boolean_value": None,
                    "boolean_value": labels,
                }
            )
            test_prediction_pd.to_parquet(test_prediction_folder / f"{index}.parquet")

    LOG.info("Computing metrics using the test set predictions at %s", test_prediction_folder)
    # Load all predictions
    test_prediction_pd = pd.read_parquet(test_prediction_folder)
    # Compute metrics and save results
    metrics = compute_metrics(
        references=test_prediction_pd.boolean_value, probs=test_prediction_pd.boolean_prediction_probability
    )
    metrics["test_loss"] = np.mean(test_losses)

    test_results_path = Path(training_args.output_dir) / "test_results.json"
    with open(test_results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    LOG.info("Test results: %s", metrics)


def load_lora_model(model_args, training_args) -> PeftModel:
    LOG.info("Loading base model from %s", model_args.model_name_or_path)
    base_model = load_finetuned_model(model_args, model_args.model_name_or_path)
    LOG.info("Loading LoRA adapter from %s", training_args.output_dir)
    return PeftModel.from_pretrained(base_model, model_id=training_args.output_dir)


if __name__ == "__main__":
    main()
