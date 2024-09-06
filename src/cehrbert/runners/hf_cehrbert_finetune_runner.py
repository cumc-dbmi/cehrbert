import json
import os
from typing import Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, get_peft_model
from scipy.special import expit as sigmoid
from sklearn.metrics import accuracy_score, auc, precision_recall_curve, roc_auc_score
from transformers import EarlyStoppingCallback, Trainer, set_seed
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.data_generators.hf_data_generator.meds_utils import create_dataset_from_meds_reader
from cehrbert.models.hf_models.config import CehrBertConfig
from cehrbert.models.hf_models.hf_cehrbert import (
    CehrBertForClassification,
    CehrBertLstmForClassification,
    CehrBertPreTrainedModel,
)
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import FineTuneModelType
from cehrbert.runners.runner_util import (
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
    parse_runner_args,
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

    return {"accuracy": accuracy, "roc_auc": roc_auc, "pr_auc": pr_auc}


def load_pretrained_model_and_tokenizer(
    model_args,
) -> Tuple[CehrBertPreTrainedModel, CehrBertTokenizer]:
    # Try to load the pretrained tokenizer
    try:
        tokenizer = CehrBertTokenizer.from_pretrained(model_args.tokenizer_name_or_path)
    except Exception:
        raise ValueError(f"Can not load the pretrained tokenizer from {model_args.tokenizer_name_or_path}")

    if model_args.finetune_model_type == FineTuneModelType.POOLING.value:
        finetune_model_cls = CehrBertForClassification
    elif model_args.finetune_model_type == FineTuneModelType.LSTM.value:
        finetune_model_cls = CehrBertLstmForClassification
    else:
        raise ValueError(
            f"finetune_model_type can be one of the following types {[e.value for e in FineTuneModelType]}"
        )

    # Try to load the pretrained model
    try:
        model = finetune_model_cls.from_pretrained(model_args.model_name_or_path)
    except Exception as e:
        LOG.warning(e)
        model_config = CehrBertConfig(
            vocab_size=tokenizer.vocab_size,
            lab_token_ids=tokenizer.lab_token_ids,
            **model_args.as_dict(),
        )
        model = finetune_model_cls(model_config)

    return model, tokenizer


def main():
    data_args, model_args, training_args = parse_runner_args()

    model, tokenizer = load_pretrained_model_and_tokenizer(model_args)

    prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)

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

    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        if data_args.streaming:
            processed_dataset = processed_dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)
        LOG.info("Prepared dataset loaded from disk...")
    else:
        # If the data is in the MEDS format, we need to convert it to the CEHR-BERT format
        if data_args.is_data_in_med:
            meds_extension_path = get_meds_extension_path(
                data_folder=data_args.cohort_folder,
                dataset_prepared_path=data_args.dataset_prepared_path,
            )
            try:
                LOG.info(f"Trying to load the MEDS extension from disk at {meds_extension_path}...")
                dataset = load_from_disk(meds_extension_path)
                if data_args.streaming:
                    dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)
            except Exception as e:
                LOG.exception(e)
                dataset = create_dataset_from_meds_reader(data_args, is_pretraining=False)
                if not data_args.streaming:
                    dataset.save_to_disk(meds_extension_path)
            train_set = dataset["train"]
            validation_set = dataset["validation"]
            test_set = dataset["test"]
        else:
            dataset = load_parquet_as_dataset(data_args.data_folder)
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

    trainer = Trainer(
        model=model,
        data_collator=collator,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["validation"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)],
        args=training_args,
    )

    checkpoint = get_last_hf_checkpoint(training_args)
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_predict:
        # If do_train is set to False, we need to load the model from the checkpoint.
        if not training_args.do_train:
            LOG.info(
                "The do_train flag is set to False. Loading the weights from %s",
                training_args.output_dir,
            )
            trainer._load_from_checkpoint(training_args.output_dir)

        test_results = trainer.predict(processed_dataset["test"])
        # Save results to JSON
        test_results_path = os.path.join(training_args.output_dir, "test_results.json")
        with open(test_results_path, "w") as f:
            json.dump(test_results.metrics, f, indent=4)

        LOG.info(f"Test results: {test_results.metrics}")

        person_ids = [row["person_id"] for row in processed_dataset["test"]]

        if isinstance(test_results.predictions, np.ndarray):
            predictions = np.squeeze(test_results.predictions).tolist()
        else:
            predictions = np.squeeze(test_results.predictions[0]).tolist()
        if isinstance(test_results.label_ids, np.ndarray):
            labels = np.squeeze(test_results.label_ids).tolist()
        else:
            labels = np.squeeze(test_results.label_ids[0]).tolist()

        prediction_pd = pd.DataFrame({"person_id ": person_ids, "prediction": predictions, "label": labels})
        prediction_pd.to_csv(os.path.join(training_args.output_dir, "test_predictions.csv"), index=False)


if __name__ == "__main__":
    main()
