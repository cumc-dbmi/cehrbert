import json
import os
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import DatasetDict, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from scipy.special import expit as sigmoid
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed
from transformers.trainer_utils import is_main_process
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset import create_cehrbert_finetuning_dataset
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import (
    CehrBertDataCollator,
    SamplePackingCehrBertDataCollator,
)
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import MedToCehrBertDatasetMapping
from cehrbert.data_generators.hf_data_generator.meds_utils import create_dataset_from_meds_reader
from cehrbert.models.hf_models.config import CehrBertConfig
from cehrbert.models.hf_models.hf_cehrbert import (
    CehrBertForClassification,
    CehrBertLstmForClassification,
    CehrBertPreTrainedModel,
)
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments, FineTuneModelType, ModelArguments
from cehrbert.runners.runner_util import (
    convert_dataset_to_iterable_dataset,
    generate_prepared_ds_path,
    get_last_hf_checkpoint,
    get_meds_extension_path,
    load_parquet_as_dataset,
    parse_runner_args,
)
from cehrbert.runners.sample_packing_trainer import SamplePackingTrainer

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
    try:
        roc_auc = roc_auc_score(references, probs)
        precision, recall, _ = precision_recall_curve(references, probs)
        pr_auc = auc(recall, precision)
        return {"roc_auc": roc_auc, "pr_auc": pr_auc}
    except Exception as e:
        LOG.exception(e)
        return {"roc_auc": None, "pr_auc": None}


def get_torch_dtype(torch_dtype: Optional[str] = None) -> Union[torch.dtype, str]:
    if torch_dtype and hasattr(torch, torch_dtype):
        return getattr(torch, torch_dtype)
    return torch.float


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
    torch_dtype = get_torch_dtype(model_args.torch_dtype)
    try:
        model = finetune_model_cls.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype, attn_implementation=model_args.attn_implementation
        )
        if torch_dtype == torch.bfloat16:
            return model.bfloat16()
        elif torch_dtype == torch.float16:
            return model.half()
        else:
            return model.float()
    except ValueError:
        raise ValueError(f"Can not load the finetuned model from {model_name_or_path}")


def main():

    cehrbert_args, data_args, model_args, training_args = parse_runner_args()

    if data_args.streaming:
        # This happens only when streaming is enabled. This is for disabling the warning message
        # https://github.com/huggingface/transformers/issues/5486
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # The iterable dataset doesn't have sharding implemented, so the number of works has to
        # be set to 0. Otherwise the trainer will throw an error
        training_args.dataloader_num_workers = 0

    tokenizer = load_pretrained_tokenizer(model_args)
    cache_file_collector = CacheFileCollector()
    processed_dataset = None
    prepared_ds_path = generate_prepared_ds_path(data_args, model_args, data_folder=data_args.cohort_folder)
    if any(prepared_ds_path.glob("*")):
        LOG.info(f"Loading prepared dataset from disk at {prepared_ds_path}...")
        processed_dataset = load_from_disk(str(prepared_ds_path))
        if data_args.streaming:
            processed_dataset = convert_dataset_to_iterable_dataset(
                processed_dataset, num_shards=training_args.dataloader_num_workers
            )
        LOG.info("Prepared dataset loaded from disk...")

    if processed_dataset is None:
        if is_main_process(training_args.local_rank):
            # Organize them into a single DatasetDict
            final_splits = prepare_finetune_dataset(data_args, training_args, cache_file_collector)

            # TODO: temp solution, this column is mixed typed and causes an issue when transforming the data
            if not data_args.streaming:
                all_columns = final_splits["train"].column_names
                if "visit_concept_ids" in all_columns:
                    final_splits = final_splits.remove_columns(["visit_concept_ids"])

            processed_dataset = create_cehrbert_finetuning_dataset(
                dataset=final_splits,
                concept_tokenizer=tokenizer,
                data_args=data_args,
                cache_file_collector=cache_file_collector,
            )

            if not data_args.streaming:
                processed_dataset.save_to_disk(str(prepared_ds_path))
                stats = processed_dataset.cleanup_cache_files()
                LOG.info(
                    "Clean up the cached files for the cehrbert fine-tuning dataset: %s",
                    stats,
                )
                processed_dataset = load_from_disk(str(prepared_ds_path))

            # Remove all the cached files collected during the data transformation if there are any
            cache_file_collector.remove_cache_files()

        # After main-process-only operations, synchronize all processes to ensure consistency
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

            # Loading tokenizer in all processes in torch distributed training
            tokenizer_name_or_path = os.path.expanduser(model_args.tokenizer_name_or_path)
            tokenizer = CehrBertTokenizer.from_pretrained(tokenizer_name_or_path)
            # Load the dataset from disk again to in torch distributed training
            processed_dataset = load_from_disk(str(prepared_ds_path))

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if not data_args.streaming and not cehrbert_args.sample_packing:
        processed_dataset.set_format("pt")

    config = CehrBertConfig.from_pretrained(model_args.model_name_or_path)
    # persist this parameter in case this is overwritten by sample packing
    per_device_eval_batch_size = training_args.per_device_eval_batch_size
    if cehrbert_args.sample_packing:
        trainer_class = partial(
            SamplePackingTrainer,
            max_tokens_per_batch=cehrbert_args.max_tokens_per_batch,
            max_position_embeddings=config.max_position_embeddings,
            train_lengths=processed_dataset["train"]["num_of_concepts"],
            validation_lengths=processed_dataset["validation"]["num_of_concepts"],
        )
        training_args.per_device_train_batch_size = 1
        training_args.per_device_eval_batch_size = 1
        data_collator_fn = partial(
            SamplePackingCehrBertDataCollator,
            cehrbert_args.max_tokens_per_batch,
            config.max_position_embeddings,
        )
    else:
        data_collator_fn = CehrBertDataCollator
        trainer_class = Trainer

    data_collator = data_collator_fn(
        tokenizer=tokenizer,
        max_length=(
            cehrbert_args.max_tokens_per_batch if cehrbert_args.sample_packing else config.max_position_embeddings
        ),
        is_pretraining=False,
        mlm_probability=config.mlm_probability,
    )

    if training_args.do_train:
        model = load_finetuned_model(model_args, model_args.model_name_or_path)
        if getattr(model.config, "cls_token_id") is None:
            model.config.cls_token_id = tokenizer.cls_token_index
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

        trainer = trainer_class(
            model=model,
            data_collator=data_collator,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=model_args.early_stopping_patience)],
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
            batch_size=per_device_eval_batch_size,
            num_workers=training_args.dataloader_num_workers,
            collate_fn=CehrBertDataCollator(
                tokenizer=tokenizer,
                max_length=config.max_position_embeddings,
                is_pretraining=False,
                mlm_probability=config.mlm_probability,
            ),
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
            # Extract and process index_dates
            index_dates = None
            if "index_date" in batch:
                try:
                    timestamps = batch.pop("index_date").numpy().squeeze().tolist()
                    # Handle potential NaN or invalid timestamps
                    index_dates = [datetime.fromtimestamp(ts) if not np.isnan(ts) else None for ts in timestamps]
                except (ValueError, OverflowError, TypeError):
                    index_dates = [None] * len(timestamps)
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
        references=test_prediction_pd.boolean_value, probs=test_prediction_pd.predicted_boolean_probability
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
