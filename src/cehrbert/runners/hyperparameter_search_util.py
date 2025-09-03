from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import optuna
from datasets import Dataset, DatasetDict
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
from cehrbert.runners.hf_runner_argument_dataclass import CehrBertArguments, ModelArguments

LOG = logging.get_logger("transformers")


class OptunaMetricCallback(TrainerCallback):
    """
    A custom callback to store the best metric in the evaluation metrics dictionary during training.

    This callback monitors the training state and updates the metrics dictionary with the `best_metric`
    (e.g., the lowest `eval_loss` or highest accuracy) observed during training. It ensures that the
    best metric value is preserved in the final evaluation results, even if early stopping occurs.

    Attributes:
        None

    Methods:
        on_evaluate(args, state, control, **kwargs):
            Called during evaluation. Adds `state.best_metric` to `metrics` if it exists.

    Example Usage:
        ```
        store_best_metric_callback = StoreBestMetricCallback()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[store_best_metric_callback]
        )
        ```
    """

    def on_evaluate(self, args, state, control, **kwargs):
        """
        During evaluation, adds the best metric value to the metrics dictionary if it exists.

        Args:
            args: Training arguments.
            state: Trainer state object that holds information about training progress.
            control: Trainer control object to modify training behavior.
            **kwargs: Additional keyword arguments, including `metrics`, which holds evaluation metrics.

        Updates:
            `metrics["best_metric"]`: Sets this to `state.best_metric` if available.
        """
        # Check if best metric is available and add it to metrics if it exists
        metrics = kwargs.get("metrics", {})
        if state.best_metric is not None:
            metrics.update({"optuna_best_metric": min(state.best_metric, metrics["eval_loss"])})
        else:
            metrics.update({"optuna_best_metric": metrics["eval_loss"]})


def get_suggestion(trial, hyperparameter_name: str, hyperparameters: List[Union[float, int]]) -> Union[float, int]:
    if len(hyperparameters) == 2:
        return trial.suggest_float(hyperparameter_name, *hyperparameters, log=True)
    return trial.suggest_categorical(hyperparameter_name, hyperparameters)


# Define the hyperparameter search space with parameters
def hp_space(
    trial: optuna.Trial,
    learning_rates: List[float],
    batch_sizes: List[int],
    weight_decays: List[float],
    num_train_epochs: Tuple[float, ...] = 10,
):
    if batch_sizes is None:
        batch_sizes = [4, 8]
    return {
        "learning_rate": get_suggestion(trial, "learning_rate", learning_rates),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", batch_sizes),
        "weight_decay": get_suggestion(trial, "weight_decay", weight_decays),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", num_train_epochs),
    }


def sample_dataset(data: Dataset, percentage: float, seed: int) -> Dataset:
    """
    Samples a subset of the given dataset based on a specified percentage.

    This function uses a random train-test split to select a subset of the dataset, returning a sample
    that is approximately `percentage` of the total dataset size. It is useful for creating smaller
    datasets for tasks such as hyperparameter tuning or quick testing.

    Args:
        data (Dataset): The input dataset to sample from.
        percentage (float): The fraction of the dataset to sample, represented as a decimal
                            (e.g., 0.1 for 10%).
        seed (int): A random seed for reproducibility in the sampling process.

    Returns:
        Dataset: A sampled subset of the input dataset containing `percentage` of the original data.

    Example:
        ```
        sampled_data = sample_dataset(my_dataset, percentage=0.1, seed=42)
        ```

    Notes:
        - The `train_test_split` method splits the dataset into "train" and "test" portions. This function
          returns the "test" portion, which is the specified percentage of the dataset.
        - Ensure that `percentage` is between 0 and 1 to avoid errors.
    """
    if percentage == 1.0:
        return data

    return data.train_test_split(
        test_size=percentage,
        seed=seed,
    )["test"]


def perform_hyperparameter_search(
    trainer_class,
    model_init: Callable,
    dataset: DatasetDict,
    data_collator: CehrBertDataCollator,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    cehrbert_args: CehrBertArguments,
) -> Tuple[TrainingArguments, Optional[str]]:
    """
    Perform hyperparameter tuning for the CehrGPT model using Optuna with the Hugging Face Trainer.

    This function initializes a Trainer with sampled training and validation sets, and performs
    a hyperparameter search using Optuna. The search tunes learning rate, batch size, and weight decay
    to optimize model performance based on a specified objective metric (e.g., validation loss).
    After the search, it updates the provided `TrainingArguments` with the best hyperparameters found.

    Args:
        trainer_class: A Trainer or its subclass
        model_init (Callable): A function to initialize the model, used for each hyperparameter trial.
        dataset (DatasetDict): A Hugging Face DatasetDict containing "train" and "validation" datasets.
        data_collator (CehrGptDataCollator): A data collator for processing batches.
        training_args (TrainingArguments): Configuration for training parameters (e.g., epochs, evaluation strategy).
        model_args (ModelArguments): Model configuration arguments, including early stopping parameters.
        cehrgpt_args (CehrGPTArguments): Additional arguments specific to CehrGPT, including hyperparameter
                                         tuning options such as learning rate range, batch sizes, and tuning percentage.

    Returns:
        TrainingArguments: Updated `TrainingArguments` instance containing the best hyperparameters found
                           from the search.

    Example:
        ```
        best_training_args = perform_hyperparameter_search(
            trainer_class=Trainer,
            model_init=my_model_init,
            dataset=my_dataset_dict,
            data_collator=my_data_collator,
            training_args=initial_training_args,
            model_args=model_args,
            cehrgpt_args=cehrgpt_args
        )
        ```

    Notes:
        - If `cehrgpt_args.hyperparameter_tuning` is set to `True`, this function samples a portion of the
          training and validation datasets for efficient tuning.
        - `EarlyStoppingCallback` is added to the Trainer if early stopping is enabled in `model_args`.
        - Optuna's `hyperparameter_search` is configured with the specified number of trials (`n_trials`)
          and learning rate and batch size ranges provided in `cehrgpt_args`.

    Logging:
        Logs the best hyperparameters found at the end of the search.
    """
    run_id = None
    if cehrbert_args.hyperparameter_tuning:
        save_total_limit_original = training_args.save_total_limit
        training_args.save_total_limit = 1
        sampled_train = sample_dataset(
            dataset["train"],
            cehrbert_args.hyperparameter_tuning_percentage,
            training_args.seed,
        )
        sampled_val = sample_dataset(
            dataset["validation"],
            cehrbert_args.hyperparameter_tuning_percentage,
            training_args.seed,
        )
        hyperparam_trainer = trainer_class(
            model_init=model_init,
            data_collator=data_collator,
            train_dataset=sampled_train,
            eval_dataset=sampled_val,
            callbacks=[
                EarlyStoppingCallback(model_args.early_stopping_patience),
                OptunaMetricCallback(),
            ],
            args=training_args,
        )
        best_trial = hyperparam_trainer.hyperparameter_search(
            direction="minimize",
            hp_space=partial(
                hp_space,
                learning_rates=cehrbert_args.hyperparameter_learning_rates,
                weight_decays=cehrbert_args.hyperparameter_weight_decays,
                batch_sizes=cehrbert_args.hyperparameter_batch_sizes,
                num_train_epochs=cehrbert_args.hyperparameter_num_train_epochs,
            ),
            backend="optuna",
            n_trials=cehrbert_args.n_trials,
            compute_objective=lambda m: m["optuna_best_metric"],
            # Ensure reproducibility
            sampler=optuna.samplers.TPESampler(seed=training_args.seed),
        )
        LOG.info("Best hyperparameters: %s", best_trial.hyperparameters)
        LOG.info("Best run_id: %s", best_trial.run_id)
        run_id = best_trial.run_id
        training_args.save_total_limit = save_total_limit_original
        # Update training arguments with best hyperparameters and set epochs based on adjusted effective epochs
        for k, v in best_trial.hyperparameters.items():
            setattr(training_args, k, v)
    return training_args, run_id
