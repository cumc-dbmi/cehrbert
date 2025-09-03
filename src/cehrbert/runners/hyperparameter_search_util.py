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


def get_suggestion(
    trial,
    hyperparameter_name: str,
    hyperparameters: List[Union[float, int]],
    is_grid: bool = False,
) -> Union[float, int]:
    """
    Get hyperparameter suggestion based on search mode.

    Args:
        trial: Optuna trial object
        hyperparameter_name: Name of the hyperparameter
        hyperparameters: List of hyperparameter values
        is_grid: Whether to use grid search mode

    Returns:
        Suggested hyperparameter value

    Raises:
        RuntimeError: If Bayesian mode is used with incorrect number of bounds
    """
    if is_grid:
        return trial.suggest_categorical(hyperparameter_name, hyperparameters)

    # For Bayesian optimization, we need exactly 2 values (lower and upper bounds)
    if len(hyperparameters) != 2:
        raise RuntimeError(
            f"{hyperparameter_name} must contain exactly two values (lower and upper bound) "
            f"for Bayesian Optimization, but {len(hyperparameters)} values were provided: {hyperparameters}"
        )

    # Ensure bounds are sorted
    lower, upper = sorted(hyperparameters)
    return trial.suggest_float(hyperparameter_name, lower, upper, log=True)


def hp_space(
    trial: optuna.Trial,
    learning_rates: List[float],
    batch_sizes: List[int],
    weight_decays: List[float],
    num_train_epochs: List[int],
    is_grid: bool = False,
):
    """
    Define the hyperparameter search space.

    Args:
        trial: Optuna trial object
        learning_rates: Learning rate values (bounds for Bayesian, discrete values for grid)
        batch_sizes: Batch size values (always discrete)
        weight_decays: Weight decay values (bounds for Bayesian, discrete values for grid)
        num_train_epochs: Number of training epochs (always discrete)
        is_grid: Whether to use grid search mode

    Returns:
        Dictionary of hyperparameter suggestions
    """
    return {
        "learning_rate": get_suggestion(trial, "learning_rate", learning_rates, is_grid),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", batch_sizes),
        "weight_decay": get_suggestion(trial, "weight_decay", weight_decays, is_grid),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", num_train_epochs),
    }


def create_grid_search_space(
    learning_rates: List[float],
    batch_sizes: List[int],
    weight_decays: List[float],
    num_train_epochs: List[int],
):
    """
    Create the search space dictionary for GridSampler.

    Args:
        learning_rates: List of learning rate values to test
        batch_sizes: List of batch size values to test
        weight_decays: List of weight decay values to test
        num_train_epochs: List of epoch values to test

    Returns:
        Dictionary defining the grid search space
    """
    return {
        "learning_rate": learning_rates,
        "weight_decay": weight_decays,
        "per_device_train_batch_size": batch_sizes,
        "num_train_epochs": num_train_epochs,
    }


def calculate_total_combinations(search_space: dict) -> int:
    """Calculate total number of combinations in grid search."""
    total = 1
    for values in search_space.values():
        total *= len(values)
    return total


def sample_dataset(data: Dataset, percentage: float, seed: int) -> Dataset:
    """Samples a subset of the given dataset based on a specified percentage."""
    if percentage >= 1.0:
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
    Perform hyperparameter tuning using either Bayesian optimization or grid search.

    This function supports two modes:
    1. Bayesian Optimization (TPE): Intelligently explores hyperparameter space using bounds
    2. Grid Search: Exhaustively tests all combinations of discrete values

    Args:
        trainer_class: A Trainer or its subclass
        model_init (Callable): A function to initialize the model, used for each hyperparameter trial.
        dataset (DatasetDict): A Hugging Face DatasetDict containing "train" and "validation" datasets.
        data_collator (CehrBertDataCollator): A data collator for processing batches.
        training_args (TrainingArguments): Configuration for training parameters.
        model_args (ModelArguments): Model configuration arguments, including early stopping parameters.
        cehrbert_args (CehrBertArguments): Additional arguments specific to CehrBert, including hyperparameter
                                         tuning options and search mode configuration.

    Returns:
        Tuple[TrainingArguments, Optional[str]]: Updated TrainingArguments with best hyperparameters
                                               and optional run_id of the best trial.

    Raises:
        ValueError: If grid search is requested but hyperparameters are not properly configured
        RuntimeError: If Bayesian optimization bounds are incorrectly specified
    """
    if not cehrbert_args.hyperparameter_tuning:
        return training_args, None

    # Validate configuration
    if cehrbert_args.hyperparameter_tuning_is_grid:
        search_space = create_grid_search_space(
            learning_rates=cehrbert_args.hyperparameter_learning_rates,
            weight_decays=cehrbert_args.hyperparameter_weight_decays,
            batch_sizes=cehrbert_args.hyperparameter_batch_sizes,
            num_train_epochs=cehrbert_args.hyperparameter_num_train_epochs,
        )
        total_combinations = calculate_total_combinations(search_space)

        LOG.info(f"Grid search mode: Testing {total_combinations} combinations")
        LOG.info(f"Search space: {search_space}")

        # Adjust n_trials for grid search if not set appropriately
        if cehrbert_args.n_trials < total_combinations:
            LOG.warning(
                f"n_trials ({cehrbert_args.n_trials}) is less than total combinations ({total_combinations}). "
                f"Setting n_trials to {total_combinations} to test all combinations."
            )
            cehrbert_args.n_trials = total_combinations

        # Configure sampler based on search mode
        sampler = optuna.samplers.GridSampler(search_space, seed=training_args.seed)
    else:
        LOG.info("Bayesian optimization mode (TPE)")
        LOG.info(f"Learning rate bounds: {cehrbert_args.hyperparameter_learning_rates}")
        LOG.info(f"Weight decay bounds: {cehrbert_args.hyperparameter_weight_decays}")
        LOG.info(f"Batch sizes: {cehrbert_args.hyperparameter_batch_sizes}")
        LOG.info(f"Epochs: {cehrbert_args.hyperparameter_num_train_epochs}")
        sampler = optuna.samplers.TPESampler(seed=training_args.seed)

    # Prepare datasets
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

    # Create trainer
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
            is_grid=cehrbert_args.hyperparameter_tuning_is_grid,
        ),
        backend="optuna",
        n_trials=cehrbert_args.n_trials,
        compute_objective=lambda m: m["optuna_best_metric"],
        sampler=sampler,
    )

    # Log results
    LOG.info("=" * 50)
    LOG.info("HYPERPARAMETER SEARCH COMPLETED")
    LOG.info("=" * 50)
    LOG.info(f"Best hyperparameters: {best_trial.hyperparameters}")
    LOG.info(f"Best metric (eval_loss): {best_trial.objective}")
    LOG.info(f"Best run_id: {best_trial.run_id}")
    LOG.info("=" * 50)

    # Restore original settings and update with best hyperparameters
    training_args.save_total_limit = save_total_limit_original
    for k, v in best_trial.hyperparameters.items():
        setattr(training_args, k, v)
        LOG.info(f"Updated training_args.{k} = {v}")

    return training_args, best_trial.run_id
