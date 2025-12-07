import datetime
import inspect
import logging
import os
import random
import re
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from tensorflow.data import Dataset
from tensorflow.keras.models import Model
from xgboost import XGBClassifier

DEFAULT_OOV_TOKEN = "-1"
DECIMAL_PLACE = 4
LOGGER = logging.getLogger(__name__)


def create_folder_if_not_exist(folder: str, sub_folder_name: str) -> Path:
    """
    Creates a subfolder if it does not exist and returns the full Path object.

    Args:
        folder (str): The parent folder where the subfolder will be created.
        sub_folder_name (str): The name of the subfolder to be created.

    Returns:
        Path: The full path to the created or existing subfolder.

    Example:
        create_sub_folder("/home/user", "new_folder")
        # Creates /home/user/new_folder if it doesn't exist and returns the path.
    """
    sub_folder = Path(folder) / sub_folder_name
    if not sub_folder.exists():
        LOGGER.info("Create folder: %s", sub_folder)
        sub_folder.mkdir(parents=True, exist_ok=True)
    return sub_folder


def log_function_decorator(function):
    def wrapper(self, *args, **kwargs):
        function_name = function.__name__
        module_name = inspect.getmodule(function).__name__
        line_no = inspect.getsourcelines(function)[1]

        beginning = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            "Started running %s: %s at line %s", module_name, function_name, line_no
        )
        output = function(self, *args, **kwargs)
        ending = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            "Took %s to run %s: %s.", ending - beginning, module_name, function_name
        )
        return output

    return wrapper


def convert_to_list_of_lists(concept_lists):
    return list(map(lambda sub_arrays: sub_arrays.tolist(), concept_lists))


@log_function_decorator
def run_model(
        model,
        dataset: Union[Dataset, Tuple[np.ndarray, np.ndarray]],
):
    if isinstance(dataset, Dataset):
        x = dataset.map(lambda _x, _y: _x)
        y = dataset.map(lambda _x, _y: _y)
        y = list(chain(*y.as_numpy_iterator()))
    elif len(dataset) == 2:
        x, y = dataset
    else:
        raise TypeError("Only numpy array and tensorflow Dataset are supported types.")

    if isinstance(model, Model):
        prob = model.predict(x)
    elif isinstance(model, (LogisticRegression, XGBClassifier, GridSearchCV)):
        prob = model.predict_proba(x)[:, 1]
    else:
        raise TypeError(f"Unknown type for the model {type(model)}")

    return np.asarray(prob), y


def calculate_pr_auc(labels, probabilities):
    """
    Calculate PR AUC given labels and probabilities.

    :param labels:
    :param probabilities:
    :return:
    """
    # Calculate precision-recall auc
    precisions, recalls, _ = metrics.precision_recall_curve(labels, np.asarray(probabilities))
    return metrics.auc(recalls, precisions)


@log_function_decorator
def compute_binary_metrics(
        model,
        test_data: Union[Dataset, Tuple[np.ndarray, np.ndarray]],
        metrics_folder,
        evaluation_model_folder: str = None,
        model_name: str = None,
        extra_info: dict = None,
        calculate_ci: bool = True,
):
    """
    Compute Recall, Precision, F1-score and PR-AUC for the test data.

    :param model:
    :param test_data:
    :param evaluation_model_folder:
    :param metrics_folder:
    :param model_name:
    :param extra_info:
    :param calculate_ci:
    :return:
    """

    def compute_confidence_interval(x, y, metric_func):
        """
        A helper function to calculate the 95% confidence interval for a given metric function.

        :param x:
        :param y:
        :param metric_func:
        :param alpha:
        :return:
        """
        # Calculate the roc-auc confidence interval using bootstrap
        bootstrap_metrics = []
        total = len(y)
        for _ in range(1001):
            x_sample, y_sample = zip(*random.choices(list(zip(x, y)), k=total))
            bootstrap_metrics.append(metric_func(x_sample, y_sample))

        bootstrap_metrics = sorted(bootstrap_metrics)

        return bootstrap_metrics[25], bootstrap_metrics[975]

    validate_folder(metrics_folder)

    probabilities, labels = run_model(model, test_data)

    predictions = (np.asarray(probabilities) > 0.5).astype(int)
    recall = metrics.recall_score(labels, predictions, average="binary")
    precision = metrics.precision_score(labels, predictions, average="binary")
    f1_score = metrics.f1_score(labels, predictions, average="binary")

    # Calculate precision-recall auc
    precisions, recalls, pr_auc_thresholds = metrics.precision_recall_curve(labels, np.asarray(probabilities))
    pr_auc = metrics.auc(recalls, precisions)

    # Calculate the best threshold for pr auc
    f_scores = (2 * precisions * recalls) / (precisions + recalls)
    f_score_ix = np.argmax(f_scores)
    pr_auc_best_threshold = pr_auc_thresholds[f_score_ix]

    # Calculate the 95% CI for pr_auc
    if calculate_ci:
        pr_auc_lower, pr_auc_upper = compute_confidence_interval(
            x=labels, y=probabilities, metric_func=calculate_pr_auc
        )
    else:
        pr_auc_lower = pr_auc_upper = pr_auc

    # Calculate roc-auc
    roc_auc = metrics.roc_auc_score(labels, probabilities)
    if calculate_ci:
        roc_auc_lower, roc_auc_upper = compute_confidence_interval(
            x=labels, y=probabilities, metric_func=metrics.roc_auc_score
        )
    else:
        roc_auc_lower = roc_auc_upper = roc_auc

    # Calculate the best threshold for roc auc
    fpr, tpr, roc_thresholds = metrics.roc_curve(labels, probabilities)
    j_measure = tpr - fpr
    ix = np.argmax(j_measure)
    roc_auc_best_threshold = roc_thresholds[ix]

    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    data_metrics = {
        "model_name": model_name,
        "time_stamp": [current_time],
        "recall": [round(recall, DECIMAL_PLACE)],
        "precision": [round(precision, DECIMAL_PLACE)],
        "f1-score": [round(f1_score, DECIMAL_PLACE)],
        "pr_auc": [round(pr_auc, DECIMAL_PLACE)],
        "pr_auc_ci": f"({round(pr_auc_lower, DECIMAL_PLACE)}, {round(pr_auc_upper, DECIMAL_PLACE)})",
        "pr_auc_best_threshold": round(pr_auc_best_threshold, DECIMAL_PLACE),
        "roc_auc": [round(roc_auc, DECIMAL_PLACE)],
        "roc_auc_ci": f"({round(roc_auc_lower, DECIMAL_PLACE)}, {round(roc_auc_upper, DECIMAL_PLACE)})",
        "roc_auc_best_threshold": round(roc_auc_best_threshold, DECIMAL_PLACE),
    }

    if extra_info:
        # Add the additional information to the metrics
        tf.print(f"Adding extra_info to the metrics folder: {extra_info}")
        data_metrics.update(extra_info)

    data_metrics_pd = pd.DataFrame(data_metrics)
    data_metrics_pd.to_parquet(os.path.join(metrics_folder, f"{current_time}.parquet"))

    if evaluation_model_folder:
        validate_folder(evaluation_model_folder)
        prediction_pd = pd.DataFrame(zip(labels, probabilities), columns=["label", "prediction"])
        prediction_pd.to_parquet(os.path.join(evaluation_model_folder, f"{current_time}.parquet"))

    return data_metrics


def save_training_history(history: Dict, history_folder, model_name: str = None):
    """
    Save the training metrics in the history dictionary as pandas dataframe to the file.

    system in parquet format

    :param history:
    :param history_folder:
    :param model_name:
    :return:
    """

    validate_folder(history_folder)

    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    history_parquet_file_path = f"{current_time}.parquet"
    data_frame = pd.DataFrame(dict(sorted(history.history.items())))
    data_frame.insert(0, "time_stamp", current_time)
    data_frame.insert(0, "model_name", model_name)
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.to_parquet(os.path.join(history_folder, history_parquet_file_path))


def validate_folder(folder):
    if not os.path.exists(folder):
        raise FileExistsError(f"{folder} does not exist!")


def create_concept_mask(mask, max_seq_length):
    # mask the third dimension
    concept_mask_1 = tf.tile(tf.expand_dims(tf.expand_dims(mask, axis=1), axis=-1), [1, 1, 1, max_seq_length])
    # mask the fourth dimension
    concept_mask_2 = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
    concept_mask = tf.cast(
        (concept_mask_1 + concept_mask_2) > 0,
        dtype=tf.int32,
        name=f'{re.sub("[^0-9a-zA-Z]+", "", mask.name)}_mask',
    )
    return concept_mask


def multimode(data):
    # Multimode of List
    # using loop + formula
    res = []
    list_1 = Counter(data)
    temp = list_1.most_common(1)[0][1]
    for ele in data:
        if data.count(ele) == temp:
            res.append(ele)
    return list(set(res))
