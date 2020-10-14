import os
import datetime
import inspect
import pathlib
import logging
import pickle

import pandas as pd
import numpy as np
from sklearn import metrics
from typing import Dict

from data_generators.tokenizer import ConceptTokenizer

LOGGER = logging.getLogger(__name__)


def create_folder_if_not_exist(folder, sub_folder_name):
    """
    Create the sub-folder if not exists. Will do not thing if the sub-folder already exists.

    :param folder:
    :param sub_folder_name:
    :return:
    """
    sub_folder = os.path.join(folder, sub_folder_name)
    if not os.path.exists(sub_folder):
        LOGGER.info(f'Create folder: {sub_folder}')
        pathlib.Path(sub_folder).mkdir(parents=True, exist_ok=True)
    return sub_folder


def log_function_decorator(function):
    def wrapper(self, *args, **kwargs):
        function_name = function.__name__
        module_name = inspect.getmodule(function).__name__
        line_no = inspect.getsourcelines(function)[1]

        beginning = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            f'Started running {module_name}: {function_name} at line {line_no}')
        output = function(self, *args, **kwargs)
        ending = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            f'Took {ending - beginning} to run {module_name}: {function_name}.')
        return output

    return wrapper


@log_function_decorator
def tokenize_concepts(training_data, column_name, tokenized_column_name, tokenizer_path,
                      oov_token='0'):
    """
    Tokenize the concept sequence and save the tokenizer as a pickle file
    :return:
    """
    tokenizer = ConceptTokenizer(oov_token=oov_token)
    training_data[column_name] = training_data[column_name].apply(
        lambda concept_ids: concept_ids.tolist())
    tokenizer.fit_on_concept_sequences(training_data[column_name])
    encoded_sequences = tokenizer.encode(training_data[column_name])
    training_data[tokenized_column_name] = encoded_sequences
    pickle.dump(tokenizer, open(tokenizer_path, 'wb'))
    return tokenizer


@log_function_decorator
def compute_binary_metrics(model, test_data, test_size, batch_size, metrics_folder):
    """
    Compute Recall, Precision, F1-score and PR-AUC for the test data

    :param model:
    :param test_data:
    :param batch_size:
    :param test_size:
    :param metrics_folder:
    :return:
    """

    def run_model():
        _step = 0
        _num_of_steps = test_size // batch_size + test_size % batch_size
        _probabilities = []
        _labels = []
        for next_batch in test_data:
            x, y = next_batch
            prediction_batch = model.predict(x)
            _probabilities.extend(prediction_batch.flatten().tolist())
            _labels.extend(y.numpy().tolist())
            _step += 1
            if _step >= _num_of_steps:
                break
        return _probabilities, _labels

    validate_folder(metrics_folder)

    probabilities, labels = run_model()
    precisions, recalls, _ = metrics.precision_recall_curve(labels, np.asarray(probabilities))
    predictions = (np.asarray(probabilities) > 0.5).astype(int)
    recall = metrics.recall_score(labels, predictions, average='binary')
    precision = metrics.precision_score(labels, predictions, average='binary')
    f1_score = metrics.f1_score(labels, predictions, average='binary')
    pr_auc = metrics.auc(recalls, precisions)

    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    data_metrics = {'time_stamp': [current_time],
                    'recall': [recall],
                    'precision': [precision],
                    'f1-score': [f1_score],
                    'pr_auc': [pr_auc]}

    pd.DataFrame(data_metrics).to_parquet(os.path.join(metrics_folder, f'{current_time}.parquet'))


def save_training_history(history: Dict, history_folder):
    """
    Save the training metrics in the history dictionary as pandas dataframe to the file
    system in parquet format

    :param history:
    :param history_folder:
    :return:
    """

    validate_folder(history_folder)

    current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    history_parquet_file_path = f'{current_time}.parquet'
    data_frame = pd.DataFrame(dict(sorted(history.history.items())))
    data_frame.columns = data_frame.columns.astype(str)
    data_frame.to_parquet(os.path.join(history_folder, history_parquet_file_path))


def validate_folder(folder):
    if not os.path.exists(folder):
        raise FileExistsError(f'{folder} does not exist!')
