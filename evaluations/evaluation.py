from typing import List, Union
import numpy as np
import tensorflow as tf
from tensorflow.data import Dataset
from evaluations.model_evaluators import ModelEvaluator


class EvaluationModels:
    """
    Evaluate the model and save the history to output_folder
    """

    def __init__(self,
                 model_evaluators: List[ModelEvaluator],
                 dataset: Union[Dataset, np.ndarray],
                 data_size: int,
                 is_k_fold=False,
                 training_percentage=0.7,
                 val_test_ratio=0.5):
        self._model_evaluators = model_evaluators
        self._dataset = dataset
        self._data_size = data_size
        self._is_k_fold = is_k_fold
        self._training_percentage = training_percentage
        self._val_test_ratio = val_test_ratio

    def _split_data(self) -> Union[Dataset, np.ndarray]:
        """
        Split data into training, val, and test sets.
        :return:
        """

        if isinstance(self._dataset, tf.data.Dataset):
            training_size = self._get_training_data_size()
            train_dataset = self._dataset.take(training_size)
            remaining = self._dataset.skip(training_size)
            val_size = self._get_val_data_size()
            val_dataset = remaining.take(val_size)
            test_dataset = remaining.skip(val_size)
            test_size = self._get_test_data_size()
            return [(train_dataset, training_size), (val_dataset, val_size),
                    (test_dataset, test_size)]

        elif isinstance(self._dataset, np.ndarray):
            pass
        else:
            raise TypeError('Only numpy array and tensorflow Dataset are supported types.')

    def evaluate(self):
        """
        Evaluate the models
        :return:
        """

        if self._is_k_fold:
            raise NotImplemented('The k_fold cross validation is not implemented!')
        else:
            split_datasets = self._split_data()
            for model_evaluator in self._model_evaluators:
                model_evaluator.evaluate(split_datasets)

    def _get_training_data_size(self):
        return int(self._training_percentage * self._data_size)

    def _get_val_data_size(self):
        rest = (1 - self._training_percentage) * self._data_size
        return int(rest * (self._val_test_ratio / (self._val_test_ratio + 1)))

    def _get_test_data_size(self):
        rest = (1 - self._training_percentage) * self._data_size
        return int(rest * (1 - (self._val_test_ratio / (self._val_test_ratio + 1))))
