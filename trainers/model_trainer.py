from abc import ABC, abstractmethod
import os
import pickle
import inspect
import pandas as pd
import datetime

import tensorflow as tf

from data_generators.tokenizer import ConceptTokenizer
from utils.logging_utils import *
from utils.utils import CosineLRSchedule


def log_function_decorator(function):
    def wrapper(self, *args, **kwargs):
        class_name = type(self).__name__
        function_name = function.__name__
        module_name = inspect.getmodule(function).__name__
        line_no = inspect.getsourcelines(function)[1]

        beginning = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            f'Started running {module_name}: {class_name}.{function_name} at line {line_no}')
        output = function(self, *args, **kwargs)
        ending = datetime.datetime.now()
        logging.getLogger(function.__name__).info(
            f'Took {ending - beginning} to run {module_name}: {class_name}.{function_name}.')
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


class AbstractModelTrainer(ABC):
    min_num_of_concepts = 5

    def __init__(self,
                 training_data_parquet_path: str,
                 model_path: str,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float,
                 tf_board_log_path: str = None,
                 shuffle_training_data: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._training_data_parquet_path = training_data_parquet_path
        self._model_path = model_path
        self._tf_board_log_path = tf_board_log_path
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._shuffle_training_data = shuffle_training_data

        self._training_data = self._load_training_data()

        # shuffle the training data
        if self._shuffle_training_data:
            self._training_data = self._training_data.sample(frac=1).reset_index(drop=True)

        self.get_logger().info(
            f'training_data_parquet_path: {training_data_parquet_path}\n'
            f'model_path: {model_path}\n'
            f'batch_size: {batch_size}\n'
            f'epochs: {epochs}\n'
            f'learning_rate: {learning_rate}\n'
            f'tf_board_log_path: {tf_board_log_path}\n'
            f'shuffle_training_data: {shuffle_training_data}\n')

    @log_function_decorator
    def _load_training_data(self):
        if not os.path.exists(self._training_data_parquet_path):
            raise FileExistsError(f'{self._training_data_parquet_path} does not exist!')
        parquet = pd.read_parquet(self._training_data_parquet_path)
        return parquet

    @abstractmethod
    def create_dataset(self):
        """
        Prepare _training_data for the model such as tokenize concepts.
        :return:
        """
        pass

    @abstractmethod
    def create_model(self):
        pass

    def train_model(self):
        """
        Train the model
        :return:
        """
        dataset, steps_per_epoch = self.create_dataset()
        model = self.create_model()

        model.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=self._epochs,
            callbacks=self.get_callbacks()
        )

    @abstractmethod
    def eval_model(self):
        pass

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)

    def __str__(self):
        return str(self.__class__.__name__)

    def get_callbacks(self):
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=self._tf_board_log_path)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self._model_path,
                                                              save_best_only=True, monitor='loss',
                                                              verbose=1)
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self._learning_rate, lr_low=1e-8, initial_period=10),
            verbose=1)
        return [
            tensor_board_callback,
            model_checkpoint,
            learning_rate_scheduler
        ]
