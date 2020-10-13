import copy
import datetime
import logging
import os
import pathlib
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Dict

import numpy as np
import pandas as pd
from sklearn import metrics
from tensorflow.data import Dataset

from models.evaluation_models import *


class ModelEvaluator(ABC):
    def __init__(self, evaluation_folder, epochs, batch_size, *args, **kwargs):
        self._model = self._create_model(*args, **kwargs)
        self._evaluation_folder = evaluation_folder
        self._epochs = epochs
        self._batch_size = batch_size

    @abstractmethod
    def _create_model(self, *args, **kwargs) -> Model:
        pass

    def _get_callbacks(self):
        """
        Standard callbacks for the evaluations
        :return:
        """
        return [tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_model_path(),
            monitor='val_loss',
            mode='auto',
            save_best_only=True,
            save_weights_only=True)]

    def _get_metrics(self):
        """
        Standard metrics used for compiling the models
        :return:
        """
        return ['binary_accuracy',
                tf.keras.metrics.Recall(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.AUC(curve='PR')]

    def evaluate(self, datasets: List[Tuple[Union[Dataset, np.ndarray], int]]):

        if len(datasets) == 3:
            # This is the scenario of training, val, and test split
            training, val, test = datasets

            training_data, train_size = training
            val_data, val_size = val
            test_data, test_size = test

        elif len(datasets) == 2:
            # This is the scenario of training, val, and test split
            training, val = datasets

            training_data, train_size = training
            val_data, val_size = val
            test_data = copy.copy(val_data)
            test_size = copy.copy(val_size)
        else:
            raise AssertionError('The number of datasets can be either 2 or 3')

        training_data = self._preprocess_dataset(training_data)
        val_data = self._preprocess_dataset(val_data)
        test_data = self._preprocess_dataset(test_data)

        history = self._model.fit(
            training_data,
            steps_per_epoch=train_size // self._batch_size,
            epochs=self._epochs,
            validation_data=val_data,
            validation_steps=val_size // self._batch_size,
            callbacks=self._get_callbacks()
        )

        self.save_training_metrics(history)
        self.compute_metrics(test_data, test_size)

    @abstractmethod
    def _preprocess_dataset(self, dataset):
        pass

    def compute_metrics(self, test_data, test_size):
        """
        Compute the metrics for the test data

        :param test_data:
        :param test_size:
        :return:
        """
        step = 0
        probabilities = []
        labels = []
        for next_batch in test_data:
            x, y = next_batch
            prediction_batch = self._model.predict(x)
            probabilities.extend(prediction_batch.flatten().tolist())
            labels.extend(y.numpy().tolist())
            step += 1
            if step >= (test_size // self._batch_size):
                break

        lr_precision, lr_recall, _ = metrics.precision_recall_curve(labels,
                                                                    np.asarray(probabilities))
        predictions = (np.asarray(probabilities) > 0.5).astype(int)
        recall = metrics.recall_score(labels, predictions, average="binary")
        precision = metrics.precision_score(labels, predictions, average="binary")
        f1_score = metrics.f1_score(labels, predictions, average="binary")
        pr_auc = metrics.auc(lr_recall, lr_precision)

        current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")

        test_data_metrics = {'time_stamp': [current_time],
                             'recall': [recall],
                             'precision': [precision],
                             'f1-score': [f1_score],
                             'pr_auc': [pr_auc]}

        pd.DataFrame(test_data_metrics).to_parquet(
            os.path.join(self.get_model_metrics_folder(), f'{current_time}.parquet'))

    def save_training_metrics(self, history: Dict):
        """
        Save the training metrics in the history dictionary as pandas dataframe to the file
        system in parquet format

        :param history:
        :return:
        """
        history_folder = self.get_model_history_folder()
        current_time = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        history_parquet_file_path = f'{current_time}.parquet'
        data_frame = pd.DataFrame(dict(sorted(history.history.items())))
        data_frame.columns = data_frame.columns.astype(str)
        data_frame.to_parquet(
            os.path.join(history_folder, history_parquet_file_path))

    def get_model_metrics_folder(self):
        metrics_folder = os.path.join(self.get_model_folder(), 'metrics')
        if not os.path.exists(metrics_folder):
            self.get_logger().info(f'Create the model history folder at {metrics_folder}')
            pathlib.Path(metrics_folder).mkdir(parents=True, exist_ok=True)
        return metrics_folder

    def get_model_history_folder(self):
        history_folder = os.path.join(self.get_model_folder(), 'history')
        if not os.path.exists(history_folder):
            self.get_logger().info(f'Create the model history folder at {history_folder}')
            pathlib.Path(history_folder).mkdir(parents=True, exist_ok=True)
        return history_folder

    def get_model_folder(self):
        model_folder = os.path.join(self._evaluation_folder, self.get_model_name())
        if not os.path.exists(model_folder):
            self.get_logger().info(f'Create the model folder at {model_folder}')
            pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
        return model_folder

    def get_model_path(self):
        model_folder = self.get_model_folder()
        return os.path.join(model_folder, f'{self.get_model_name()}.h5')

    def get_model_name(self):
        return self._model.name

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)


class BiLstmModelEvaluator(ModelEvaluator):

    def __init__(self,
                 max_seq_length,
                 vocab_size,
                 embedding_size,
                 time_aware_model_path=None,
                 *args, **kwargs):
        super(BiLstmModelEvaluator, self).__init__(max_seq_length=max_seq_length,
                                                   vocab_size=vocab_size,
                                                   embedding_size=embedding_size,
                                                   time_aware_model_path=time_aware_model_path,
                                                   *args, **kwargs)

    def _create_model(self,
                      max_seq_length,
                      vocab_size,
                      embedding_size,
                      time_aware_model_path) -> Model:

        def get_concept_embeddings():
            concept_embeddings = None
            if time_aware_model_path:
                another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
                with another_strategy.scope():
                    time_aware_model = tf.keras.models.load_model(time_aware_model_path,
                                                                  custom_objects=dict(
                                                                      **get_custom_objects()))
                    concept_embeddings = time_aware_model.get_layer('embedding_layer') \
                        .get_weights()[0]
            return concept_embeddings

        embeddings = get_concept_embeddings()
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_bi_lstm_model(max_seq_length,
                                         vocab_size,
                                         embedding_size,
                                         embeddings)
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=self._get_metrics())
            return model

    def _preprocess_dataset(self, dataset):
        if isinstance(dataset, tf.data.Dataset):
            return dataset.map(lambda x, y: (x['concept_ids'], y['label']))
        elif isinstance(dataset, np.ndarray):
            raise NotImplemented('Support for numpy.ndarray is not implemented.')
        else:
            raise TypeError('Only numpy array and tensorflow Dataset are supported types.')


class VanillaBertBiLstmModelEvaluator(ModelEvaluator):

    def __init__(self,
                 max_seq_length,
                 vanilla_bert_model_path,
                 *args, **kwargs):
        super(BiLstmModelEvaluator, self).__init__(max_seq_length=max_seq_length,
                                                   vanilla_bert_model_path=vanilla_bert_model_path,
                                                   *args, **kwargs)

    def _create_model(self,
                      max_seq_length,
                      vanilla_bert_model_path) -> Model:
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_vanilla_bert_bi_lstm_model(max_seq_length,
                                                      vanilla_bert_model_path)
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=self._get_metrics())
            return model

    def _preprocess_dataset(self, dataset):
        return dataset
