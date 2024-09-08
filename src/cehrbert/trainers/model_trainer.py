import copy
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import tensorflow as tf

from cehrbert.data_generators.data_generator_base import AbstractDataGeneratorBase
from cehrbert.models.layers.custom_layers import get_custom_objects
from cehrbert.models.loss_schedulers import CosineLRSchedule
from cehrbert.utils.checkpoint_utils import MODEL_CONFIG_FILE, get_checkpoint_epoch
from cehrbert.utils.logging_utils import logging
from cehrbert.utils.model_utils import create_folder_if_not_exist, log_function_decorator, save_training_history


class AbstractModel(ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._model = self._create_model(*args, **kwargs)

    @abstractmethod
    def _create_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def train_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_model_folder(self):
        pass

    def get_model_metrics_folder(self):
        return create_folder_if_not_exist(self.get_model_folder(), "metrics")

    def get_model_test_metrics_folder(self):
        return create_folder_if_not_exist(self.get_model_folder(), "test_metrics")

    def get_model_test_prediction_folder(self):
        return create_folder_if_not_exist(self.get_model_folder(), "test_prediction")

    def get_model_history_folder(self):
        return create_folder_if_not_exist(self.get_model_folder(), "history")

    @classmethod
    def get_logger(cls):
        return logging.getLogger(cls.__name__)

    def __str__(self):
        return str(self.__class__.__name__)


class AbstractConceptEmbeddingTrainer(AbstractModel):
    min_num_of_concepts = 5

    def __init__(
        self,
        training_data_parquet_path: str,
        model_folder: str,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        checkpoint_name: str = None,
        val_data_parquet_path: str = None,
        tf_board_log_path: str = None,
        shuffle_training_data: bool = True,
        cache_dataset: bool = False,
        use_dask: bool = False,
        save_checkpoint: bool = False,
        save_freq: int = 0,
        shuffle_records: bool = False,
        *args,
        **kwargs,
    ):

        self._training_data_parquet_path = training_data_parquet_path
        self._val_data_parquet_path = val_data_parquet_path
        self._checkpoint_name = checkpoint_name
        self._tf_board_log_path = tf_board_log_path
        self._batch_size = batch_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._model_folder = model_folder
        self._shuffle_training_data = shuffle_training_data
        self._cache_dataset = cache_dataset
        self._use_dask = use_dask
        self._save_checkpoint = save_checkpoint
        self._save_freq = save_freq
        self._training_data = self._load_data(self._training_data_parquet_path)
        self._current_epoch = 0
        self._shuffle_records = shuffle_records

        if self._val_data_parquet_path:
            self._val_data = self._load_data(self._val_data_parquet_path)
        else:
            self._val_data = None

        # shuffle the training data
        if self._shuffle_training_data and not self._use_dask:
            self._training_data = self._training_data.sample(frac=1).reset_index(drop=True)

        self._load_dependencies()

        super(AbstractConceptEmbeddingTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f"training_data_parquet_path: {training_data_parquet_path}\n"
            f"val_data_parquet_path: {val_data_parquet_path}\n"
            f"batch_size: {batch_size}\n"
            f"epochs: {epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"model_folder: {model_folder}\n"
            f"checkpoint_name: {checkpoint_name}\n"
            f"tf_board_log_path: {tf_board_log_path}\n"
            f"shuffle_training_data: {shuffle_training_data}\n"
            f"cache_dataset: {cache_dataset}\n"
            f"use_dask: {use_dask}\n"
            f"save_checkpoint: {save_checkpoint}\n"
            f"save_freq: {save_freq}\n"
            f"shuffle_records: {shuffle_records}\n"
        )

        self.get_logger().info("Saving the model configuration")
        self.save_model_config()

    @abstractmethod
    def _load_dependencies(self):
        pass

    @log_function_decorator
    def _load_data(self, data_parquet_path):
        if not os.path.exists(data_parquet_path):
            raise FileExistsError(f"{data_parquet_path} does not exist!")

        if self._use_dask:
            return dd.read_parquet(data_parquet_path)
        else:
            return pd.read_parquet(data_parquet_path)

    @abstractmethod
    def create_data_generator(self) -> AbstractDataGeneratorBase:
        """
        Prepare _training_data for the model such as tokenize concepts.

        :return:
        """

    def create_val_data_generator(self) -> AbstractDataGeneratorBase:
        """
        Prepare _training_data for the model such as tokenize concepts.

        :return:
        """
        return None

    def train_model(self):
        """
        Train the model and save the history metrics into the model folder.

        :return:
        """
        data_generator = self.create_data_generator()
        steps_per_epoch = data_generator.get_steps_per_epoch()
        dataset = tf.data.Dataset.from_generator(
            data_generator.create_batch_generator,
            output_types=(data_generator.get_tf_dataset_schema()),
        ).prefetch(tf.data.experimental.AUTOTUNE)

        if self._cache_dataset:
            dataset = dataset.take(data_generator.get_steps_per_epoch()).cache().repeat()
            dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = None
        val_steps_per_epoch = None
        val_data_generator = self.create_val_data_generator()
        if val_data_generator:
            val_steps_per_epoch = val_data_generator.get_steps_per_epoch()
            val_dataset = tf.data.Dataset.from_generator(
                val_data_generator.create_batch_generator,
                output_types=(val_data_generator.get_tf_dataset_schema()),
            ).prefetch(tf.data.experimental.AUTOTUNE)

        history = self._model.fit(
            x=dataset,
            validation_data=val_dataset,
            validation_steps=val_steps_per_epoch,
            steps_per_epoch=steps_per_epoch,
            epochs=self._epochs,
            callbacks=self._get_callbacks(),
            validation_freq=1 if val_dataset is not None else None,
            initial_epoch=self._current_epoch,
            use_multiprocessing=True,
        )

        save_training_history(history, self.get_model_history_folder())

    def restore_from_checkpoint(self):
        existing_model_path = os.path.join(self.get_model_folder(), self._checkpoint_name)
        current_epoch = get_checkpoint_epoch(existing_model_path)
        self._current_epoch = current_epoch
        self._epochs += current_epoch
        self.get_logger().info(f"The {self} model will be loaded from {existing_model_path}")
        model = tf.keras.models.load_model(existing_model_path, custom_objects=get_custom_objects())
        return model

    def _get_callbacks(self):
        tensor_board_callback = tf.keras.callbacks.TensorBoard(log_dir=self._tf_board_log_path)

        model_checkpoint_args = {
            "filepath": self.get_model_path_epoch(),
            "save_best_only": True,
            "monitor": "loss",
            "verbose": 1,
        }
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(**model_checkpoint_args)
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self._learning_rate, lr_low=1e-8, initial_period=10),
            verbose=1,
        )

        callbacks = [tensor_board_callback, model_checkpoint, learning_rate_scheduler]

        # Additional step-based checkpoint callback
        if self._save_checkpoint:

            def on_epoch_begin(self, epoch, logs=None):
                self._current_epoch = epoch
                self._last_batch_seen = -1
                self._batches_seen_since_last_saving = 0

            frequency_checkpoint_args = copy.deepcopy(model_checkpoint_args)
            frequency_checkpoint_args["filepath"] = self.get_model_path_step()
            frequency_checkpoint_args["save_freq"] = self._save_freq
            frequency_checkpoint_args["name"] = " "
            # Monkey patch the on_epoch_begin in ModelCheckpoint because we need to clear out _last_batch_seen and
            # _batches_seen_since_last_saving So the batch number in the model checkpoints created is a multiple of
            # save_freq
            frequencyModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
            frequencyModelCheckpoint.on_epoch_begin = on_epoch_begin
            callbacks.append(frequencyModelCheckpoint(**frequency_checkpoint_args))
        return callbacks

    def get_model_folder(self):
        """
        Infer the model folder from the property model_path.

        :return:
        """
        return str(Path(self._model_folder))

    def get_model_path_epoch(self):
        model_name = f"{self.get_model_name()}" + "_epoch_{epoch:02d}_batch_final.h5"
        return os.path.join(self.get_model_folder(), model_name)

    def get_model_path_step(self):
        model_name = f"{self.get_model_name()}" + "_epoch_{epoch:02d}_batch_{batch:02d}.h5"
        return os.path.join(self.get_model_folder(), model_name)

    def get_tokenizer_name(self):
        return f"{self.get_model_name()}_tokenizer.pickle"

    def get_tokenizer_path(self):
        return os.path.join(self.get_model_folder(), self.get_tokenizer_name())

    def get_visit_tokenizer_name(self):
        return f"{self.get_model_name()}_visit_tokenizer.pickle"

    def get_visit_tokenizer_path(self):
        return os.path.join(self.get_model_folder(), self.get_visit_tokenizer_name())

    def checkpoint_exists(self):
        if self._checkpoint_name:
            existing_model_path = os.path.join(self.get_model_folder(), self._checkpoint_name)
            return os.path.exists(existing_model_path)
        return False

    def get_model_config(self):
        def remove_first_underscore(name):
            if name[0] == "_":
                return name[1:]
            return name

        model_config = {
            remove_first_underscore(k): v
            for k, v in self.__dict__.items()
            if type(v) in (int, float, str, bool, type(None))
        }
        model_config.update(
            {
                "model_name": self.get_model_name(),
                "tokenizer": self.get_tokenizer_name(),
            }
        )
        return model_config

    def save_model_config(self):
        model_config = self.get_model_config()
        model_config_path = os.path.join(self.get_model_folder(), MODEL_CONFIG_FILE)
        if not os.path.exists(model_config_path):
            with open(model_config_path, "w") as f:
                f.write(json.dumps(model_config))

    @abstractmethod
    def get_model_name(self):
        pass
