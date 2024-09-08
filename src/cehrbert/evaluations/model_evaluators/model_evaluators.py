import copy
import os
import pathlib
from abc import abstractmethod

import tensorflow as tf

from cehrbert.trainers.model_trainer import AbstractModel


def get_metrics():
    """
    Standard metrics used for compiling the models.

    :return:
    """

    return [
        "binary_accuracy",
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        tf.keras.metrics.AUC(name="auc"),
    ]


class AbstractModelEvaluator(AbstractModel):
    def __init__(
        self,
        dataset,
        evaluation_folder,
        num_of_folds,
        is_transfer_learning: bool = False,
        training_percentage: float = 1.0,
        learning_rate: float = 1e-4,
        is_chronological_test: bool = False,
        k_fold_test: bool = False,
        test_person_ids=None,
        *args,
        **kwargs,
    ):
        self._dataset = copy.copy(dataset)
        self._evaluation_folder = evaluation_folder
        self._num_of_folds = num_of_folds
        self._training_percentage = min(training_percentage, 1.0)
        self._is_transfer_learning = is_transfer_learning
        self._learning_rate = learning_rate
        self._is_chronological_test = is_chronological_test
        self._k_fold_test = k_fold_test
        self._test_person_ids = test_person_ids

        if is_transfer_learning:
            extension = "transfer_learning_{:.2f}".format(self._training_percentage).replace(".", "_")
            self._evaluation_folder = os.path.join(self._evaluation_folder, extension)

        self.get_logger().info(
            f"evaluation_folder: {self._evaluation_folder}\n"
            f"num_of_folds: {self._num_of_folds}\n"
            f"is_transfer_learning {self._is_transfer_learning}\n"
            f"training_percentage: {self._training_percentage}\n"
            f"learning_rate: {self._learning_rate}\n"
            f"is_chronological_test: {is_chronological_test}\n"
            f"k_fold_test: {k_fold_test}\n"
        )

        if self._is_chronological_test:
            self.get_logger().info(f"Start sorting dataset chronologically using index date")
            self._dataset = self._dataset.sort_values("index_date").reset_index()
            self.get_logger().info(f"Finish sorting dataset chronologically")

        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_model_name(self):
        pass

    def get_model_folder(self):
        model_folder = os.path.join(self._evaluation_folder, self.get_model_name())
        if not os.path.exists(model_folder):
            self.get_logger().info(f"Create the model folder at {model_folder}")
            pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
        return model_folder

    def get_model_path(self):
        model_folder = self.get_model_folder()
        return os.path.join(model_folder, f"{self.get_model_name()}.h5")

    @abstractmethod
    def k_fold(self, features, labels):
        pass

    @abstractmethod
    def eval_model_cross_validation_test(self):
        pass
