import copy
from abc import abstractmethod

from trainers.model_trainer import AbstractModel
from utils.model_utils import *


def get_metrics():
    """
    Standard metrics used for compiling the models
    :return:
    """

    return ['binary_accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(curve='PR', name='pr_auc'),
            tf.keras.metrics.AUC(name='auc')]


class AbstractModelEvaluator(AbstractModel):
    def __init__(self,
                 dataset,
                 evaluation_folder,
                 num_of_folds,
                 is_transfer_learning: bool = False,
                 training_percentage: float = 1.0,
                 learning_rate: float = 1e-4,
                 *args, **kwargs):
        self._dataset = copy.copy(dataset)
        self._evaluation_folder = evaluation_folder
        self._num_of_folds = num_of_folds
        self._training_percentage = min(training_percentage, 1.0)
        self._is_transfer_learning = is_transfer_learning
        self._learning_rate = learning_rate

        if is_transfer_learning:
            extension = 'transfer_learning_{:.2f}'.format(self._training_percentage).replace('.',
                                                                                             '_')
            self._evaluation_folder = os.path.join(self._evaluation_folder, extension)

        self.get_logger().info(f'evaluation_folder: {self._evaluation_folder}\n'
                               f'num_of_folds: {self._num_of_folds}\n'
                               f'is_transfer_learning {self._is_transfer_learning}\n'
                               f'training_percentage: {self._training_percentage}\n')

        super().__init__(*args, **kwargs)

    @abstractmethod
    def get_model_name(self):
        pass

    def get_model_folder(self):
        model_folder = os.path.join(self._evaluation_folder, self.get_model_name())
        if not os.path.exists(model_folder):
            self.get_logger().info(f'Create the model folder at {model_folder}')
            pathlib.Path(model_folder).mkdir(parents=True, exist_ok=True)
        return model_folder

    def get_model_path(self):
        model_folder = self.get_model_folder()
        return os.path.join(model_folder, f'{self.get_model_name()}.h5')

    @abstractmethod
    def k_fold(self):
        pass
