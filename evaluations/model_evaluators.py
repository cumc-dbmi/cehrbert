import copy
from abc import abstractmethod
from typing import List, Tuple, Union

from tensorflow.data import Dataset

from models.evaluation_models import *
from trainers.model_trainer import AbstractModel
from utils.model_utils import *


def get_metrics():
    """
    Standard metrics used for compiling the models
    :return:
    """
    return ['binary_accuracy',
            tf.keras.metrics.Recall(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.AUC(curve='PR')]


class ModelEvaluator(AbstractModel):
    def __init__(self,
                 evaluation_folder,
                 epochs,
                 batch_size,
                 *args, **kwargs):

        self._evaluation_folder = evaluation_folder
        self._epochs = epochs
        self._batch_size = batch_size

        super().__init__(*args, **kwargs)

    def evaluate(self, datasets: List[Tuple[Union[Dataset, np.ndarray], int]]):

        if len(datasets) == 3:
            # This is the scenario of training, val, and test split
            training, val, test = datasets

        elif len(datasets) == 2:
            # This is the scenario of training, val, and test split
            training, val = datasets
            val_data, val_size = val
            test = (copy.copy(val_data), copy.copy(val_size))
        else:
            raise AssertionError('The number of datasets can be either 2 or 3')

        self.train_model(training, val)
        self.eval_model(test)

    def train_model(self, training: Tuple[Union[Dataset, np.ndarray], int],
                    val: Tuple[Union[Dataset, np.ndarray], int]):

        training_data, train_size = training
        val_data, val_size = val

        training_data = self._preprocess_dataset(training_data)
        val_data = self._preprocess_dataset(val_data)

        history = self._model.fit(
            training_data,
            steps_per_epoch=train_size // self._batch_size,
            epochs=self._epochs,
            validation_data=val_data,
            validation_steps=val_size // self._batch_size,
            callbacks=self._get_callbacks()
        )

        save_training_history(history, self.get_model_history_folder())

    def eval_model(self, test: Tuple[Union[Dataset, np.ndarray], int]):

        test_data, test_size = test
        test_data = self._preprocess_dataset(test_data)
        compute_binary_metrics(self._model, test_data, test_size, self._batch_size,
                               self.get_model_metrics_folder())

    @abstractmethod
    def _preprocess_dataset(self, dataset):
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

    def get_model_name(self):
        return self._model.name

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


class BiLstmModelEvaluator(ModelEvaluator):

    def __init__(self,
                 max_seq_length,
                 vocab_size,
                 embedding_size,
                 time_aware_model_path=None,
                 *args, **kwargs):

        self._max_seq_length = max_seq_length
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._time_aware_model_path = time_aware_model_path

        super(BiLstmModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self) -> Model:

        def get_concept_embeddings():
            concept_embeddings = None
            if self._time_aware_model_path:
                another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
                with another_strategy.scope():
                    time_aware_model = tf.keras.models.load_model(self._time_aware_model_path,
                                                                  custom_objects=dict(
                                                                      **get_custom_objects()))
                    concept_embeddings = time_aware_model.get_layer('embedding_layer') \
                        .get_weights()[0]
            return concept_embeddings

        embeddings = get_concept_embeddings()
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_bi_lstm_model(self._max_seq_length,
                                         self._vocab_size,
                                         self._embedding_size,
                                         embeddings)
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=get_metrics())
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
        self._max_seq_length = max_seq_length
        self._vanilla_bert_model_path = vanilla_bert_model_path

        super(VanillaBertBiLstmModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self) -> Model:
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_vanilla_bert_bi_lstm_model(self._max_seq_length,
                                                      self._vanilla_bert_model_path)
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=get_metrics())
            return model

    def _preprocess_dataset(self, dataset):
        return dataset
