from abc import ABC, abstractmethod

from sklearn.model_selection import KFold, train_test_split
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from data_generators.learning_objective import post_pad_pre_truncate
from evaluations.model_evaluators.model_evaluators import AbstractModelEvaluator, get_metrics
from models.evaluation_models import create_bi_lstm_model
from utils.model_utils import *
from models.loss_schedulers import CosineLRSchedule


class SequenceModelEvaluator(AbstractModelEvaluator, ABC):

    def __init__(self,
                 epochs,
                 batch_size,
                 sequence_model_name=None,
                 *args, **kwargs):
        self.get_logger().info(f'epochs: {epochs}\n'
                               f'batch_size: {batch_size}\n'
                               f'sequence_model_name: {sequence_model_name}\n')
        self._epochs = epochs
        self._batch_size = batch_size
        self._sequence_model_name = sequence_model_name
        super(SequenceModelEvaluator, self).__init__(*args, **kwargs)

    def train_model(self, training_data: Dataset, val_data: Dataset, **kwargs):
        """
        Training the model for the keras based sequence models
        :param training_data:
        :param val_data:
        :return:
        """
        history = self._model.fit(
            training_data,
            epochs=self._epochs,
            validation_data=val_data,
            callbacks=self._get_callbacks(),
            **kwargs
        )
        save_training_history(history, self.get_model_history_folder())

    def eval_model(self):
        for train, val, test in self.k_fold():
            self._model = self._create_model()
            self.train_model(train, val)
            compute_binary_metrics(self._model, test, self.get_model_metrics_folder())

    def k_fold(self):
        inputs, labels = self.extract_model_inputs()
        k_fold = KFold(n_splits=self._num_of_folds, shuffle=True)

        for train, val_test in k_fold.split(labels):
            # further split val_test using a 2:3 ratio between val and test
            val, test = train_test_split(val_test, test_size=0.6)

            if self._is_transfer_learning:
                size = int(len(train) * self._training_percentage)
                train = np.random.choice(train, size, replace=False)

            training_input = {k: v[train] for k, v in inputs.items()}
            val_input = {k: v[val] for k, v in inputs.items()}
            test_input = {k: v[test] for k, v in inputs.items()}

            tf.print(f'{self}: The train size is {len(train)}')
            tf.print(f'{self}: The val size is {len(val)}')
            tf.print(f'{self}: The test size is {len(test)}')

            training_set = tf.data.Dataset.from_tensor_slices(
                (training_input, labels[train])).cache().batch(self._batch_size)
            val_set = tf.data.Dataset.from_tensor_slices(
                (val_input, labels[val])).cache().batch(self._batch_size)
            test_set = tf.data.Dataset.from_tensor_slices(
                (test_input, labels[test])).cache().batch(self._batch_size)

            yield training_set, val_set, test_set

    def get_model_name(self):
        return self._sequence_model_name if self._sequence_model_name else self._model.name

    def _get_callbacks(self):
        """
        Standard callbacks for the evaluations
        :return:
        """
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self._learning_rate, lr_low=1e-8, initial_period=10),
            verbose=1)

        # learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
        #     tf.keras.optimizers.schedules.ExponentialDecay(
        #         self._learning_rate,
        #         decay_steps=self._epochs,
        #         decay_rate=0.5,
        #         staircase=True
        #     ),
        #     verbose=1)

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=1,
                                                          restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.get_model_path(),
                                                              monitor='val_loss', mode='auto',
                                                              save_best_only=True, verbose=1)
        return [learning_rate_scheduler, early_stopping, model_checkpoint]

    @abstractmethod
    def extract_model_inputs(self):
        pass


class BiLstmModelEvaluator(SequenceModelEvaluator):

    def __init__(self,
                 max_seq_length: int,
                 time_aware_model_path: str,
                 tokenizer_path: str,
                 *args, **kwargs):
        self._max_seq_length = max_seq_length
        self._time_aware_model_path = time_aware_model_path
        self._tokenizer = pickle.load(open(tokenizer_path, 'rb'))

        self.get_logger().info(f'max_seq_length: {max_seq_length}\n'
                               f'time_aware_model_path: {time_aware_model_path}\n'
                               f'tokenizer_path: {tokenizer_path}\n')

        super(BiLstmModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self):
        def get_concept_embeddings():
            another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
            with another_strategy.scope():
                time_aware_model = tf.keras.models.load_model(self._time_aware_model_path,
                                                              custom_objects=dict(
                                                                  **get_custom_objects()))
                embedding_layer = time_aware_model.get_layer('embedding_layer')
            return embedding_layer.get_weights()[0]

        embeddings = get_concept_embeddings()
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            _, embedding_size = np.shape(embeddings)
            model = create_bi_lstm_model(self._max_seq_length,
                                         self._tokenizer.get_vocab_size(),
                                         embedding_size,
                                         embeddings)
            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=get_metrics())
            return model

    def extract_model_inputs(self):
        token_ids = self._tokenizer.encode(
            self._dataset.concept_ids.apply(lambda concept_ids: concept_ids.tolist()))
        ages = np.asarray(((self._dataset['age'] - self._dataset['age'].mean()) / self._dataset[
            'age'].std()).astype(float).apply(lambda c: [c]).tolist())
        labels = self._dataset.label
        padded_token_ides = post_pad_pre_truncate(token_ids, self._tokenizer.get_unused_token_id(),
                                                  self._max_seq_length)
        inputs = {
            'age': ages,
            'concept_ids': padded_token_ides
        }
        return inputs, labels
