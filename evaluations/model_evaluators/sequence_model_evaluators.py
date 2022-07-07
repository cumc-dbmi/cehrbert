from abc import ABC, abstractmethod

from statistics import mode
from sklearn.model_selection import StratifiedShuffleSplit, RepeatedStratifiedKFold, \
    train_test_split
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from data_generators.learning_objective import post_pad_pre_truncate
from evaluations.model_evaluators.model_evaluators import AbstractModelEvaluator, get_metrics
from models.evaluation_models import create_bi_lstm_model
from utils.model_utils import *
from models.loss_schedulers import CosineLRSchedule


class SequenceModelEvaluator(AbstractModelEvaluator, ABC):

    def __init__(
            self,
            epochs,
            batch_size,
            sequence_model_name: bool = None,
            cross_validation_test: bool = False,
            num_of_repeats: int = 1,
            *args, **kwargs
    ):
        self.get_logger().info(
            f'epochs: {epochs}\n'
            f'batch_size: {batch_size}\n'
            f'sequence_model_name: {sequence_model_name}\n'
            f'cross_validation_test: {cross_validation_test}\n'
            f'num_of_repeats: {num_of_repeats}\n'
        )
        self._epochs = epochs
        self._batch_size = batch_size
        self._sequence_model_name = sequence_model_name
        self._cross_validation_test = cross_validation_test
        self._num_of_repeats = num_of_repeats
        super(SequenceModelEvaluator, self).__init__(*args, **kwargs)

    def train_model(
            self,
            training_data: Dataset,
            val_data: Dataset,
            model_name,
            **kwargs
    ):
        """
        Training the model for the keras based sequence models
        :param training_data:
        :param val_data:
        :param model_name:
        :return:
        """
        history = self._model.fit(
            training_data,
            epochs=self._epochs,
            validation_data=val_data,
            callbacks=self._get_callbacks(),
            **kwargs
        )

        save_training_history(
            history,
            self.get_model_history_folder(),
            model_name
        )
        return history

    def eval_model(self):

        # If cross_validation_test is enabled, use this approach otherwise use the default k-fold
        # validations
        if self._cross_validation_test:
            self.eval_model_cross_validation_test()
        else:
            inputs, labels = self.extract_model_inputs()
            for i, (train, val, test) in enumerate(self.k_fold(features=inputs, labels=labels)):
                self._model = self._create_model()
                self.train_model(
                    training_data=train,
                    val_data=val,
                    model_name=f'{self._sequence_model_name}_{i}')
                compute_binary_metrics(
                    self._model,
                    test,
                    self.get_model_metrics_folder(),
                    model_name=f'{self._sequence_model_name}_{i}'
                )

    def eval_model_cross_validation_test(self):
        """
        The data is split into train_val and test partitions. It carries out a k-fold cross
        validation on the train_val partition first, then

        :return:
        """
        features, labels = self.extract_model_inputs()

        # Hold out 20% of the data for testing
        stratified_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.2,
            random_state=1
        )
        training_val_test_set_idx, held_out_set_idx = next(
            stratified_splitter.split(
                X=labels,
                y=labels
            )
        )

        # Use the remaining 80% of the training data for optimizing
        training_val_test_set_inputs = {
            k: v[training_val_test_set_idx]
            for k, v in features.items()
        }
        training_val_test_set_labels = labels[training_val_test_set_idx]

        # Conduct k-fold cross validation to get a sense of the model
        num_of_epochs = []
        for i, (train, val, test) in enumerate(
                self.k_fold(
                    features=training_val_test_set_inputs,
                    labels=training_val_test_set_labels
                )
        ):
            self._model = self._create_model()
            history = self.train_model(
                training_data=train,
                val_data=val,
                model_name=f'{self._sequence_model_name}_{i}'
            )
            # This captures the number of epochs each fold trained
            num_of_epochs.append(len(history.history['loss']) - 1)
            compute_binary_metrics(
                self._model,
                test,
                self.get_model_metrics_folder(),
                model_name=f'{self._sequence_model_name}_{i}'
            )

        # Now that we know the most optimal configurations. Let's retrain the model with the full
        # set using the most frequent number of epochs in k-fold validation
        self._epochs = mode(
            num_of_epochs
        )

        # Recreate the model
        self._model = self._create_model()

        # Train using the full training set
        full_training_set = tf.data.Dataset.from_tensor_slices(
            (training_val_test_set_inputs,
             training_val_test_set_labels)
        ).cache().batch(self._batch_size)

        # Retrain the model and set the epoch size to the most optimal one derived from the
        # k-fold cross validation
        self.train_model(
            training_data=full_training_set,
            val_data=full_training_set.take(10),
            model_name=f'{self._sequence_model_name}_final'
        )

        # Construct the held-out tensorflow dataset to calculate the metrics
        held_out_set_inputs = {
            k: v[held_out_set_idx]
            for k, v in features.items()
        }
        held_out_set_labels = labels[held_out_set_idx]

        hold_out_set = tf.data.Dataset.from_tensor_slices(
            (held_out_set_inputs,
             held_out_set_labels)
        ).cache().batch(self._batch_size)

        compute_binary_metrics(
            self._model,
            hold_out_set,
            self.get_model_test_metrics_folder(),
            model_name=f'{self._sequence_model_name}_final'
        )

    def k_fold(
            self,
            features,
            labels
    ):
        """

        :param features:
        :param labels:
        :param n_repeats:

        """
        # This preserves the percentage of samples for each class (0 and 1 for binary
        # classification)
        stratified_k_fold = RepeatedStratifiedKFold(
            n_splits=self._num_of_folds,
            n_repeats=self._num_of_repeats,
            random_state=1
        )

        for train, val_test in stratified_k_fold.split(
                X=labels,
                y=labels
        ):
            # further split val_test using a 1:1 ratio between val and test
            val, test = train_test_split(
                val_test,
                test_size=0.6,
                random_state=1,
                stratify=labels[val_test]
            )

            if self._is_transfer_learning:
                size = int(len(train) * self._training_percentage)
                train = np.random.choice(train, size, replace=False)

            training_input = {k: v[train] for k, v in features.items()}
            val_input = {k: v[val] for k, v in features.items()}
            test_input = {k: v[test] for k, v in features.items()}

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

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=1,
            restore_best_weights=True
        )
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_model_path(),
            monitor='val_loss', mode='auto',
            save_best_only=True, verbose=1
        )
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
