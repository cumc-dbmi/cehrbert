import math
from abc import ABC, abstractmethod
from itertools import product

from scipy import stats
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

from cehrbert.config.grid_search_config import GridSearchConfig
from cehrbert.data_generators.learning_objective import post_pad_pre_truncate
from cehrbert.evaluations.model_evaluators.model_evaluators import AbstractModelEvaluator, get_metrics
from cehrbert.models.evaluation_models import create_bi_lstm_model
from cehrbert.models.loss_schedulers import CosineLRSchedule
from cehrbert.utils.model_utils import compute_binary_metrics, multimode, np, os, pd, pickle, save_training_history, tf

# Define a list of learning rates to fine-tune the model with
LEARNING_RATES = [0.5e-4, 0.8e-4, 1.0e-4, 1.2e-4]
# Define whether the LSTM is uni-directional or bi-directional
LSTM_BI_DIRECTIONS = [True]
# Define a list of LSTM units
LSTM_UNITS = [128]


class SequenceModelEvaluator(AbstractModelEvaluator, ABC):

    def __init__(
        self,
        epochs,
        batch_size,
        sequence_model_name: bool = None,
        cross_validation_test: bool = False,
        grid_search_config: GridSearchConfig = None,
        freeze_pretrained_model=False,
        multiple_test_run=False,
        *args,
        **kwargs,
    ):
        self.get_logger().info(
            f"epochs: {epochs}\n"
            f"batch_size: {batch_size}\n"
            f"sequence_model_name: {sequence_model_name}\n"
            f"cross_validation_test: {cross_validation_test}\n"
            f"grid_search_config: {grid_search_config}\n"
            f"freeze_pretrained_model: {freeze_pretrained_model}\n"
            f"multiple_test_run: {multiple_test_run}\n"
        )
        self._epochs = epochs
        self._batch_size = batch_size
        self._sequence_model_name = sequence_model_name
        self._cross_validation_test = cross_validation_test
        self._freeze_pretrained_model = freeze_pretrained_model
        self._multiple_test_run = multiple_test_run

        if grid_search_config:
            self._grid_search_config = grid_search_config
        else:
            self._grid_search_config = GridSearchConfig()
            self.get_logger().info(f"grid_search_config is None and initializing default " f"GridSearchConfig")

        # Set the GPU to memory growth to true to prevent the entire GPU memory from being
        # allocated
        try:
            [
                tf.config.experimental.set_memory_growth(device, True)
                for device in tf.config.list_physical_devices("GPU")
            ]
        except (ValueError, RuntimeError) as error:
            # Invalid device or cannot modify virtual devices once initialized.
            tf.print(error)

        super(SequenceModelEvaluator, self).__init__(*args, **kwargs)

    def train_model(
        self,
        training_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        model_name,
        **kwargs,
    ):
        """
        Training the model for the keras based sequence models.

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
            **kwargs,
        )

        save_training_history(history, self.get_model_history_folder(), model_name)
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
                    model_name=f"{self._sequence_model_name}_{i}",
                )
                compute_binary_metrics(
                    self._model,
                    test,
                    self.get_model_metrics_folder(),
                    model_name=f"{self._sequence_model_name}_{i}",
                )

    def eval_model_cross_validation_test(self):
        """
        The data is split into train_val and test partitions.

        It carries out a k-fold cross
        validation on the train_val partition first, then

        :return:
        """
        features, labels = self.extract_model_inputs()

        # Hold out 15% of the data for testing
        if self._is_chronological_test:
            training_stop = math.ceil(self._dataset.index.stop * 0.85)
            training_val_test_set_idx = np.asarray(range(training_stop))
            held_out_set_idx = np.asarray(range(training_stop, len(self._dataset)))
        else:
            stratified_splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=10)
            training_val_test_set_idx, held_out_set_idx = next(stratified_splitter.split(X=labels, y=labels))

        # Use the remaining 85% of the training data for optimizing
        training_val_test_set_inputs = {k: v[training_val_test_set_idx] for k, v in features.items()}
        training_val_test_set_labels = labels[training_val_test_set_idx]

        # Conduct a grid search to find the best combination of hyperparameters
        all_param_configs_pd = self.grid_search_cross_validation(
            features=training_val_test_set_inputs, labels=training_val_test_set_labels
        )

        # Now that we know the most optimal configurations. Let's retrain the model with the full
        # set using the most frequent number of epochs in k-fold validation. In case of multiple
        # modes, we always take the smallest mode
        optimal_hyperparam_combination = all_param_configs_pd.sort_values("roc_auc", ascending=False).iloc[0]

        self._epochs = optimal_hyperparam_combination.epoch
        self._learning_rate = optimal_hyperparam_combination.learning_rate

        with tf.device("/CPU:0"):
            # Train using the full training set
            full_training_set = (
                tf.data.Dataset.from_tensor_slices((training_val_test_set_inputs, training_val_test_set_labels))
                .cache()
                .batch(self._batch_size)
            )

        for _ in range(self._num_of_folds):
            # Recreate the model
            self._model = self._create_model(
                is_bi_directional=optimal_hyperparam_combination.is_bi_directional,
                lstm_unit=optimal_hyperparam_combination.lstm_unit,
            )

            if self._is_transfer_learning:
                size = int(len(full_training_set) * self._training_percentage)
                training_data = full_training_set.shuffle(512, seed=10).take(size)
            else:
                training_data = full_training_set

            # Retrain the model and set the epoch size to the most optimal one derived from the
            # k-fold cross validation
            self.train_model(
                training_data=training_data,
                val_data=training_data.take(10),
                model_name=f"{self._sequence_model_name}_final",
            )

            # Construct the held-out tensorflow dataset to calculate the metrics
            held_out_set_inputs = {k: v[held_out_set_idx] for k, v in features.items()}
            held_out_set_labels = labels[held_out_set_idx]

            with tf.device("/CPU:0"):
                hold_out_set = (
                    tf.data.Dataset.from_tensor_slices((held_out_set_inputs, held_out_set_labels))
                    .cache()
                    .batch(self._batch_size)
                )

            compute_binary_metrics(
                self._model,
                hold_out_set,
                self.get_model_test_metrics_folder(),
                evaluation_model_folder=self.get_model_test_prediction_folder(),
                model_name=f"{self._sequence_model_name}_final",
                calculate_ci=not self._multiple_test_run,
            )
            # If multiple test run is not enabled, we break out of the loop
            if not self._multiple_test_run:
                break

    def grid_search_cross_validation(self, features, labels):
        """
        This method conducts a grid search via cross validation to determine the best combination.

        of hyperparameters

        :param features:
        :param labels:
        :return:
        """
        all_param_configs = []
        for idx, (lr, is_bi_directional, lstm_unit) in enumerate(
            product(
                self._grid_search_config.learning_rates,
                self._grid_search_config.lstm_directions,
                self._grid_search_config.lstm_units,
            )
        ):
            # Print out the model hyperparameters
            tf.print(f"learning_rate: {lr}")
            tf.print(f"is_bi_directional: {is_bi_directional}")
            tf.print(f"lstm_unit: {lstm_unit}")

            # Remember this configuration in a dict
            param_config = {
                "learning_rate": lr,
                "is_bi_directional": is_bi_directional,
                "lstm_unit": lstm_unit,
            }
            # Update the learning rate
            self._learning_rate = lr
            # Conduct k-fold cross validation to get a sense of the model
            num_of_epochs = []
            roc_auc_scores = []

            # Run the k-fold 10 times until we discover a single mode
            max_iter = 10
            while max_iter > 0:
                for i, (train, val, test) in enumerate(self.k_fold(features=features, labels=labels)):
                    self._model = self._create_model(is_bi_directional=is_bi_directional, lstm_unit=lstm_unit)
                    history = self.train_model(
                        training_data=train,
                        val_data=val,
                        model_name=f"{self._sequence_model_name}_param_{idx}_iter_{i}",
                    )
                    # This captures the number of epochs each fold trained
                    num_of_epochs.append(len(history.history["loss"]) - 1)
                    fold_metrics = compute_binary_metrics(
                        self._model,
                        test,
                        self.get_model_metrics_folder(),
                        model_name=f"{self._sequence_model_name}_param_{idx}_iter_{i}",
                        extra_info=param_config,
                        calculate_ci=False,
                    )
                    roc_auc_scores.append(fold_metrics["roc_auc"])

                max_iter = max_iter - 1

                # If we find a single mode, we exit the loop
                if len(multimode(num_of_epochs)) == 1:
                    self.get_logger().info(
                        f"Found the best epoch for lr={lr},"
                        f" is_bi_directional={is_bi_directional}, lstm_unit={lstm_unit}"
                    )
                    break

            if max_iter == 0:
                raise RuntimeError(
                    f"Failed to find the best epoch for lr={lr},"
                    f" is_bi_directional={is_bi_directional}, lstm_unit={lstm_unit}"
                )

            # Add the number of epochs and average roc_auc to this combination
            param_config.update(
                {
                    "epoch": stats.mode(num_of_epochs).mode[0],
                    "roc_auc": np.mean(roc_auc_scores),
                }
            )

            all_param_configs.append(param_config)
        # Save all the parameter combinations to the model folder
        all_param_configs_pd = pd.DataFrame(all_param_configs)
        all_param_configs_pd.to_parquet(
            os.path.join(
                self.get_model_folder(),
                f"{self._sequence_model_name}_parameter_combinations.parquet",
            )
        )
        return all_param_configs_pd

    def k_fold(self, features, labels):
        """
        :param features:

        :param labels:
        """
        # This preserves the percentage of samples for each class (0 and 1 for binary
        # classification)
        if self._k_fold_test:
            stratified_splitter = StratifiedKFold(n_splits=self._num_of_folds, random_state=10)
        else:
            stratified_splitter = StratifiedShuffleSplit(n_splits=self._num_of_folds, test_size=0.15, random_state=10)

        for train, val_test in stratified_splitter.split(X=labels, y=labels):
            if self._k_fold_test:
                # further split val_test using a 1:1 ratio between val and test
                val, test = train_test_split(val_test, test_size=0.5, random_state=10, stratify=labels[val_test])
            else:
                test = val_test
                val = val_test

            if self._is_transfer_learning:
                size = int(len(train) * self._training_percentage)
                train = np.random.choice(train, size, replace=False)

            training_input = {k: v[train] for k, v in features.items()}
            val_input = {k: v[val] for k, v in features.items()}
            test_input = {k: v[test] for k, v in features.items()}

            tf.print(f"{self}: The train size is {len(train)}")
            tf.print(f"{self}: The val size is {len(val)}")
            tf.print(f"{self}: The test size is {len(test)}")

            with tf.device("/CPU:0"):
                training_set = (
                    tf.data.Dataset.from_tensor_slices((training_input, labels[train])).cache().batch(self._batch_size)
                )
                val_set = tf.data.Dataset.from_tensor_slices((val_input, labels[val])).cache().batch(self._batch_size)
                test_set = (
                    tf.data.Dataset.from_tensor_slices((test_input, labels[test])).cache().batch(self._batch_size)
                )

            yield training_set, val_set, test_set

    def get_model_name(self):
        return self._sequence_model_name if self._sequence_model_name else self._model.name

    def _get_callbacks(self):
        """
        Standard callbacks for the evaluations.

        :return:
        """
        learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self._learning_rate, lr_low=1e-8, initial_period=10),
            verbose=1,
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.get_model_path(),
            monitor="val_loss",
            mode="auto",
            save_best_only=True,
            verbose=1,
        )
        return [learning_rate_scheduler, early_stopping, model_checkpoint]

    @abstractmethod
    def extract_model_inputs(self):
        pass


class BiLstmModelEvaluator(SequenceModelEvaluator):

    def __init__(
        self,
        max_seq_length: int,
        time_aware_model_path: str,
        tokenizer_path: str,
        embedding_size: int,
        *args,
        **kwargs,
    ):
        self._max_seq_length = max_seq_length
        self._embedding_size = embedding_size
        self._time_aware_model_path = time_aware_model_path
        self._tokenizer = pickle.load(open(tokenizer_path, "rb"))

        self.get_logger().info(
            f"max_seq_length: {max_seq_length}\n"
            f"embedding_size: {embedding_size}\n"
            f"time_aware_model_path: {time_aware_model_path}\n"
            f"tokenizer_path: {tokenizer_path}\n"
        )

        super(BiLstmModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self, **kwargs):
        def get_concept_embeddings():
            try:
                another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
                with another_strategy.scope():
                    time_aware_model = tf.keras.models.load_model(
                        self._time_aware_model_path,
                        custom_objects=dict(**get_custom_objects()),
                    )
                    embedding_layer = time_aware_model.get_layer("embedding_layer")

                return embedding_layer.get_weights()[0]
            except (IOError, ImportError) as e:
                self.get_logger().info(f"Cannot load the time attention model, return None. Error: {e}")
                return None

        embeddings = get_concept_embeddings()
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info("Number of devices: {}".format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_bi_lstm_model(
                self._max_seq_length,
                self._tokenizer.get_vocab_size(),
                self._embedding_size,
                embeddings,
                **kwargs,
            )
            model.compile(
                loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(1e-4),
                metrics=get_metrics(),
            )
            return model

    def extract_model_inputs(self):
        token_ids = self._tokenizer.encode(self._dataset.concept_ids.apply(lambda concept_ids: concept_ids.tolist()))
        labels = self._dataset.label.to_numpy()
        padded_token_ides = post_pad_pre_truncate(
            token_ids, self._tokenizer.get_unused_token_id(), self._max_seq_length
        )
        inputs = {
            "age": np.expand_dims(self._dataset.age, axis=-1),
            "concept_ids": padded_token_ides,
        }
        return inputs, labels
