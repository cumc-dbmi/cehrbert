import copy
from abc import abstractmethod, ABC
from sklearn.model_selection import KFold

from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import normalize, StandardScaler
from tensorflow.python.keras.preprocessing.text import Tokenizer

from models.evaluation_models import *
from trainers.model_trainer import AbstractModel
from utils.model_utils import *
from data_generators.learning_objective import post_pad_pre_truncate


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
                 *args, **kwargs):
        self._dataset = copy.copy(dataset)
        self._evaluation_folder = evaluation_folder
        self._num_of_folds = num_of_folds

        self.get_logger().info(f'evaluation_folder: {evaluation_folder}\n'
                               f'num_of_folds: {num_of_folds}\n')

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


class SequenceModelEvaluator(AbstractModelEvaluator, ABC):

    def __init__(self,
                 epochs,
                 batch_size,
                 *args, **kwargs):
        self.get_logger().info(f'epochs: {epochs}\n'
                               f'batch_size: {batch_size}\n')
        self._epochs = epochs
        self._batch_size = batch_size
        super(SequenceModelEvaluator, self).__init__(*args, **kwargs)

    def train_model(self, training_data: Dataset, val_data: Dataset):
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
            callbacks=self._get_callbacks()
        )
        save_training_history(history, self.get_model_history_folder())

    def eval_model(self):
        for train, test in self.k_fold():
            self._model = self._create_model()
            self.train_model(train, test)
            compute_binary_metrics(self._model, test, self.get_model_metrics_folder())

    def k_fold(self):
        inputs, labels = self.extract_model_inputs()
        k_fold = KFold(n_splits=self._num_of_folds, shuffle=True)
        for train, test in k_fold.split(self._dataset):
            training_input = {k: v[train] for k, v in inputs.items()}
            val_input = {k: v[test] for k, v in inputs.items()}
            train = tf.data.Dataset.from_tensor_slices((training_input, labels[train])) \
                .cache().batch(self._batch_size)
            test = tf.data.Dataset.from_tensor_slices((val_input, labels[test])) \
                .cache().batch(self._batch_size)
            yield train, test

    def get_model_name(self):
        return self._model.name

    def _get_callbacks(self):
        """
        Standard callbacks for the evaluations
        :return:
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1)
        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=self.get_model_path(),
                                                              monitor='val_loss', mode='auto',
                                                              save_best_only=True, verbose=1)
        return [early_stopping, model_checkpoint]

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


class BertLstmModelEvaluator(SequenceModelEvaluator):

    def __init__(self,
                 max_seq_length: str,
                 bert_model_path: str,
                 tokenizer_path: str,
                 is_temporal: bool = True,
                 *args, **kwargs):
        self._max_seq_length = max_seq_length
        self._bert_model_path = bert_model_path
        self._tokenizer = pickle.load(open(tokenizer_path, 'rb'))
        self._is_temporal = is_temporal

        self.get_logger().info(f'max_seq_length: {max_seq_length}\n'
                               f'vanilla_bert_model_path: {bert_model_path}\n'
                               f'tokenizer_path: {tokenizer_path}\n'
                               f'is_temporal: {is_temporal}\n')

        super(BertLstmModelEvaluator, self).__init__(*args, **kwargs)

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        self.get_logger().info('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            create_model_fn = (create_temporal_bert_bi_lstm_model if self._is_temporal
                               else create_vanilla_bert_bi_lstm_model)
            try:
                model = create_model_fn(self._max_seq_length, self._bert_model_path)
            except ValueError as e:
                self.get_logger().exception(e)
                model = create_model_fn(self._max_seq_length, self._bert_model_path)

            model.compile(loss='binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4),
                          metrics=get_metrics())
            return model

    def extract_model_inputs(self):
        token_ids = self._tokenizer.encode(
            self._dataset.concept_ids.apply(lambda concept_ids: concept_ids.tolist()))
        visit_segments = self._dataset.visit_segments
        ages = np.asarray(((self._dataset['age'] - self._dataset['age'].mean()) / self._dataset[
            'age'].std()).astype(float).apply(lambda c: [c]).tolist())
        labels = self._dataset.label
        padded_token_ides = post_pad_pre_truncate(token_ids, self._tokenizer.get_unused_token_id(),
                                                  self._max_seq_length)
        padded_visit_segments = post_pad_pre_truncate(visit_segments, 0, self._max_seq_length)
        mask = (padded_token_ides == self._tokenizer.get_unused_token_id()).astype(int)

        inputs = {
            'age': ages,
            'concept_ids': padded_token_ides,
            'masked_concept_ids': padded_token_ides,
            'mask': mask,
            'visit_segments': padded_visit_segments
        }

        if self._is_temporal:
            inputs['time_stamps'] = post_pad_pre_truncate(self._dataset.dates, 0,
                                                          self._max_seq_length)
        return inputs, labels


class BaselineModelEvaluator(AbstractModelEvaluator, ABC):

    def __init__(self, *args, **kwargs):
        super(BaselineModelEvaluator, self).__init__(*args, **kwargs)

    def train_model(self, *args, **kwargs):
        pass

    def eval_model(self):
        for train, test in self.k_fold():
            x, y = train
            self._model = self._create_model()
            self._model.fit(x, y)
            compute_binary_metrics(self._model, test, self.get_model_metrics_folder())

    def get_model_name(self):
        return type(self._model).__name__

    def k_fold(self):
        inputs, age, labels = self.extract_model_inputs()
        k_fold = KFold(n_splits=self._num_of_folds, shuffle=True)
        for train, test in k_fold.split(self._dataset):
            train_data = (csr_matrix(hstack([inputs[train], age[train]])), labels[train])
            test_data = (csr_matrix(hstack([inputs[test], age[test]])), labels[test])
            yield train_data, test_data

    def extract_model_inputs(self):
        # Load the training data
        self._dataset.concept_ids = self._dataset.concept_ids.apply(
            lambda concept_ids: concept_ids.tolist())
        self._dataset.race_concept_id = self._dataset.race_concept_id.astype(str)
        self._dataset.gender_concept_id = self._dataset.gender_concept_id.astype(str)

        # Tokenize the concepts
        tokenizer = Tokenizer(filters='', lower=False)
        tokenizer.fit_on_texts(self._dataset['concept_ids'])
        self._dataset['token_ids'] = tokenizer.texts_to_sequences(self._dataset['concept_ids'])

        # Create the row index
        dataset = self._dataset.reset_index().reset_index()
        dataset['row_index'] = dataset[['token_ids', 'level_0']].apply(
            lambda tup: [tup[1]] * len(tup[0]), axis=1)

        row_index = list(chain(*dataset['row_index'].tolist()))
        col_index = list(chain(*dataset['token_ids'].tolist()))
        values = list(chain(*dataset['frequencies'].tolist()))

        data_size = len(dataset)
        vocab_size = len(tokenizer.word_index) + 1
        row_index, col_index, values = zip(
            *sorted(zip(row_index, col_index, values), key=lambda tup: (tup[0], tup[1])))

        concept_freq_count = csr_matrix((values, (row_index, col_index)),
                                        shape=(data_size, vocab_size))
        normalized_concept_freq_count = normalize(concept_freq_count)

        # one_hot_gender_race = OneHotEncoder(handle_unknown='ignore') \
        #     .fit_transform(dataset[['gender_concept_id', 'race_concept_id']].to_numpy())
        scaled_age = StandardScaler().fit_transform(dataset[['age']].to_numpy())

        y = dataset['label'].to_numpy()

        return normalized_concept_freq_count, scaled_age, y


class LogisticRegressionModelEvaluator(BaselineModelEvaluator):
    def _create_model(self, *args, **kwargs):
        return LogisticRegression(random_state=0, n_jobs=20, verbose=1)


class XGBClassifierEvaluator(BaselineModelEvaluator):
    def _create_model(self, *args, **kwargs):
        return XGBClassifier()
