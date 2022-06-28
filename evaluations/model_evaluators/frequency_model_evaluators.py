from abc import ABC
from itertools import chain

import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize, StandardScaler
from tensorflow.python.keras.preprocessing.text import Tokenizer
from xgboost import XGBClassifier

from evaluations.model_evaluators.model_evaluators import AbstractModelEvaluator
from utils.model_utils import compute_binary_metrics


class BaselineModelEvaluator(AbstractModelEvaluator, ABC):

    def __init__(self, *args, **kwargs):
        super(BaselineModelEvaluator, self).__init__(*args, **kwargs)

    def train_model(self, *args, **kwargs):
        pass

    def eval_model(self):
        for train, test in self.k_fold():
            x, y = train
            self._model = self._create_model()
            if isinstance(self._model, GridSearchCV):
                self._model = self._model.fit(x, y)
            else:
                self._model.fit(x, y)
            compute_binary_metrics(self._model, test, self.get_model_metrics_folder())

    def get_model_name(self):
        return type(self._model).__name__

    def k_fold(self):
        inputs, age, labels = self.extract_model_inputs()
        k_fold = KFold(n_splits=self._num_of_folds, shuffle=True, random_state=1)

        for train, val_test in k_fold.split(labels):
            # further split val_test using a 2:3 ratio between val and test
            val, test = train_test_split(val_test, test_size=0.6, random_state=1)
            train = np.concatenate([train, val])
            if self._is_transfer_learning:
                size = int(len(train) * self._training_percentage)
                train = np.random.choice(train, size, replace=False)
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
        pipe = Pipeline([('classifier', LogisticRegression())])
        # Create param grid.
        param_grid = [
            {'classifier': [LogisticRegression()],
             'classifier__penalty': ['l1', 'l2'],
             'classifier__C': np.logspace(-4, 4, 20),
             'classifier__solver': ['liblinear'],
             'classifier__max_iter': [500]
             }
        ]
        # Create grid search object
        clf = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
        return clf


class XGBClassifierEvaluator(BaselineModelEvaluator):
    def _create_model(self, *args, **kwargs):
        return XGBClassifier()