import os
import json
import pickle
import collections
from functools import partial
from typing import Sequence, Union, List, Dict, Any
from itertools import islice

import transformers
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers.tokenization_utils_base import PushToHubMixin

from models.hf_models.tokenization_utils import load_json_file, _agg_helper, map_statistics, agg_statistics
from runner.hf_runner_argument_dataclass import DataTrainingArguments
from data_generators.gpt_utils import (
    is_att_token, extract_time_interval_in_days, convert_time_interval_to_time_tuple, is_inpatient_att_token
)

START_TOKEN = "[START]"
END_TOKEN = "[END]"
PAD_TOKEN = "[PAD]"
OUT_OF_VOCABULARY_TOKEN = "[OOV]"

TOKENIZER_FILE_NAME = "cehrgpt_tokenizer.json"
TIME_TOKENIZER_FILE_NAME = "cehrgpt_time_tokenizer.json"
TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME = "token_to_sub_time_token_mapping.json"
LAB_STATS_FILE_NAME = "cehrgpt_lab_stats.json"
CONCEPT_MAPPING_FILE_NAME = "concept_name_mapping.json"


class CehrGptTokenizer(PushToHubMixin):

    def __init__(
            self,
            tokenizer: Tokenizer,
            att_tokenizer: Tokenizer,
            token_to_sub_time_token_mapping: Dict[str, List[str]],
            lab_stats: List[Dict[str, Any]],
            concept_name_mapping: Dict[str, str]
    ):
        self._tokenizer = tokenizer
        self._att_tokenizer = att_tokenizer
        self._token_to_sub_time_token_mapping = token_to_sub_time_token_mapping
        self._lab_stats = lab_stats
        self._lab_stat_mapping = {
            lab_stat["concept_id"]: {
                'unit': lab_stat["unit"],
                'mean': lab_stat['mean'],
                'std': lab_stat['std']
            } for lab_stat in lab_stats
        }
        self._concept_name_mapping = concept_name_mapping
        self._oov_token_id = self._tokenizer.token_to_id(OUT_OF_VOCABULARY_TOKEN)
        self._padding_token_id = self._tokenizer.token_to_id(PAD_TOKEN)
        self._start_token_id = self._tokenizer.token_to_id(START_TOKEN)
        self._end_token_id = self._tokenizer.token_to_id(END_TOKEN)

        super().__init__()

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    @property
    def time_token_vocab_size(self) -> int:
        return self._att_tokenizer.get_vocab_size()

    @property
    def start_token_id(self):
        return self._start_token_id

    @property
    def end_token_id(self):
        return self._end_token_id

    @property
    def pad_token_id(self):
        return self._padding_token_id

    @property
    def lab_token_ids(self):
        return self.encode([_['concept_id'] for _ in self._lab_stats])

    @property
    def token_to_time_token_mapping(self) -> Dict[int, List[int]]:
        default_mapping = {-1: [0, 0, 0]}
        default_mapping.update({
            self._tokenizer.token_to_id(time_token): list(map(self._att_tokenizer.token_to_id, sub_time_tokens))
            for time_token, sub_time_tokens in self._token_to_sub_time_token_mapping.items()
        })
        return default_mapping

    def encode(self, concept_ids: Sequence[str]) -> Sequence[int]:
        encoded = self._tokenizer.encode(concept_ids, is_pretokenized=True)
        return encoded.ids

    def decode(self, concept_token_ids: List[int]) -> List[str]:
        return self._tokenizer.decode(concept_token_ids).split(' ')

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        token_id = self._tokenizer.token_to_id(token)
        return token_id if token_id else self._oov_token_id

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._tokenizer.id_to_token(index)
        return token if token else OUT_OF_VOCABULARY_TOKEN

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join([self._concept_name_mapping[t] for t in tokens])
        return out_string

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save the Cehrbert tokenizer.


        This method make sure the batch processor can then be re-loaded using the
        .from_pretrained class method.

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`PushToHubMixin.push_to_hub`] method.
        """
        assert not os.path.isfile(save_directory), f"Provided path ({save_directory}) should be a directory, not a file"

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", str(save_directory).split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        self._tokenizer.save(os.path.join(save_directory, TOKENIZER_FILE_NAME))

        self._att_tokenizer.save(os.path.join(save_directory, TIME_TOKENIZER_FILE_NAME))

        with open(os.path.join(save_directory, TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME), "w") as f:
            json.dump(self._token_to_sub_time_token_mapping, f)

        with open(os.path.join(save_directory, LAB_STATS_FILE_NAME), "w") as f:
            json.dump(self._lab_stats, f)

        with open(os.path.join(save_directory, CONCEPT_MAPPING_FILE_NAME), "w") as f:
            json.dump(self._concept_name_mapping, f)

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
            cls,
            pretrained_model_name_or_path: Union[str, os.PathLike],
            **kwargs,
    ):
        """
        Load the CehrBert tokenizer.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                      Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                      user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing tokenization data saved using
                      [`save_pretrained`], e.g., `./my_data_directory/`.
            kwargs: Arguments for loading to pass to transformers.utils.hub.cached_file

        Returns:
            A CehrBert Tokenizer
        """

        tokenizer_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TOKENIZER_FILE_NAME, **kwargs
        )

        if not tokenizer_file:
            return None

        tokenizer = Tokenizer.from_file(tokenizer_file)

        att_tokenizer_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TIME_TOKENIZER_FILE_NAME, **kwargs
        )
        if not att_tokenizer_file:
            return None

        att_tokenizer = Tokenizer.from_file(att_tokenizer_file)

        token_to_sub_time_token_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, TOKEN_TO_SUB_TIME_TOKEN_MAPPING_FILE_NAME, **kwargs
        )
        if not token_to_sub_time_token_mapping_file:
            return None

        lab_stats_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, LAB_STATS_FILE_NAME, **kwargs
        )
        if not lab_stats_file:
            return None

        concept_name_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_MAPPING_FILE_NAME, **kwargs
        )
        if not concept_name_mapping_file:
            return None

        token_to_sub_time_token_mapping = load_json_file(token_to_sub_time_token_mapping_file)

        concept_name_mapping = load_json_file(concept_name_mapping_file)

        lab_stats = load_json_file(lab_stats_file)

        return CehrGptTokenizer(
            tokenizer,
            att_tokenizer,
            token_to_sub_time_token_mapping,
            lab_stats,
            concept_name_mapping
        )

    @classmethod
    def train_tokenizer(
            cls,
            dataset: Union[Dataset, DatasetDict],
            feature_names: List[str],
            concept_name_mapping: Dict[str, str],
            data_args: DataTrainingArguments
    ):
        """
        Train a huggingface word level tokenizer. To use their tokenizer, we need to concatenate all the concepts
        together and treat it as a sequence.
        """

        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']

        lab_stats = []
        # Use the Fast Tokenizer from the Huggingface tokenizers Rust implementation.
        # https://github.com/huggingface/tokenizers
        concept_tokenizer = Tokenizer(WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict()))
        concept_tokenizer.pre_tokenizer = WhitespaceSplit()
        concept_trainer = WordLevelTrainer(
            special_tokens=[PAD_TOKEN, OUT_OF_VOCABULARY_TOKEN, START_TOKEN, END_TOKEN],
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.min_frequency,
            show_progress=True
        )
        for feature_name in feature_names:
            batch_concat_concepts_partial_func = partial(cls.batch_concat_concepts, feature_name=feature_name)
            if data_args.streaming:
                concatenated_features = dataset.map(
                    batch_concat_concepts_partial_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size
                )

                def batched_generator():
                    iterator = iter(concatenated_features)
                    while True:
                        batch = list(islice(iterator, data_args.preprocessing_batch_size))
                        if not batch:
                            break
                        yield [
                            example[feature_name] for example in batch
                        ]

                # We pass a generator of list of texts (concatenated concept_ids) to train_from_iterator
                # for efficient training
                generator = batched_generator()
            else:
                concatenated_features = dataset.map(
                    batch_concat_concepts_partial_func,
                    num_proc=data_args.preprocessing_num_workers,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=dataset.column_names
                )
                generator = concatenated_features[feature_name]

            concept_tokenizer.train_from_iterator(generator, trainer=concept_trainer)

            if data_args.streaming:
                parts = dataset.map(
                    partial(_agg_helper, map_func=map_statistics),
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    keep_in_memory=True,
                    new_fingerprint="invalid",
                    remove_columns=dataset.column_names
                )
            else:
                parts = dataset.map(
                    partial(_agg_helper, map_func=map_statistics),
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=dataset.column_names,
                    num_proc=data_args.preprocessing_num_workers,
                    keep_in_memory=True,
                    new_fingerprint="invalid",
                )
            current = None
            for stat in parts:
                fixed_stat = pickle.loads(stat["data"])
                if current is None:
                    current = fixed_stat
                else:
                    current = agg_statistics(current, fixed_stat)
            lab_stats = [
                {
                    'concept_id': concept_id,
                    'unit': unit,
                    'mean': online_stats.mean(),
                    'std': online_stats.standard_deviation(),
                    'count': online_stats.count
                }
                for (concept_id, unit), online_stats in current['numeric_stats_by_lab'].items()
            ]

        # We will train a tokenizer specifically for time intervals
        sub_time_token_data = []
        token_to_sub_time_token_mapping = collections.defaultdict(list)
        for token, token_id in concept_tokenizer.get_vocab().items():
            if is_att_token(token):
                time_interval = extract_time_interval_in_days(token)
                time_tuple = convert_time_interval_to_time_tuple(time_interval, is_inpatient_att_token(token))
                token_to_sub_time_token_mapping[token] = list(time_tuple)
                sub_time_token_data.append(" ".join(time_tuple))

        att_tokenizer = Tokenizer(WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict()))
        att_tokenizer.pre_tokenizer = WhitespaceSplit()
        att_trainer = WordLevelTrainer(
            special_tokens=[OUT_OF_VOCABULARY_TOKEN],
            vocab_size=data_args.vocab_size,
            min_frequency=0,
            show_progress=True
        )
        att_tokenizer.train_from_iterator(sub_time_token_data, trainer=att_trainer)

        return CehrGptTokenizer(
            concept_tokenizer,
            att_tokenizer,
            token_to_sub_time_token_mapping,
            lab_stats,
            concept_name_mapping
        )

    def normalize(self, concept_id, concept_value) -> float:
        if concept_id in self._lab_stat_mapping:
            mean_ = (concept_value - self._lab_stat_mapping[concept_id]['mean'])
            std = self._lab_stat_mapping[concept_id]['std']
            if std > 0:
                normalized_value = mean_ / self._lab_stat_mapping[concept_id]['std']
            else:
                normalized_value = mean_
            return normalized_value
        return concept_value

    def denormalize(self, concept_id, value) -> float:
        if concept_id in self._lab_stat_mapping:
            value = (
                    value * self._lab_stat_mapping[concept_id]['std']
                    + self._lab_stat_mapping[concept_id]['mean']
            )
        return value

    @classmethod
    def batch_concat_concepts(cls, records: Dict[str, List], feature_name) -> Dict[str, List]:
        return {feature_name: [" ".join(map(str, _)) for _ in records[feature_name]]}
