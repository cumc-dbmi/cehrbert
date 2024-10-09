import collections
import json
import os
import pickle
from functools import partial
from itertools import islice
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import transformers
from cehrbert_data.const.common import NA
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.trainers import WordLevelTrainer
from tqdm import tqdm
from transformers.tokenization_utils_base import PushToHubMixin

from cehrbert.models.hf_models.tokenization_utils import agg_helper, agg_statistics, map_statistics
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
MASK_TOKEN = "[MASK]"
UNUSED_TOKEN = "[UNUSED]"
OUT_OF_VOCABULARY_TOKEN = "[OOV]"

TOKENIZER_FILE_NAME = "tokenizer.json"
CONCEPT_MAPPING_FILE_NAME = "concept_name_mapping.json"
LAB_STATS_FILE_NAME = "cehrbert_lab_stats.json"


def load_json_file(json_file) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Loads a JSON file and returns the parsed JSON object.

    Args:
       json_file (str): The path to the JSON file.

    Returns:
       dict: The parsed JSON object.

    Raises:
       RuntimeError: If the JSON file cannot be read.
    """
    try:
        with open(json_file, "r", encoding="utf-8") as reader:
            file_contents = reader.read()
            parsed_json = json.loads(file_contents)
            return parsed_json
    except RuntimeError as e:
        raise RuntimeError(f"Can't load the json file at {json_file}") from e


def create_numeric_concept_unit_mapping(
    lab_stats: List[Dict[str, Any]]
) -> Tuple[Dict[str, List[float]], Dict[str, List[str]]]:
    numeric_concept_unit_mapping = collections.defaultdict(list)
    for each_lab_stat in lab_stats:
        numeric_concept_unit_mapping[each_lab_stat["concept_id"]].append(
            (each_lab_stat["count"], each_lab_stat["unit"])
        )

    concept_prob_mapping = dict()
    concept_unit_mapping = dict()
    for concept_id in numeric_concept_unit_mapping.keys():
        counts, units = zip(*numeric_concept_unit_mapping[concept_id])
        total_count = sum(counts)
        if total_count == 0:
            probs = [1.0]
        else:
            probs = [float(c) / total_count for c in counts]
        concept_prob_mapping[concept_id] = probs
        concept_unit_mapping[concept_id] = units
    return concept_prob_mapping, concept_unit_mapping


class NumericEventStatistics:
    def __init__(self, lab_stats: List[Dict[str, Any]]):
        self._lab_stats = lab_stats
        self._lab_stats_mapping = {
            (lab_stat["concept_id"], lab_stat["unit"]): {
                "unit": lab_stat["unit"],
                "mean": lab_stat["mean"],
                "std": lab_stat["std"],
                "value_outlier_std": lab_stat["value_outlier_std"],
                "lower_bound": lab_stat["lower_bound"],
                "upper_bound": lab_stat["upper_bound"],
            }
            for lab_stat in lab_stats
        }
        self._concept_prob_mapping, self._concept_unit_mapping = create_numeric_concept_unit_mapping(lab_stats)

    def get_numeric_concept_ids(self) -> List[str]:
        return [_["concept_id"] for _ in self._lab_stats]

    def get_random_unit(self, concept_id: str) -> str:
        if concept_id in self._concept_prob_mapping:
            unit_probs = self._concept_prob_mapping[concept_id]
            return np.random.choice(self._concept_unit_mapping[concept_id], p=unit_probs)
        return NA

    def normalize(self, concept_id: str, unit: str, concept_value: float) -> float:
        if (concept_id, unit) in self._lab_stats_mapping:
            concept_unit_stats = self._lab_stats_mapping[(concept_id, unit)]
            mean_ = concept_value - concept_unit_stats["mean"]
            std = concept_unit_stats["std"]
            if std > 0:
                value_outlier_std = concept_unit_stats["value_outlier_std"]
                normalized_value = mean_ / std
                # Clip the value between the lower and upper bounds of the corresponding lab
                normalized_value = max(-value_outlier_std, min(value_outlier_std, normalized_value))
            else:
                # If there is not a valid standard deviation,
                # we just the normalized value to the mean of the standard normal
                normalized_value = 0.0
            return normalized_value
        return concept_value

    def denormalize(self, concept_id: str, value: float) -> Tuple[float, str]:
        unit = self.get_random_unit(concept_id)
        if (concept_id, unit) in self._lab_stats_mapping:
            stats = self._lab_stats_mapping[(concept_id, unit)]
            value = value * stats["std"] + stats["mean"]
        return value, unit


class CehrBertTokenizer(PushToHubMixin):

    def __init__(
        self,
        tokenizer: Tokenizer,
        lab_stats: List[Dict[str, Any]],
        concept_name_mapping: Dict[str, str],
    ):
        self._tokenizer = tokenizer
        self._lab_stats = lab_stats
        self._numeric_event_statistics = NumericEventStatistics(lab_stats)
        self._concept_name_mapping = concept_name_mapping
        self._oov_token_index = self._tokenizer.token_to_id(OUT_OF_VOCABULARY_TOKEN)
        self._padding_token_index = self._tokenizer.token_to_id(PAD_TOKEN)
        self._mask_token_index = self._tokenizer.token_to_id(MASK_TOKEN)
        self._unused_token_index = self._tokenizer.token_to_id(UNUSED_TOKEN)
        self._cls_token_index = self._tokenizer.token_to_id(CLS_TOKEN)

        super().__init__()

    @property
    def vocab_size(self):
        return self._tokenizer.get_vocab_size()

    @property
    def oov_token_index(self):
        return self._oov_token_index

    @property
    def mask_token_index(self):
        return self._mask_token_index

    @property
    def unused_token_index(self):
        return self._unused_token_index

    @property
    def pad_token_index(self):
        return self._padding_token_index

    @property
    def cls_token_index(self):
        return self._cls_token_index

    @property
    def lab_token_ids(self):
        reserved_tokens = [
            OUT_OF_VOCABULARY_TOKEN,
            PAD_TOKEN,
            UNUSED_TOKEN,
            OUT_OF_VOCABULARY_TOKEN,
        ]
        return self.encode(
            [
                concept_id
                for concept_id in self._numeric_event_statistics.get_numeric_concept_ids()
                if concept_id not in reserved_tokens
            ]
        )

    def encode(self, concept_ids: Sequence[str]) -> Sequence[int]:
        encoded = self._tokenizer.encode(concept_ids, is_pretokenized=True)
        return encoded.ids

    def decode(self, concept_token_ids: List[int]) -> List[str]:
        return self._tokenizer.decode(concept_token_ids).split(" ")

    def convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        token_id = self._tokenizer.token_to_id(token)
        return token_id if token_id else self._oov_token_index

    def convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        token = self._tokenizer.id_to_token(index)
        return token if token else OUT_OF_VOCABULARY_TOKEN

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        out_string = " ".join([self._concept_name_mapping[t] for t in tokens])
        return out_string

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs,
    ):
        """
        Save the Cehrbert tokenizer.

        This method make sure the batch processor can then be re-loaded using the
        .from_pretrained class method.

        Args:
            save_directory (`str` or `os.PathLike`): The path to a directory where the tokenizer will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to push your model to the Hugging Face model hub after saving it. You can specify the
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
            raise RuntimeError(f"tokenizer_file does not exist: {tokenizer_file}")

        tokenizer = Tokenizer.from_file(tokenizer_file)

        lab_stats_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, LAB_STATS_FILE_NAME, **kwargs
        )
        if not lab_stats_file:
            raise RuntimeError(f"lab_stats_file does not exist: {lab_stats_file}")

        concept_name_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_MAPPING_FILE_NAME, **kwargs
        )
        if not concept_name_mapping_file:
            raise RuntimeError(f"concept_name_mapping_file does not exist: {concept_name_mapping_file}")

        lab_stats = load_json_file(lab_stats_file)

        concept_name_mapping = load_json_file(concept_name_mapping_file)

        return CehrBertTokenizer(tokenizer, lab_stats, concept_name_mapping)

    @classmethod
    def train_tokenizer(
        cls,
        dataset: Union[Dataset, DatasetDict],
        feature_names: List[str],
        concept_name_mapping: Dict[str, str],
        data_args: DataTrainingArguments,
    ):
        """
        Train a huggingface word level tokenizer.

        To use their tokenizer, we need to concatenate all the concepts
        together and treat it as a sequence.
        """

        if isinstance(dataset, DatasetDict):
            dataset = dataset["train"]

        # Use the Fast Tokenizer from the Huggingface tokenizers Rust implementation.
        # https://github.com/huggingface/tokenizers
        tokenizer = Tokenizer(WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict()))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = WordLevelTrainer(
            special_tokens=[
                PAD_TOKEN,
                MASK_TOKEN,
                OUT_OF_VOCABULARY_TOKEN,
                CLS_TOKEN,
                UNUSED_TOKEN,
            ],
            vocab_size=data_args.vocab_size,
            min_frequency=data_args.min_frequency,
            show_progress=True,
        )
        for feature_name in feature_names:
            batch_concat_concepts_partial_func = partial(cls.batch_concat_concepts, feature_name=feature_name)
            if data_args.streaming:
                concatenated_features = dataset.map(
                    batch_concat_concepts_partial_func,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                )

                def batched_generator():
                    iterator = iter(concatenated_features)
                    while True:
                        batch = list(islice(iterator, data_args.preprocessing_batch_size))
                        if not batch:
                            break
                        yield [example[feature_name] for example in batch]

                # We pass a generator of list of texts (concatenated concept_ids) to train_from_iterator
                # for efficient training
                generator = batched_generator()
            else:
                concatenated_features = dataset.map(
                    batch_concat_concepts_partial_func,
                    num_proc=data_args.preprocessing_num_workers,
                    batched=True,
                    batch_size=data_args.preprocessing_batch_size,
                    remove_columns=dataset.column_names,
                )
                generator = concatenated_features[feature_name]

            tokenizer.train_from_iterator(generator, trainer=trainer)

        map_statistics_partial = partial(
            map_statistics,
            capacity=data_args.offline_stats_capacity,
            value_outlier_std=data_args.value_outlier_std,
        )

        if data_args.streaming:
            parts = dataset.map(
                partial(agg_helper, map_func=map_statistics_partial),
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=dataset.column_names,
            )
        else:
            parts = dataset.map(
                partial(agg_helper, map_func=map_statistics_partial),
                batched=True,
                batch_size=data_args.preprocessing_batch_size,
                remove_columns=dataset.column_names,
                num_proc=data_args.preprocessing_num_workers,
                keep_in_memory=True,
                new_fingerprint="invalid",
            )
        current = None
        for stat in tqdm(parts, desc="Aggregating the lab statistics"):
            fixed_stat = pickle.loads(stat["data"])
            if current is None:
                current = fixed_stat
            else:
                current = agg_statistics(current, fixed_stat)

        lab_stats = [
            {
                "concept_id": concept_id,
                "unit": unit,
                "mean": online_stats.mean(),
                "std": online_stats.standard_deviation(),
                "count": online_stats.count,
                "value_outlier_std": data_args.value_outlier_std,
                "lower_bound": online_stats.mean() - data_args.value_outlier_std * online_stats.standard_deviation(),
                "upper_bound": online_stats.mean() + data_args.value_outlier_std * online_stats.standard_deviation(),
            }
            for (concept_id, unit), online_stats in current["numeric_stats_by_lab"].items()
        ]

        return CehrBertTokenizer(tokenizer, lab_stats, concept_name_mapping)

    @classmethod
    def batch_concat_concepts(cls, records: Dict[str, List], feature_name) -> Dict[str, List]:
        return {feature_name: [" ".join(map(str, _)) for _ in records[feature_name]]}

    def normalize(self, concept_id: str, unit: str, concept_value: float) -> float:
        return self._numeric_event_statistics.normalize(concept_id, unit, concept_value)
