import os
import json
from tqdm import tqdm
import collections
import transformers
from transformers.tokenization_utils_base import PushToHubMixin
from typing import Sequence, Union, List, Dict
from datasets import Dataset, DatasetDict

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
MASK_TOKEN = "[MASK]"
UNUSED_TOKEN = '[UNUSED]'
OUT_OF_VOCABULARY_TOKEN = "[OOV]"
VS_TOKEN = "[VS]"
VE_TOKEN = "[VE]"

VOCAB_FILE_NAME = "vocab.json"
CONCEPT_MAPPING_FILE_NAME = "concept_name_mapping.json"


def load_json_file(
        json_file
):
    try:
        with open(json_file, "r", encoding="utf-8") as reader:
            file_contents = reader.read()
            parsed_json = json.loads(file_contents)
            return parsed_json
    except Exception as e:
        raise RuntimeError(f"Can't load the json file at {json_file} due to {e}")


class CehrBertTokenizer(PushToHubMixin):

    def __init__(
            self,
            word_index: Dict[str, int],
            concept_name_mapping: Dict[str, str]
    ):
        self._word_index = word_index
        self._concept_name_mapping = concept_name_mapping
        self._index_word = collections.OrderedDict([(ids, tok) for tok, ids in self._word_index.items()])
        self._oov_token_index = self._word_index[OUT_OF_VOCABULARY_TOKEN]
        self._padding_token_index = self._word_index[PAD_TOKEN]
        self._mask_token_index = self._word_index[MASK_TOKEN]
        self._unused_token_index = self._word_index[UNUSED_TOKEN]

        super().__init__()

    @property
    def vocab_size(self):
        return len(self._word_index)

    @property
    def mask_token_index(self):
        return self._mask_token_index

    @property
    def unused_token_index(self):
        return self._unused_token_index

    @property
    def pad_token_index(self):
        return self._padding_token_index

    def encode(self, concept_ids: Sequence[str]) -> Sequence[int]:
        return list(map(self._convert_token_to_id, concept_ids))

    def decode(self, concept_token_ids: List[int]) -> List[str]:
        return list(map(self._convert_id_to_token, concept_token_ids))

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self._word_index.get(token, self._oov_token_index)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self._index_word.get(index, OUT_OF_VOCABULARY_TOKEN)

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

        with open(os.path.join(save_directory, VOCAB_FILE_NAME), "w") as f:
            json.dump(self._word_index, f)

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

        vocab_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, VOCAB_FILE_NAME, **kwargs
        )

        if not vocab_file:
            return None

        word_index = load_json_file(vocab_file)

        concept_name_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_MAPPING_FILE_NAME, **kwargs
        )
        if not concept_name_mapping_file:
            return None

        concept_name_mapping = load_json_file(concept_name_mapping_file)

        return CehrBertTokenizer(word_index, concept_name_mapping)

    @classmethod
    def train_tokenizer(
            cls,
            dataset: Dataset,
            feature_names: List[str],
            concept_name_mapping: Dict[str, str]
    ):

        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']

        words = set()
        for record in tqdm(dataset, total=len(dataset)):
            for feature_name in feature_names:
                for concept_id in record[feature_name]:
                    words.add(concept_id)

        vocabulary = [PAD_TOKEN, MASK_TOKEN, OUT_OF_VOCABULARY_TOKEN, CLS_TOKEN, UNUSED_TOKEN, VS_TOKEN, VE_TOKEN]
        vocabulary.extend(words)
        word_index = collections.OrderedDict(zip(vocabulary, list(range(0, len(vocabulary)))))
        return CehrBertTokenizer(word_index, concept_name_mapping)
