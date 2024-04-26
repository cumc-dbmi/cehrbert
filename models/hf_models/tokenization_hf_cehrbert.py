import os
import json
import transformers
from typing import Sequence, Union, List, Dict
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers.tokenization_utils_base import PushToHubMixin

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
MASK_TOKEN = "[MASK]"
UNUSED_TOKEN = '[UNUSED]'
OUT_OF_VOCABULARY_TOKEN = "[OOV]"

TOKENIZER_FILE_NAME = "tokenizer.json"
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
            tokenizer: Tokenizer,
            concept_name_mapping: Dict[str, str]
    ):
        self._tokenizer = tokenizer
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

    def encode(self, concept_ids: Sequence[str]) -> Sequence[int]:
        encoded = self._tokenizer.encode(concept_ids, is_pretokenized=True)
        return encoded.ids

    def decode(self, concept_token_ids: List[int]) -> List[str]:
        return self._tokenizer.decode(concept_token_ids).split(' ')

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        token_id = self._tokenizer.token_to_id(token)
        return token_id if token_id else self._oov_token_index

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

        concept_name_mapping_file = transformers.utils.hub.cached_file(
            pretrained_model_name_or_path, CONCEPT_MAPPING_FILE_NAME, **kwargs
        )
        if not concept_name_mapping_file:
            return None

        concept_name_mapping = load_json_file(concept_name_mapping_file)

        return CehrBertTokenizer(tokenizer, concept_name_mapping)

    @classmethod
    def train_tokenizer(
            cls,
            dataset: Union[Dataset, DatasetDict],
            feature_names: List[str],
            concept_name_mapping: Dict[str, str],
            num_proc: int = 16,
            vocab_size: int = 50_000
    ):
        if isinstance(dataset, DatasetDict):
            dataset = dataset['train']

        # Use the Fast Tokenizer from the Huggingface tokenizers Rust implementation.
        # https://github.com/huggingface/tokenizers
        tokenizer = Tokenizer(WordLevel(unk_token=OUT_OF_VOCABULARY_TOKEN, vocab=dict()))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        trainer = WordLevelTrainer(
            special_tokens=[PAD_TOKEN, MASK_TOKEN, OUT_OF_VOCABULARY_TOKEN, CLS_TOKEN, UNUSED_TOKEN],
            vocab_size=vocab_size
        )
        for feature in feature_names:
            concatenated_features = dataset.map(
                lambda x: {feature: ' '.join(map(str, x[feature]))}, num_proc=num_proc,
                remove_columns=dataset.column_names
            )
            tokenizer.train_from_iterator(concatenated_features[feature], trainer=trainer)
        return CehrBertTokenizer(tokenizer, concept_name_mapping)
