from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_folder: Optional[str] = field(
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_prepared_path: Optional[str] = field(
        metadata={"help": "The folder in which the prepared dataset is cached"}
    )
    test_data_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the test dataset to use (via the datasets library)."}
    )
    validation_split_percentage: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    test_eval_ratio: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name_or_path: Optional[str] = field(
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        }
    )
    num_hidden_layers: Optional[int] = field(
        default=12,
        metadata={"help": "The number of layers used in the transformer model"}
    )
    max_position_embeddings: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum length of the sequence allowed for the transformer model"}
    )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
