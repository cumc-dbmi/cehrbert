from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, Literal
from spark_apps.decorators.patient_event_decorator import AttType


class FineTuneModelType(Enum):
    POOLING = "pooling"
    LSTM = "lstm"


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
    validation_split_num: Optional[int] = field(
        default=1000,
        metadata={
            "help": "The number of the train set used as validation set in case there's no validation split"
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
    att_function_type: Literal[AttType.CEHR_BERT.value, AttType.DAY.value, AttType.NONE.value] = field(
        default=AttType.CEHR_BERT.value,
        metadata={
            "help": "The ATT type to choose the level of granularity to use for creating the "
                    "artificial time tokens between visits",
            "choices": f"choices={[e.value for e in AttType]}"
        }
    )
    is_data_in_med: Optional[bool] = field(
        default=False,
        metadata={"help": "The boolean indicator to indicate whether the data is in the MED format"}
    )
    inpatient_att_function_type: Literal[AttType.CEHR_BERT.value, AttType.DAY.value, AttType.NONE.value] = field(
        default=AttType.NONE,
        metadata={
            "help": "The ATT type to choose the level of granularity to use for creating the "
                    "artificial time tokens between neighboring events within inpatient visits."
                    "Default to None, meaning the inpatient artificial time tokens are not created.",
            "choices": f"choices={[e.value for e in AttType]}"
        }
    )
    include_auxiliary_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "The boolean indicator to indicate whether visit type should be included "
                    "at the beginning of the visit and discharge facility should be included at the end of the visit"
        }
    )
    include_demographic_prompt: Optional[bool] = field(
        default=False,
        metadata={
            "help": "The boolean indicator to indicate whether the demographic tokens should be added "
                    "at the beginning of the sequence including start_year, start_age, gender, race"
        }
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={
            "help": "The boolean indicator to indicate whether the data should be streamed"
        }
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
    finetune_model_type: Literal[FineTuneModelType.POOLING.value, FineTuneModelType.LSTM.value] = field(
        default=FineTuneModelType.POOLING.value,
        metadata={
            "help": "The finetune model type to choose from",
            "choices": f"choices={[e.value for e in FineTuneModelType]}"
        }
    )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)
