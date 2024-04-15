import os
import glob

from datasets import load_dataset
from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint

LOG = logging.get_logger("transformers")


def load_parquet_as_dataset(data_folder, split="train"):
    data_abspath = os.path.abspath(data_folder)
    data_files = glob.glob(os.path.join(data_abspath, "*.parquet"))
    dataset = load_dataset('parquet', data_files=data_files, split=split)
    return dataset


def get_last_hf_checkpoint(training_args):
    last_checkpoint = None
    output_dir_abspath = os.path.abspath(training_args.output_dir)
    if os.path.isdir(output_dir_abspath) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(output_dir_abspath)
        if last_checkpoint is None and len([_ for _ in os.listdir(output_dir_abspath) if os.path.isdir(_)]) > 0:
            raise ValueError(
                f"Output directory ({output_dir_abspath}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            LOG.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    return last_checkpoint
