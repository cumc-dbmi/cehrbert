from typing import Optional, Union

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer
from transformers.trainer_utils import has_length
from transformers.utils import import_utils, logging

from cehrbert.data_generators.hf_data_generator.sample_packing_sampler import SamplePackingBatchSampler

DEFAULT_MAX_TOKENS_PER_BATCH = 16384

LOG = logging.get_logger("transformers")


class SamplePackingTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        if "max_tokens_per_batch" in kwargs:
            self.max_tokens_per_batch = kwargs.pop("max_tokens_per_batch")
            LOG.info("max_tokens_per_batch: %s", self.max_tokens_per_batch)
        else:
            self.max_tokens_per_batch = DEFAULT_MAX_TOKENS_PER_BATCH
            LOG.info(
                "max_tokens_per_batch is not provided to SamplePackingTrainer and will default to %s",
                DEFAULT_MAX_TOKENS_PER_BATCH,
            )

        if "max_position_embeddings" in kwargs:
            self.max_position_embeddings = kwargs.pop("max_position_embeddings")
            LOG.info("max_position_embeddings: %s", self.max_position_embeddings)
        else:
            self.max_position_embeddings = self.max_tokens_per_batch
            LOG.info(
                "max_position_embeddings is not provided to SamplePackingTrainer and will default to %s",
                self.max_tokens_per_batch,
            )

        self.train_lengths = kwargs.pop("train_lengths", None)
        self.validation_lengths = kwargs.pop("validation_lengths", None)
        super().__init__(*args, **kwargs)
        self.accelerator.even_batches = False

    def num_examples(self, dataloader: DataLoader) -> int:
        if has_length(dataloader):
            return len(dataloader)
        raise RuntimeError("DataLoader in SamplePackingTrainer must have length")

    def get_train_dataloader(self) -> DataLoader:
        """Returns the training dataloader with our custom batch sampler."""
        train_dataset = self.train_dataset

        if self.train_lengths is None:
            LOG.info("Started computing lengths for the train dataset")
            # Calculate lengths of all sequences in dataset
            if "num_of_concepts" in train_dataset.column_names:
                lengths = train_dataset["num_of_concepts"]
            else:
                lengths = [len(sample["input_ids"]) for sample in train_dataset]

            LOG.info("Finished computing lengths for the train dataset")
        else:
            lengths = self.train_lengths

        data_collator = self.data_collator
        if import_utils.is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")
        # Create our custom batch sampler
        batch_sampler = SamplePackingBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_position_embeddings=self.max_position_embeddings,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )
        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": batch_sampler,
        }
        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )

        if self.validation_lengths is None:
            LOG.info("Started computing lengths for the train dataset")
            # Calculate lengths of all sequences in dataset
            if "num_of_concepts" in eval_dataset.column_names:
                lengths = eval_dataset["num_of_concepts"]
            else:
                lengths = [len(sample["input_ids"]) for sample in eval_dataset]

            LOG.info("Finished computing lengths for the train dataset")
        else:
            lengths = self.validation_lengths

        data_collator = self.data_collator

        if import_utils.is_datasets_available() and isinstance(eval_dataset, Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        # Create our custom batch sampler
        batch_sampler = SamplePackingBatchSampler(
            lengths=lengths,
            max_tokens_per_batch=self.max_tokens_per_batch,
            max_position_embeddings=self.max_position_embeddings,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )
        dataloader_params = {
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
            "batch_sampler": batch_sampler,
        }
        # accelerator.free_memory() will destroy the references, so
        # we need to store the non-prepared version
        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        if self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = eval_dataloader
            else:
                self._eval_dataloaders = {dataloader_key: eval_dataloader}

        return self.accelerator.prepare(eval_dataloader)
