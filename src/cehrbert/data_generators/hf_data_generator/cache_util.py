import os
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Union

from datasets import Dataset, DatasetDict, IterableDataset
from transformers.utils import logging

LOG = logging.get_logger("transformers")


@dataclass
class CacheFileCollector:
    cache_files: List[Dict[str, str]] = field(default_factory=list)

    def add_cache_files(self, dataset: Union[Dataset, IterableDataset, DatasetDict]) -> None:
        if isinstance(dataset, Dataset):
            self.cache_files.extend(dataset.cache_files)
        elif isinstance(dataset, DatasetDict):
            for dataset_split in dataset.values():
                self.add_cache_files(dataset_split)

    def remove_cache_files(self) -> None:
        for cache_file in self.cache_files:
            file_name = cache_file.get("filename", None)
            if file_name and os.path.exists(file_name):
                try:
                    if os.path.isdir(file_name):
                        shutil.rmtree(file_name)
                        LOG.debug(f"Removed cache directory: {file_name}")
                    else:
                        os.remove(file_name)
                        LOG.debug(f"Removed cache file: {file_name}")
                except OSError as e:
                    LOG.warning(f"Error removing {file_name}: {e}")
        self.cache_files = []
