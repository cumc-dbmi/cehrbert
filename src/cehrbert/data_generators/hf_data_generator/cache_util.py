import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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

    @staticmethod
    def get_dataset_version_folder(file_name: str) -> Optional[str]:
        found_dataset_folder = False
        parents = []
        for p in Path(file_name).parents:
            if str(p).endswith("datasets"):
                found_dataset_folder = True
                break
            parents.append(str(p))
        if len(parents) < 2:
            LOG.warning(f"the number of parents is less than 2 for {file_name}")
            return None
        if (parents[-1].endswith("generator") or parents[-1].endswith("parquet")) and found_dataset_folder:
            return parents[-2]
        return None

    def remove_cache_files(self) -> None:
        dataset_dirs_to_delete = set()
        for cache_file in self.cache_files:
            file_name = cache_file.get("filename", None)
            if file_name and os.path.exists(file_name):
                dataset_dir = self.get_dataset_version_folder(file_name)
                if dataset_dir:
                    dataset_dirs_to_delete.add(dataset_dir)
                try:
                    if os.path.isdir(file_name):
                        shutil.rmtree(file_name)
                        LOG.debug(f"Removed cache directory: {file_name}")
                    else:
                        os.remove(file_name)
                        LOG.debug(f"Removed cache file: {file_name}")
                except OSError as e:
                    LOG.warning(f"Error removing {file_name}: {e}")

        for dataset_dir in dataset_dirs_to_delete:
            try:
                if os.path.isdir(dataset_dir):
                    shutil.rmtree(dataset_dir)
                    LOG.debug(f"Removed cache directory: {dataset_dir}")
            except OSError as e:
                LOG.warning(f"Error removing {dataset_dir}: {e}")
        self.cache_files = []
