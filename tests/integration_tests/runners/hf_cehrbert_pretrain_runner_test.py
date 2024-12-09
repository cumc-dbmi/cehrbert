import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import disable_caching

from cehrbert.runners.hf_cehrbert_finetune_runner import main as finetune_main
from cehrbert.runners.hf_cehrbert_pretrain_runner import main

disable_caching()
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


class HfCehrBertRunnerIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        self.pretrain_data_folder = os.path.join(root_folder, "sample_data", "pretrain")
        self.finetune_data_folder = os.path.join(root_folder, "sample_data", "finetune", "full")
        # Create a temporary directory to store model and tokenizer
        self.temp_dir = tempfile.mkdtemp()
        self.model_folder_path = os.path.join(self.temp_dir, "model")
        self.finetuned_model_folder_path = os.path.join(self.temp_dir, "model_finetuned")
        Path(self.model_folder_path).mkdir(parents=True, exist_ok=True)
        self.dataset_prepared_path = os.path.join(self.temp_dir, "dataset_prepared_path")
        Path(self.dataset_prepared_path).mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_train_model(self):
        sys.argv = [
            "hf_cehrbert_pretraining_runner.py",
            "--model_name_or_path",
            self.model_folder_path,
            "--tokenizer_name_or_path",
            self.model_folder_path,
            "--output_dir",
            self.model_folder_path,
            "--data_folder",
            self.pretrain_data_folder,
            "--dataset_prepared_path",
            self.dataset_prepared_path,
            "--max_steps",
            "10",
        ]
        main()
        sys.argv = [
            "hf_cehrbert_finetune_runner.py",
            "--model_name_or_path",
            self.model_folder_path,
            "--tokenizer_name_or_path",
            self.model_folder_path,
            "--output_dir",
            self.finetuned_model_folder_path,
            "--data_folder",
            self.finetune_data_folder,
            "--dataset_prepared_path",
            self.dataset_prepared_path,
            "--max_steps",
            "10",
            "--save_strategy",
            "steps",
            "--evaluation_strategy",
            "steps",
            "--do_train",
            "true",
            "--do_predict",
            "true",
            "--load_best_model_at_end",
            "true",
        ]
        finetune_main()


if __name__ == "__main__":
    unittest.main()
