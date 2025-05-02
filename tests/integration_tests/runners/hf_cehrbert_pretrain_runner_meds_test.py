import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from datasets import disable_caching

from cehrbert.runners.hf_cehrbert_pretrain_runner import main

disable_caching()
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WANDB_MODE"] = "disabled"
os.environ["TRANSFORMERS_VERBOSITY"] = "info"


class HfCehrBertRunnerIntegrationTest(unittest.TestCase):
    def setUp(self):
        # Get the root folder of the project
        root_folder = Path(os.path.abspath(__file__)).parent.parent.parent.parent
        data_folder = os.path.join(root_folder, "sample_data", "MIMIC-IV-meds", "meds_reader")
        # Create a temporary directory to store model and tokenizer
        self.temp_dir = tempfile.mkdtemp()
        self.model_folder_path = os.path.join(self.temp_dir, "model")
        Path(self.model_folder_path).mkdir(parents=True, exist_ok=True)
        self.dataset_prepared_path = os.path.join(self.temp_dir, "dataset_prepared_path")
        Path(self.dataset_prepared_path).mkdir(parents=True, exist_ok=True)
        sys.argv = [
            "hf_cehrbert_pretraining_runner.py",
            "--model_name_or_path",
            self.model_folder_path,
            "--tokenizer_name_or_path",
            self.model_folder_path,
            "--output_dir",
            self.model_folder_path,
            "--data_folder",
            data_folder,
            "--dataset_prepared_path",
            self.dataset_prepared_path,
            "--save_strategy",
            "steps",
            "--eval_strategy",
            "steps",
            "--max_steps",
            "10",
            "--is_data_in_meds",
            "--load_best_model_at_end",
            "true",
            "--report_to",
            "none",
        ]

    def tearDown(self):
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_train_model(self):
        main()


if __name__ == "__main__":
    unittest.main()
