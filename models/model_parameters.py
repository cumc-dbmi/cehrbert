import os

from config.parameters import *
from types import SimpleNamespace


class ModelPathConfig(SimpleNamespace):
    def __init__(self, input_folder, output_folder):
        super().__init__(
            parquet_data_path=os.path.join(input_folder, parquet_data_path),
            feather_data_path=os.path.join(input_folder, feather_data_path),
            tokenizer_path=os.path.join(output_folder, tokenizer_path),
            visit_tokenizer_path=os.path.join(output_folder, visit_tokenizer_path),
            model_path=os.path.join(output_folder, bert_model_path),
            concept_similarity_path=os.path.join(input_folder, concept_similarity_path)
        )
