import os

import config.parameters
from types import SimpleNamespace


class ModelPathConfig(SimpleNamespace):
    def __init__(self, input_folder, output_folder):
        super().__init__(parquet_data_path=os.path.join(input_folder,
                                                        config.parameters.parquet_data_path),
                         feather_data_path=os.path.join(input_folder,
                                                        config.parameters.feather_data_path),
                         tokenizer_path=os.path.join(output_folder,
                                                     config.parameters.tokenizer_path),
                         visit_tokenizer_path=os.path.join(output_folder,
                                                           config.parameters.visit_tokenizer_path),
                         model_path=os.path.join(output_folder, config.parameters.bert_model_path))

