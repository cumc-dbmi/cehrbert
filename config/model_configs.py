import os

import spark_apps.parameters as p
from types import SimpleNamespace


class Config(SimpleNamespace):

    def __init__(self, parquet_data_path: str,
                 feather_data_path: str,
                 tokenizer_path: str,
                 model_path: str,
                 tf_board_log_path: str,
                 concept_embedding_size: int,
                 max_seq_length: int,
                 time_window_size: int,
                 batch_size: int,
                 epochs: int,
                 learning_rate: float):
        super().__init__(parquet_data_path=parquet_data_path,
                         feather_data_path=feather_data_path,
                         tokenizer_path=tokenizer_path,
                         model_path=model_path,
                         tf_board_log_path=tf_board_log_path,
                         concept_embedding_size=concept_embedding_size,
                         max_seq_length=max_seq_length,
                         time_window_size=time_window_size,
                         batch_size=batch_size,
                         epochs=epochs,
                         learning_rate=learning_rate)


class BertConfig(Config):

    def __init__(self, visit_tokenizer_path,
                 depth, num_heads,
                 include_visit_prediction,
                 use_time_embedding,
                 use_behrt,
                 use_dask,
                 *args, **kwargs):
        super(BertConfig, self).__init__(*args, **kwargs)
        self.visit_tokenizer_path = visit_tokenizer_path
        self.depth = depth
        self.num_heads = num_heads
        self.include_visit_prediction = include_visit_prediction
        self.use_time_embedding = use_time_embedding
        self.use_behrt = use_behrt
        self.use_dask = use_dask


class TemporalBertConfig(BertConfig):

    def __init__(self, time_attention_model_path, *args, **kwargs):
        super(TemporalBertConfig, self).__init__(*args, **kwargs)
        self.time_attention_model_path = time_attention_model_path


def create_time_attention_model_config(args):
    """
    Create a named tuple config to store the parameters for training the time attention model
    :param args:
    :return:
    """
    parquet_data_path = os.path.join(args.input_folder, p.parquet_data_path)
    feather_data_path = os.path.join(args.input_folder, p.feather_data_path)
    tokenizer_path = os.path.join(args.output_folder, p.tokenizer_path)
    model_path = os.path.join(args.output_folder, p.time_attention_model_path)

    return Config(parquet_data_path=parquet_data_path,
                  feather_data_path=feather_data_path,
                  tokenizer_path=tokenizer_path,
                  model_path=model_path,
                  tf_board_log_path=args.tf_board_log_path,
                  concept_embedding_size=args.concept_embedding_size,
                  max_seq_length=args.max_seq_length,
                  time_window_size=args.time_window_size,
                  batch_size=args.batch_size,
                  epochs=args.epochs,
                  learning_rate=args.learning_rate)


def create_bert_model_config(args):
    """
    Create a named tuple config to store the parameters for training the vanilla bert model
    :param args:
    :return:
    """
    parquet_data_path = os.path.join(args.input_folder, p.parquet_data_path)
    feather_data_path = os.path.join(args.input_folder, p.feather_data_path)
    tokenizer_path = os.path.join(args.output_folder, p.tokenizer_path)
    visit_tokenizer_path = os.path.join(args.output_folder, p.visit_tokenizer_path)
    model_path = os.path.join(args.output_folder, p.bert_model_path)

    return BertConfig(parquet_data_path=parquet_data_path,
                      feather_data_path=feather_data_path,
                      tokenizer_path=tokenizer_path,
                      visit_tokenizer_path=visit_tokenizer_path,
                      model_path=model_path,
                      tf_board_log_path=args.tf_board_log_path,
                      concept_embedding_size=args.concept_embedding_size,
                      max_seq_length=args.max_seq_length,
                      time_window_size=args.time_window_size,
                      depth=args.depth,
                      num_heads=args.num_heads,
                      batch_size=args.batch_size,
                      epochs=args.epochs,
                      learning_rate=args.learning_rate,
                      include_visit_prediction=args.include_visit_prediction,
                      use_time_embedding=args.use_time_embedding,
                      use_behrt=args.use_behrt,
                      use_dask=args.use_dask)


def create_temporal_bert_model_config(args):
    """
    Create a named tuple config to store the parameters for training the temporal bert model
    :param args:
    :return:
    """
    parquet_data_path = os.path.join(args.input_folder, p.parquet_data_path)
    feather_data_path = os.path.join(args.input_folder, p.feather_data_path)
    tokenizer_path = os.path.join(args.output_folder, p.tokenizer_path)
    visit_tokenizer_path = os.path.join(args.output_folder, p.visit_tokenizer_path)
    model_path = os.path.join(args.output_folder, p.temporal_bert_model_path)
    time_attention_model_path = os.path.join(args.time_attention_folder,
                                             p.time_attention_model_path)

    return TemporalBertConfig(parquet_data_path=parquet_data_path,
                              feather_data_path=feather_data_path,
                              tokenizer_path=tokenizer_path,
                              visit_tokenizer_path=visit_tokenizer_path,
                              model_path=model_path,
                              tf_board_log_path=args.tf_board_log_path,
                              time_attention_model_path=time_attention_model_path,
                              concept_embedding_size=args.concept_embedding_size,
                              max_seq_length=args.max_seq_length,
                              time_window_size=args.time_window_size,
                              depth=args.depth,
                              num_heads=args.num_heads,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              learning_rate=args.learning_rate,
                              include_visit_prediction=args.include_visit_prediction)
