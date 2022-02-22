import argparse
from sys import argv

from evaluations.evaluation_parameters import *


def create_evaluation_args():
    main_parser = argparse.ArgumentParser(
        description='Arguments for evaluating the models')

    sequence_model_required = BASELINE_MODEL not in argv
    baseline_model_required = SEQUENCE_MODEL not in argv
    lstm_model_required = LSTM in argv
    vanilla_bert_lstm = VANILLA_BERT_LSTM in argv
    temporal_bert_lstm = TEMPORAL_BERT_LSTM in argv
    sliding_bert = SLIDING_BERT in argv
    hierarchical_bert = HIERARCHICAL_BERT_LSTM in argv

    main_parser.add_argument('-a',
                             '--action',
                             dest='action',
                             action='store',
                             choices=EVALUATION_CHOICES,
                             help='The action that determines the evaluation process',
                             required=True)
    main_parser.add_argument('-d',
                             '--data_path',
                             dest='data_path',
                             action='store',
                             help='The training data path',
                             required=baseline_model_required)
    main_parser.add_argument('-ef',
                             '--evaluation_folder',
                             dest='evaluation_folder',
                             action='store',
                             required=True)
    main_parser.add_argument('-n',
                             '--num_of_folds',
                             dest='num_of_folds',
                             action='store',
                             required=False,
                             type=int,
                             default=4)
    main_parser.add_argument('--is_transfer_learning',
                             dest='is_transfer_learning',
                             action='store_true')
    main_parser.add_argument('--training_percentage',
                             dest='training_percentage',
                             required=False,
                             action='store',
                             type=float,
                             default=1.0)

    group = main_parser.add_argument_group('sequence model')
    group.add_argument('-me',
                       '--model_evaluators',
                       dest='model_evaluators',
                       action='store',
                       nargs='+',
                       choices=SEQUENCE_MODEL_EVALUATORS,
                       required=sequence_model_required)
    group.add_argument('-sd',
                       '--sequence_model_data_path',
                       dest='sequence_model_data_path',
                       action='store',
                       required=sequence_model_required)
    group.add_argument('-smn',
                       '--sequence_model_name',
                       dest='sequence_model_name',
                       action='store',
                       default=None)
    group.add_argument('-m',
                       '--max_seq_length',
                       dest='max_seq_length',
                       type=int,
                       action='store',
                       required=sequence_model_required)
    group.add_argument('-b',
                       '--batch_size',
                       dest='batch_size',
                       action='store',
                       type=int,
                       required=sequence_model_required)
    group.add_argument('-p',
                       '--epochs',
                       dest='epochs',
                       action='store',
                       type=int,
                       required=sequence_model_required)
    group.add_argument('-ti',
                       '--time_attention_model_folder',
                       dest='time_attention_model_folder',
                       action='store',
                       required=lstm_model_required)
    group.add_argument('-vb',
                       '--vanilla_bert_model_folder',
                       dest='vanilla_bert_model_folder',
                       action='store',
                       required=vanilla_bert_lstm)
    group.add_argument('-tb',
                       '--temporal_bert_model_folder',
                       dest='temporal_bert_model_folder',
                       action='store',
                       required=temporal_bert_lstm)
    group.add_argument('--stride',
                       dest='stride',
                       action='store',
                       type=int,
                       required=sliding_bert)
    group.add_argument('--context_window',
                       dest='context_window',
                       action='store',
                       type=int,
                       required=sliding_bert)
    group.add_argument('--max_num_of_visits',
                       dest='max_num_of_visits',
                       action='store',
                       type=int,
                       required=hierarchical_bert)
    group.add_argument('--max_num_of_concepts',
                       dest='max_num_of_concepts',
                       action='store',
                       type=int,
                       required=hierarchical_bert)
    group.add_argument('--depth',
                       dest='depth',
                       action='store',
                       type=int,
                       default=5,
                       required=False)
    group.add_argument('-nh',
                       '--num_heads',
                       dest='num_heads',
                       action='store',
                       type=int,
                       default=8,
                       required=False)
    group.add_argument('-iv',
                       '--include_visit',
                       dest='include_visit_prediction',
                       action='store_true',
                       required=False)
    group.add_argument('-ut',
                       '--use_time_embedding',
                       dest='use_time_embedding',
                       action='store_true',
                       required=False)
    group.add_argument('--time_embeddings_size',
                       dest='time_embeddings_size',
                       action='store',
                       type=int,
                       default=16,
                       required=False)
    group.add_argument('--embedding_size',
                       dest='embedding_size',
                       action='store',
                       type=int,
                       default=128,
                       required=False)

    return main_parser
