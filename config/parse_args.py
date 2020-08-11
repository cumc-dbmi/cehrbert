import argparse


def create_parse_args():
    parser = argparse.ArgumentParser(description='Arguments for concept embedding model')
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the raw data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The output folder that stores the domain tables download destination',
                        required=True)
    parser.add_argument('-m',
                        '--max_seq_length',
                        dest='max_seq_length',
                        action='store',
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument('-t',
                        '--time_window_size',
                        dest='time_window_size',
                        action='store',
                        type=int,
                        default=100,
                        required=False)
    parser.add_argument('-c',
                        '--concept_embedding_size',
                        dest='concept_embedding_size',
                        action='store',
                        type=int,
                        default=128,
                        required=False)
    parser.add_argument('-e',
                        '--epochs',
                        dest='epochs',
                        action='store',
                        type=int,
                        default=50,
                        required=False)
    parser.add_argument('-b',
                        '--batch_size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=128,
                        required=False)
    parser.add_argument('-lr',
                        '--learning_rate',
                        dest='learning_rate',
                        action='store',
                        type=float,
                        default=2e-4,
                        required=False)
    parser.add_argument('-bl',
                        '--tf_board_log_path',
                        dest='tf_board_log_path',
                        action='store',
                        default='./logs',
                        required=False)
    return parser


def create_parse_args_base_bert():
    parser = create_parse_args()

    parser.add_argument('-d',
                        '--depth',
                        dest='depth',
                        action='store',
                        default=5,
                        required=False)

    parser.add_argument('-nh',
                        '--num_heads',
                        dest='num_heads',
                        action='store',
                        default=8,
                        required=False)
    return parser


def create_parse_args_temporal_bert():
    parser = create_parse_args_base_bert()
    parser.add_argument('-ti',
                        '--time_attention_folder',
                        dest='time_attention_folder',
                        action='store',
                        help='The path for your time attention input_folder where the raw data is',
                        required=True)
    return parser
