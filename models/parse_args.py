import argparse
from sys import argv

from data_generators.graph_sample_method import SimilarityType


def create_parse_args():
    parser = argparse.ArgumentParser(
        description='Arguments for concept embedding model')
    parser.add_argument(
        '-i',
        '--input_folder',
        dest='input_folder',
        action='store',
        help='The path for your input_folder where the raw data is',
        required=True)
    parser.add_argument(
        '-o',
        '--output_folder',
        dest='output_folder',
        action='store',
        help=
        'The output folder that stores the domain tables download destination',
        required=True)
    parser.add_argument(
        '-m',
        '--max_seq_length',
        dest='max_seq_length',
        action='store',
        type=int,
        default=100,
        required=False
    )
    parser.add_argument(
        '-t',
        '--time_window_size',
        dest='time_window_size',
        action='store',
        type=int,
        default=100,
        required=False
    )
    parser.add_argument(
        '-c',
        '--embedding_size',
        dest='embedding_size',
        action='store',
        type=int,
        default=128,
        required=False
    )
    parser.add_argument(
        '-e',
        '--epochs',
        dest='epochs',
        action='store',
        type=int,
        default=50,
        required=False
    )
    parser.add_argument(
        '-b',
        '--batch_size',
        dest='batch_size',
        action='store',
        type=int,
        default=128,
        required=False
    )
    parser.add_argument(
        '-lr',
        '--learning_rate',
        dest='learning_rate',
        action='store',
        type=float,
        default=2e-4,
        required=False
    )
    parser.add_argument(
        '-bl',
        '--tf_board_log_path',
        dest='tf_board_log_path',
        action='store',
        default='./logs',
        required=False
    )
    return parser


def create_parse_args_base_bert():
    parser = create_parse_args()
    parser.add_argument(
        '--min_num_of_concepts',
        dest='min_num_of_concepts',
        action='store',
        type=int,
        default=5,
        required=False
    )
    parser.add_argument(
        '-d',
        '--depth',
        dest='depth',
        action='store',
        type=int,
        default=5,
        required=False
    )
    parser.add_argument(
        '-nh',
        '--num_heads',
        dest='num_heads',
        action='store',
        type=int,
        default=8,
        required=False
    )
    parser.add_argument(
        '-iv',
        '--include_visit',
        dest='include_visit_prediction',
        action='store_true'
    )
    parser.add_argument(
        '--include_prolonged_length_stay',
        dest='include_prolonged_length_stay',
        action='store_true'
    )
    parser.add_argument(
        '-ut',
        '--use_time_embedding',
        dest='use_time_embedding',
        action='store_true'
    )
    parser.add_argument(
        '--use_behrt',
        dest='use_behrt',
        action='store_true'
    )
    parser.add_argument(
        '--use_dask',
        dest='use_dask',
        action='store_true'
    )
    parser.add_argument(
        '--time_embeddings_size',
        dest='time_embeddings_size',
        action='store',
        type=int,
        default=16,
        required=False
    )
    return parser


def create_parse_args_gpt():
    parser = create_parse_args()
    parser.add_argument(
        '--min_num_of_concepts',
        dest='min_num_of_concepts',
        action='store',
        type=int,
        default=5,
        required=False
    )
    parser.add_argument(
        '-d',
        '--depth',
        dest='depth',
        action='store',
        type=int,
        default=5,
        required=False
    )
    parser.add_argument(
        '-nh',
        '--num_heads',
        dest='num_heads',
        action='store',
        type=int,
        default=8,
        required=False
    )
    parser.add_argument(
        '--concept_path',
        dest='concept_path',
        action='store',
        help='The path for the concept',
        required=True
    )
    parser.add_argument(
        '--use_dask',
        dest='use_dask',
        action='store_true'
    )
    parser.add_argument(
        '--min_num_of_visits',
        dest='min_num_of_visits',
        action='store',
        type=int,
        default=2,
        required=False
    )
    parser.add_argument(
        '--max_num_of_visits',
        dest='max_num_of_visits',
        action='store',
        type=int,
        default=20,
        required=False
    )
    parser.add_argument(
        '--print_every',
        dest='print_every',
        action='store',
        type=int,
        default=500,
        required=False
    )
    parser.add_argument(
        '--num_of_patients',
        dest='num_of_patients',
        action='store',
        type=int,
        default=1024,
        required=False
    )
    parser.add_argument(
        '--sampling_batch_size',
        dest='sampling_batch_size',
        action='store',
        type=int,
        default=256,
        required=False
    )
    parser.add_argument(
        '--including_long_sequence',
        dest='including_long_sequence',
        action='store_true'
    )
    parser.add_argument(
        '--save_checkpoint',
        dest='save_checkpoint',
        action='store_true'
    )
    parser.add_argument(
        '--save_freq',
        dest='save_freq',
        action='store',
        type=int,
        default=0,
        required='--save_checkpoint' in argv
    )
    parser.add_argument(
        '--sampling_dataset_enabled',
        dest='sampling_dataset_enabled',
        action='store_true'
    )
    return parser


def create_parse_args_temporal_bert():
    parser = create_parse_args_base_bert()
    parser.add_argument(
        '-ti',
        '--time_attention_folder',
        dest='time_attention_folder',
        action='store',
        help=
        'The path for your time attention input_folder where the raw data is',
        required=True)
    return parser


def create_parse_args_hierarchical_bert():
    def check_prob(value):
        ivalue = float(value)
        if ivalue < 0 or ivalue > 1:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return ivalue

    parser = create_parse_args_base_bert()
    parser.add_argument(
        '--max_num_visits',
        dest='max_num_visits',
        action='store',
        type=int,
        help='Max no.of visits per patient',
        required=True
    )
    parser.add_argument(
        '--max_num_concepts',
        dest='max_num_concepts',
        action='store',
        type=int,
        help='Max no.of concepts per visit per patient',
        required=True
    )
    parser.add_argument(
        '--min_num_of_visits',
        dest='min_num_of_visits',
        action='store',
        type=int,
        default=1,
        required=False
    )
    parser.add_argument(
        '--include_att_prediction',
        dest='include_att_prediction',
        action='store_true'
    )
    parser.add_argument(
        '--include_readmission',
        dest='include_readmission',
        action='store_true'
    )
    parser.add_argument(
        '--random_mask_prob',
        dest='random_mask_prob',
        type=check_prob,
        required='include_readmission' in argv or 'include_prolonged_length_stay' in argv,
        default=1.0,
        help='The probability the secondary learning objective uses. The value 0.2 '
             'indicates there is a 20% chance of masking in pre-training '
             'for secondary learning objectives'
    )
    parser.add_argument(
        '--concept_similarity_type',
        dest='concept_similarity_type',
        action='store',
        choices=[
            member.value for member in SimilarityType
        ],
        help='The concept similarity measures to use for masking',
        default=SimilarityType.NONE.value,
        required=False
    )
    parser.add_argument(
        '--secondary_learning_warmup_step',
        dest='warmup_step',
        action='store',
        type=int,
        help='The number steps before secondary learning objectives start',
        default=-1,
        required=False
    )
    return parser


def create_parse_args_hierarchical_bert_phenotype():
    parser = create_parse_args_hierarchical_bert()
    parser.add_argument(
        '--num_of_phenotypes',
        dest='num_of_phenotypes',
        action='store',
        type=int,
        help='Num of phenotypes',
        default=20,
        required=False
    )
    parser.add_argument(
        '--num_of_phenotype_neighbors',
        dest='num_of_phenotype_neighbors',
        action='store',
        type=int,
        help='Num of phenotype neighbors to consider when driving the phenotypes apart from each '
             'other',
        default=3,
        required=False
    )
    parser.add_argument(
        '--num_of_concept_neighbors',
        dest='num_of_concept_neighbors',
        action='store',
        type=int,
        help='Num of concept neighbors to consider when minimizing the phenotype-concept distances',
        default=10,
        required=False
    )
    return parser
