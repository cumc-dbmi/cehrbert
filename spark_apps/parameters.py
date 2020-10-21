import argparse
import datetime

parquet_data_path = 'patient_sequence'
feather_data_path = 'patient_sequence.pickle'
tokenizer_path = 'tokenizer.pickle'
visit_tokenizer_path = 'visit_tokenizer.pickle'
time_attention_model_path = 'time_aware_model.h5'
bert_model_path = 'bert_model.h5'
temporal_bert_model_path = 'temporal_bert_model.h5'

mortality_data_path = 'mortality'
heart_failure_data_path = 'heart_failure'
hospitalization_data_path = 'hospitalization'


def valid_date(s):
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        msg = "Not a valid date: '{0}'.".format(s)
        raise argparse.ArgumentTypeError(msg)


def create_spark_args():
    parser = argparse.ArgumentParser(
        description='Arguments for spark applications for generating cohort definitions')
    parser.add_argument('-c',
                        '--cohort_name',
                        dest='cohort_name',
                        action='store',
                        help='The cohort name',
                        required=True)
    parser.add_argument('-i',
                        '--input_folder',
                        dest='input_folder',
                        action='store',
                        help='The path for your input_folder where the sequence data is',
                        required=True)
    parser.add_argument('-o',
                        '--output_folder',
                        dest='output_folder',
                        action='store',
                        help='The path for your output_folder',
                        required=True)
    parser.add_argument('-dl',
                        '--date_lower_bound',
                        dest='date_lower_bound',
                        action='store',
                        help='The date filter lower bound for filtering training data',
                        required=True,
                        type=valid_date)
    parser.add_argument('-du',
                        '--date_upper_bound',
                        dest='date_upper_bound',
                        action='store',
                        help='The date filter upper bound for filtering training data',
                        required=True,
                        type=valid_date)
    parser.add_argument('-l',
                        '--lower_bound',
                        dest='lower_bound',
                        action='store',
                        help='The age lower bound',
                        required=False,
                        type=int,
                        default=0)
    parser.add_argument('-u',
                        '--upper_bound',
                        dest='upper_bound',
                        action='store',
                        help='The age upper bound',
                        required=False,
                        type=int,
                        default=100)
    parser.add_argument('-ow',
                        '--observation_window',
                        dest='observation_window',
                        action='store',
                        help='The observation window in days for extracting features',
                        required=False,
                        type=int,
                        default=365)
    parser.add_argument('-pw',
                        '--prediction_window',
                        dest='prediction_window',
                        action='store',
                        help='The prediction window in days prior the index date',
                        required=False,
                        type=int,
                        default=180)
    parser.add_argument('-im',
                        '--index_date_match_window',
                        dest='index_date_match_window',
                        action='store',
                        help='The window for matching index dates of '
                             'the incident and control cases',
                        required=False,
                        type=int,
                        default=30)

    parser.add_argument('-iv',
                        '--include_visit_type',
                        dest='include_visit_type',
                        action='store_true',
                        help='Specify whether to include visit types for '
                             'generating the training data')

    parser.add_argument('-f',
                        '--is_feature_concept_frequency',
                        dest='is_feature_concept_frequency',
                        action='store_true',
                        help='Specify whether the features are concept counts or not')

    parser.add_argument('-ir',
                        '--is_roll_up_concept',
                        dest='is_roll_up_concept',
                        action='store_true',
                        help='Specify whether to roll up the concepts to their ancestors')

    return parser.parse_args()
