import argparse
import datetime

parquet_data_path = 'patient_sequence'
feather_data_path = 'patient_sequence.pickle'
tokenizer_path = 'tokenizer.pickle'
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
    parser.add_argument('-f',
                        '--date_filter',
                        dest='date_filter',
                        action='store',
                        help='The path for your output_folder',
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
    return parser.parse_args()