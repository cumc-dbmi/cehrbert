"""
This module defines functions for parsing command-line arguments for Spark applications.

that generate cohort definitions. It includes argument parsing for cohort specifications,
date ranges, patient information, and EHR data extraction settings.

Functions:
    valid_date: Validates and converts a date string into a datetime object.
    create_spark_args: Defines and parses command-line arguments for cohort generation and EHR
    processing.
"""

import argparse
import datetime

from .decorators.patient_event_decorator import AttType


def valid_date(s):
    """
    Validates and converts a date string into a datetime object.

    Args:
        s (str): The date string in the format 'YYYY-MM-DD'.
    Returns:
        datetime.datetime: The parsed date.
    Raises:
        argparse.ArgumentTypeError: If the date string is not valid.
    """
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d")
    except ValueError as e:
        raise argparse.ArgumentTypeError(e)


def create_spark_args():
    """
    Defines and parses the command-line arguments for Spark applications.

    that generate cohort definitions based on EHR data.

    Returns:
        argparse.Namespace: The parsed arguments as a namespace object containing the user
        inputs.

    Command-line Arguments:
        -c, --cohort_name: The name of the cohort being generated.
        -i, --input_folder: The folder path containing the input data.
        --patient_splits_folder: The folder containing patient splits data.
        -o, --output_folder: The folder path to store the output data.
        --ehr_table_list: List of EHR domain tables for feature extraction.
        -dl, --date_lower_bound: The lower bound for date filtering.
        -du, --date_upper_bound: The upper bound for date filtering.
        -l, --age_lower_bound: The minimum age filter for cohort inclusion.
        -u, --age_upper_bound: The maximum age filter for cohort inclusion.
        -ow, --observation_window: The observation window duration in days.
        -pw, --prediction_window: The prediction window duration in days.
        -ps, --prediction_start_days: The start point of the prediction window in days.
        -hw, --hold_off_window: The hold-off window for excluding certain features.
        --num_of_visits: The minimum number of visits required for cohort inclusion.
        --num_of_concepts: The minimum number of concepts required for cohort inclusion.
        -iw, --is_window_post_index: Whether the observation window is post-index.
        -iv, --include_visit_type: Whether to include visit types in feature generation.
        -ev, --exclude_visit_tokens: Whether to exclude certain visit tokens (VS and VE).
        -f, --is_feature_concept_frequency: Whether the features are based on concept counts.
        -ir, --is_roll_up_concept: Whether to roll up concepts to their ancestors.
        -ip, --is_new_patient_representation: Whether to use a new patient representation.
        --gpt_patient_sequence: Whether to generate GPT sequences for EHR records.
        -ih, --is_hierarchical_bert: Whether to use a hierarchical patient representation for BERT.
        -cbs, --classic_bert_seq: Whether to use classic BERT sequence representation with SEP.
        --is_first_time_outcome: Whether the outcome is the first-time occurrence.
        --is_remove_index_prediction_starts: Whether to remove outcomes between index and prediction
        start.
        --is_prediction_window_unbounded: Whether the prediction window end is unbounded.
        --is_observation_window_unbounded: Whether the observation window is unbounded.
        --include_concept_list: Whether to apply filters for low-frequency concepts.
        --allow_measurement_only: Whether patients with only measurements are allowed.
        --is_population_estimation: Whether cohort is constructed for population-level estimation.
        --att_type: The attribute type used for cohort definitions.
        --exclude_demographic: Whether to exclude demographic prompts in patient sequences.
        --use_age_group: Whether to represent age using age groups in patient sequences.
        --single_contribution: Whether patients should contribute only once to the training data.
    """
    parser = argparse.ArgumentParser(
        description="Arguments for spark applications for generating cohort definitions"
    )
    parser.add_argument(
        "-c",
        "--cohort_name",
        dest="cohort_name",
        action="store",
        help="The cohort name",
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input_folder",
        dest="input_folder",
        action="store",
        help="The path for your input_folder where the sequence data is",
        required=True,
    )
    parser.add_argument(
        "--patient_splits_folder",
        dest="patient_splits_folder",
        action="store",
        help="The folder that contains the patient_splits data",
        required=False,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The path for your output_folder",
        required=True,
    )
    parser.add_argument(
        "--ehr_table_list",
        dest="ehr_table_list",
        nargs="+",
        action="store",
        help="The list of domain tables you want to include for feature extraction",
        required=False,
    )
    parser.add_argument(
        "-dl",
        "--date_lower_bound",
        dest="date_lower_bound",
        action="store",
        help="The date filter lower bound for filtering training data",
        required=True,
        type=valid_date,
    )
    parser.add_argument(
        "-du",
        "--date_upper_bound",
        dest="date_upper_bound",
        action="store",
        help="The date filter upper bound for filtering training data",
        required=True,
        type=valid_date,
    )
    parser.add_argument(
        "-l",
        "--age_lower_bound",
        dest="age_lower_bound",
        action="store",
        help="The age lower bound",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "-u",
        "--age_upper_bound",
        dest="age_upper_bound",
        action="store",
        help="The age upper bound",
        required=False,
        type=int,
        default=100,
    )
    parser.add_argument(
        "-ow",
        "--observation_window",
        dest="observation_window",
        action="store",
        help="The observation window in days for extracting features",
        required=False,
        type=int,
        default=365,
    )
    parser.add_argument(
        "-pw",
        "--prediction_window",
        dest="prediction_window",
        action="store",
        help="The prediction window in which the prediction is made",
        required=False,
        type=int,
        default=180,
    )
    parser.add_argument(
        "-ps",
        "--prediction_start_days",
        dest="prediction_start_days",
        action="store",
        help="The prediction start days in which the prediction is made",
        required=False,
        type=int,
        default=1,
    )
    parser.add_argument(
        "-hw",
        "--hold_off_window",
        dest="hold_off_window",
        action="store",
        help="The hold off window for excluding the features",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_of_visits",
        dest="num_of_visits",
        action="store",
        help="The number of visits to qualify for the inclusion of the cohorts",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--num_of_concepts",
        dest="num_of_concepts",
        action="store",
        help="The number of concepts to qualify for the inclusion of the cohorts",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "-iw",
        "--is_window_post_index",
        dest="is_window_post_index",
        action="store_true",
        help="Indicate if the observation window is pre/post the index date",
    )
    parser.add_argument(
        "-iv",
        "--include_visit_type",
        dest="include_visit_type",
        action="store_true",
        help="Specify whether to include visit types for " "generating the training data",
    )
    parser.add_argument(
        "-ev",
        "--exclude_visit_tokens",
        dest="exclude_visit_tokens",
        action="store_true",
        help="Specify whether or not to exclude the VS and VE tokens",
    )
    parser.add_argument(
        "-f",
        "--is_feature_concept_frequency",
        dest="is_feature_concept_frequency",
        action="store_true",
        help="Specify whether the features are concept counts or not",
    )
    parser.add_argument(
        "-ir",
        "--is_roll_up_concept",
        dest="is_roll_up_concept",
        action="store_true",
        help="Specify whether to roll up the concepts to their ancestors",
    )
    parser.add_argument(
        "-ip",
        "--is_new_patient_representation",
        dest="is_new_patient_representation",
        action="store_true",
        help="Specify whether to generate the sequence of "
        "EHR records using the new patient representation",
    )
    parser.add_argument(
        "--gpt_patient_sequence",
        dest="gpt_patient_sequence",
        action="store_true",
        help="Specify whether to generate the GPT sequence of "
        "EHR records using the new patient representation",
    )
    parser.add_argument(
        "-ih",
        "--is_hierarchical_bert",
        dest="is_hierarchical_bert",
        action="store_true",
        help="Specify whether to generate the sequence of "
        "EHR records using the hierarchical patient representation",
    )
    parser.add_argument(
        "-cbs",
        "--classic_bert_seq",
        dest="classic_bert_seq",
        action="store_true",
        help="Specify whether to generate the sequence of "
        "EHR records using the classic BERT sequence representation where "
        "visits are separated by a SEP token",
    )
    parser.add_argument(
        "--is_first_time_outcome",
        dest="is_first_time_outcome",
        action="store_true",
        help="is the outcome the first time occurrence?",
    )
    parser.add_argument(
        "--is_remove_index_prediction_starts",
        dest="is_remove_index_prediction_starts",
        action="store_true",
        help="is outcome between index_date and prediction start window removed?",
    )
    parser.add_argument(
        "--is_prediction_window_unbounded",
        dest="is_prediction_window_unbounded",
        action="store_true",
        help="is the end of the prediction window unbounded?",
    )
    parser.add_argument(
        "--is_observation_window_unbounded",
        dest="is_observation_window_unbounded",
        action="store_true",
        help="is the observation window unbounded?",
    )
    parser.add_argument(
        "--include_concept_list",
        dest="include_concept_list",
        action="store_true",
        help="Apply the filter to remove low-frequency concepts",
    )
    parser.add_argument(
        "--allow_measurement_only",
        dest="allow_measurement_only",
        action="store_true",
        help="Indicate whether we allow patients with measurements only",
    )
    parser.add_argument(
        "--is_population_estimation",
        dest="is_population_estimation",
        action="store_true",
        help="Indicate whether the cohort is constructed for population level " "estimation",
    )
    parser.add_argument(
        "--att_type",
        dest="att_type",
        action="store",
        choices=[e.value for e in AttType],
    )
    parser.add_argument(
        "--exclude_demographic",
        dest="exclude_demographic",
        action="store_true",
        help="Indicate whether we should exclude the demographic prompt of the patient sequence",
    )
    parser.add_argument(
        "--use_age_group",
        dest="use_age_group",
        action="store_true",
        help="Indicate whether we should age group to represent the age at the first event in the "
        "patient sequence",
    )
    parser.add_argument(
        "--single_contribution",
        dest="single_contribution",
        action="store_true",
        help="Indicate whether we should contribute once to the training data",
    )
    return parser.parse_args()
