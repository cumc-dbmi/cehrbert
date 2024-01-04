import argparse
from typing import Union

import dask.dataframe as dd
from analyses.gpt.privacy.patient_index.whoosh_index import PatientDataIndex


def create_argparser():
    parser = argparse.ArgumentParser(
        description='Patient index builder arguments'
    )
    parser.add_argument(
        '--patient_sequence_folder',
        dest='patient_sequence_folder',
        action='store',
        help='The path for where the patient sequence folder',
        required=True
    )
    parser.add_argument(
        '--index_folder',
        dest='index_folder',
        action='store',
        help='The output folder that stores the index',
        required=True
    )
    parser.add_argument(
        '--rebuilt',
        dest='rebuilt',
        action='store_true',
        help='Indicate whether the index should be overwritten and rebuilt'
    )
    parser.add_argument(
        '--set_unique_concepts',
        dest='set_unique_concepts',
        action='store_true',
        help='Indicate whether to use the unique set of concepts for each patient'
    )
    parser.add_argument(
        '--attribute_config',
        dest='attribute_config',
        action='store',
        help='The configuration yaml file for common and sensitive attributes',
        required=False
    )
    return parser


if __name__ == "__main__":
    import yaml

    args = create_argparser().parse_args()

    common_attributes = None
    sensitive_attributes = None

    if args.attribute_config:
        try:
            with open(args.attribute_config, 'r') as file:
                data = yaml.safe_load(file)
            if 'common_attributes' in data:
                common_attributes = data['common_attributes']
            if 'sensitive_attributes' in data:
                sensitive_attributes = data['sensitive_attributes']
        except FileNotFoundError:
            print(f"The file {args.attribute_config} was not found")
        except Union[PermissionError, OSError] as e:
            print(e)

    patient_data_index = PatientDataIndex(
        index_folder=args.index_folder,
        rebuilt=args.rebuilt,
        set_unique_concepts=args.set_unique_concepts,
        common_attributes=common_attributes,
        sensitive_attributes=sensitive_attributes
    )
    dataset = dd.read_parquet(args.patient_sequence_folder)
    patient_data_index.build_index(dataset)
