import argparse
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
    return parser


if __name__ == "__main__":
    args = create_argparser().parse_args()
    patient_data_index = PatientDataIndex(
        index_folder=args.index_folder,
        rebuilt=args.rebuilt
    )
    dataset = dd.read_parquet(args.patient_sequence_folder)
    patient_data_index.build_index(dataset)
