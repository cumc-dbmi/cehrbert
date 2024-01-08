import argparse
import sys
from typing import Union

import dask.dataframe as dd
from analyses.gpt.privacy.patient_index.weaviate_indexer import PatientDataWeaviateDocumentIndex


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
        '--server_name',
        dest='server_name',
        action='store',
        required=True
    )
    parser.add_argument(
        '--index_name',
        dest='index_name',
        action='store',
        help='The output folder that stores the index',
        required=True
    )
    parser.add_argument(
        '--tokenizer_path',
        dest='tokenizer_path',
        action='store',
        help='The path to ConceptTokenizer',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        action='store',
        type=int,
        default=1024,
        required=False
    )
    parser.add_argument(
        '--rebuilt',
        dest='rebuilt',
        action='store_true',
        help='Indicate whether the index should be overwritten and rebuilt'
    )
    parser.add_argument(
        '--incremental_built',
        dest='incremental_built',
        action='store_true',
        help='Indicate whether the index should be built incrementally'
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
    import pickle
    import traceback

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

    try:
        concept_tokenizer = pickle.load(open(args.tokenizer_path, 'rb'))
    except (AttributeError, EOFError, ImportError, IndexError, OSError) as e:
        sys.exit(traceback.format_exc(e))
    except Exception as e:
        # everything else, possibly fatal
        sys.exit(traceback.format_exc(e))

    patient_data_index = PatientDataWeaviateDocumentIndex(
        index_name=args.index_name,
        server_name=args.server_name,
        rebuilt=args.rebuilt,
        incremental_built=args.incremental_built,
        concept_tokenizer=concept_tokenizer,
        set_unique_concepts=args.set_unique_concepts,
        common_attributes=common_attributes,
        sensitive_attributes=sensitive_attributes,
        batch_size=args.batch_size
    )
    dataset = dd.read_parquet(args.patient_sequence_folder)
    patient_data_index.build_index(dataset)
