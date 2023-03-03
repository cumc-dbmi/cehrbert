import os
import argparse
import random
import pandas as pd
import pickle
import tensorflow as tf
from models.layers.custom_layers import get_custom_objects
from models.gpt_model import generate_patient_history
from omop_converter import gpt_to_omop_converter


def detokenize(
        number,
        tokenizer,
        concept_map
):
    concept_id = tokenizer.decode([[number]])[0]
    if concept_id in concept_map:
        return concept_map[concept_id]
    return concept_id


def main(
        args
):
    tokenizer_path = os.path.join(args.model_folder, 'tokenizer.pickle')
    model_path = os.path.join(args.model_folder, 'bert_model.h5')
    model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    concept_map = dict()
    concept_ids = tokenizer.tokenizer.word_index.keys()
    concept = pd.read_parquet(args.concept_path)
    for t in concept.itertuples():
        if str(t.concept_id) in concept_ids:
            concept_map[str(t.concept_id)] = t.concept_name

    demographic_info = None
    if args.demographic_data_path:
        demographic_info = pd.read_parquet(
            args.demographic_data_path
        ).concept_ids.apply(lambda concept_list: concept_list[0:4])
        demographic_info = tokenizer.encode(map(list, demographic_info))
    person_id: int = 0
    visit_occurrence_id: int = 1
    condition_occurrence_id: int = 1
    procedure_occurrence_id: int = 1
    drug_exposure_id: int = 1
    while True:
        person_id += 1
        start_tokens = [tokenizer.get_start_token_id()]
        if demographic_info is not None:
            # Randomly sample a patient from the population
            start_tokens.extend(random.sample(demographic_info, 1)[0])
        tokens_generated = generate_patient_history(
            model,
            start_tokens,
            tokenizer,
            args.context_window,
            args.top_k
        )
        [visit_occurrence_id, condition_occurrence_id, procedure_occurrence_id, drug_exposure_id] = gpt_to_omop_converter(concept,
                                                                                                     person_id,
                                                                                                     tokenizer,
                                                                                                     start_tokens,
                                                                                                     tokens_generated,
                                                                                                     visit_occurrence_id,
                                                                                                     condition_occurrence_id,
                                                                                                     procedure_occurrence_id,
                                                                                                     drug_exposure_id)

        txt = '\n'.join(
            [detokenize(_, tokenizer, concept_map) for _ in start_tokens[:5] + tokens_generated]
        )
        print(txt)

        x = input('Enter yes to generate another patient sequence:')
        if x != 'yes':
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for generating patient sequences')

    parser.add_argument(
        '--model_folder',
        dest='model_folder',
        action='store',
        help='The path for your model_folder',
        required=True
    )
    parser.add_argument(
        '--context_window',
        dest='context_window',
        action='store',
        type=int,
        help='The context window of the gpt model',
        required=True
    )
    parser.add_argument(
        '--top_k',
        dest='top_k',
        action='store',
        default=10,
        type=int,
        help='The number of top concepts to sample',
        required=False
    )
    parser.add_argument(
        '--concept_path',
        dest='concept_path',
        action='store',
        help='The path for your concept_path',
        required=True
    )
    parser.add_argument(
        '--demographic_data_path',
        dest='demographic_data_path',
        action='store',
        help='The path for your concept_path',
        required=False
    )

    main(parser.parse_args())
