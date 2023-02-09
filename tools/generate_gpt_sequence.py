import os
import argparse
import pandas as pd
import pickle
import tensorflow as tf
from models.layers.custom_layers import get_custom_objects
from models.gpt_model import generate_patient_history


def detokenize(
        number,
        tokenizer,
        concept_map
):
    concept_id = tokenizer.decode([[number]])[0]
    if concept_id in concept_map:
        return concept_map[concept_id]
    return concept_id


def main(args):
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

    while True:

        tokens_generated = generate_patient_history(
            model,
            [tokenizer.get_start_token_id()],
            tokenizer,
            100,
            10
        )

        txt = '\n'.join(
            [detokenize(_, tokenizer, concept_map) for _ in tokens_generated]
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
        '--concept_path',
        dest='concept_path',
        action='store',
        help='The path for your concept_path',
        required=True
    )

    args = parser.parse_args()

    main(args)
