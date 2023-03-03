import os
import argparse
import random
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import uuid
from models.layers.custom_layers import get_custom_objects
from data_generators.learning_objective import post_pad_pre_truncate


def generate_single_batch(
        model,
        tokenizer,
        context_window,
        batch_size,
        demographic_info
):
    start_tokens = np.tile(
        np.asarray([[tokenizer.get_start_token_id()]]),
        [batch_size, 1]
    )
    random_prompts = random.sample(
        demographic_info,
        batch_size
    )
    prompt_batch = np.hstack([start_tokens, random_prompts])
    _, length = np.shape(
        prompt_batch
    )
    while length < context_window:

        token_index = length - 1

        padded_prompt_batch = post_pad_pre_truncate(
            prompt_batch,
            0,
            context_window
        )

        predictions = model.predict(padded_prompt_batch)

        pred_logits, indices = tf.math.top_k(predictions, k=10, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = tf.keras.activations.softmax(pred_logits)
        preds = np.asarray(preds).astype("float32")

        next_token_indices = indices[:, token_index, :]
        next_token_logits = preds[:, token_index, :]

        next_tokens = tf.gather(
            next_token_indices,
            tf.random.categorical(next_token_logits, 1),
            axis=1,
            batch_dims=1
        ).numpy()

        prompt_batch = np.hstack(
            [prompt_batch, next_tokens]
        )

        _, length = np.shape(
            prompt_batch
        )

        # This indicates all the sequences have ended
        if np.all(np.any(prompt_batch == tokenizer.get_end_token_id(), axis=-1)):
            break

    return [s.split(' ') for s in tokenizer.decode(prompt_batch)]


def main(
        args
):
    tokenizer_path = os.path.join(args.model_folder, 'tokenizer.pickle')
    model_path = os.path.join(args.model_folder, 'bert_model.h5')
    model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))

    demographic_info = pd.read_parquet(
        args.demographic_data_path
    ).concept_ids.apply(lambda concept_list: concept_list[0:4])
    demographic_info = tokenizer.encode(map(list, demographic_info))

    num_of_batches = args.num_of_patients // args.batch_size + 1

    sequence_to_flush = []
    for i in range(num_of_batches):
        print(f'Batch: {i} started\n')
        batch_sequence = generate_single_batch(
            model,
            tokenizer,
            args.context_window,
            args.batch_size,
            demographic_info
        )
        for seq in batch_sequence:
            seq_copy = []
            for token in seq:
                if token == tokenizer.end_token:
                    break
                seq_copy.append(token)
            sequence_to_flush.append({'concept_ids': seq_copy})

        if len(sequence_to_flush) > args.buffer_size:
            pd.DataFrame(
                sequence_to_flush,
                columns=['concept_ids']
            ).to_parquet(os.path.join(args.output_folder, f'{uuid.uuid4()}.parquet'))
            sequence_to_flush.clear()

    if len(sequence_to_flush) > 0:
        pd.DataFrame(
            sequence_to_flush,
            columns=['concept_ids']
        ).to_parquet(os.path.join(args.output_folder, f'{uuid.uuid4()}.parquet'))


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
        '--output_folder',
        dest='output_folder',
        action='store',
        help='The path for your generated data',
        required=True
    )
    parser.add_argument(
        '--num_of_patients',
        dest='num_of_patients',
        action='store',
        type=int,
        help='The number of patients that will be generated',
        required=True
    )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        action='store',
        type=int,
        help='batch_size',
        required=True
    )
    parser.add_argument(
        '--buffer_size',
        dest='buffer_size',
        action='store',
        type=int,
        default=100,
        help='buffer_size',
        required=False
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
        '--demographic_data_path',
        dest='demographic_data_path',
        action='store',
        help='The path for your concept_path',
        required=True
    )

    main(parser.parse_args())
