import argparse
import datetime
import os
import pickle
import random
import uuid

import numpy as np
import pandas as pd
import tensorflow as tf

from models.gpt_model import GptInferenceModel, TopKStrategy, TopPStrategy
from models.layers.custom_layers import get_custom_objects


def generate_single_batch(
        gpt_inference_model,
        tokenizer,
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

    prompt_batch = tf.cast(prompt_batch, dtype=tf.int32)

    prompt_batch = gpt_inference_model(
        prompt_batch
    )

    return [s.split(' ') for s in tokenizer.decode(prompt_batch)]


def main(
        args
):
    tokenizer_path = os.path.join(args.model_folder, 'tokenizer.pickle')
    model_path = os.path.join(args.model_folder, 'bert_model.h5')
    tokenizer = pickle.load(open(tokenizer_path, 'rb'))
    strategy = tf.distribute.MirroredStrategy()
    # atexit.register(strategy._extended._collective_ops._pool.close)  # type: ignore
    # atexit.register(strategy._extended._cross_device_ops._pool.close) # type: ignore
    # atexit.register(strategy._extended._host_cross_device_ops._pool.close) #type: ignore
    print(f'{datetime.datetime.now()}: Loading tokenizer at {tokenizer_path}')
    print(f'{datetime.datetime.now()}: Loading model at {model_path}')
    print(f'{datetime.datetime.now()}: Write sequences to {args.output_folder}')
    print(f'{datetime.datetime.now()}: Context window {args.context_window}')
    print(f'{datetime.datetime.now()}: Temperature {args.temperature}')
    print(f'{datetime.datetime.now()}: Sampling Strategy {args.sampling_strategy}')
    print(f'{datetime.datetime.now()}: Top P {args.top_p}')
    print(f'{datetime.datetime.now()}: Top K {args.top_k}')
    with strategy.scope():
        model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())

        sampling_strategy = None
        if args.sampling_strategy == TopKStrategy.__name__:
            sampling_strategy = TopKStrategy(top_k=args.top_k, temperature=args.temperature)
        if args.sampling_strategy == TopPStrategy.__name__:
            sampling_strategy = TopPStrategy(top_p=args.top_p, temperature=args.temperature)
        if not sampling_strategy:
            raise RuntimeError(
                'sampling_strategy has to be one of the following two options TopKStrategy or TopPStrategy'
            )

        gpt_inference_model = GptInferenceModel(
            gpt_model=model,
            tokenizer=tokenizer,
            context_window=args.context_window,
            sampling_strategy=sampling_strategy
        )

    print(f'{datetime.datetime.now()}: Loading demographic_info at {args.demographic_data_path}')
    # data = dd.read_parquet(
    #     args.demographic_data_path,
    #     engine='pyarrow'
    # )
    data = pd.read_parquet(
        args.demographic_data_path
    )
    demographic_info = data.concept_ids.apply(lambda concept_list: concept_list[0:4])
    demographic_info = tokenizer.encode(map(list, demographic_info))

    num_of_batches = args.num_of_patients // args.batch_size + 1
    sequence_to_flush = []
    for i in range(num_of_batches):
        print(f'{datetime.datetime.now()}: Batch {i} started')
        batch_sequence = generate_single_batch(
            gpt_inference_model,
            tokenizer,
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

        if len(sequence_to_flush) >= args.buffer_size:
            print(f'{datetime.datetime.now()}: Flushing to the Disk at Batch {i}')
            pd.DataFrame(
                sequence_to_flush,
                columns=['concept_ids']
            ).to_parquet(os.path.join(args.output_folder, f'{uuid.uuid4()}.parquet'))
            sequence_to_flush.clear()

    if len(sequence_to_flush) > 0:
        print(f'{datetime.datetime.now()}: Flushing to the Disk at Final Batch')
        pd.DataFrame(
            sequence_to_flush,
            columns=['concept_ids']
        ).to_parquet(os.path.join(args.output_folder, f'{uuid.uuid4()}-last.parquet'))


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
        '--sampling_strategy',
        dest='sampling_strategy',
        action='store',
        choices=[TopPStrategy.__name__, TopKStrategy.__name__],
        help='Pick the sampling strategy between top_k and top_p',
        required=True
    )
    parser.add_argument(
        '--top_k',
        dest='top_k',
        action='store',
        default=100,
        type=int,
        help='The number of top concepts to sample',
        required=False
    )
    parser.add_argument(
        '--top_p',
        dest='top_p',
        action='store',
        default=1.0,
        type=float,
        help='The accumulative probability of top concepts to sample',
        required=False
    )
    parser.add_argument(
        '--demographic_data_path',
        dest='demographic_data_path',
        action='store',
        help='The path for your concept_path',
        required=True
    )
    parser.add_argument(
        '--temperature',
        dest='temperature',
        action='store',
        default=1.0,
        type=float,
        help='The temperature parameter for softmax',
        required=False
    )

    main(parser.parse_args())
