import logging
from statistics import mean

import os
import sys
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

from utils.model_utils import tokenize_one_field
from data_generators.data_generator_base import GptDataGenerator
from models.layers.custom_layers import get_custom_objects
from trainers.model_trainer import find_tokenizer_path

logger = logging.getLogger("model_overfitting_analysis")


def main(args):
    @tf.function
    def distributed_inference(dist_inputs):
        per_replica_losses = strategy.run(model, args=(dist_inputs,))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        )

    validation_data = pd.read_parquet(args.validation_data_folder)
    tokenizer_path = find_tokenizer_path(args.model_folder)
    tokenizer = tokenize_one_field(
        validation_data,
        "concept_ids",
        "token_ids",
        tokenizer_path
    )
    gpt_data_generator = GptDataGenerator(
        training_data=validation_data,
        batch_size=args.batch_size,
        max_seq_len=512,
        concept_tokenizer=tokenizer,
        min_num_of_visits=2,
        max_num_of_visits=1e10,
        min_num_of_concepts=20,
        including_long_sequence=False,
        sampling_dataset_enabled=False,
        is_random_cursor=False,
        is_pretraining=False,
    )

    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        existing_model_path = os.path.join(args.model_folder, "bert_model.h5")
        if not os.path.exists(existing_model_path):
            sys.exit(f"The model can not be loaded from {existing_model_path}!")
        logger.info(f"The model is loaded from {existing_model_path}")
        model = tf.keras.models.load_model(
            existing_model_path, custom_objects=get_custom_objects()
        )

    losses = []

    batch_iterator = gpt_data_generator.create_batch_generator()
    for i, each_batch in tqdm(
            enumerate(batch_iterator), total=gpt_data_generator.get_steps_per_epoch()
    ):
        inputs, outputs = each_batch
        predictions = distributed_inference(inputs)
        y_true_val = outputs["concept_predictions"][:, :, 0]
        mask = tf.cast(outputs["concept_predictions"][:, :, 1], dtype=tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_val, predictions)
        masked_loss = tf.reduce_mean(loss * mask, axis=-1).numpy().tolist()
        losses.extend(masked_loss)
        print(f"Batch {i}: Val loss: {mean(masked_loss)}\n")

    print(f"Average Val loss: {mean(losses)}\n")


def create_argparser():
    import argparse

    parser = argparse.ArgumentParser(
        description="Model Overfitting Analysis Arguments using the GPT model"
    )
    parser.add_argument(
        "--validation_data_folder",
        dest="validation_data_folder",
        action="store",
        help="The path for where the validation data folder",
        required=True,
    )
    parser.add_argument(
        "--model_folder",
        dest="model_folder",
        action="store",
        help="The path to trained model folder",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        action="store",
        required=False,
        default=32,
    )
    return parser


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    main(create_argparser().parse_args())
