import logging
from statistics import mean

import os
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from keras_transformer.bert import masked_perplexity

from utils.model_utils import tokenize_one_field
from data_generators.data_generator_base import GptDataGenerator
from data_generators.learning_objective import CustomLearningObjective
from models.layers.custom_layers import get_custom_objects
from utils.checkpoint_utils import find_tokenizer_path, find_latest_checkpoint_path, checkpoint_exists

logger = logging.getLogger("model_overfitting_analysis")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        max_seq_len=args.max_seq_len,
        concept_tokenizer=tokenizer,
        min_num_of_visits=args.min_num_of_visits,
        max_num_of_visits=int(1e10),
        min_num_of_concepts=args.min_num_of_concepts,
        including_long_sequence=args.including_long_sequence,
        sampling_dataset_enabled=False,
        is_random_cursor=False,
        is_pretraining=False,
    )

    gpt_data_generator._learning_objectives.append(
        CustomLearningObjective(input_schema={'person_id': tf.int32}, output_schema={'label': tf.int32})
    )

    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))
    with strategy.scope():
        if args.checkpoint_name:
            if checkpoint_exists(args.model_folder, args.checkpoint_name):
                existing_model_path = os.path.join(args.model_folder, args.checkpoint_name)
            else:
                logger.info(
                    f'The checkpoint {args.checkpoint_name} does not exist in {args.model_folder}. '
                    f'The latest checkpoint will be used.'
                )
                existing_model_path = find_latest_checkpoint_path(args.model_folder)
        else:
            logger.info(
                f'The checkpoint {args.checkpoint_name} is not provided. '
                f'The latest checkpoint will be used in {args.model_folder}.'
            )
            existing_model_path = find_latest_checkpoint_path(args.model_folder)

        logger.info(f"The model is loaded from {existing_model_path}")
        model = tf.keras.models.load_model(
            existing_model_path, custom_objects=get_custom_objects()
        )

    person_ids = []
    losses = []
    labels = []
    perplexities = []

    batch_iterator = gpt_data_generator.create_batch_generator()
    for i, each_batch in tqdm(
            enumerate(batch_iterator), total=gpt_data_generator.get_steps_per_epoch()
    ):
        inputs, outputs = each_batch
        person_ids.extend(inputs['person_id'])
        labels.extend(outputs['label'])
        predictions = distributed_inference(inputs)
        y_true_val = outputs["concept_predictions"][:, :, 0]
        mask = tf.cast(outputs["concept_predictions"][:, :, 1], dtype=tf.float32)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_val, predictions)
        masked_loss = tf.reduce_mean(loss * mask, axis=-1).numpy().tolist()
<<<<<<< Updated upstream
        perplexity = masked_perplexity(predictions)
        losses.extend(masked_loss)
        perplexities.extend(perplexity)
=======
        perplexity = masked_perplexity(y_true=outputs['concept_predictions'], y_pred=predictions)
>>>>>>> Stashed changes
        print(f"Batch {i}: Val loss: {mean(masked_loss)}. Val perplexity: {mean(perplexity)}\n")

        results_df = pd.DataFrame(
            zip(person_ids, losses, perplexities, labels),
            columns=['person_id', 'loss', 'perplexity', 'label']
        )
        current_time = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        results_df.to_parquet(os.path.join(args.output_folder, f'{current_time}.parquet'))
        # Clear the lists for the next batch
        person_ids.clear()
        losses.clear()
        labels.clear()

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
        "--output_folder",
        dest="output_folder",
        action="store",
        help="The output folder for storing the example losses",
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
        "--checkpoint_name",
        dest="checkpoint_name",
        action="store",
        help="The name of the checkpoint",
        required=False,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        action="store",
        required=False,
        default=32,
    )
    parser.add_argument(
        "--max_seq_len",
        dest="max_seq_len",
        type=int,
        action="store",
        required=False,
        default=512
    )
    parser.add_argument(
        "--min_num_of_visits",
        dest="min_num_of_visits",
        type=int,
        action="store",
        required=False,
        default=1
    )
    parser.add_argument(
        "--min_num_of_concepts",
        dest="min_num_of_concepts",
        type=int,
        action="store",
        required=False,
        default=10
    )
    parser.add_argument(
        "--including_long_sequence",
        dest="including_long_sequence",
        action="store_true"
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
