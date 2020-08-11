from keras_transformer.bert import (masked_perplexity,
                                    MaskedPenalizedSparseCategoricalCrossentropy)

from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

from trainers.train_time_aware_embeddings import *
from models.bert_models import *
from utils.utils import CosineLRSchedule
from models.custom_layers import get_custom_objects

from data_generators.data_generator import BertBatchGenerator


class BertTrainer(Trainer):
    confidence_penalty = 0.1

    def __init__(self, input_folder,
                 output_folder,
                 concept_embedding_size,
                 max_seq_length,
                 time_window_size,
                 depth,
                 num_heads,
                 batch_size,
                 epochs,
                 learning_rate,
                 tf_board_log_path):

        super(BertTrainer, self).__init__(input_folder=input_folder,
                                          output_folder=output_folder,
                                          concept_embedding_size=concept_embedding_size,
                                          max_seq_length=max_seq_length,
                                          time_window_size=time_window_size,
                                          batch_size=batch_size,
                                          epochs=epochs,
                                          learning_rate=learning_rate,
                                          tf_board_log_path=tf_board_log_path)
        self.depth = depth
        self.num_heads = num_heads

    def create_tf_dataset(self, tokenizer, training_data):
        data_generator = BertBatchGenerator(patient_event_sequence=training_data,
                                            mask_token_id=tokenizer.get_mask_token_id(),
                                            unused_token_id=tokenizer.get_unused_token_id(),
                                            max_sequence_length=self.max_seq_length,
                                            batch_size=self.batch_size,
                                            first_token_id=tokenizer.get_first_token_index(),
                                            last_token_id=tokenizer.get_last_token_index())

        dataset = tf.data.Dataset.from_generator(data_generator.batch_generator,
                                                 output_types=({'masked_concept_ids': tf.int32,
                                                                'concept_ids': tf.int32,
                                                                'time_stamps': tf.int32,
                                                                'visit_orders': tf.int32,
                                                                'mask': tf.int32}, tf.int32))
        return dataset, data_generator.get_steps_per_epoch()

    def train(self, vocabulary_size, dataset, val_dataset, steps_per_epoch, val_steps_per_epoch):

        model = self.create_model(vocabulary_size)

        lr_scheduler = callbacks.LearningRateScheduler(
            CosineLRSchedule(lr_high=self.learning_rate, lr_low=1e-8,
                             initial_period=10),
            verbose=1)

        model_callbacks = [
            callbacks.ModelCheckpoint(
                filepath=self.model_path,
                save_best_only=True,
                verbose=1),
            lr_scheduler,
        ]

        model.fit(
            dataset,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            callbacks=model_callbacks,
            validation_data=val_dataset,
            validation_steps=val_steps_per_epoch
        )

    def create_model(self, vocabulary_size):
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            if os.path.exists(self.model_path):
                model = tf.keras.models.load_model(self.model_path, custom_objects=get_custom_objects())
            else:
                optimizer = optimizers.Adam(
                    lr=self.learning_rate, beta_1=0.9, beta_2=0.999)

                model = transformer_bert_model(
                    max_seq_length=self.max_seq_length,
                    vocabulary_size=vocabulary_size,
                    concept_embedding_size=self.concept_embedding_size,
                    depth=self.depth,
                    num_heads=self.num_heads)

                model.compile(
                    optimizer,
                    loss=MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty),
                    metrics={'concept_predictions': masked_perplexity})
        return model


def create_parse_args_base_bert():
    parser = create_parse_args()

    parser.add_argument('-d',
                        '--depth',
                        dest='depth',
                        action='store',
                        default=5,
                        required=False)

    parser.add_argument('-nh',
                        '--num_heads',
                        dest='num_heads',
                        action='store',
                        default=8,
                        required=False)
    return parser


def main(args):
    trainer = BertTrainer(input_folder=args.input_folder,
                          output_folder=args.output_folder,
                          concept_embedding_size=args.concept_embedding_size,
                          max_seq_length=args.max_seq_length,
                          time_window_size=args.time_window_size,
                          depth=args.depth,
                          num_heads=args.num_heads,
                          batch_size=args.batch_size,
                          epochs=args.epochs,
                          learning_rate=args.learning_rate,
                          tf_board_log_path=args.tf_board_log_path)

    trainer.run()


if __name__ == "__main__":
    main(create_parse_args_base_bert().parse_args())
