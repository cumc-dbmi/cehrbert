from trainers.train_bert_only import *
from models.bert_models import *
from models.custom_layers import get_custom_objects


class TemporalBertTrainer(BertTrainer):
    confidence_penalty = 0.1

    def __init__(self, input_folder,
                 time_attention_folder,
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

        super(TemporalBertTrainer, self).__init__(input_folder=input_folder,
                                                  output_folder=output_folder,
                                                  concept_embedding_size=concept_embedding_size,
                                                  max_seq_length=max_seq_length,
                                                  time_window_size=time_window_size,
                                                  depth=depth,
                                                  num_heads=num_heads,
                                                  batch_size=batch_size,
                                                  epochs=epochs,
                                                  learning_rate=learning_rate,
                                                  tf_board_log_path=tf_board_log_path)
        self.time_attention_folder = time_attention_folder
        self.time_attention_model_path = os.path.join(time_attention_folder, 'model_time_aware_embeddings.h5')

    def create_model(self, vocabulary_size):
        another_strategy = tf.distribute.OneDeviceStrategy("/cpu:0")
        with another_strategy.scope():
            time_attention_model = tf.keras.models.load_model(self.time_attention_model_path,
                                                              custom_objects=dict(**get_custom_objects()))
            pre_trained_embedding_layer = time_attention_model.get_layer('time_attention').embedding_layer
            weights = pre_trained_embedding_layer.get_weights()

        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            if os.path.exists(self.model_path):
                model = tf.keras.models.load_model(self.model_path, custom_objects=get_custom_objects())
            else:
                model = transformer_temporal_bert_model(
                    max_seq_length=self.max_seq_length,
                    time_window_size=self.time_window_size,
                    vocabulary_size=vocabulary_size,
                    concept_embedding_size=self.concept_embedding_size,
                    depth=self.depth,
                    num_heads=self.num_heads,
                    time_attention_trainable=False)

                optimizer = optimizers.Adam(
                    lr=self.learning_rate, beta_1=0.9, beta_2=0.999)

                model.compile(
                    optimizer,
                    loss=MaskedPenalizedSparseCategoricalCrossentropy(self.confidence_penalty),
                    metrics={'concept_predictions': masked_perplexity})

                self_attention_layer_name = [layer.name for layer in model.layers if
                                             'time_self_attention' in layer.name]
                if self_attention_layer_name:
                    model.get_layer(self_attention_layer_name[0]).set_weights(weights)
        return model


def main(args):
    trainer = TemporalBertTrainer(input_folder=args.input_folder,
                                  time_attention_folder=args.time_attention_folder,
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


def create_parse_args_temporal_bert():
    parser = create_parse_args_base_bert()
    parser.add_argument('-ti',
                        '--time_attention_folder',
                        dest='time_attention_folder',
                        action='store',
                        help='The path for your time attention input_folder where the raw data is',
                        required=True)
    return parser


if __name__ == "__main__":
    main(create_parse_args_temporal_bert().parse_args())
