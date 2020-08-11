from config.parse_args import create_parse_args_temporal_bert
from config.model_configs import TemporalBertConfig, create_temporal_bert_model_config

from trainers.train_bert_only import *
from models.bert_models import *
from models.custom_layers import get_custom_objects


class TemporalBertTrainer(BertTrainer):
    confidence_penalty = 0.1

    def __init__(self, config: TemporalBertConfig):
        super(TemporalBertTrainer, self).__init__(config)
        self.time_attention_model_path = config.time_attention_model_path

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
    TemporalBertTrainer(create_temporal_bert_model_config(args)).run()


if __name__ == "__main__":
    main(create_parse_args_temporal_bert().parse_args())
