from config.parse_args import create_parse_args_temporal_bert
from config.model_configs import create_temporal_bert_model_config

from trainers.train_bert_only import *
from models.bert_models_visit_prediction import transformer_temporal_bert_model_visit_prediction
from models.bert_models import transformer_temporal_bert_model
from models.custom_layers import get_custom_objects


class TemporalBertTrainer(VanillaBertTrainer):
    confidence_penalty = 0.1

    def __init__(self, time_attention_model_path, *args, **kwargs):

        self._time_attention_model_path = time_attention_model_path
        self._time_attention_weights = self._load_time_attention_model()
        self._time_window_size = np.shape(self._time_attention_weights[0])[1] - 1

        super(TemporalBertTrainer, self).__init__(*args, **kwargs)

        self.get_logger().info(
            f'time_attention_model_path: {time_attention_model_path}\n'
            f'time_window_size: {self._time_window_size}\n')

    def _load_time_attention_model(self):

        def extract_time_attention_weights():
            with tf.distribute.OneDeviceStrategy("/cpu:0").scope():
                time_attention_model = tf.keras.models.load_model(self._time_attention_model_path,
                                                                  custom_objects=dict(
                                                                      **get_custom_objects()))
                pre_trained_embedding_layer = time_attention_model.get_layer(
                    'time_attention').embedding_layer

                return pre_trained_embedding_layer.get_weights()

        if not os.path.exists(self._time_attention_model_path):
            raise FileExistsError(f'{self._time_attention_model_path} does not exist')

        return extract_time_attention_weights()

    def _create_model(self):
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            if os.path.exists(self._model_path):
                self.get_logger().info(
                    f'The {self.__class__.__name__} model will be loaded from {self._model_path}')
                model = tf.keras.models.load_model(self._model_path,
                                                   custom_objects=get_custom_objects())
            else:

                optimizer = optimizers.Adam(
                    lr=self._learning_rate, beta_1=0.9, beta_2=0.999)

                if self._include_visit_prediction:
                    model = transformer_temporal_bert_model_visit_prediction(
                        max_seq_length=self._context_window_size,
                        time_window_size=self._time_window_size,
                        concept_vocab_size=self._tokenizer.get_vocab_size(),
                        visit_vocab_size=self._visit_tokenizer.get_vocab_size(),
                        embedding_size=self._embedding_size,
                        depth=self._depth,
                        num_heads=self._num_heads,
                        time_attention_trainable=False)

                    losses = {
                        'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty),
                        'visit_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty)
                    }
                else:
                    model = transformer_temporal_bert_model(
                        max_seq_length=self._context_window_size,
                        time_window_size=self._time_window_size,
                        vocab_size=self._tokenizer.get_vocab_size(),
                        embedding_size=self._embedding_size,
                        depth=self._depth,
                        num_heads=self._num_heads,
                        time_attention_trainable=False)

                    losses = {
                        'concept_predictions': MaskedPenalizedSparseCategoricalCrossentropy(
                            self.confidence_penalty)
                    }

                model.compile(optimizer, loss=losses,
                              metrics={'concept_predictions': masked_perplexity})

                self_attention_layer_name = [layer.name for layer in model.layers if
                                             'time_self_attention' in layer.name]
                if self_attention_layer_name:
                    model.get_layer(self_attention_layer_name[0]).set_weights(
                        self._time_attention_weights)
        return model

    def create_data_generator(self) -> TemporalBertDataGenerator:

        if self._include_visit_prediction:
            data_generator_class = TemporalVisitPredictionBertDataGenerator
        else:
            data_generator_class = TemporalBertDataGenerator

        data_generator = data_generator_class(
            training_data=self._training_data,
            batch_size=self._batch_size,
            max_seq_len=self._context_window_size,
            min_num_of_concepts=self.min_num_of_concepts,
            concept_tokenizer=self._tokenizer,
            visit_tokenizer=self._visit_tokenizer,
            time_window_size=self._time_window_size)

        return data_generator


def main(args):
    config = create_temporal_bert_model_config(args)
    TemporalBertTrainer(training_data_parquet_path=config.parquet_data_path,
                        time_attention_model_path=config.time_attention_model_path,
                        model_path=config.model_path,
                        tokenizer_path=config.tokenizer_path,
                        visit_tokenizer_path=config.visit_tokenizer_path,
                        embedding_size=config.concept_embedding_size,
                        context_window_size=config.max_seq_length,
                        depth=config.depth,
                        num_heads=config.num_heads,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        learning_rate=config.learning_rate,
                        include_visit_prediction=config.include_visit_prediction,
                        tf_board_log_path=config.tf_board_log_path).train_model()


if __name__ == "__main__":
    main(create_parse_args_temporal_bert().parse_args())
