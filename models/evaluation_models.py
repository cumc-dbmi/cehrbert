import tensorflow as tf

from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

from models.custom_layers import get_custom_objects


def create_bi_lstm_model(max_seq_length, vocab_size, embedding_size, concept_embeddings):
    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='concept_ids')

    if concept_embeddings is not None:
        embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                                    embedding_size,
                                                    embeddings_initializer=Constant(
                                                        concept_embeddings),
                                                    mask_zero=True)
    else:
        embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                                    embedding_size,
                                                    mask_zero=True)

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation='relu')

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    next_input = embedding_layer(concept_ids)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    model = Model(inputs=concept_ids, outputs=output, name='Vanilla_BI_LSTM')

    return model


def create_vanilla_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path):
    age_of_visit_input = tf.keras.layers.Input(name='ages', shape=(1,))

    vanilla_bert_model = tf.keras.models.load_model(vanilla_bert_model_path,
                                                    custom_objects=dict(**get_custom_objects()))
    contextualized_embeddings, _ = vanilla_bert_model.get_layer('encoder').output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    mask_input = vanilla_bert_model.inputs[-1]
    mask_embeddings = tf.tile(tf.expand_dims(mask_input == 0, -1, name='expand_mask'),
                              [1, 1, embedding_size], name='tile_mask')
    contextualized_embeddings = tf.math.multiply(contextualized_embeddings,
                                                 tf.cast(mask_embeddings, dtype=tf.float32,
                                                         name='cast_mask'))

    masking_layer = tf.keras.layers.Masking(mask_value=0.,
                                            input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation='relu')

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    next_input = masking_layer(contextualized_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, age_of_visit_input])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_vanilla_bert = Model(inputs=vanilla_bert_model.inputs + [age_of_visit_input],
                                   outputs=output, name='Vanilla_BERT_PLUS_BI_LSTM')

    return lstm_with_vanilla_bert


def create_temporal_bert_bi_lstm_model(max_seq_length, temporal_bert_model_path):
    temporal_bert_model = tf.keras.models.load_model(temporal_bert_model_path,
                                                     custom_objects=dict(**get_custom_objects()))
    _, _, embedding_size = temporal_bert_model.get_layer('temporal_encoder').output[
        0].get_shape().as_list()
    contextualized_embeddings, _, _ = temporal_bert_model.get_layer('temporal_encoder').output

    age_of_visit_input = tf.keras.layers.Input(name='ages', shape=(1,))

    mask_input = temporal_bert_model.inputs[-1]
    mask_embeddings = tf.cast(
        tf.tile(tf.expand_dims(mask_input == 0, -1, name='expand_mask'), [1, 1, embedding_size],
                name='tile_mask'), tf.float32, name='cast_mask_embeddings')
    contextualized_embeddings = tf.math.multiply(contextualized_embeddings, mask_embeddings)

    masking_layer = tf.keras.layers.Masking(mask_value=0.,
                                            input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.1)

    dense_layer = tf.keras.layers.Dense(64, activation='tanh')

    dropout_dense_layer = tf.keras.layers.Dropout(0.1)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    next_input = masking_layer(contextualized_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, age_of_visit_input])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    return Model(inputs=temporal_bert_model.inputs + [age_of_visit_input], outputs=output,
                 name='TEMPORAL_BERT_PLUS_BI_LSTM')
