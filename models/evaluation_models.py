import tensorflow as tf

from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

from models.custom_layers import get_custom_objects
from models.custom_layers import ConvolutionBertLayer
from models.bert_models_visit_prediction import transformer_bert_model_visit_prediction


def create_bi_lstm_model(max_seq_length, vocab_size, embedding_size, concept_embeddings):
    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name='age_batch_norm_layer')

    normalized_index_age = age_batch_norm_layer(age_of_visit_input)

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

    next_input = tf.keras.layers.concatenate([next_input, normalized_index_age])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    model = Model(inputs=[concept_ids, age_of_visit_input], outputs=output, name='Vanilla_BI_LSTM')

    return model


def create_vanilla_feed_forward_model(vanilla_bert_model_path):
    """
    BERT + Feedforward model for binary prediction
    :param vanilla_bert_model_path:
    :return:
    """
    age_at_index_date = tf.keras.layers.Input(name='age', shape=(1,))

    vanilla_bert_model = tf.keras.models.load_model(vanilla_bert_model_path,
                                                    custom_objects=dict(**get_custom_objects()))
    bert_inputs = [i for i in vanilla_bert_model.inputs if
                   'visit' not in i.name or ('visit_segment' in i.name
                                             or 'visit_concept_order' in i.name)]

    contextualized_embeddings, _ = vanilla_bert_model.get_layer('encoder').output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    mask_input = [i for i in bert_inputs if
                  'mask' in i.name and 'concept' not in i.name][0]
    mask_embeddings = tf.tile(tf.expand_dims(mask_input == 0, -1, name='bert_expand_ff'),
                              [1, 1, embedding_size], name='bert_tile_ff')
    contextualized_embeddings = tf.math.multiply(contextualized_embeddings,
                                                 tf.cast(mask_embeddings, dtype=tf.float32,
                                                         name='bert_cast_ff'),
                                                 name='bert_multiply_ff')

    output_layer = tf.keras.layers.Dense(1, name='prediction', activation='sigmoid')

    output = output_layer(tf.reduce_sum(contextualized_embeddings, axis=-2))

    lstm_with_vanilla_bert = Model(inputs=bert_inputs + [age_at_index_date],
                                   outputs=output, name='Vanilla_BERT_PLUS_BI_LSTM')

    return lstm_with_vanilla_bert


def create_sliding_bert_model(model_path, max_seq_length, context_window, stride):
    age_at_index_date = tf.keras.layers.Input(name='age', shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name='age_batch_norm_layer')

    normalized_index_age = age_batch_norm_layer(age_at_index_date)

    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                        name='concept_ids')
    visit_segments = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                           name='visit_segments')
    time_stamps = tf.keras.layers.Input(shape=(max_seq_length,),
                                        dtype='int32',
                                        name='time_stamps')
    ages = tf.keras.layers.Input(shape=(max_seq_length,),
                                 dtype='int32',
                                 name='ages')
    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32', name='mask')

    convolution_bert_layer = ConvolutionBertLayer(model_path=model_path,
                                                  seq_len=max_seq_length,
                                                  context_window=context_window,
                                                  stride=stride)

    conv_bert_output = convolution_bert_layer([concept_ids,
                                               visit_segments,
                                               time_stamps,
                                               ages,
                                               mask])

    next_input = tf.keras.layers.concatenate([conv_bert_output, normalized_index_age])

    dropout_conv_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation='tanh')

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, name='prediction', activation='sigmoid')

    output = output_layer(dropout_dense_layer(dense_layer(dropout_conv_layer(next_input))))

    model_inputs = [concept_ids, visit_segments, time_stamps, ages, mask]
    ffd_bert_model = tf.keras.models.Model(inputs=model_inputs + [normalized_index_age],
                                           outputs=output)

    return ffd_bert_model


def create_vanilla_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path):
    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name='age_batch_norm_layer')

    normalized_index_age = age_batch_norm_layer(age_of_visit_input)

    vanilla_bert_model = tf.keras.models.load_model(vanilla_bert_model_path,
                                                    custom_objects=dict(**get_custom_objects()))
    bert_inputs = [i for i in vanilla_bert_model.inputs if
                   'visit' not in i.name or ('visit_segment' in i.name
                                             or 'visit_concept_order' in i.name)]
    contextualized_embeddings, _ = vanilla_bert_model.get_layer('encoder').output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    mask_input = [i for i in bert_inputs if
                  'mask' in i.name and 'concept' not in i.name][0]
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

    # attach a property to the concept embeddings to indicate where are the masks, send the flag to the downstream layer
    next_input = masking_layer(
        contextualized_embeddings
    )

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, normalized_index_age])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_vanilla_bert = Model(inputs=bert_inputs + [age_of_visit_input],
                                   outputs=output, name='Vanilla_BERT_PLUS_BI_LSTM')

    return lstm_with_vanilla_bert


def create_random_vanilla_bert_bi_lstm_model(max_seq_length,
                                             embedding_size,
                                             depth,
                                             tokenizer,
                                             visit_tokenizer,
                                             num_heads,
                                             use_time_embedding,
                                             time_embeddings_size):
    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name='age_batch_norm_layer')

    normalized_index_age = age_batch_norm_layer(age_of_visit_input)

    vanilla_bert_model = transformer_bert_model_visit_prediction(
        max_seq_length=max_seq_length,
        concept_vocab_size=tokenizer.get_vocab_size(),
        visit_vocab_size=visit_tokenizer.get_vocab_size(),
        embedding_size=embedding_size,
        depth=depth,
        num_heads=num_heads,
        use_time_embedding=use_time_embedding,
        time_embeddings_size=time_embeddings_size
    )
    bert_inputs = [i for i in vanilla_bert_model.inputs if
                   'visit' not in i.name or ('visit_segment' in i.name
                                             or 'visit_concept_order' in i.name)]
    contextualized_embeddings, _ = vanilla_bert_model.get_layer('encoder').output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    # mask_input = bert_inputs[-1]
    mask_input = [i for i in bert_inputs if
                  'mask' in i.name and 'concept' not in i.name][0]
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

    # attach a property to the concept embeddings to indicate where are the masks, send the flag to the downstream layer
    next_input = masking_layer(
        contextualized_embeddings
    )

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, normalized_index_age])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_vanilla_bert = Model(inputs=bert_inputs + [age_of_visit_input],
                                   outputs=output, name='Vanilla_BERT_PLUS_BI_LSTM')

    return lstm_with_vanilla_bert
