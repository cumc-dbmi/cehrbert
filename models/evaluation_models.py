import tensorflow as tf

from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

from models.custom_layers import get_custom_objects
from models.custom_layers import ConvolutionBertLayer
from models.bert_models_visit_prediction import transformer_bert_model_visit_prediction


def create_bi_lstm_model(max_seq_length, vocab_size, embedding_size, concept_embeddings):
    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

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

    next_input = tf.keras.layers.concatenate([next_input, age_of_visit_input])

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

    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                        name='concept_ids')
    visit_segments = tf.keras.layers.Input(shape=(max_seq_length,), dtype='int32',
                                           name='visit_segments')
    time_stamps = tf.keras.layers.Input(shape=(max_seq_length,),
                                        dtype='int32',
                                        name='time_stamps')
    visit_concept_orders = tf.keras.layers.Input(
        shape=(max_seq_length,),
        dtype='int32',
        name='visit_concept_orders')
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
                                               visit_concept_orders,
                                               time_stamps,
                                               ages,
                                               mask])

    next_input = tf.keras.layers.concatenate([conv_bert_output, age_at_index_date])

    dropout_conv_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation='tanh')

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, name='prediction', activation='sigmoid')

    output = output_layer(dropout_dense_layer(dense_layer(dropout_conv_layer(next_input))))

    model_inputs = [concept_ids, visit_segments, visit_concept_orders, time_stamps, ages, mask]
    ffd_bert_model = tf.keras.models.Model(inputs=model_inputs + [age_at_index_date],
                                           outputs=output)

    return ffd_bert_model


def create_vanilla_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path):
    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

    vanilla_bert_model = tf.keras.models.load_model(vanilla_bert_model_path,
                                                    custom_objects=dict(**get_custom_objects()))
    bert_inputs = [i for i in vanilla_bert_model.inputs if
                   'visit' not in i.name or ('visit_segment' in i.name
                                             or 'visit_concept_order' in i.name)]
    #     bert_inputs = vanilla_bert_model.inputs
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

    next_input = masking_layer(
        contextualized_embeddings)  # attach a property to the concept embeddings to indicate where are the masks, send the flag to the downstream layer

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, age_of_visit_input])

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
    #     bert_inputs = vanilla_bert_model.inputs
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

    next_input = masking_layer(
        contextualized_embeddings)  # attach a property to the concept embeddings to indicate where are the masks, send the flag to the downstream layer

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, age_of_visit_input])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_vanilla_bert = Model(inputs=bert_inputs + [age_of_visit_input],
                                   outputs=output, name='Vanilla_BERT_PLUS_BI_LSTM')

    return lstm_with_vanilla_bert


def create_cher_bert_bi_lstm_model(bert_model_path):
    model = tf.keras.models.load_model(bert_model_path, custom_objects=get_custom_objects())
    return create_cher_bert_bi_lstm_model_with_model(model)


def create_cher_bert_bi_lstm_model_with_model(model):
    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

    contextualized_embeddings = model.get_layer(
        'global_concept_embeddings_normalization'
    ).output
    _, num_of_visits, num_of_concepts, embedding_size = model.get_layer(
        'temporal_transformation'
    ).output.shape
    max_seq_length = num_of_visits * num_of_concepts

    pat_mask = model.get_layer('pat_mask').output

    pat_mask_reshaped = tf.reshape(pat_mask, (-1, max_seq_length))

    pat_mask_reordered = tf.sort(pat_mask_reshaped, axis=1)

    sorted_index = tf.argsort(pat_mask_reshaped, axis=1)

    index_1d = tf.cast(tf.where(sorted_index >= 0)[:, 0], dtype=tf.int32)

    index_2d = tf.stack([index_1d, tf.reshape(sorted_index, [-1])], axis=-1)

    contextualized_embeddings = tf.reshape(
        tf.gather_nd(contextualized_embeddings, index_2d),
        (-1, max_seq_length, embedding_size)
    )

    mask_embeddings = tf.cast(
        tf.math.logical_not(
            tf.cast(
                tf.reshape(pat_mask_reordered, (-1, max_seq_length)),
                dtype=tf.bool
            )
        ),
        dtype=tf.float32
    )[:, :, tf.newaxis]

    contextualized_embeddings = tf.math.multiply(contextualized_embeddings, mask_embeddings)

    masking_layer = tf.keras.layers.Masking(mask_value=0.,
                                            input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation='tanh')

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='label')

    next_input = masking_layer(contextualized_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, tf.reshape(age_of_visit_input, (-1, 1))])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_cher_bert = tf.keras.models.Model(inputs=model.inputs + [age_of_visit_input],
                                                outputs=output, name='CHER_BERT_PLUS_BI_LSTM')

    return lstm_with_cher_bert


def create_temporal_bert_bi_lstm_model(max_seq_length, temporal_bert_model_path):
    temporal_bert_model = tf.keras.models.load_model(temporal_bert_model_path,
                                                     custom_objects=dict(**get_custom_objects()))
    bert_inputs = temporal_bert_model.inputs[0:5]
    _, _, embedding_size = temporal_bert_model.get_layer('temporal_encoder').output[
        0].get_shape().as_list()
    contextualized_embeddings, _, _ = temporal_bert_model.get_layer('temporal_encoder').output

    age_of_visit_input = tf.keras.layers.Input(name='age', shape=(1,))

    mask_input = bert_inputs[-1]
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

    return Model(inputs=bert_inputs + [age_of_visit_input], outputs=output,
                 name='TEMPORAL_BERT_PLUS_BI_LSTM')


def create_probabilistic_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path):
    age_of_visit_input = tf.keras.layers.Input(
        name='age',
        shape=(1,)
    )

    bert_model = tf.keras.models.load_model(
        vanilla_bert_model_path,
        custom_objects=dict(**get_custom_objects())
    )
    bert_inputs = bert_model.inputs

    # Get phenotype embeddings and probability distribution
    _, phenotype_probability_dist = bert_model.get_layer('hidden_phenotype_layer').output

    num_hidden_state = bert_model.get_layer('hidden_phenotype_layer').hidden_unit

    # (batch * num_hidden_state, max_sequence, embedding_size)
    contextualized_embeddings, _ = bert_model.get_layer('encoder').output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    # mask_input = bert_inputs[-1]
    mask_input = [i for i in bert_inputs if
                  'mask' in i.name and 'concept' not in i.name][0]

    # (batch * num_hidden_state, max_sequence, embedding_size)
    mask_embeddings = tf.reshape(
        tf.tile(
            (mask_input == 0)[:, tf.newaxis, :],
            [1, num_hidden_state, 1]
        ),
        (-1, max_seq_length)
    )[:, tf.newaxis, tf.newaxis, :]

    # (batch * num_hidden_state, max_sequence, embedding_size)
    contextualized_embeddings = tf.math.multiply(
        contextualized_embeddings,
        tf.cast(mask_embeddings, dtype=tf.float32)
    )
    # Masking layer for LSTM
    masking_layer = tf.keras.layers.Masking(
        mask_value=0.,
        input_shape=(max_seq_length, embedding_size)
    )

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation='relu')

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    # (batch * num_hidden_state, 1)
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    # attach a property to the concept embeddings to indicate where are the masks, send the flag
    # to the downstream layer (batch * num_hidden_state, embedding_size)
    next_input = masking_layer(
        contextualized_embeddings)

    # (batch * num_hidden_state, 256)
    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    # (batch * num_hidden_state, 1)
    duplicate_age_of_visit_input = tf.reshape(
        tf.tile(
            age_of_visit_input[:, tf.newaxis, :],
            [1, num_hidden_state, 1]
        ),
        (-1, 1)
    )
    # (batch * num_hidden_state, 256 + 1)
    next_input = tf.keras.layers.concatenate(
        [next_input, duplicate_age_of_visit_input]
    )

    # (batch * num_hidden_state, 64)
    next_input = dropout_dense_layer(dense_layer(next_input))

    # (batch * num_hidden_state, 1)
    output = output_layer(next_input)

    # (batch, num_hidden_state, 1)
    reshaped_output = tf.reshape(
        output,
        (-1, num_hidden_state, max_seq_length, embedding_size)
    )
    # (batch, 1)
    weighted_output = tf.squeeze(
        tf.reduce_sum(
            phenotype_probability_dist[:, :, tf.newaxis] * reshaped_output,
            axis=1
        )
    )

    lstm_with_prob_bert = Model(
        inputs=bert_inputs + [age_of_visit_input],
        outputs=weighted_output, name='Probabilistic_BERT_PLUS_BI_LSTM'
    )

    return lstm_with_prob_bert
