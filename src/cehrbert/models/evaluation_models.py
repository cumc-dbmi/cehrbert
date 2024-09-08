import tensorflow as tf
from tensorflow.keras.initializers import Constant
from tensorflow.keras.models import Model

from cehrbert.models.bert_models_visit_prediction import transformer_bert_model_visit_prediction
from cehrbert.models.layers.custom_layers import ConvolutionBertLayer, get_custom_objects


def create_bi_lstm_model(
    max_seq_length,
    vocab_size,
    embedding_size,
    concept_embeddings,
    dropout_rate=0.2,
    lstm_unit=128,
    activation="relu",
    is_bi_directional=True,
):
    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name="age_batch_norm_layer")

    normalized_index_age = age_batch_norm_layer(age_of_visit_input)

    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="concept_ids")

    if concept_embeddings is not None:
        embedding_layer = tf.keras.layers.Embedding(
            vocab_size,
            embedding_size,
            embeddings_initializer=Constant(concept_embeddings),
            mask_zero=True,
        )
    else:
        embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)

    bi_lstm_layer = tf.keras.layers.LSTM(lstm_unit)

    if is_bi_directional:
        bi_lstm_layer = tf.keras.layers.Bidirectional(bi_lstm_layer)

    dropout_lstm_layer = tf.keras.layers.Dropout(dropout_rate)

    dense_layer = tf.keras.layers.Dense(64, activation=activation)

    dropout_dense_layer = tf.keras.layers.Dropout(dropout_rate)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    next_input = embedding_layer(concept_ids)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, normalized_index_age])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    model = Model(inputs=[concept_ids, age_of_visit_input], outputs=output, name="Vanilla_BI_LSTM")

    return model


def create_vanilla_feed_forward_model(vanilla_bert_model_path):
    """
    BERT + Feedforward model for binary prediction.

    :param vanilla_bert_model_path:
    :return:
    """
    age_at_index_date = tf.keras.layers.Input(name="age", shape=(1,))

    vanilla_bert_model = tf.keras.models.load_model(
        vanilla_bert_model_path, custom_objects=dict(**get_custom_objects())
    )
    bert_inputs = [
        i
        for i in vanilla_bert_model.inputs
        if "visit" not in i.name or ("visit_segment" in i.name or "visit_concept_order" in i.name)
    ]

    contextualized_embeddings, _ = vanilla_bert_model.get_layer("encoder").output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    mask_input = [i for i in bert_inputs if "mask" in i.name and "concept" not in i.name][0]
    mask_embeddings = tf.tile(
        tf.expand_dims(mask_input == 0, -1, name="bert_expand_ff"),
        [1, 1, embedding_size],
        name="bert_tile_ff",
    )
    contextualized_embeddings = tf.math.multiply(
        contextualized_embeddings,
        tf.cast(mask_embeddings, dtype=tf.float32, name="bert_cast_ff"),
        name="bert_multiply_ff",
    )

    output_layer = tf.keras.layers.Dense(1, name="prediction", activation="sigmoid")

    output = output_layer(tf.reduce_sum(contextualized_embeddings, axis=-2))

    lstm_with_vanilla_bert = Model(
        inputs=bert_inputs + [age_at_index_date],
        outputs=output,
        name="Vanilla_BERT_PLUS_BI_LSTM",
    )

    return lstm_with_vanilla_bert


def create_sliding_bert_model(model_path, max_seq_length, context_window, stride):
    age_at_index_date = tf.keras.layers.Input(name="age", shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name="age_batch_norm_layer")

    normalized_index_age = age_batch_norm_layer(age_at_index_date)

    concept_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="concept_ids")
    visit_segments = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="visit_segments")
    time_stamps = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="time_stamps")
    visit_concept_orders = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="visit_concept_orders")
    ages = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="ages")
    mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype="int32", name="mask")

    convolution_bert_layer = ConvolutionBertLayer(
        model_path=model_path,
        seq_len=max_seq_length,
        context_window=context_window,
        stride=stride,
    )

    conv_bert_output = convolution_bert_layer(
        [concept_ids, visit_segments, visit_concept_orders, time_stamps, ages, mask]
    )

    next_input = tf.keras.layers.concatenate([conv_bert_output, normalized_index_age])

    dropout_conv_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation="tanh")

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, name="prediction", activation="sigmoid")

    output = output_layer(dropout_dense_layer(dense_layer(dropout_conv_layer(next_input))))

    model_inputs = [
        concept_ids,
        visit_segments,
        visit_concept_orders,
        time_stamps,
        ages,
        mask,
    ]
    ffd_bert_model = tf.keras.models.Model(inputs=model_inputs + [age_at_index_date], outputs=output)

    return ffd_bert_model


def create_vanilla_bert_bi_lstm_model(
    max_seq_length,
    vanilla_bert_model_path,
    dropout_rate=0.2,
    lstm_unit=128,
    activation="relu",
    is_bi_directional=True,
):
    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name="age_batch_norm_layer")

    normalized_index_age = age_batch_norm_layer(age_of_visit_input)

    vanilla_bert_model = tf.keras.models.load_model(
        vanilla_bert_model_path, custom_objects=dict(**get_custom_objects())
    )
    bert_inputs = [
        i
        for i in vanilla_bert_model.inputs
        if "visit" not in i.name or ("visit_segment" in i.name or "visit_concept_order" in i.name)
    ]
    #     bert_inputs = vanilla_bert_model.inputs
    contextualized_embeddings, _ = vanilla_bert_model.get_layer("encoder").output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    # mask_input = bert_inputs[-1]
    mask_input = [i for i in bert_inputs if "mask" in i.name and "concept" not in i.name][0]
    mask_embeddings = tf.tile(
        tf.expand_dims(mask_input == 0, -1, name="expand_mask"),
        [1, 1, embedding_size],
        name="tile_mask",
    )

    contextualized_embeddings = tf.math.multiply(
        contextualized_embeddings,
        tf.cast(mask_embeddings, dtype=tf.float32, name="cast_mask"),
    )

    masking_layer = tf.keras.layers.Masking(mask_value=0.0, input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.LSTM(lstm_unit)

    if is_bi_directional:
        bi_lstm_layer = tf.keras.layers.Bidirectional(bi_lstm_layer)

    dropout_lstm_layer = tf.keras.layers.Dropout(dropout_rate)

    dense_layer = tf.keras.layers.Dense(64, activation=activation)

    dropout_dense_layer = tf.keras.layers.Dropout(dropout_rate)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    # attach a property to the concept embeddings to indicate where are the masks, send the flag
    # to the downstream layer
    next_input = masking_layer(contextualized_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, normalized_index_age])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_vanilla_bert = Model(
        inputs=bert_inputs + [age_of_visit_input],
        outputs=output,
        name="Vanilla_BERT_PLUS_BI_LSTM",
    )

    return lstm_with_vanilla_bert


def create_random_vanilla_bert_bi_lstm_model(
    max_seq_length,
    embedding_size,
    depth,
    tokenizer,
    visit_tokenizer,
    num_heads,
    use_time_embedding,
    time_embeddings_size,
):
    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name="age_batch_norm_layer")

    normalized_index_age = age_batch_norm_layer(age_of_visit_input)

    vanilla_bert_model = transformer_bert_model_visit_prediction(
        max_seq_length=max_seq_length,
        concept_vocab_size=tokenizer.get_vocab_size(),
        visit_vocab_size=visit_tokenizer.get_vocab_size(),
        embedding_size=embedding_size,
        depth=depth,
        num_heads=num_heads,
        use_time_embedding=use_time_embedding,
        time_embeddings_size=time_embeddings_size,
    )
    bert_inputs = [
        i
        for i in vanilla_bert_model.inputs
        if "visit" not in i.name or ("visit_segment" in i.name or "visit_concept_order" in i.name)
    ]
    #     bert_inputs = vanilla_bert_model.inputs
    contextualized_embeddings, _ = vanilla_bert_model.get_layer("encoder").output
    _, _, embedding_size = contextualized_embeddings.get_shape().as_list()

    # mask_input = bert_inputs[-1]
    mask_input = [i for i in bert_inputs if "mask" in i.name and "concept" not in i.name][0]
    mask_embeddings = tf.tile(
        tf.expand_dims(mask_input == 0, -1, name="expand_mask"),
        [1, 1, embedding_size],
        name="tile_mask",
    )
    contextualized_embeddings = tf.math.multiply(
        contextualized_embeddings,
        tf.cast(mask_embeddings, dtype=tf.float32, name="cast_mask"),
    )

    masking_layer = tf.keras.layers.Masking(mask_value=0.0, input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation="relu")

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    next_input = masking_layer(
        contextualized_embeddings
    )  # attach a property to the concept embeddings to indicate where are the masks, send the flag to the downstream layer

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, normalized_index_age])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_vanilla_bert = Model(
        inputs=bert_inputs + [age_of_visit_input],
        outputs=output,
        name="Vanilla_BERT_PLUS_BI_LSTM",
    )

    return lstm_with_vanilla_bert


def create_hierarchical_bert_bi_lstm_model(bert_model_path, **kwargs):
    model = tf.keras.models.load_model(bert_model_path, custom_objects=get_custom_objects())
    return create_hierarchical_bert_bi_lstm_model_with_model(model, **kwargs)


def create_hierarchical_bert_bi_lstm_model_with_model(
    hierarchical_bert_model,
    dropout_rate=0.2,
    lstm_unit=128,
    activation="relu",
    is_bi_directional=True,
    include_att_tokens=False,
    freeze_pretrained_model=False,
):
    index_age_input = tf.keras.layers.Input(name="age", shape=(1,))

    age_batch_norm_layer = tf.keras.layers.BatchNormalization(name="age_batch_norm_layer")

    normalized_index_age = age_batch_norm_layer(index_age_input)

    _, num_of_visits, num_of_concepts, embedding_size = hierarchical_bert_model.get_layer(
        "temporal_transformation_layer"
    ).output.shape

    is_phenotype_enabled = "hidden_visit_embeddings" in [layer.name for layer in hierarchical_bert_model.layers]

    # Freeze the weight of the pretrained model if enabled
    if freeze_pretrained_model:
        hierarchical_bert_model.trainable = False

    if is_phenotype_enabled:
        contextualized_visit_embeddings, _ = hierarchical_bert_model.get_layer("hidden_visit_embeddings").output
    else:
        contextualized_visit_embeddings, _ = hierarchical_bert_model.get_layer("visit_encoder").output
        if not include_att_tokens:
            # Pad contextualized_visit_embeddings on axis 1 with one extra visit so we can extract the
            # visit embeddings using the reshape trick
            expanded_contextualized_visit_embeddings = tf.concat(
                [
                    contextualized_visit_embeddings,
                    contextualized_visit_embeddings[:, 0:1, :],
                ],
                axis=1,
            )
            # Extract the visit embeddings elements
            contextualized_visit_embeddings = tf.reshape(
                expanded_contextualized_visit_embeddings,
                (-1, num_of_visits, 3 * embedding_size),
            )[:, :, embedding_size : embedding_size * 2]

    visit_mask = hierarchical_bert_model.get_layer("visit_mask").output

    if include_att_tokens:
        visit_mask = tf.reshape(tf.tile(visit_mask[:, :, tf.newaxis], [1, 1, 3]), (-1, num_of_visits * 3))[:, 1:]

    mask_embeddings = tf.cast(tf.math.logical_not(tf.cast(visit_mask, dtype=tf.bool)), dtype=tf.float32)[
        :, :, tf.newaxis
    ]

    contextualized_embeddings = tf.math.multiply(contextualized_visit_embeddings, mask_embeddings)

    masking_layer = tf.keras.layers.Masking(mask_value=0.0, input_shape=(num_of_visits, embedding_size))

    bi_lstm_layer = tf.keras.layers.LSTM(lstm_unit)

    if is_bi_directional:
        bi_lstm_layer = tf.keras.layers.Bidirectional(bi_lstm_layer)

    dropout_lstm_layer = tf.keras.layers.Dropout(dropout_rate)

    dense_layer = tf.keras.layers.Dense(64, activation=activation)

    dropout_dense_layer = tf.keras.layers.Dropout(dropout_rate)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="label")

    next_input = masking_layer(contextualized_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, tf.reshape(normalized_index_age, (-1, 1))])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_hierarchical_bert = tf.keras.models.Model(
        inputs=hierarchical_bert_model.inputs + [index_age_input],
        outputs=output,
        name="HIERARCHICAL_BERT_PLUS_BI_LSTM",
    )

    return lstm_with_hierarchical_bert


def create_hierarchical_bert_model_with_pooling(
    bert_model_path,
    dropout_rate=0.2,
    activation="tanh",
    freeze_pretrained_model=False,
    **kwargs,
):
    hierarchical_bert_model = tf.keras.models.load_model(bert_model_path, custom_objects=get_custom_objects())

    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    _, num_of_visits, num_of_concepts, embedding_size = hierarchical_bert_model.get_layer(
        "temporal_transformation_layer"
    ).output.shape

    is_phenotype_enabled = "hidden_visit_embeddings" in [layer.name for layer in hierarchical_bert_model.layers]

    # Freeze the weight of the pretrained model if enabled
    if freeze_pretrained_model:
        hierarchical_bert_model.trainable = False

    if is_phenotype_enabled:
        contextualized_visit_embeddings, _ = hierarchical_bert_model.get_layer("hidden_visit_embeddings").output
    else:
        contextualized_visit_embeddings, _ = hierarchical_bert_model.get_layer("visit_encoder").output

    visit_masks = hierarchical_bert_model.get_layer("visit_mask").output
    # Get the first embedding from the visit embedding sequence
    # [batch_size, embedding_size]
    visit_embedding_pooling = tf.gather_nd(
        contextualized_visit_embeddings,
        indices=tf.argmax(visit_masks, axis=1)[:, tf.newaxis],
        batch_dims=1,
    )

    dense_layer_1 = tf.keras.layers.Dense(128, activation=activation)
    dropout_dense_layer_1 = tf.keras.layers.Dropout(dropout_rate)
    dense_layer_2 = tf.keras.layers.Dense(64, activation=activation)
    dropout_dense_layer_2 = tf.keras.layers.Dropout(dropout_rate)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="label")

    next_input = tf.keras.layers.concatenate([visit_embedding_pooling, tf.reshape(age_of_visit_input, (-1, 1))])

    next_input = dropout_dense_layer_1(dense_layer_1(next_input))
    next_input = dropout_dense_layer_2(dense_layer_2(next_input))

    output = output_layer(next_input)

    hierarchical_bert_with_pooling = tf.keras.models.Model(
        inputs=hierarchical_bert_model.inputs + [age_of_visit_input],
        outputs=output,
        name="HIERARCHICAL_BERT_POOLING",
    )

    return hierarchical_bert_with_pooling


def create_prob_phenotype_bi_lstm_model_with_model(bert_model_path):
    """
    :param bert_model_path:

    :return:
    """
    model = tf.keras.models.load_model(bert_model_path, custom_objects=get_custom_objects())

    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    _, _, contextual_visit_embeddings, _ = model.get_layer("visit_phenotype_layer").output
    embedding_size = contextual_visit_embeddings.shape[-1]

    visit_mask = [i for i in model.inputs if i.name == "visit_mask"][0]
    num_of_visits = visit_mask.shape[1]
    # Expand dimension for masking MultiHeadAttention in Visit Encoder
    visit_mask_with_att = tf.reshape(tf.stack([visit_mask, visit_mask], axis=2), shape=(-1, num_of_visits * 2))[:, 1:]

    mask_embeddings = tf.cast(
        tf.math.logical_not(tf.cast(visit_mask_with_att, dtype=tf.bool)),
        dtype=tf.float32,
    )[:, :, tf.newaxis]

    contextual_visit_embeddings = tf.math.multiply(contextual_visit_embeddings, mask_embeddings)

    masking_layer = tf.keras.layers.Masking(mask_value=0.0, input_shape=(num_of_visits, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation="tanh")

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="label")

    next_input = masking_layer(contextual_visit_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, tf.reshape(age_of_visit_input, (-1, 1))])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    lstm_with_cher_bert = tf.keras.models.Model(
        inputs=model.inputs + [age_of_visit_input],
        outputs=output,
        name="PROB_PHENOTYPE_PLUS_BI_LSTM",
    )

    return lstm_with_cher_bert


def create_temporal_bert_bi_lstm_model(max_seq_length, temporal_bert_model_path):
    temporal_bert_model = tf.keras.models.load_model(
        temporal_bert_model_path, custom_objects=dict(**get_custom_objects())
    )
    bert_inputs = temporal_bert_model.inputs[0:5]
    _, _, embedding_size = temporal_bert_model.get_layer("temporal_encoder").output[0].get_shape().as_list()
    contextualized_embeddings, _, _ = temporal_bert_model.get_layer("temporal_encoder").output

    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    mask_input = bert_inputs[-1]
    mask_embeddings = tf.cast(
        tf.tile(
            tf.expand_dims(mask_input == 0, -1, name="expand_mask"),
            [1, 1, embedding_size],
            name="tile_mask",
        ),
        tf.float32,
        name="cast_mask_embeddings",
    )
    contextualized_embeddings = tf.math.multiply(contextualized_embeddings, mask_embeddings)

    masking_layer = tf.keras.layers.Masking(mask_value=0.0, input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.1)

    dense_layer = tf.keras.layers.Dense(64, activation="tanh")

    dropout_dense_layer = tf.keras.layers.Dropout(0.1)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    next_input = masking_layer(contextualized_embeddings)

    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    next_input = tf.keras.layers.concatenate([next_input, age_of_visit_input])

    next_input = dropout_dense_layer(dense_layer(next_input))

    output = output_layer(next_input)

    return Model(
        inputs=bert_inputs + [age_of_visit_input],
        outputs=output,
        name="TEMPORAL_BERT_PLUS_BI_LSTM",
    )


def create_probabilistic_bert_bi_lstm_model(max_seq_length, vanilla_bert_model_path):
    age_of_visit_input = tf.keras.layers.Input(name="age", shape=(1,))

    bert_model = tf.keras.models.load_model(vanilla_bert_model_path, custom_objects=dict(**get_custom_objects()))
    bert_inputs = bert_model.inputs

    # Get phenotype embeddings and probability distribution
    _, phenotype_probability_dist = bert_model.get_layer("hidden_phenotype_layer").output

    num_hidden_state = bert_model.get_layer("hidden_phenotype_layer").hidden_unit

    # (batch, max_sequence, embedding_size)
    concept_embeddings, _ = bert_model.get_layer("concept_embeddings").output
    _, _, embedding_size = concept_embeddings.get_shape().as_list()

    # (batch * num_hidden_state, max_sequence, embedding_size)
    contextualized_embeddings = bert_model.get_layer("tf.reshape").output

    # mask_input = bert_inputs[-1]
    mask_input = [i for i in bert_inputs if "mask" in i.name and "concept" not in i.name][0]

    # (batch * num_hidden_state, max_sequence, 1)
    mask_embeddings = tf.reshape(
        tf.tile((mask_input == 0)[:, tf.newaxis, :], [1, num_hidden_state, 1]),
        (-1, max_seq_length),
    )[:, :, tf.newaxis]

    # (batch * num_hidden_state, max_sequence, embedding_size)
    contextualized_embeddings = tf.math.multiply(contextualized_embeddings, tf.cast(mask_embeddings, dtype=tf.float32))
    # Masking layer for LSTM
    masking_layer = tf.keras.layers.Masking(mask_value=0.0, input_shape=(max_seq_length, embedding_size))

    bi_lstm_layer = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))

    dropout_lstm_layer = tf.keras.layers.Dropout(0.2)

    dense_layer = tf.keras.layers.Dense(64, activation="relu")

    dropout_dense_layer = tf.keras.layers.Dropout(0.2)

    # (batch * num_hidden_state, 1)
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    # attach a property to the concept embeddings to indicate where are the masks, send the flag
    # to the downstream layer: (batch * num_hidden_state, max_sequence, embedding_size)
    next_input = masking_layer(contextualized_embeddings)

    # (batch * num_hidden_state, 256)
    next_input = dropout_lstm_layer(bi_lstm_layer(next_input))

    # (batch * num_hidden_state, 1)
    duplicate_age_of_visit_input = tf.reshape(
        tf.tile(age_of_visit_input[:, tf.newaxis, :], [1, num_hidden_state, 1]), (-1, 1)
    )
    # (batch * num_hidden_state, 256 + 1)
    next_input = tf.keras.layers.concatenate([next_input, duplicate_age_of_visit_input])

    # (batch * num_hidden_state, 64)
    next_input = dropout_dense_layer(dense_layer(next_input))

    # (batch * num_hidden_state, 1)
    output = output_layer(next_input)

    # (batch, num_hidden_state, 1)
    reshaped_output = tf.reshape(output, (-1, num_hidden_state, 1))
    # (batch, 1)
    weighted_output = tf.squeeze(tf.reduce_sum(phenotype_probability_dist[:, :, tf.newaxis] * reshaped_output, axis=1))

    lstm_with_prob_bert = Model(
        inputs=bert_inputs + [age_of_visit_input],
        outputs=weighted_output,
        name="Probabilistic_BERT_PLUS_BI_LSTM",
    )

    return lstm_with_prob_bert
