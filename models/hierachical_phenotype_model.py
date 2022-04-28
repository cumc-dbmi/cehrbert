from models.layers.custom_layers import *


def create_probabilistic_phenotype_model(
        num_of_visits,
        num_of_concepts,
        concept_vocab_size,
        embedding_size,
        depth: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4,
        time_embeddings_size: int = 16,
        include_visit_type_prediction: bool = False,
        include_readmission: bool = False,
        include_prolonged_length_stay: bool = False,
        visit_vocab_size: int = None,
        num_of_phenotypes: int = 20
):
    """
    Create a hierarchical bert model

    :param num_of_visits:
    :param num_of_concepts:
    :param concept_vocab_size:
    :param embedding_size:
    :param depth:
    :param num_heads:
    :param num_of_exchanges:
    :param transformer_dropout:
    :param embedding_dropout:
    :param l2_reg_penalty:
    :param time_embeddings_size:
    :param include_visit_type_prediction:
    :param include_readmission:
    :param include_prolonged_length_stay:
    :param visit_vocab_size:
    :param num_of_phenotypes:
    :return:
    """
    # If the second tiered learning objectives are enabled, visit_vocab_size needs to be provided
    if include_visit_type_prediction and not visit_vocab_size:
        raise RuntimeError(f'visit_vocab_size can not be null '
                           f'when the second learning objectives are enabled')

    pat_seq = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_seq'
    )
    pat_seq_age = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_seq_age'
    )
    pat_seq_time = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_seq_time'
    )
    pat_mask = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='pat_mask'
    )
    concept_values = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='float32',
        name='concept_values'
    )
    concept_value_masks = tf.keras.layers.Input(
        shape=(num_of_visits, num_of_concepts,),
        dtype='int32',
        name='concept_value_masks'
    )
    visit_segment = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_segment'
    )

    visit_mask = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_mask')

    visit_time_delta_att = tf.keras.layers.Input(
        shape=(num_of_visits - 1,),
        dtype='int32',
        name='visit_time_delta_att'
    )

    visit_rank_order = tf.keras.layers.Input(
        shape=(num_of_visits,),
        dtype='int32',
        name='visit_rank_order')

    # Create a list of inputs so the model could reference these later
    default_inputs = [pat_seq, pat_seq_age, pat_seq_time, pat_mask,
                      concept_values, concept_value_masks, visit_segment, visit_mask,
                      visit_time_delta_att, visit_rank_order]

    # Expand dimensions for masking MultiHeadAttention in Concept Encoder
    pat_concept_mask = tf.reshape(
        pat_mask,
        shape=(-1, num_of_concepts)
    )[:, tf.newaxis, tf.newaxis, :]

    # output the embedding_matrix:
    l2_regularizer = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size,
        embedding_size,
        name='concept_embedding_layer',
        embeddings_regularizer=l2_regularizer
    )

    # Look up the embeddings for the concepts
    concept_embeddings, embedding_matrix = concept_embedding_layer(
        pat_seq
    )

    concept_value_transformation_layer = ConceptValueTransformationLayer(
        embedding_size=embedding_size,
        name='concept_value_transformation_layer'
    )

    # Transform the concept embeddings by combining their concept embeddings with the
    # corresponding val
    concept_embeddings = concept_value_transformation_layer(
        concept_embeddings=concept_embeddings,
        concept_values=concept_values,
        concept_value_masks=concept_value_masks
    )

    # Look up the embeddings for the att tokens
    att_embeddings, _ = concept_embedding_layer(
        visit_time_delta_att
    )

    # Repurpose token id 0 as the visit start embedding
    visit_start_embeddings, _ = concept_embedding_layer(
        tf.zeros_like(
            visit_mask,
            dtype=tf.int32
        )
    )

    temporal_transformation_layer = TemporalTransformationLayer(
        time_embeddings_size=time_embeddings_size,
        embedding_size=embedding_size,
        name='temporal_transformation_layer'
    )

    # (batch, num_of_visits, num_of_concepts, embedding_size)
    concept_embeddings = temporal_transformation_layer(
        concept_embeddings,
        pat_seq_age,
        pat_seq_time,
        visit_rank_order
    )

    # The first bert applied at the visit level
    concept_encoder = Encoder(
        name='concept_encoder',
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout
    )

    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_concepts, embedding_size)
    )

    concept_embeddings, _ = concept_encoder(
        concept_embeddings,  # be reused
        pat_concept_mask  # not change
    )

    # (batch_size, num_of_visits, num_of_concepts, embedding_size)
    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_visits, num_of_concepts, embedding_size)
    )

    # Step 2 generate visit embeddings
    # Slice out the first contextualized embedding of each visit
    # (batch_size, num_of_visits, embedding_size)
    visit_embeddings = concept_embeddings[:, :, 0]

    # (batch_size, num_of_visits, embedding_size)
    expanded_att_embeddings = tf.concat([att_embeddings, att_embeddings[:, 0:1, :]], axis=1)

    # Insert the att embeddings between visit embeddings
    # (batch_size, num_of_visits + num_of_visits + num_of_visits - 1, embedding_size)
    contextualized_visit_embeddings = tf.reshape(
        tf.concat(
            [visit_start_embeddings,
             visit_embeddings,
             expanded_att_embeddings],
            axis=-1
        ),
        (-1, 3 * num_of_visits, embedding_size)
    )[:, :-1, :]

    # Expand dimension for masking MultiHeadAttention in Visit Encoder
    visit_mask_with_att = tf.reshape(
        tf.tile(visit_mask[:, :, tf.newaxis], [1, 1, 3]),
        (-1, num_of_visits * 3)
    )[:, tf.newaxis, tf.newaxis, 1:]

    # (num_of_visits_with_att, num_of_visits_with_att)
    look_ahead_mask_base = tf.cast(
        1 - tf.linalg.band_part(tf.ones((num_of_visits, num_of_visits)), -1, 0),
        dtype=tf.int32
    )
    look_ahead_visit_mask_with_att = tf.reshape(
        tf.tile(
            look_ahead_mask_base[:, tf.newaxis, :, tf.newaxis],
            [1, 3, 1, 3]
        ),
        shape=(num_of_visits * 3, num_of_visits * 3)
    )[:-1, :-1]

    look_ahead_concept_mask = tf.reshape(
        tf.tile(
            look_ahead_mask_base[:, tf.newaxis, :, tf.newaxis],
            [1, num_of_concepts, 1, 1]
        ),
        (num_of_concepts * num_of_visits, -1)
    )

    # (batch_size, 1, num_of_visits_with_att, num_of_visits_with_att)
    look_ahead_visit_mask_with_att = tf.maximum(
        visit_mask_with_att,
        look_ahead_visit_mask_with_att
    )

    # (batch_size, 1, num_of_visits * num_of_concepts, num_of_visits)
    look_ahead_concept_mask = tf.maximum(
        visit_mask[:, tf.newaxis, tf.newaxis, :],
        look_ahead_concept_mask
    )

    # (batch_size, 1, num_of_visits * num_of_concepts, num_of_visits)
    # look_ahead_concept_mask = tf.maximum(
    #     tf.cast(
    #         tf.reshape(
    #             tf.tile(
    #                 tf.expand_dims(
    #                     1 - tf.eye(num_of_visits),
    #                     axis=1
    #                 ),
    #                 [1, num_of_concepts, 1]
    #             ),
    #             (num_of_concepts * num_of_visits, num_of_visits)
    #         ),
    #         dtype=tf.int32
    #     ),
    #     look_ahead_concept_mask
    # )

    # Second bert applied at the patient level to the visit embeddings
    visit_encoder = Encoder(
        name='visit_encoder',
        num_layers=depth,
        d_model=embedding_size,
        num_heads=num_heads,
        dropout_rate=transformer_dropout
    )

    # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
    contextualized_visit_embeddings, _ = visit_encoder(
        contextualized_visit_embeddings,
        look_ahead_visit_mask_with_att
    )

    # Pad contextualized_visit_embeddings on axis 1 with one extra visit so we can extract the
    # visit embeddings using the reshape trick
    expanded_contextualized_visit_embeddings = tf.concat(
        [contextualized_visit_embeddings,
         contextualized_visit_embeddings[:, 0:1, :]],
        axis=1
    )

    # Extract the visit embeddings elements
    visit_embeddings_without_att = tf.reshape(
        expanded_contextualized_visit_embeddings, (-1, num_of_visits, 3 * embedding_size)
    )[:, :, embedding_size: embedding_size * 2]

    # Step 4: Assuming there is a generative process that generates diagnosis embeddings from a
    # Multivariate Gaussian Distribution Declare phenotype distribution prior
    visit_phenotype_layer = VisitPhenotypeLayer(
        num_of_phenotypes=num_of_phenotypes,
        embedding_size=embedding_size,
        transformer_dropout=transformer_dropout,
        name='hidden_visit_embeddings'
    )

    # (batch_size, num_of_visits, vocab_size)
    visit_embeddings_without_att = visit_phenotype_layer(
        [visit_embeddings_without_att, visit_mask, embedding_matrix]
    )

    # # Step 3 decoder applied to patient level
    # Reshape the data in visit view back to patient view:
    # (batch, num_of_visits * num_of_concepts, embedding_size)
    concept_embeddings = tf.reshape(
        concept_embeddings,
        shape=(-1, num_of_visits * num_of_concepts, embedding_size)
    )

    # Let local concept embeddings access the global representatives of each visit
    multi_head_attention_layer = MultiHeadAttention(
        d_model=embedding_size,
        num_heads=num_heads
    )

    mha_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    mha_dropout = tf.keras.layers.Dropout(transformer_dropout)

    global_concept_embeddings, _ = multi_head_attention_layer(
        visit_embeddings_without_att,
        visit_embeddings_without_att,
        concept_embeddings,
        look_ahead_concept_mask
    )

    global_attn_norm = mha_layernorm(
        mha_dropout(global_concept_embeddings) + concept_embeddings
    )

    ffn = point_wise_feed_forward_network(embedding_size, 512)
    ffn_layernorm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        name='global_concept_embeddings_normalization'
    )
    ffn_dropout = tf.keras.layers.Dropout(
        transformer_dropout
    )

    ffn_dropout_output = ffn_dropout(ffn(global_attn_norm))

    global_attn = ffn_layernorm(
        ffn_dropout_output + global_attn_norm
    )

    concept_output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularizer,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits')

    concept_softmax_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    concept_predictions = concept_softmax_layer(
        concept_output_layer([global_attn, embedding_matrix])
    )

    outputs = [concept_predictions]

    if include_visit_type_prediction:
        # Slice out the the visit embeddings (CLS tokens)
        visit_prediction_dense = tf.keras.layers.Dense(
            visit_vocab_size,
            activation='tanh',
            name='visit_prediction_dense'
        )

        visit_softmax_layer = tf.keras.layers.Softmax(
            name='visit_predictions'
        )

        visit_predictions = visit_softmax_layer(
            visit_prediction_dense(visit_embeddings_without_att)
        )

        outputs.append(visit_predictions)

    if include_readmission:
        is_readmission_layer = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='is_readmission'
        )

        is_readmission_output = is_readmission_layer(
            visit_embeddings_without_att
        )

        outputs.append(is_readmission_output)

    if include_prolonged_length_stay:
        visit_prolonged_stay_layer = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            name='visit_prolonged_stay'
        )

        visit_prolonged_stay_output = visit_prolonged_stay_layer(
            visit_embeddings_without_att
        )

        outputs.append(visit_prolonged_stay_output)

    hierarchical_bert = tf.keras.Model(
        inputs=default_inputs,
        outputs=outputs
    )

    return hierarchical_bert
