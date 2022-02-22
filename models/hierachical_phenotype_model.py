from models.custom_layers import *


def create_probabilistic_phenotype_model(num_of_visits,
                                         num_of_concepts,
                                         concept_vocab_size,
                                         embedding_size,
                                         depth: int,
                                         num_heads: int,
                                         transformer_dropout: float = 0.1,
                                         embedding_dropout: float = 0.6,
                                         l2_reg_penalty: float = 1e-4,
                                         time_embeddings_size: int = 16,
                                         num_hidden_state: int = 50):
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
                      visit_segment, visit_mask, visit_time_delta_att,
                      visit_rank_order]

    # Expand dimensions for masking MultiHeadAttention in Concept Encoder
    pat_concept_mask = tf.reshape(
        pat_mask,
        shape=(-1, num_of_concepts)
    )[:, tf.newaxis, tf.newaxis, :]

    # output the embedding_matrix:
    l2_regularization = (tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None)
    concept_embedding_layer = ReusableEmbedding(
        concept_vocab_size,
        embedding_size,
        name='concept_embedding_layer',
        embeddings_regularizer=l2_regularization
    )

    # define the time embedding layer for absolute time stamps (since 1970)
    time_embedding_layer = TimeEmbeddingLayer(
        embedding_size=time_embeddings_size,
        name='time_embedding_layer'
    )
    # define the age embedding layer for the age w.r.t the medical record
    age_embedding_layer = TimeEmbeddingLayer(
        embedding_size=time_embeddings_size,
        name='age_embedding_layer'
    )
    # define positional encoding layer for visit numbers, the visit numbers are normalized
    # by subtracting visit numbers off the first visit number
    positional_encoding_layer = PositionalEncodingLayer(
        max_sequence_length=num_of_visits * num_of_concepts,
        embedding_size=time_embeddings_size,
        name='positional_encoding_layer'
    )
    # Temporal transformation
    temporal_transformation_layer = tf.keras.layers.Dense(
        embedding_size,
        activation='tanh',
        name='temporal_transformation'
    )

    # Look up the embeddings for the concepts
    concept_embeddings, embedding_matrix = concept_embedding_layer(
        pat_seq
    )
    # Look up the embeddings for the att tokens
    att_embeddings, _ = concept_embedding_layer(
        visit_time_delta_att
    )

    pt_seq_age_embeddings = age_embedding_layer(
        pat_seq_age
    )
    pt_seq_time_embeddings = time_embedding_layer(
        pat_seq_time
    )
    visit_positional_encoding = positional_encoding_layer(
        visit_rank_order
    )
    visit_positional_encoding = tf.tile(
        visit_positional_encoding[:, :, tf.newaxis, :], [1, 1, num_of_concepts, 1])

    # (batch, num_of_visits, num_of_concepts, embedding_size)
    concept_embeddings = temporal_transformation_layer(
        tf.concat(
            [concept_embeddings,
             pt_seq_age_embeddings,
             pt_seq_time_embeddings,
             visit_positional_encoding],
            axis=-1
        )
    )

    # The first encoder applied at the visit level
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

    visit_phenotype_layer = VisitPhenotypeLayer(
        hidden_unit=num_hidden_state,
        embedding_size=embedding_size,
        depth=depth,
        num_heads=num_heads,
        num_of_visits=num_of_visits,
        name='visit_phenotype_layer'
    )

    phenotype_embeddings, _ = visit_phenotype_layer(
        [visit_embeddings, att_embeddings, visit_mask]
    )

    output_layer = TiedOutputEmbedding(
        projection_regularizer=l2_regularization,
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits'
    )

    concept_softmax_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    # (batch_size, num_of_visits, vocab_size)
    concept_predictions = concept_softmax_layer(
        output_layer(
            [phenotype_embeddings, embedding_matrix]
        )
    )
    probabilistic_phenotype_model = tf.keras.Model(
        inputs=default_inputs,
        outputs=[concept_predictions])

    return probabilistic_phenotype_model
