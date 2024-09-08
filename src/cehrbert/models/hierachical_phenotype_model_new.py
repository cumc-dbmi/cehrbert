import tensorflow as tf

from cehrbert.models.hierachical_bert_model_v2 import create_att_concept_mask
from cehrbert.models.layers.custom_layers import (
    ConceptValueTransformationLayer,
    Encoder,
    ReusableEmbedding,
    SimpleDecoderLayer,
    TemporalTransformationLayer,
    TiedOutputEmbedding,
    VisitPhenotypeLayer,
)


def create_visit_masks(visit_mask, num_of_visits):
    # Expand dimension for masking MultiHeadAttention in Visit Encoder
    visit_mask_with_att = tf.reshape(tf.tile(visit_mask[:, :, tf.newaxis], [1, 1, 3]), (-1, num_of_visits * 3))[
        :, tf.newaxis, tf.newaxis, 1:
    ]

    # (num_of_visits_with_att, num_of_visits_with_att)
    look_ahead_mask_base = tf.cast(
        1 - tf.linalg.band_part(tf.ones((num_of_visits, num_of_visits)), -1, 0),
        dtype=tf.int32,
    )
    look_ahead_visit_mask_with_att = tf.reshape(
        tf.tile(look_ahead_mask_base[:, tf.newaxis, :, tf.newaxis], [1, 3, 1, 3]),
        shape=(num_of_visits * 3, num_of_visits * 3),
    )[:-1, :-1]

    # (batch_size, 1, num_of_visits_with_att, num_of_visits_with_att)
    look_ahead_visit_mask_with_att = tf.maximum(visit_mask_with_att, look_ahead_visit_mask_with_att)

    return look_ahead_visit_mask_with_att


def create_concept_masks(visit_mask, num_of_visits, num_of_concepts):
    # Expand dimension for masking MultiHeadAttention in Visit Encoder
    # (num_of_visits_with_att, num_of_visits_with_att)
    look_ahead_mask_base = tf.cast(
        1 - tf.linalg.band_part(tf.ones((num_of_visits, num_of_visits)), -1, 0),
        dtype=tf.int32,
    )

    look_ahead_concept_mask = tf.reshape(
        tf.tile(
            look_ahead_mask_base[:, tf.newaxis, :, tf.newaxis],
            [1, num_of_concepts, 1, 1],
        ),
        (num_of_concepts * num_of_visits, -1),
    )

    # (batch_size, 1, num_of_visits * num_of_concepts, num_of_visits)
    look_ahead_concept_mask = tf.maximum(visit_mask[:, tf.newaxis, tf.newaxis, :], look_ahead_concept_mask)
    return look_ahead_concept_mask


class HierarchicalBertModel(tf.keras.Model):
    def __init__(
        self,
        num_of_visits: int,
        num_of_concepts: int,
        concept_vocab_size: int,
        visit_vocab_size: int,
        embedding_size: int,
        depth: int,
        num_heads: int,
        transformer_dropout: float = 0.1,
        embedding_dropout: float = 0.6,
        l2_reg_penalty: float = 1e-4,
        time_embeddings_size: int = 16,
        num_of_phenotypes: int = 20,
        num_of_phenotype_neighbors: int = 3,
        num_of_concept_neighbors: int = 10,
        phenotype_entropy_weight: float = 2e-05,
        phenotype_euclidean_weight: float = 2e-05,
        phenotype_concept_distance_weight: float = 1e-04,
        include_att_prediction: bool = True,
        **kwargs,
    ):
        super(HierarchicalBertModel, self).__init__(**kwargs)

        self.num_of_visits = num_of_visits
        self.num_of_concepts = num_of_concepts
        self.concept_vocab_size = concept_vocab_size
        self.visit_vocab_size = visit_vocab_size
        self.embedding_size = embedding_size
        self.time_embeddings_size = time_embeddings_size
        self.depth = depth
        self.num_heads = num_heads
        self.transformer_dropout = transformer_dropout
        self.embedding_dropout = embedding_dropout
        self.l2_reg_penalty = l2_reg_penalty
        self.num_of_phenotypes = num_of_phenotypes
        self.num_of_phenotype_neighbors = num_of_phenotype_neighbors
        self.num_of_concept_neighbors = num_of_concept_neighbors
        self.phenotype_entropy_weight = phenotype_entropy_weight
        self.phenotype_euclidean_weight = phenotype_euclidean_weight
        self.phenotype_concept_distance_weight = phenotype_concept_distance_weight
        self.include_att_prediction = include_att_prediction

        # output the embedding_matrix:
        self.l2_regularizer = tf.keras.regularizers.l2(l2_reg_penalty) if l2_reg_penalty else None

        self.concept_embedding_layer = ReusableEmbedding(
            concept_vocab_size,
            embedding_size,
            name="concept_embedding_layer",
            embeddings_regularizer=self.l2_regularizer,
        )

        self.visit_type_embedding_layer = ReusableEmbedding(
            visit_vocab_size,
            embedding_size,
            name="visit_type_embedding_layer",
            embeddings_regularizer=self.l2_regularizer,
        )

        # a layer for combining concept values with concept embeddings
        self.concept_value_transformation_layer = ConceptValueTransformationLayer(
            embedding_size=embedding_size, name="concept_value_transformation_layer"
        )

        # a layer for combining time/age embeddings with the concept embeddings
        self.temporal_transformation_layer = TemporalTransformationLayer(
            time_embeddings_size=time_embeddings_size,
            embedding_size=embedding_size,
            name="temporal_transformation_layer",
        )

        # (batch, num_of_visits, embedding_size)
        # The first bert applied at the visit level
        self.concept_encoder = Encoder(
            name="concept_encoder",
            num_layers=depth,
            d_model=embedding_size,
            num_heads=num_heads,
            dropout_rate=transformer_dropout,
        )

        # Second bert applied at the patient level to the visit embeddings
        self.visit_encoder = Encoder(
            name="visit_encoder",
            num_layers=depth,
            d_model=embedding_size,
            num_heads=num_heads,
            dropout_rate=transformer_dropout,
        )

        self.visit_type_embedding_dense_layer = tf.keras.layers.Dense(
            embedding_size, name="visit_type_embedding_dense_layer"
        )

        # A hidden phenotype layer that each visit embedding needs to go through
        self.visit_phenotype_layer = VisitPhenotypeLayer(
            num_of_phenotypes=num_of_phenotypes,
            num_of_phenotype_neighbors=num_of_phenotype_neighbors,
            num_of_concept_neighbors=num_of_concept_neighbors,
            embedding_size=embedding_size,
            transformer_dropout=transformer_dropout,
            phenotype_entropy_weight=phenotype_entropy_weight,
            phenotype_euclidean_weight=phenotype_euclidean_weight,
            phenotype_concept_distance_weight=phenotype_concept_distance_weight,
            name="hidden_visit_embeddings",
        )

        # Let local concept embeddings access the global representatives of each visit
        self.global_concept_embeddings_layer = SimpleDecoderLayer(
            d_model=embedding_size,
            num_heads=num_heads,
            rate=transformer_dropout,
            dff=512,
            name="global_concept_embeddings_layer",
        )

        self.concept_output_layer = TiedOutputEmbedding(
            projection_regularizer=self.l2_regularizer,
            projection_dropout=embedding_dropout,
            name="concept_prediction_logits",
        )

        self.concept_softmax_layer = tf.keras.layers.Softmax(name="concept_predictions")

        if include_att_prediction:
            self.global_att_embeddings_layer = SimpleDecoderLayer(
                d_model=embedding_size,
                num_heads=num_heads,
                rate=transformer_dropout,
                dff=512,
                name="global_att_embeddings_layer",
            )
            self.att_prediction_layer = tf.keras.layers.Softmax(
                name="att_predictions",
            )

    def call(self, inputs, **kwargs):

        pat_seq = inputs["pat_seq"]
        pat_seq_age = inputs["pat_seq_age"]
        pat_seq_time = inputs["pat_seq_time"]
        pat_mask = inputs["pat_mask"]
        concept_values = inputs["concept_values"]
        concept_value_masks = inputs["concept_value_masks"]
        visit_mask = inputs["visit_mask"]
        visit_time_delta_att = inputs["visit_time_delta_att"]
        visit_rank_order = inputs["visit_rank_order"]
        visit_visit_type = inputs["masked_visit_type"]

        # Expand dimensions for masking MultiHeadAttention in Concept Encoder
        pat_concept_mask = tf.reshape(pat_mask, shape=(-1, self.num_of_concepts))[:, tf.newaxis, tf.newaxis, :]

        # Look up the embeddings for the concepts
        concept_embeddings, embedding_matrix = self.concept_embedding_layer(pat_seq)

        # Look up the embeddings for the att tokens
        att_embeddings, _ = self.concept_embedding_layer(visit_time_delta_att)

        # Re-purpose token id 0 as the visit start embedding
        visit_start_embeddings, _ = self.concept_embedding_layer(tf.zeros_like(visit_mask, dtype=tf.int32))

        # Transform the concept embeddings by combining their concept embeddings with the
        # corresponding val
        concept_embeddings = self.concept_value_transformation_layer(
            concept_embeddings=concept_embeddings,
            concept_values=concept_values,
            concept_value_masks=concept_value_masks,
        )

        # (batch, num_of_visits, num_of_concepts, embedding_size)
        concept_embeddings = self.temporal_transformation_layer(
            self.num_of_concepts,
            concept_embeddings,
            pat_seq_age,
            pat_seq_time,
            visit_rank_order,
        )

        # (batch * num_of_visits, num_of_concepts, embedding_size)
        concept_embeddings = tf.reshape(concept_embeddings, shape=(-1, self.num_of_concepts, self.embedding_size))

        # Step 1: apply the first bert to the concept embeddings for each visit
        concept_embeddings, _ = self.concept_encoder(
            concept_embeddings, pat_concept_mask, **kwargs  # be reused  # not change,
        )

        # (batch, num_of_visits, num_of_concepts, embedding_size)
        concept_embeddings = tf.reshape(
            concept_embeddings,
            shape=(-1, self.num_of_visits, self.num_of_concepts, self.embedding_size),
        )

        # Step 2: generate visit embeddings
        # Slice out the first contextualized embedding of each visit
        # (batch_size, num_of_visits, embedding_size)
        visit_embeddings = concept_embeddings[:, :, 0]

        # (batch_size, num_of_visits, embedding_size)
        visit_type_embeddings, visit_type_embedding_matrix = self.visit_type_embedding_layer(visit_visit_type, **kwargs)

        # Combine visit_type_embeddings with visit_embeddings
        visit_embeddings = self.visit_type_embedding_dense_layer(
            tf.concat([visit_embeddings, visit_type_embeddings], axis=-1), **kwargs
        )

        # (batch_size, num_of_visits, embedding_size)
        expanded_att_embeddings = tf.concat([att_embeddings, att_embeddings[:, 0:1, :]], axis=1)

        # Insert the att embeddings between visit embeddings
        # (batch_size, num_of_visits + num_of_visits + num_of_visits - 1, embedding_size)
        visit_embeddings = tf.reshape(
            tf.concat(
                [visit_start_embeddings, visit_embeddings, expanded_att_embeddings],
                axis=-1,
            ),
            (-1, 3 * self.num_of_visits, self.embedding_size),
        )[:, :-1, :]

        look_ahead_visit_mask_with_att = create_visit_masks(visit_mask, self.num_of_visits)

        # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
        visit_embeddings, _ = self.visit_encoder(visit_embeddings, look_ahead_visit_mask_with_att, **kwargs)

        # Pad contextualized_visit_embeddings on axis 1 with one extra visit, we can extract the
        # visit embeddings using reshape trick
        padded_visit_embeddings = tf.concat([visit_embeddings, visit_embeddings[:, 0:1, :]], axis=1)

        # Extract the visit embeddings elements
        visit_embeddings_without_att = tf.reshape(
            padded_visit_embeddings, (-1, self.num_of_visits, 3 * self.embedding_size)
        )[:, :, self.embedding_size : self.embedding_size * 2]

        # (batch_size, num_of_visits, vocab_size)
        (
            visit_embeddings_without_att,
            _,
        ) = self.visit_phenotype_layer([visit_embeddings_without_att, visit_mask, embedding_matrix], **kwargs)

        # # Step 3 decoder applied to patient level
        # Reshape the data in visit view back to patient view:
        # (batch, num_of_visits * num_of_concepts, embedding_size)
        concept_embeddings = tf.reshape(
            concept_embeddings,
            shape=(-1, self.num_of_visits * self.num_of_concepts, self.embedding_size),
        )

        look_ahead_concept_mask = create_concept_masks(visit_mask, self.num_of_visits, self.num_of_concepts)

        global_concept_embeddings, _ = self.global_concept_embeddings_layer(
            concept_embeddings,
            visit_embeddings_without_att,
            look_ahead_concept_mask,
            **kwargs,
        )

        concept_predictions = self.concept_softmax_layer(
            self.concept_output_layer([global_concept_embeddings, embedding_matrix])
        )

        outputs = {
            "concept_predictions": concept_predictions,
            # 'padded_visit_embeddings': padded_visit_embeddings,
            # 'visit_embeddings_without_att': visit_embeddings_without_att
        }

        if self.include_att_prediction:
            # Extract the ATT embeddings
            contextualized_att_embeddings = tf.reshape(
                padded_visit_embeddings,
                (-1, self.num_of_visits, 3 * self.embedding_size),
            )[:, :-1, self.embedding_size * 2 :]

            # Create the att to concept mask ATT tokens only attend to the concepts in the
            # neighboring visits
            att_concept_mask = create_att_concept_mask(self.num_of_concepts, self.num_of_visits, visit_mask)

            # Use the simple decoder layer to decode att embeddings using the neighboring concept
            # embeddings
            contextualized_att_embeddings, _ = self.global_att_embeddings_layer(
                contextualized_att_embeddings,
                concept_embeddings,
                att_concept_mask,
                **kwargs,
            )

            att_predictions = self.att_prediction_layer(
                self.concept_output_layer([contextualized_att_embeddings, embedding_matrix])
            )
            outputs["att_predictions"] = att_predictions

        return outputs

    def get_config(self):
        config = super().get_config()
        config["concept_vocab_size"] = self.concept_vocab_size
        config["visit_vocab_size"] = self.visit_vocab_size
        config["embedding_size"] = self.embedding_size
        config["time_embeddings_size"] = self.time_embeddings_size
        config["depth"] = self.depth
        config["num_heads"] = self.num_heads
        config["transformer_dropout"] = self.transformer_dropout
        config["embedding_dropout"] = self.embedding_dropout
        config["l2_reg_penalty"] = self.l2_reg_penalty
        config["num_of_phenotypes"] = self.num_of_phenotypes
        config["num_of_phenotype_neighbors"] = self.num_of_phenotype_neighbors
        config["num_of_concept_neighbors"] = self.num_of_concept_neighbors
        config["phenotype_entropy_weight"] = self.phenotype_entropy_weight
        config["phenotype_euclidean_weight"] = self.phenotype_euclidean_weight
        config["phenotype_concept_distance_weight"] = self.phenotype_concept_distance_weight
        config["include_att_prediction"] = self.include_att_prediction
        return config


class MultiTaskHierarchicalBertModel(HierarchicalBertModel):
    def __init__(
        self,
        include_visit_prediction: bool,
        include_readmission: bool,
        include_prolonged_length_stay: bool,
        *args,
        **kwargs,
    ):
        super(MultiTaskHierarchicalBertModel, self).__init__(*args, **kwargs)

        self.include_visit_prediction = include_visit_prediction
        self.include_readmission = include_readmission
        self.include_prolonged_length_stay = include_prolonged_length_stay

        if include_visit_prediction:
            self.visit_prediction_output_layer = TiedOutputEmbedding(
                projection_regularizer=self.l2_regularizer,
                projection_dropout=self.embedding_dropout,
                name="visit_type_prediction_logits",
            )

            self.visit_softmax_layer = tf.keras.layers.Softmax(name="visit_predictions")

        if include_readmission:
            self.is_readmission_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="is_readmission")

        if include_prolonged_length_stay:
            self.visit_prolonged_stay_layer = tf.keras.layers.Dense(
                1, activation="sigmoid", name="visit_prolonged_stay"
            )

    def get_config(self):
        config = super().get_config()
        config["include_visit_prediction"] = self.include_visit_prediction
        config["include_readmission"] = self.include_readmission
        config["include_prolonged_length_stay"] = self.include_prolonged_length_stay
        return config

    def call(self, inputs, **kwargs):
        # Get the outputs from the super class
        outputs = super(MultiTaskHierarchicalBertModel, self).call(inputs, **kwargs)

        visit_embeddings_without_att = outputs["visit_embeddings_without_att"]

        if self.include_visit_prediction:
            # Slice out the visit embeddings (CLS tokens)
            visit_type_embedding_matrix = self.visit_type_embedding_layer.embeddings
            visit_predictions = self.visit_softmax_layer(
                self.visit_prediction_output_layer([visit_embeddings_without_att, visit_type_embedding_matrix])
            )
            outputs["visit_predictions"] = visit_predictions

        if self.include_readmission:
            is_readmission_output = self.is_readmission_layer(visit_embeddings_without_att)
            outputs["is_readmission"] = is_readmission_output

        if self.include_prolonged_length_stay:
            visit_prolonged_stay_output = self.visit_prolonged_stay_layer(visit_embeddings_without_att)
            outputs["visit_prolonged_stay"] = visit_prolonged_stay_output

        return outputs
