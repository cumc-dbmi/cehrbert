import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

from .custom_layers import Encoder


class HierarchicalBertLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_of_exchanges,
        num_of_visits,
        num_of_concepts,
        depth,
        embedding_size,
        num_heads,
        dropout_rate=0.1,
        *args,
        **kwargs,
    ):
        super(HierarchicalBertLayer, self).__init__(*args, **kwargs)

        assert embedding_size % num_heads == 0

        self.num_of_visits = num_of_visits
        self.num_of_concepts = num_of_concepts
        self.num_of_exchanges = num_of_exchanges
        self.embedding_size = embedding_size
        self.depth = depth
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.concept_encoder_layer = Encoder(
            name="concept_encoder",
            num_layers=depth,
            d_model=embedding_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.visit_encoder_layer = Encoder(
            name="visit_encoder",
            num_layers=depth,
            d_model=embedding_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
        )
        self.mha_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embedding_size // num_heads, name="mha"
        )

        # Insert the att embeddings between the visit embeddings using the following trick
        self.identity = tf.constant(
            np.insert(
                np.identity(self.num_of_visits),
                obj=range(1, self.num_of_visits),
                values=0,
                axis=1,
            ),
            dtype=tf.float32,
        )

        # Create the inverse "identity" matrix for inserting att embeddings
        self.identity_inverse = tf.constant(
            np.insert(
                np.identity(self.num_of_visits - 1),
                obj=range(0, self.num_of_visits),
                values=0,
                axis=1,
            ),
            dtype=tf.float32,
        )

        self.merge_matrix = tf.constant([1] + [0] * (self.num_of_concepts - 1), dtype=tf.float32)[
            tf.newaxis, tf.newaxis, :, tf.newaxis
        ]

        self.merge_matrix_inverse = tf.constant([0] + [1] * (self.num_of_concepts - 1), dtype=tf.float32)[
            tf.newaxis, tf.newaxis, :, tf.newaxis
        ]

        self.global_embedding_dropout_layer = tf.keras.layers.Dropout(dropout_rate)
        self.global_concept_embeddings_normalization = tf.keras.layers.LayerNormalization(
            name="global_concept_embeddings_normalization", epsilon=1e-6
        )

    def get_config(self):
        config = super().get_config()
        config["num_of_visits"] = self.num_of_visits
        config["num_of_concepts"] = self.num_of_concepts
        config["num_of_exchanges"] = self.num_of_exchanges
        config["embedding_size"] = self.embedding_size
        config["depth"] = self.depth
        config["num_heads"] = self.num_heads
        config["dropout_rate"] = self.dropout_rate

        return config

    def call(
        self,
        temporal_concept_embeddings,
        att_embeddings,
        pat_concept_mask,
        visit_concept_mask,
        **kwargs,
    ):
        for i in range(self.num_of_exchanges):
            # Step 1
            # (batch_size * num_of_visits, num_of_concepts, embedding_size)
            contextualized_concept_embeddings, _ = self.concept_encoder_layer(
                temporal_concept_embeddings,  # be reused
                pat_concept_mask,  # not change
                **kwargs,
            )

            # (batch_size, num_of_visits, num_of_concepts, embedding_size)
            contextualized_concept_embeddings = tf.reshape(
                contextualized_concept_embeddings,
                shape=(
                    -1,
                    self.num_of_visits,
                    self.num_of_concepts,
                    self.embedding_size,
                ),
            )
            # Step 2 generate augmented visit embeddings
            # Slice out the first contextualized embedding of each visit
            # (batch_size, num_of_visits, embedding_size)
            visit_embeddings = contextualized_concept_embeddings[:, :, 0]

            # Reshape the data in visit view back to patient view:
            # (batch, num_of_visits * num_of_concepts, embedding_size)
            contextualized_concept_embeddings = tf.reshape(
                contextualized_concept_embeddings,
                shape=(
                    -1,
                    self.num_of_visits * self.num_of_concepts,
                    self.embedding_size,
                ),
            )

            # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
            expanded_visit_embeddings = tf.transpose(
                tf.transpose(visit_embeddings, perm=[0, 2, 1]) @ self.identity,
                perm=[0, 2, 1],
            )

            # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
            expanded_att_embeddings = tf.transpose(
                tf.transpose(att_embeddings, perm=[0, 2, 1]) @ self.identity_inverse,
                perm=[0, 2, 1],
            )

            # Insert the att embeddings between visit embedidngs
            # (batch_size, num_of_visits + num_of_visits - 1, embedding_size)
            augmented_visit_embeddings = expanded_visit_embeddings + expanded_att_embeddings

            # Step 3 encoder applied to patient level
            # Feed augmented visit embeddings into encoders to get contextualized visit embeddings
            visit_embeddings, _ = self.visit_encoder_layer(augmented_visit_embeddings, visit_concept_mask, **kwargs)
            # v, k, q
            global_concept_embeddings = self.mha_layer(
                value=visit_embeddings,
                key=visit_embeddings,
                query=contextualized_concept_embeddings,
                attention_mask=visit_concept_mask,
                return_attention_scores=False,
            )

            global_concept_embeddings = self.global_embedding_dropout_layer(global_concept_embeddings)

            global_concept_embeddings = self.global_concept_embeddings_normalization(
                global_concept_embeddings + contextualized_concept_embeddings
            )

            att_embeddings = self.identity_inverse @ visit_embeddings

            visit_embeddings_without_att = self.identity @ visit_embeddings

            global_concept_embeddings = tf.reshape(
                global_concept_embeddings,
                (-1, self.num_of_visits, self.num_of_concepts, self.embedding_size),
            )

            global_concept_embeddings += (
                global_concept_embeddings * self.merge_matrix_inverse
                + tf.expand_dims(visit_embeddings_without_att, axis=-2) * self.merge_matrix
            )

            temporal_concept_embeddings = tf.reshape(
                global_concept_embeddings,
                (-1, self.num_of_concepts, self.embedding_size),
            )

        global_concept_embeddings = tf.reshape(
            global_concept_embeddings,
            (-1, self.num_of_visits * self.num_of_concepts, self.embedding_size),
        )

        return global_concept_embeddings, self.identity @ visit_embeddings


get_custom_objects().update({"HierarchicalBertLayer": HierarchicalBertLayer})
