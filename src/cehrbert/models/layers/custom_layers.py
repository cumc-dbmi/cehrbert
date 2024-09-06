import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects

from ...keras_transformer.bert import MaskedPenalizedSparseCategoricalCrossentropy
from ...keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from ...utils.model_utils import create_concept_mask


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential(
        [
            tf.keras.layers.Dense(dff, activation="relu"),  # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model),  # (batch_size, seq_len, d_model)
        ]
    )


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(EncoderLayer, self).__init__(*args, **kwargs)

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            output_shape=d_model,
            attention_axes=1,
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["dff"] = self.dff
        config["rate"] = self.rate
        return config

    def call(self, x, mask, **kwargs):
        # The reason we are doing this is that tensorflow on Mac doesn't seem to recognize the rank correctly
        # if platform.system() == 'Darwin':
        batch, length = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (batch, -1, self.d_model))
        mask = tf.reshape(mask, (batch, -1, length))

        attn_output, attn_weights = self.mha(
            query=x,
            key=x,
            value=x,
            attention_mask=mask,
            return_attention_scores=True,
            **kwargs,
        )
        attn_output = self.dropout1(attn_output, **kwargs)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, **kwargs)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attn_weights


class Encoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff=2148,
        dropout_rate=0.1,
        *args,
        **kwargs,
    ):
        super(Encoder, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, dff, dropout_rate, name="transformer" + str(i)) for i in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config["num_layers"] = self.num_layers
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["dff"] = self.dff
        config["dropout_rate"] = self.dropout_rate
        return config

    def call(self, x, mask, **kwargs):
        attention_weights = []
        for i in range(self.num_layers):
            x, attn_weights = self.enc_layers[i](x, mask, **kwargs)
            attention_weights.append(attn_weights)
        return x, tf.stack(attention_weights, axis=0)  # (batch_size, input_seq_len, d_model)


class GptDecoder(tf.keras.layers.Layer):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff=2148,
        dropout_rate=0.1,
        *args,
        **kwargs,
    ):
        super(GptDecoder, self).__init__(*args, **kwargs)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.decoder_layers = [
            GptDecoderLayer(d_model, num_heads, dff, dropout_rate, name="transformer" + str(i))
            for i in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super().get_config()
        config["num_layers"] = self.num_layers
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["dff"] = self.dff
        config["dropout_rate"] = self.dropout_rate
        return config

    def call(self, x, **kwargs):
        attention_weights = []
        layer_contexts = []
        for i in range(self.num_layers):
            x, attn_weights = self.decoder_layers[i](x, x, x, **kwargs)
            attention_weights.append(attn_weights)
            layer_contexts.append(x)
        return x, tf.stack(layer_contexts, axis=0), tf.stack(attention_weights, axis=0)


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(DecoderLayer, self).__init__(*args, **kwargs)

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            output_shape=d_model,
            attention_axes=1,
        )
        self.mha2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            output_shape=d_model,
            attention_axes=1,
        )

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["dff"] = self.dff
        config["rate"] = self.rate
        return config

    def call(self, x, enc_output, decoder_mask, encoder_mask, **kwargs):
        # The reason we are doing this is that tensorflow on Mac doesn't seem to recognize the rank correctly
        # if platform.system() == 'Darwin':
        batch, length = tf.shape(x)[0], tf.shape(x)[1]
        x = tf.reshape(x, (batch, -1, self.d_model))
        decoder_mask = tf.reshape(decoder_mask, (batch, -1, length))
        encoder_mask = tf.reshape(encoder_mask, (batch, -1, length))

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(
            query=x,
            key=x,
            value=x,
            attention_mask=decoder_mask,
            return_attention_scores=True,
            **kwargs,
        )  # (batch_size, target_seq_len, d_model)

        attn1 = self.dropout1(attn1, **kwargs)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            value=enc_output,
            key=enc_output,
            query=out1,
            attention_mask=encoder_mask,
            return_attention_scores=True,
            **kwargs,
        )  # (batch_size, target_seq_len, d_model)

        attn2 = self.dropout2(attn2, **kwargs)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, **kwargs)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class GptDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1, *args, **kwargs):
        super(GptDecoderLayer, self).__init__(*args, **kwargs)

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["dff"] = self.dff
        config["rate"] = self.rate
        return config

    def call(self, query, key, value, decoder_mask=None, **kwargs):
        # Supports backward compatibility
        if "mask" in kwargs:
            kwargs.pop("mask")

        # (batch_size, target_seq_len, d_model)
        attn, attn_weights_block = self.mha(
            value=value,
            key=key,
            query=query,
            attention_mask=decoder_mask,
            use_causal_mask=decoder_mask is None,
            return_attention_scores=True,
            **kwargs,
        )

        attn = self.dropout1(attn, **kwargs)

        # The reason we are doing this is that tensorflow on Mac doesn't seem to recognize the rank correctly
        batch = tf.shape(query)[0]
        attn = tf.reshape(attn, (batch, -1, self.d_model))
        query = tf.reshape(query, (batch, -1, self.d_model))

        out = self.layernorm1(attn + query)

        ffn_output = self.ffn(out)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, **kwargs)
        out2 = self.layernorm2(ffn_output + out)  # (batch_size, target_seq_len, d_model)

        return out2, attn_weights_block


class NonTrainablePositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._maxlen = maxlen
        self._embed_dim = embed_dim
        self.pos_emb = tf.squeeze(positional_encoding(maxlen, embed_dim))

    def get_config(self):
        config = super().get_config()
        config["maxlen"] = self._maxlen
        config["embed_dim"] = self._embed_dim
        return config

    def call(self, x, **kwargs):
        # kwargs is needed for backward compatability
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = tf.gather(self.pos_emb, positions, axis=0)
        return position_embeddings


class TrainablePositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_emb = tf.keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self._maxlen = maxlen
        self._embed_dim = embed_dim

    def get_config(self):
        config = super().get_config()
        config["maxlen"] = self._maxlen
        config["embed_dim"] = self._embed_dim
        return config

    def call(self, x, **kwargs):
        # kwargs is needed for backward compatability
        maxlen = tf.shape(x)[1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions, **kwargs)
        return positions


class SimpleDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff=512, rate=0.1, *args, **kwargs):
        super(SimpleDecoderLayer, self).__init__(*args, **kwargs)
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        self.multi_head_attention_layer = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.mha_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha_dropout_layer = tf.keras.layers.Dropout(rate)
        self.ffn_dropout_layer = tf.keras.layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["dff"] = self.dff
        config["rate"] = self.rate
        return config

    def call(self, decoder_input, enc_output, encoder_mask, **kwargs):
        # The reason we are doing this is that tensorflow on Mac doesn't seem to recognize the rank correctly
        # if platform.system() == 'Darwin':
        batch, enc_length = tf.shape(enc_output)[0], tf.shape(enc_output)[1]
        enc_output = tf.reshape(enc_output, (batch, -1, self.d_model))
        encoder_mask = tf.reshape(encoder_mask, (batch, -1, enc_length))

        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn, attn_weights_block = self.multi_head_attention_layer(
            value=enc_output,
            key=enc_output,
            query=decoder_input,
            attention_mask=encoder_mask,
            return_attention_scores=True,
            **kwargs,
        )  # (batch_size, target_seq_len, d_model)
        attn = self.mha_dropout_layer(attn, **kwargs)
        out2 = self.mha_layernorm(attn + decoder_input)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn_dropout_layer(ffn_output, **kwargs)
        out3 = self.ffn_layernorm(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, max_sequence_length=512, *args, **kwargs):
        super(PositionalEncodingLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.max_sequence_length = max_sequence_length
        # TODO: change this to dynamic in the future
        self.pos_encoding = tf.squeeze(positional_encoding(10000, self.embedding_size))

    def get_config(self):
        config = super().get_config()
        config["max_sequence_length"] = self.max_sequence_length
        config["embedding_size"] = self.embedding_size
        return config

    def call(self, visit_concept_orders):
        # Normalize the visit_orders using the smallest visit_concept_orders
        # Take the absolute value to make sure the padded values are not negative after
        # normalization
        visit_concept_orders = tf.abs(
            visit_concept_orders - tf.reduce_min(visit_concept_orders, axis=-1)[..., tf.newaxis]
        )
        # Get the same positional encodings for the concepts with the same visit_order
        positional_embeddings = tf.gather(self.pos_encoding, visit_concept_orders, axis=0)
        return positional_embeddings


class TimeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, is_time_delta=False, *args, **kwargs):
        super(TimeEmbeddingLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.is_time_delta = is_time_delta
        self.w = self.add_weight(
            shape=(1, self.embedding_size),
            trainable=True,
            initializer=tf.keras.initializers.GlorotNormal(),
            name=f"time_embedding_weight_{self.name}",
        )
        self.phi = self.add_weight(
            shape=(1, self.embedding_size),
            trainable=True,
            initializer=tf.keras.initializers.GlorotNormal(),
            name=f"time_embedding_phi_{self.name}",
        )

    def get_config(self):
        config = super().get_config()
        config["embedding_size"] = self.embedding_size
        config["is_time_delta"] = self.is_time_delta
        return config

    def call(self, time_stamps):
        time_stamps = tf.cast(time_stamps, tf.float32)
        if self.is_time_delta:
            time_stamps = tf.concat(
                [time_stamps[:, 0:1] * 0, time_stamps[:, 1:] - time_stamps[:, :-1]],
                axis=-1,
            )
        next_input = tf.expand_dims(time_stamps, axis=-1) * self.w + self.phi
        return tf.sin(next_input)


class VisitEmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, visit_order_size: int, embedding_size: int, *args, **kwargs):
        super(VisitEmbeddingLayer, self).__init__(*args, **kwargs)
        self.visit_order_size = visit_order_size
        self.embedding_size = embedding_size

        self.visit_embedding_layer = tf.keras.layers.Embedding(self.visit_order_size, self.embedding_size)

    def get_config(self):
        config = super().get_config()
        config["visit_order_size"] = self.visit_order_size
        config["embedding_size"] = self.embedding_size
        return config

    def call(self, inputs, **kwargs):
        visit_orders, concept_embeddings = inputs
        return self.visit_embedding_layer(visit_orders, **kwargs) + concept_embeddings


class ConceptValuePredictionLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, *args, **kwargs):
        super(ConceptValuePredictionLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.concept_value_decoder_layer = tf.keras.Sequential(
            layers=[
                tf.keras.layers.Dense(self.embedding_size, activation="tanh"),
                tf.keras.layers.Dense(self.embedding_size, activation="tanh"),
                tf.keras.layers.Dense(1),
            ],
            name="value_decoder_layer",
        )

    def get_config(self):
        config = super().get_config()
        config["embedding_size"] = self.embedding_size
        return config

    def call(self, original_concept_embeddings, concept_val_embeddings, concept_value_masks):
        # (batch_size, context_window, 2 * embedding_size)
        context = tf.concat([original_concept_embeddings, concept_val_embeddings], axis=-1)
        # (batch_size, context_window, 1)
        concept_vals = self.concept_value_decoder_layer(context)

        # (batch_size, context_window, 1)
        concept_value_masks = tf.expand_dims(concept_value_masks, axis=-1)
        # Zero out the positions without a val
        concept_vals = tf.multiply(concept_vals, tf.cast(concept_value_masks, dtype=tf.float32))
        return concept_vals


class ConceptValueTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size, *args, **kwargs):
        super(ConceptValueTransformationLayer, self).__init__(*args, **kwargs)
        self.embedding_size = embedding_size
        self.merge_value_transformation_layer = tf.keras.layers.Dense(
            embedding_size, name="merge_value_transformation_layer"
        )

    def get_config(self):
        config = super().get_config()
        config["embedding_size"] = self.embedding_size
        return config

    def call(self, concept_embeddings, concept_values, concept_value_masks):
        # Mask out the concept embeddings without a value
        # Combine the concept embeddings with concept_values

        # (batch_size, num_of_visits, num_of_concepts, 1)
        concept_values = tf.expand_dims(concept_values, axis=-1)
        # (batch_size, num_of_visits, num_of_concepts, 1)
        concept_value_masks = tf.expand_dims(concept_value_masks, axis=-1)
        # (batch_size, num_of_visits, num_of_concepts, 1 + embedding_size)
        concept_embeddings_with_val = tf.concat([concept_embeddings, concept_values], axis=-1)
        # Run through a dense layer to bring the dimension back to embedding_size
        concept_embeddings_with_val = self.merge_value_transformation_layer(concept_embeddings_with_val)
        # Zero out the positions without a val
        concept_embeddings_with_val = tf.multiply(
            concept_embeddings_with_val, tf.cast(concept_value_masks, dtype=tf.float32)
        )
        # Derive the inverse concept value masks for zeroing out the embeddings without a val
        inverse_concept_value_masks = tf.cast(
            tf.logical_not(tf.cast(concept_value_masks, dtype=tf.bool)),
            dtype=tf.float32,
        )

        # Zero out the position of concept embeddings with a val
        concept_embeddings_without_val = tf.multiply(inverse_concept_value_masks, concept_embeddings)

        # Merge two sets of concept embeddings
        concept_embeddings = concept_embeddings_without_val + concept_embeddings_with_val

        return concept_embeddings


class TemporalTransformationLayer(tf.keras.layers.Layer):
    def __init__(self, time_embeddings_size, embedding_size, *args, **kwargs):
        super(TemporalTransformationLayer, self).__init__(*args, **kwargs)

        self.time_embeddings_size = time_embeddings_size
        self.embedding_size = embedding_size

        # define the time embedding layer for absolute time stamps (since 1970)
        self.time_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size, name="time_embedding_layer")
        # define the age embedding layer for the age w.r.t the medical record
        self.age_embedding_layer = TimeEmbeddingLayer(embedding_size=time_embeddings_size, name="age_embedding_layer")

        # define positional encoding layer for visit numbers, the visit numbers are normalized
        # by subtracting visit numbers off the first visit number
        self.positional_encoding_layer = PositionalEncodingLayer(
            embedding_size=time_embeddings_size, name="positional_encoding_layer"
        )
        # Temporal transformation
        self.temporal_transformation_layer = tf.keras.layers.Dense(
            embedding_size, activation="tanh", name="temporal_transformation"
        )

    def get_config(self):
        config = super().get_config()
        config["time_embeddings_size"] = self.time_embeddings_size
        config["embedding_size"] = self.embedding_size
        return config

    def call(self, concept_embeddings, pat_seq_age, pat_seq_time, visit_rank_order, **kwargs):
        _, _, num_of_concepts = pat_seq_age.shape

        pt_seq_age_embeddings = self.age_embedding_layer(pat_seq_age, **kwargs)
        pt_seq_time_embeddings = self.time_embedding_layer(pat_seq_time, **kwargs)
        visit_positional_encoding = self.positional_encoding_layer(visit_rank_order, **kwargs)

        visit_positional_encoding = tf.tile(visit_positional_encoding[:, :, tf.newaxis, :], [1, 1, num_of_concepts, 1])

        # (batch, num_of_visits, num_of_concepts, embedding_size)
        temporal_concept_embeddings = self.temporal_transformation_layer(
            tf.concat(
                [
                    concept_embeddings,
                    pt_seq_age_embeddings,
                    pt_seq_time_embeddings,
                    visit_positional_encoding,
                ],
                axis=-1,
            )
        )

        return temporal_concept_embeddings


class BertLayer(tf.keras.layers.Layer):

    def __init__(self, model_path: str, *args, **kwargs):
        super(BertLayer, self).__init__(*args, **kwargs)
        bert_model = tf.keras.models.load_model(model_path, custom_objects=get_custom_objects())

        self.model_path = model_path
        self.concept_embedding_layer = bert_model.get_layer("concept_embeddings")
        self.visit_segment_layer = [
            layer for layer in bert_model.layers if layer.name in ["visit_embedding_layer", "visit_segment_layer"]
        ][0]
        self.positional_encoding_layer = bert_model.get_layer("positional_encoding_layer")
        self.time_embedding_layer = bert_model.get_layer("time_embedding_layer")
        self.age_embedding_layer = bert_model.get_layer("age_embedding_layer")
        self.scale_pat_seq_layer = bert_model.get_layer("scale_pat_seq_layer")
        self.encoder_layer = bert_model.get_layer("encoder")
        #         self.conv_1d = tf.keras.layers.Conv1D(1, 1)
        self.attention_dense = tf.keras.layers.Dense(self.scale_pat_seq_layer.units, activation="tanh")
        self.dense = tf.keras.layers.Dense(self.scale_pat_seq_layer.units, activation="tanh")

    def get_config(self):
        config = super().get_config()
        config["model_path"] = self.model_path
        return config

    def call(self, inputs, **kwargs):
        (
            local_concept_ids,
            local_visit_segments,
            local_visit_concept_orders,
            local_time_stamps,
            local_ages,
            local_mask,
        ) = inputs

        batch_size, max_seq_length = local_mask.get_shape().as_list()

        concept_embeddings, _ = self.concept_embedding_layer(local_concept_ids)
        time_embeddings = self.time_embedding_layer(local_time_stamps)
        age_embeddings = self.age_embedding_layer(local_ages)
        positional_encoddings = self.positional_encoding_layer(local_visit_concept_orders)
        concept_mask = create_concept_mask(local_mask, max_seq_length)

        input_for_encoder = self.scale_pat_seq_layer(
            tf.concat(
                [
                    concept_embeddings,
                    time_embeddings,
                    age_embeddings,
                    positional_encoddings,
                ],
                axis=-1,
            )
        )
        input_for_encoder = self.visit_segment_layer([local_visit_segments, input_for_encoder])
        contextualized_embeddings, _ = self.encoder_layer(input_for_encoder, concept_mask)
        _, _, embedding_size = contextualized_embeddings.get_shape().as_list()
        mask_embeddings = tf.tile(tf.expand_dims(local_mask == 0, -1), [1, 1, embedding_size])
        contextualized_embeddings = tf.math.multiply(
            contextualized_embeddings, tf.cast(mask_embeddings, dtype=tf.float32)
        )

        # (batch, seq_len, embeddings_size)
        multi_dim_att = tf.nn.softmax(
            self.attention_dense(contextualized_embeddings)
            + (tf.cast(tf.expand_dims(local_mask, axis=-1), dtype="float32") * -1e9),
            axis=1,
        )
        context_representation = tf.reduce_sum(multi_dim_att * contextualized_embeddings, axis=1)

        #         conv_output = self.conv_1d(contextualized_embeddings)
        #         conv_output += (tf.cast(tf.expand_dims(local_mask, axis=-1), dtype='float32') * -1e9)
        #         context_representation = tf.reshape(
        #             tf.transpose(tf.nn.softmax(conv_output, axis=1), [0, 2, 1]) @ contextualized_embeddings,
        #             (-1, self.conv_1d.filters * embedding_size))

        return self.dense(context_representation)


class ConvolutionBertLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        model_path: str,
        seq_len: int,
        context_window: int,
        stride: int,
        *args,
        **kwargs,
    ):
        super(ConvolutionBertLayer, self).__init__(*args, **kwargs)
        self.model_path = model_path
        self.seq_len = seq_len
        self.context_window = context_window
        self.stride = stride
        self.step = (seq_len - context_window) // stride + 1
        self.bert_layer = BertLayer(model_path=model_path)
        #         self.conv_1d = tf.keras.layers.Conv1D(1, 1)
        self.attention_dense = tf.keras.layers.Dense(self.bert_layer.scale_pat_seq_layer.units, activation="tanh")

        assert (self.step - 1) * self.stride + self.context_window == self.seq_len

    def get_config(self):
        config = super().get_config()
        config["model_path"] = self.model_path
        config["seq_len"] = self.seq_len
        config["context_window"] = self.context_window
        config["stride"] = self.stride
        return config

    def call(self, inputs, **kwargs):
        concept_ids, visit_segments, visit_concept_orders, time_stamps, ages, mask = inputs

        bert_outputs = []
        bert_output_masking = []
        for i in range(self.step):
            start_index = i * self.stride
            end_index = i * self.stride + self.context_window

            concept_ids_step = concept_ids[:, start_index:end_index]
            visit_segments_step = visit_segments[:, start_index:end_index]
            time_stamps_step = time_stamps[:, start_index:end_index]
            ages_step = ages[:, start_index:end_index]
            visit_concept_orders_step = visit_concept_orders[:, start_index:end_index]
            mask_step = mask[:, start_index:end_index]

            inputs_step = [
                concept_ids_step,
                visit_segments_step,
                visit_concept_orders_step,
                time_stamps_step,
                ages_step,
                mask_step,
            ]

            output_masking = tf.cast(tf.reduce_all(mask_step == 1, axis=-1), dtype=tf.int32)

            output_step = self.bert_layer(inputs_step)
            bert_outputs.append(output_step)
            bert_output_masking.append(output_masking)

        # (batch, step, embedding_size)
        bert_output_tensor = tf.stack(bert_outputs, axis=1)
        # (batch, step)
        bert_output_masking_tensor = tf.stack(bert_output_masking, axis=1)
        # (batch, step, 1)
        #         conv_output = self.conv_1d(bert_output_tensor)

        attn = self.attention_dense(bert_output_tensor)

        attn += tf.cast(tf.expand_dims(bert_output_masking_tensor, axis=-1), dtype="float32") * -1e9

        _, _, embedding_size = bert_output_tensor.get_shape().as_list()

        context_representation = tf.reduce_sum(tf.nn.softmax(attn, axis=1) * bert_output_tensor, axis=1)

        #         context_representation = tf.reshape(
        #             tf.transpose(tf.nn.softmax(conv_output, axis=1), [0, 2, 1]) @ bert_output_tensor,
        #             (-1, self.conv_1d.filters * embedding_size))

        return context_representation


class VisitPhenotypeLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        num_of_phenotypes: int,
        num_of_phenotype_neighbors: int,
        num_of_concept_neighbors: int,
        embedding_size: int,
        transformer_dropout: float,
        dff: int = 2148,
        *args,
        **kwargs,
    ):
        super(VisitPhenotypeLayer, self).__init__(*args, **kwargs)
        self.num_of_phenotypes = num_of_phenotypes
        self.embedding_size = embedding_size
        self.transformer_dropout = transformer_dropout
        self.dff = dff
        self.num_of_concept_neighbors = num_of_concept_neighbors
        self.num_of_phenotype_neighbors = num_of_phenotype_neighbors

        # We assume there exists hidden phenotype embeddings
        # (num_of_phenotypes, embedding_size)
        self.phenotype_embeddings = self.add_weight(
            shape=(num_of_phenotypes, embedding_size),
            initializer=tf.keras.initializers.GlorotUniform(seed=0),
            trainable=True,
            name="phenotype_embeddings_matrix",
        )

        self.ffn = point_wise_feed_forward_network(embedding_size, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(transformer_dropout)
        self.dropout2 = tf.keras.layers.Dropout(transformer_dropout)

    def get_config(self):
        config = super().get_config()
        config["num_of_phenotypes"] = self.num_of_phenotypes
        config["embedding_size"] = self.embedding_size
        config["transformer_dropout"] = self.transformer_dropout
        config["dff"] = self.dff
        config["num_of_concept_neighbors"] = self.num_of_concept_neighbors
        config["num_of_phenotype_neighbors"] = self.num_of_phenotype_neighbors
        return config

    def call(self, inputs, **kwargs):
        visit_embeddings, visit_mask, embedding_matrix = inputs

        # Do not compute the entropy for the masked visits
        converted_visit_mask = tf.cast(tf.logical_not(tf.cast(visit_mask, dtype=tf.bool)), dtype=tf.float32)[
            :, :, tf.newaxis
        ]

        # (batch_size, num_of_visits, num_of_phenotypes)
        visit_phenotype_probs = tf.nn.softmax(
            visit_embeddings @ tf.transpose(self.phenotype_embeddings, [1, 0]) * converted_visit_mask
        )

        # calculate phenotype concept distance matrix (num_of_phenotypes, top_k)
        phenotype_concept_dist = tf.reduce_mean(
            -tf.math.top_k(
                -distance_matrix(self.phenotype_embeddings, embedding_matrix),
                k=self.num_of_concept_neighbors,
            ).values
        )

        self.add_metric(phenotype_concept_dist, name="phenotype_concept_dist")

        # Calculate the probability distribution entropy
        phenotype_prob_entropy = -tf.reduce_sum(
            visit_phenotype_probs * tf.math.log(visit_phenotype_probs) * converted_visit_mask,
            axis=-1,
        )
        # Add the entropy to the model metrics
        self.add_metric(phenotype_prob_entropy, name="phenotype_probability_entropy")

        # Get phenotype pairwise distance metrics
        phe_inv_loss, phe_dist_metric, phe_dist_var = self.get_inverse_phenotype_dist_loss_metric()

        self.add_metric(phe_dist_metric, name="phenotype_euclidean_distance")

        self.add_metric(phe_dist_var, name="phenotype_euclidean_variance")

        # Calculate the contextualized visit embeddings using the pre-defined phenotype embeddings
        # (batch_size, num_of_visits, embedding_size)
        contextualized_phenotype_embeddings = self.dropout1(
            visit_phenotype_probs @ self.phenotype_embeddings,
            training=kwargs.get("training"),
        )

        out1 = self.layernorm1(visit_embeddings + contextualized_phenotype_embeddings)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=kwargs.get("training"))
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, visit_phenotype_probs

    def get_inverse_phenotype_dist_loss_metric(self):
        r = tf.reduce_sum(self.phenotype_embeddings * self.phenotype_embeddings, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        euclidean_distances_full = (
            r - 2 * tf.matmul(self.phenotype_embeddings, tf.transpose(self.phenotype_embeddings)) + tf.transpose(r)
        )

        euclidean_distances = -tf.math.top_k(-euclidean_distances_full, k=self.num_of_phenotype_neighbors).values

        inv_loss = tf.reduce_mean(tf.math.exp(-euclidean_distances))

        var_loss = tf.math.reduce_variance(euclidean_distances)

        dist_metric = tf.reduce_mean(euclidean_distances)

        return inv_loss, dist_metric, var_loss


def distance_matrix(matrix_1, matrix_2):
    m = matrix_1.shape[0]
    n = matrix_2.shape[0]

    assert (
        matrix_1.shape[1] == matrix_2.shape[1]
    ), f"The number of components for vectors in A \
            {matrix_1.shape[1]} does not match that of B {matrix_2.shape[1]}!"

    matrix_1_dots = tf.reshape(tf.reduce_sum(matrix_1 * matrix_1, axis=1), (m, 1)) * tf.ones((1, n))
    matrix_2_dots = tf.reduce_sum(matrix_2 * matrix_2, axis=1) * tf.ones((m, 1))

    matrix_distance_squared = matrix_1_dots + matrix_2_dots - 2 * matrix_1 @ tf.transpose(matrix_2)

    return tf.sqrt(matrix_distance_squared)


get_custom_objects().update(
    {
        "Encoder": Encoder,
        "GptDecoder": GptDecoder,
        "GptDecoderLayer": GptDecoderLayer,
        "TrainablePositionEmbedding": TrainablePositionEmbedding,
        "EncoderLayer": EncoderLayer,
        "DecoderLayer": DecoderLayer,
        "SimpleDecoderLayer": SimpleDecoderLayer,
        "VisitEmbeddingLayer": VisitEmbeddingLayer,
        "PositionalEncodingLayer": PositionalEncodingLayer,
        "NonTrainablePositionEmbedding": NonTrainablePositionEmbedding,
        "TimeEmbeddingLayer": TimeEmbeddingLayer,
        "TemporalTransformationLayer": TemporalTransformationLayer,
        "ConceptValueTransformationLayer": ConceptValueTransformationLayer,
        "ReusableEmbedding": ReusableEmbedding,
        "TiedOutputEmbedding": TiedOutputEmbedding,
        "MaskedPenalizedSparseCategoricalCrossentropy": MaskedPenalizedSparseCategoricalCrossentropy,
        "BertLayer": BertLayer,
        "ConvolutionBertLayer": ConvolutionBertLayer,
        "VisitPhenotypeLayer": VisitPhenotypeLayer,
        "ConceptValuePredictionLayer": ConceptValuePredictionLayer,
    }
)
