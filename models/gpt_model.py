import datetime
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from data_generators.tokenizer import ConceptTokenizer
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from models.layers.custom_layers import GptDecoder, TrainablePositionEmbedding


class GptInferenceModel(tf.keras.Model):
    def __init__(
            self,
            gpt_model: tf.keras.Model,
            tokenizer: ConceptTokenizer,
            context_window: int,
            top_k: int,
            *args,
            **kwargs
    ):
        super(GptInferenceModel, self).__init__(*args, **kwargs)

        self.context_window = context_window
        self.tokenizer = tokenizer
        self.top_k = top_k
        self.concept_embedding_layer = self._get_concept_embedding(gpt_model)
        self.positional_encoding_layer = self._get_positional_encoding_layer(gpt_model)
        self.output_layer = self._get_output_layer(gpt_model)
        self.gpt_decoder = self._get_gpt_decoder(gpt_model)
        self.vocab_size = self.concept_embedding_layer.input_dim
        self.att_token_ids = self._get_att_token_ids()

    def _get_att_token_ids(self):
        """
        We assume we create artificial tokens before 1080 days
        :return:
        """
        day_tokens = [f'D{i}' for i in range(1080)]
        month_tokens = [f'M{i}' for i in range(40)]
        quarter_tokens = [f'Q{i}' for i in range(12)]
        year_tokens = [f'Y{i}' for i in range(3)]

        att_tokens = day_tokens + month_tokens + quarter_tokens + year_tokens

        att_token_ids = []
        for token, token_id in self.tokenizer.tokenizer.word_index.items():
            if token in att_tokens:
                att_token_ids.append(token_id)

        return att_token_ids

    def _generate_next_token(
            self,
            generated_tokens,
            cached_contexts
    ):
        current_length = tf.shape(generated_tokens)[-1]
        previous_length = tf.shape(cached_contexts)[2] if cached_contexts is not None else 0

        first_layer_context, concept_embedding_matrix = self.concept_embedding_layer(
            generated_tokens,
            training=False
        )
        # Add the positional embeddings
        first_layer_context += self.positional_encoding_layer(
            first_layer_context
        )

        look_ahead_mask_base = tf.cast(
            1 - tf.linalg.band_part(
                tf.ones((current_length, current_length)), -1, 0
            ),
            dtype=tf.int32
        )[tf.newaxis, tf.newaxis, previous_length:current_length, :]

        if cached_contexts is not None:
            # Slice out the new token representations
            x = first_layer_context[:, previous_length: current_length]
        else:
            x = first_layer_context

        layer_contexts = []
        for i in range(self.gpt_decoder.num_layers):
            if i == 0:
                value = first_layer_context
            elif cached_contexts is not None:
                # Concat the previous context with the x representation
                value = tf.concat([cached_contexts[i - 1], x], axis=1)
            else:
                value = x

            x, attn_weights = self.gpt_decoder.decoder_layers[i](
                query=x,
                key=value,
                value=value,
                decoder_mask=look_ahead_mask_base,
                training=False
            )
            layer_contexts.append(x)

        layer_contexts = tf.stack(layer_contexts, axis=0)

        if cached_contexts is not None:
            new_cached_contexts = tf.concat([cached_contexts, layer_contexts], axis=2)
        else:
            new_cached_contexts = layer_contexts

        logtis = self.output_layer(
            [x, concept_embedding_matrix],
            training=False
        )

        return logtis, new_cached_contexts

    def _block_recurring_tokens(
            self,
            inputs,
            outputs,
            disallowed_token_ids,
            vs_token_id
    ):
        batch, length = tf.shape(inputs)
        # Get the index of the last occurrence of VS token in the sequence
        last_vs_token_index = self._find_last_index_of_token(
            sequence_in_batch=inputs,
            token_id=vs_token_id
        )

        # If this is a new visit, indicated by last_vs_token_index == length - 1, we will clear out the memory
        new_visit_indicator = tf.cast(last_vs_token_index == length - 1, dtype=tf.float32)
        disallowed_token_ids = (1 - new_visit_indicator) * disallowed_token_ids

        # Mask token ids by adding -1e9 to their corresponding logits
        masked_outputs = (disallowed_token_ids * -1e9)[:, tf.newaxis, :] + outputs
        return masked_outputs, disallowed_token_ids

    def call(
            self,
            inputs
    ):

        # Get the current sequence length
        batch, length = tf.shape(
            inputs
        )
        #  Create a cache contexts to store the previous states
        cached_contexts = None
        # disallowed_token_ids = tf.zeros((
        #     batch,
        #     self.tokenizer.get_vocab_size()
        # ))

        while length < self.context_window:
            # Generate the next batch of tokens and update the contexts
            outputs, cached_contexts = self._generate_next_token(
                inputs,
                cached_contexts
            )
            # # Block the sampling of the tokens that already appear within the same visit (defined
            # # as the tokens that appear since the last VS)
            # outputs, disallowed_token_ids = self._block_recurring_tokens(
            #     inputs,
            #     outputs,
            #     disallowed_token_ids,
            #     self.tokenizer.get_visit_start_token_id()
            # )
            # Randomly sample a batch of tokens
            pred_logits, indices = tf.math.top_k(outputs[:, -1, :], k=self.top_k, sorted=True)
            indices = np.asarray(indices).astype('int32')

            next_tokens = tf.gather(
                indices,
                tf.random.categorical(pred_logits, 1),
                axis=1,
                batch_dims=1
            ).numpy()
            #
            # disallowed_token_ids += tf.one_hot(
            #     tf.squeeze(next_tokens),
            #     self.tokenizer.get_vocab_size()
            # )

            # Check if any of the current token is att tokens
            att_token_indicators = tf.cast(
                tf.reduce_any(inputs[:, -1:] == self.att_token_ids, axis=-1),
                dtype=tf.int32
            )[:, tf.newaxis]

            # Replace the next tokens with VS token if the current token is an ATT token
            next_tokens = (
                    att_token_indicators * self.tokenizer.get_visit_start_token_id() +
                    (1 - att_token_indicators) * next_tokens
            )
            #
            # new_visit_indicator = tf.cast(
            #     next_tokens == self.tokenizer.get_visit_start_token_id(),
            #     dtype=tf.int32
            # )

            # Stitch up the new tokens and previously generated tokens
            inputs = np.hstack(
                [inputs, next_tokens]
            )

            # Get the new length of the sequence
            _, length = np.shape(
                inputs
            )

            # This indicates all the sequences have ended
            if np.all(np.any(inputs == self.tokenizer.get_end_token_id(), axis=-1)):
                break

        return inputs

    @staticmethod
    def _find_last_index_of_token(
            sequence_in_batch,
            token_id
    ):
        length = tf.shape(sequence_in_batch)[1].numpy()
        argmax_in_reverse = tf.argmax(tf.reverse(sequence_in_batch == token_id, [-1]), axis=-1)
        last_token_index = length - argmax_in_reverse[:, tf.newaxis] - 1
        return tf.cast(last_token_index, dtype=tf.int32)

    @staticmethod
    def _get_concept_embedding(gpt_model):
        layers = [layer for layer in gpt_model.layers if isinstance(layer, ReusableEmbedding)]
        if len(layers) == 0:
            raise RuntimeError(f'Could not find ReusableEmbedding')
        return layers[0]

    @staticmethod
    def _get_output_layer(gpt_model):
        layers = [layer for layer in gpt_model.layers if isinstance(layer, TiedOutputEmbedding)]
        if len(layers) == 0:
            raise RuntimeError(f'Could not find TiedOutputEmbedding')
        return layers[0]

    @staticmethod
    def _get_positional_encoding_layer(gpt_model):
        gpt_model.get_layer('positional_encoding_layer')
        layers = [layer for layer in gpt_model.layers if
                  isinstance(layer, TrainablePositionEmbedding)]
        if len(layers) == 0:
            raise RuntimeError(f'Could not find GptPositionEmbedding')
        return layers[0]

    @staticmethod
    def _get_gpt_decoder(gpt_model):
        layers = [layer for layer in gpt_model.layers if isinstance(layer, GptDecoder)]
        if len(layers) == 0:
            raise RuntimeError(f'Could not find GPT Decoder')
        return layers[0]


def create_model(
        context_window_size,
        vocab_size,
        embedding_size,
        num_heads,
        depth,
        embedding_dropout: float = 0.6,
        confidence_penalty_weight: float = 0.1
):
    """
    model = create_model(
        max_len=100,
        vocab_size=100,
        embed_dim=128,
        num_heads=16,
        num_of_layers=5
    )
    :param context_window_size:
    :param vocab_size:
    :param embedding_size:
    :param num_heads:
    :param depth:
    :param confidence_penalty_weight:
    :param embedding_dropout:
    :return:
    """
    concept_inputs = tf.keras.layers.Input(
        shape=(context_window_size,),
        dtype=tf.int32,
        name='concept_ids'
    )

    look_ahead_mask_base = tf.cast(
        1 - tf.linalg.band_part(tf.ones((context_window_size, context_window_size)), -1, 0),
        dtype=tf.int32
    )[tf.newaxis, tf.newaxis, :, :]

    concept_embedding_layer = ReusableEmbedding(
        vocab_size, embedding_size,
        input_length=context_window_size,
        name='concept_embeddings',
        # Regularization is based on paper "A Comparative Study on
        # Regularization Strategies for Embedding-based Neural Networks"
        # https://arxiv.org/pdf/1508.03721.pdf
        embeddings_regularizer=tf.keras.regularizers.l2(1e-4)
    )
    positional_encoding_layer = TrainablePositionEmbedding(
        maxlen=context_window_size,
        embed_dim=embedding_size,
        name='positional_encoding_layer'
    )

    output_layer = TiedOutputEmbedding(
        projection_regularizer=tf.keras.regularizers.l2(1e-4),
        projection_dropout=embedding_dropout,
        name='concept_prediction_logits'
    )

    # embeddings for encoder input
    x, concept_embedding_matrix = concept_embedding_layer(concept_inputs)

    x += positional_encoding_layer(
        x
    )

    transformer_block = GptDecoder(
        depth,
        embedding_size,
        num_heads,
        name='decoder'
    )
    contextualized_embeddings, _, _ = transformer_block(
        x,
        look_ahead_mask_base
    )

    concept_prediction_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    outputs = concept_prediction_layer(
        output_layer([contextualized_embeddings, concept_embedding_matrix])
    )

    model = tf.keras.Model(inputs=[concept_inputs], outputs=[outputs])

    # # Penalty for confidence of the output distribution, as described in
    # # "Regularizing Neural Networks by Penalizing Confident
    # # Output Distributions" (https://arxiv.org/abs/1701.06548)
    # confidence_penalty = K.mean(
    #     confidence_penalty_weight * K.sum(outputs * K.log(outputs), axis=-1)
    # )
    #
    # model.add_loss(confidence_penalty)
    #
    # model.add_metric(
    #     confidence_penalty,
    #     name='Output distribution penalty'
    # )

    return model


def sample_predicted_probabibility(
        pred_logits,
        top_k
):
    pred_logits, indices = tf.math.top_k(pred_logits, k=top_k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = tf.keras.activations.softmax(tf.expand_dims(pred_logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)


def generate_artificial_time_tokens():
    """
    Generate all the time tokens used in training
    :return:
    """
    day_tokens = [f'D{i}' for i in range(2000)]
    week_tokens = [f'W{i}' for i in range(4)]
    month_tokens = [f'M{i}' for i in range(12)]
    long_term_tokens = ['LT']
    return day_tokens + week_tokens + month_tokens + long_term_tokens


class PatientHistoryGenerator(tf.keras.callbacks.Callback):
    def __init__(
            self,
            demographic_info,
            max_seq,
            concept_tokenizer: ConceptTokenizer,
            concept_map: dict,
            top_k=10,
            print_every=1
    ):
        self.max_seq = max_seq
        self.concept_tokenizer = concept_tokenizer
        self.concept_map = concept_map
        self.print_every = print_every
        self.k = top_k
        self.genders = np.squeeze(concept_tokenizer.encode([
            '8532',  # FEMALE,
            '8507'  # MALE
        ])).tolist()

        self.races = np.squeeze(concept_tokenizer.encode([
            '0',  # No matching concept
            '8527',  # white
            '8552',  # Unknown
            '8516',  # Black or African American,
            '44814653',  # Unknown
            '8522',  # Other Race
            '8515',  # Asian
        ])).tolist()

        self.starting_ages = np.squeeze(concept_tokenizer.encode(
            list(map(str, range(30, 50))) + list(map(lambda a: f'age:{a}', range(30, 50)))
        )).tolist()

        self.starting_years = np.squeeze(concept_tokenizer.encode(
            list(map(str, range(2000, 2020))) + list(map(lambda a: f'year:{a}', range(2000, 2020)))
        )).tolist()

    def detokenize(self, number):
        concept_id = self.concept_tokenizer.decode([[number]])[0]
        if concept_id in self.concept_map:
            return self.concept_map[concept_id]
        return concept_id

    def on_batch_end(self, batch, logs=None):
        if batch == 0 or batch % self.print_every != 0:
            return
        print(f'\nGenerating text for {batch}\n')
        inference_model = GptInferenceModel(
            self.model,
            tokenizer=self.concept_tokenizer,
            context_window=self.max_seq,
            top_k=self.k
        )
        start_tokens = [
            self.concept_tokenizer.get_start_token_id(),
            random.sample(self.starting_years, 1)[0],
            random.sample(self.starting_ages, 1)[0],
            random.sample(self.genders, 1)[0],
            random.sample(self.races, 1)[0]
        ]
        start_tokens = tf.reshape(
            start_tokens,
            (1, -1)
        )
        prompt_batch = inference_model(
            start_tokens
        )

        txt = '\n'.join(
            [self.detokenize(_) for _ in prompt_batch[0]]
        )

        print(f"generated text:\n{txt}\n")


class ComputeMarginalDistribution(tf.keras.callbacks.Callback):
    def __init__(
            self,
            demographic_info,
            max_seq,
            concept_tokenizer: ConceptTokenizer,
            concept_map: dict,
            batch_size,
            num_of_patients=1024,
            top_k=10,
            print_every=1
    ):
        self.demographic_info = demographic_info
        self.max_seq = max_seq
        self.concept_tokenizer = concept_tokenizer
        self.concept_map = concept_map
        self.print_every = print_every
        self.k = top_k
        self.batch_size = batch_size
        self.num_of_patients = num_of_patients

    def detokenize(self, number):
        concept_id = self.concept_tokenizer.decode([[number]])[0]
        if concept_id in self.concept_map:
            return self.concept_map[concept_id]
        return concept_id

    def on_batch_end(self, batch, logs=None):
        if batch == 0 or batch % self.print_every != 0:
            return
        inference_model = GptInferenceModel(
            self.model,
            tokenizer=self.concept_tokenizer,
            context_window=self.max_seq,
            top_k=self.k
        )

        num_of_batches = self.num_of_patients // self.batch_size + 1
        sequence_to_flush = []
        for i in range(num_of_batches):

            print(f'{datetime.datetime.now()}: Patient generation batch {i} started')

            start_tokens = np.tile(
                np.asarray([[self.concept_tokenizer.get_start_token_id()]]),
                [self.batch_size, 1]
            )
            random_prompts = random.sample(
                self.demographic_info,
                self.batch_size
            )
            prompt_batch = np.hstack([start_tokens, random_prompts])
            _, length = np.shape(
                prompt_batch
            )

            prompt_batch = tf.cast(prompt_batch, dtype=tf.int32)

            prompt_batch = inference_model(
                prompt_batch
            )
            for seq in prompt_batch.tolist():
                seq_copy = []
                for token in seq:
                    if token == self.concept_tokenizer.get_end_token_id():
                        break
                    seq_copy.append(self.detokenize(token))
                sequence_to_flush.append({'token_ids': seq_copy})

        generated_patient_sequences = pd.DataFrame(
            sequence_to_flush,
            columns=['token_ids']
        )
        dist = generated_patient_sequences.token_ids.explode().value_counts() / len(generated_patient_sequences)
        print(f'{datetime.datetime.now()}: The marginal distribution is below:\n {dist.head(60)}\n')
        txt = '\n'.join(sequence_to_flush[0]['token_ids'])
        print(f'{datetime.datetime.now()}: The generated patient sequence:\n{txt}\n')
