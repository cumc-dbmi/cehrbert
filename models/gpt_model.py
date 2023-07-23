import copy
import random

import numpy as np
import tensorflow as tf

from data_generators.tokenizer import ConceptTokenizer
from keras_transformer.extras import ReusableEmbedding, TiedOutputEmbedding
from models.layers.custom_layers import GptDecoder, PositionalEncodingLayer


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
            visit_concept_orders,
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
            visit_concept_orders
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

        logtis = self.output_layer([x, concept_embedding_matrix])

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
        visit_concept_orders = tf.zeros_like(
            inputs
        )

        while length < self.context_window:
            # Generate the next batch of tokens and update the contexts
            outputs, cached_contexts = self._generate_next_token(
                inputs,
                visit_concept_orders,
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

            new_visit_indicator = tf.cast(
                next_tokens == self.tokenizer.get_visit_start_token_id(),
                dtype=tf.int32
            )

            visit_concept_orders = np.hstack([
                visit_concept_orders,
                visit_concept_orders[:, -1:] + new_visit_indicator]
            )

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
                  isinstance(layer, PositionalEncodingLayer)]
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

    visit_concept_orders = tf.keras.layers.Input(
        shape=(context_window_size,),
        dtype='int32',
        name='visit_concept_orders'
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
    positional_encoding_layer = PositionalEncodingLayer(
        embedding_size=embedding_size,
        max_sequence_length=context_window_size,
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
        visit_concept_orders
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

    model = tf.keras.Model(inputs=[concept_inputs, visit_concept_orders], outputs=[outputs])

    # Penalty for confidence of the output distribution, as described in
    # "Regularizing Neural Networks by Penalizing Confident
    # Output Distributions" (https://arxiv.org/abs/1701.06548)
    # confidence_penalty = K.mean(
    #     confidence_penalty_weight * K.sum(outputs * K.log(outputs), axis=-1)
    # )
    #
    # model.add_loss(confidence_penalty)

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


def generate_visit_boundary_tokens():
    return ['VS', 'VE']


def finding_patterns(
        sample_token,
        tokens_generated,
        visit_boundary_tokens,
        artificial_time_tokens
):
    visit_start_token, visit_end_token, = visit_boundary_tokens
    cursor = len(tokens_generated) - 1
    while cursor >= 0:
        prev_token = tokens_generated[cursor]

        # The pattern restriction is enforced
        # VE -> ATT
        if sample_token in artificial_time_tokens:
            if prev_token == visit_end_token:
                return True
            return False

        # Multiple occurrences of the same concept within the same visit restriction is prohibited
        # Not allowed C1->C1
        if sample_token not in visit_boundary_tokens and sample_token not in artificial_time_tokens:
            if sample_token == prev_token:
                return False

        # The pattern restriction is enforced
        # ATT -> VS
        if prev_token in artificial_time_tokens:
            if sample_token == visit_end_token:
                return True
            return False

        # The pattern restriction is allowed
        # Concept -> VS
        if visit_start_token == prev_token and visit_start_token != sample_token:
            return True

        # The pattern restriction is prohibited
        # VS -> VS
        if visit_start_token == prev_token and visit_start_token == sample_token:
            return False

        # The pattern restriction is prohibited
        # VE -> VE
        if visit_end_token == prev_token and visit_end_token == sample_token:
            return False
        cursor -= 1

    return True


def generate_patient_history(
        model,
        start_tokens,
        concept_tokenizer,
        max_seq,
        top_k,
        prohibited_tokens=None
):
    visit_boundary_tokens = np.squeeze(
        concept_tokenizer.encode(generate_visit_boundary_tokens())
    ).tolist()

    artificial_time_tokens = np.squeeze(
        concept_tokenizer.encode(generate_artificial_time_tokens())
    ).tolist()

    tokens_generated = []
    while len(tokens_generated) <= max_seq:
        sample_token = None
        # The pattern restriction is enforced ATT -> VS.
        # We manually set the next token to be VS if the previous one is an ATT
        if len(tokens_generated) > 0:
            prev_token = tokens_generated[-1]
            if prev_token in artificial_time_tokens:
                sample_token = visit_boundary_tokens[0]
        else:
            sample_token = visit_boundary_tokens[0]

        # We randomly sample a token from the predicted distribution
        if not sample_token:
            pad_len = max_seq - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:max_seq]
                sample_index = max_seq - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens

            x = np.array([x])
            y = model.predict(x)

            max_num_iter = 10
            # If the generated token is the same as the previous one, skip it
            while max_num_iter > 0:
                sample_token = sample_predicted_probabibility(
                    y[0][sample_index],
                    top_k
                )
                if len(tokens_generated) == 0 or tokens_generated[-1] != sample_token:
                    break
                # if finding_patterns(
                #         sample_token,
                #         tokens_generated,
                #         visit_boundary_tokens,
                #         artificial_time_tokens
                # ):
                max_num_iter = max_num_iter - 1

        # # Prohibit the tokens from being generated
        if prohibited_tokens and sample_token in prohibited_tokens:
            continue

        if sample_token == concept_tokenizer.get_end_token_id():
            break

        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)

    return tokens_generated


class PatientHistoryGenerator(tf.keras.callbacks.Callback):
    def __init__(
            self,
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
        print(f'Generating text for {batch}\n')
        start_tokens = [
            self.concept_tokenizer.get_start_token_id(),
            random.sample(self.starting_years, 1)[0],
            random.sample(self.starting_ages, 1)[0],
            random.sample(self.genders, 1)[0],
            random.sample(self.races, 1)[0]
        ]

        prohibited_tokens = [
            t for t in self.genders + self.races + self.starting_ages + self.starting_years
            if t != self.concept_tokenizer.tokenizer.word_index['0']
        ]

        tokens_generated = generate_patient_history(
            model=self.model,
            start_tokens=copy.deepcopy(start_tokens),
            concept_tokenizer=self.concept_tokenizer,
            max_seq=self.max_seq,
            top_k=self.k,
            prohibited_tokens=prohibited_tokens
        )

        txt = '\n'.join(
            [self.detokenize(_) for _ in start_tokens + tokens_generated]
        )

        print(f"generated text:\n{txt}\n")
