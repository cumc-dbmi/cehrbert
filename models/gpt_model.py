import numpy as np
from pandas import DataFrame
import tensorflow as tf

from models.layers.custom_layers import (
    GptDecoder,
    TokenAndPositionEmbedding
)

from data_generators.tokenizer import ConceptTokenizer


def create_model(
        context_window_size,
        vocab_size,
        embedding_size,
        num_heads,
        depth
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

    embedding_layer = TokenAndPositionEmbedding(context_window_size, vocab_size, embedding_size)
    x = embedding_layer(concept_inputs)

    transformer_block = GptDecoder(depth, embedding_size, num_heads)
    x, _ = transformer_block(x, look_ahead_mask_base)

    concept_prediction_layer = tf.keras.layers.Softmax(
        name='concept_predictions'
    )

    outputs = tf.keras.layers.Dense(vocab_size)(x)

    outputs = concept_prediction_layer(outputs)

    return tf.keras.Model(inputs=[concept_inputs], outputs=[outputs])


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
    week_tokens = [f'W{i}' for i in range(4)]
    month_tokens = [f'M{i}' for i in range(12)]
    long_term_tokens = ['LT']
    return week_tokens + month_tokens + long_term_tokens


def generate_visit_boundary_tokens():
    return ['VS', 'VE']


def token_exists(
        sample_token,
        tokens_generated,
        visit_boundary_tokens,
        artificial_time_tokens
):
    visit_start_token, visit_end_token, = visit_boundary_tokens
    cursor = len(tokens_generated) - 1
    while True:
        # If there are no tokens generated, return true
        if len(tokens_generated) == 0:
            return True
        prev_token = tokens_generated[cursor]

        # The pattern restriction is enforced
        # ATT -> VS
        if sample_token == visit_start_token:
            if prev_token in artificial_time_tokens:
                return True
            return False

        # The pattern restriction is enforced
        # concepts -> VE
        if sample_token == visit_end_token:
            if prev_token not in artificial_time_tokens and prev_token not in visit_boundary_tokens:
                return True
            return False

        # This indicates this is a time artificial token
        if sample_token in artificial_time_tokens:
            # This pattern VE -> ATT is enforced
            if prev_token == visit_end_token:
                return True
            return False

        # sample_token has to be a valid concept at this point
        # This indicates backtracking has reached the previous visit boundary
        if prev_token == visit_end_token:
            return True

        # The sample_token can't be the same as the prev_token within the same visit
        if prev_token == sample_token:
            return False

        cursor = cursor - 1

    return False


def generate_patient_history(
        model,
        start_tokens,
        concept_tokenizer,
        max_seq,
        top_k
):
    visit_boundary_tokens = np.squeeze(
        concept_tokenizer.encode(generate_visit_boundary_tokens())
    ).tolist()

    artificial_time_tokens = np.squeeze(
        concept_tokenizer.encode(generate_artificial_time_tokens())
    ).tolist()

    tokens_generated = []
    while len(tokens_generated) <= max_seq:
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

        # If the generated token is the same as the previous one, skip it
        while True:
            sample_token = sample_predicted_probabibility(
                y[0][sample_index],
                top_k
            )
            if token_exists(
                    sample_token,
                    tokens_generated,
                    visit_boundary_tokens,
                    artificial_time_tokens
            ):
                break

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

    def detokenize(self, number):
        concept_id = self.concept_tokenizer.decode([[number]])[0]
        if concept_id in self.concept_map:
            return self.concept_map[concept_id]
        return concept_id

    def on_batch_end(self, batch, logs=None):
        if batch != 0 and batch % self.print_every != 0:
            return
        print(f'Generating text for {batch}\n')
        tokens_generated = generate_patient_history(
            model=self.model,
            start_tokens=[self.concept_tokenizer.get_start_token_id()],
            concept_tokenizer=self.concept_tokenizer,
            max_seq=self.max_seq,
            top_k=self.k
        )

        txt = '\n'.join(
            [self.detokenize(_) for _ in tokens_generated]
        )

        print(f"generated text:\n{txt}\n")
