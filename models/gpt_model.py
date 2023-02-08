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


def generate_patient_history(
        model,
        start_tokens,
        concept_tokenizer,
        max_seq,
        top_k
):
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
            if len(tokens_generated) == 0 or tokens_generated[-1] != sample_token:
                break

        if sample_token == concept_tokenizer.end_token:
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
        if batch % self.print_every != 0:
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
