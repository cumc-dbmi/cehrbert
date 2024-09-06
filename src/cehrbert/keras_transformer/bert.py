"""
BERT stands for Bidirectional Encoder Representations from Transformers.

It's a way of pre-training Transformer to model a language, described in
paper [BERT: Pre-training of Deep Bidirectional Transformers for
Language Understanding](https://arxiv.org/abs/1810.04805). A quote from it:

> BERT is designed to pre-train deep bidirectional representations
> by jointly conditioning on both left and right context in all layers.
> As a result, the pre-trained BERT representations can be fine-tuned
> with just one additional output layer to create state-of-the art
> models for a wide range of tasks, such as question answering
> and language inference, without substantial task-specific architecture
> modifications.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import get_custom_objects


def masked_perplexity(y_true, y_pred):
    """
    Masked version of popular metric for evaluating performance of.

    language modelling architectures. It assumes that y_pred has shape
    (batch_size, sequence_length, 2), containing both
      - the original token ids
      - and the mask (0s and 1s, indicating places where
        a word has been replaced).
    both stacked along the last dimension.
    Masked perplexity ignores all but masked words.

    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
    """
    y_true_value = y_true[:, :, 0]
    mask = K.cast(y_true[:, :, 1], dtype="float32")
    cross_entropy = K.sparse_categorical_crossentropy(y_true_value, y_pred)
    batch_perplexities = K.exp(K.sum(mask * cross_entropy, axis=-1) / (K.sum(mask, axis=-1) + 1e-6))
    return K.mean(batch_perplexities)


class MaskedMeanSquaredError(object):
    def __call__(self, y_true, y_pred):
        y_true_val = K.cast(y_true[:, :, 0], dtype="float32")
        mask = K.cast(y_true[:, :, 1], dtype="float32")

        num_items_masked = tf.reduce_sum(mask, axis=-1) + 1e-6
        masked_mse = tf.reduce_sum(tf.square(y_true_val - y_pred) * mask, axis=-1) / num_items_masked

        return masked_mse


class MaskedPenalizedSparseCategoricalCrossentropy(object):
    """
    Masked cross-entropy (see `masked_perplexity` for more details).

    loss function with penalized confidence.
    Combines two loss functions: cross-entropy and negative entropy
    (weighted by `penalty_weight` parameter), following paper
    "Regularizing Neural Networks by Penalizing Confident Output Distributions"
    (https://arxiv.org/abs/1701.06548)

    how to use:
    >>> model.compile(
    >>>     optimizer,
    >>>     loss=MaskedPenalizedSparseCategoricalCrossentropy(0.1))
    """

    def __init__(self, penalty_weight: float):
        self.__name__ = "MaskedPenalizedSparseCategoricalCrossentropy"
        self.penalty_weight = penalty_weight

    def __call__(self, y_true, y_pred):
        y_true_val = K.cast(y_true[:, :, 0], dtype="float32")
        mask = K.cast(y_true[:, :, 1], dtype="float32")

        # masked per-sample means of each loss
        num_items_masked = K.sum(mask, axis=-1) + 1e-6
        masked_cross_entropy = (
            K.sum(mask * K.sparse_categorical_crossentropy(y_true_val, y_pred), axis=-1) / num_items_masked
        )
        masked_entropy = K.sum(mask * -K.sum(y_pred * K.log(y_pred), axis=-1), axis=-1) / num_items_masked
        return masked_cross_entropy - self.penalty_weight * masked_entropy

    def get_config(self):
        return {"penalty_weight": self.penalty_weight}


class SequenceCrossentropy(object):
    def __init__(self):
        self.__name__ = "SequenceCrossentropy"

    def __call__(self, y_true, y_pred):
        y_true_val = K.cast(y_true[:, :, 0], dtype="float32")
        mask = K.cast(y_true[:, :, 1], dtype="float32")
        num_items_masked = K.sum(mask, axis=-1) + 1e-6
        loss = K.sum(binary_crossentropy(y_true_val[:, :, tf.newaxis], y_pred) * mask, axis=-1)
        return loss / num_items_masked


get_custom_objects().update(
    {
        "MaskedPenalizedSparseCategoricalCrossentropy": MaskedPenalizedSparseCategoricalCrossentropy,
        "masked_perplexity": masked_perplexity,
        "SequenceCrossentropy": SequenceCrossentropy,
    }
)
