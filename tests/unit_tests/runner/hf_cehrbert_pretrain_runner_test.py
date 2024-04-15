import unittest
import numpy as np
from runner.hf_cehrbert_pretrain_runner import compute_metrics

import numpy as np


# Manual calculation for softmax and cross-entropy
def manual_perplexity_calculation(logits, labels):
    # Compute softmax
    max_logits = np.max(logits, axis=1, keepdims=True)
    exps = np.exp(logits - max_logits)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # Compute cross-entropy
    cross_entropy = -np.log(probs[np.arange(len(labels)), labels])

    # Calculate perplexity
    perplexity = np.exp(np.mean(cross_entropy))
    return perplexity


class TestComputeMetrics(unittest.TestCase):
    def test_perplexity_calculation(self):
        logits = np.array([[10, 1], [1, 1], [1, 10], [10, 1]])
        labels = np.array([0, -100, 1, 0])
        # Transformers Trainer will remove the loss from the model output
        # We need to take the first entry of the model output, which is logits
        eval_pred = ([logits], labels)

        # Call the compute_metrics function
        results = compute_metrics(eval_pred)

        # Manually verified expected perplexity
        expected_perplexity = manual_perplexity_calculation(
            np.array([[10, 1], [1, 10], [10, 1]]),  # Actual processed logits
            np.array([0, 1, 0])  # Corresponding labels
        )

        # Check if the calculated perplexity is close to the expected value
        self.assertAlmostEqual(results['perplexity'], expected_perplexity, places=5,
                               msg="The perplexity was not calculated correctly.")

    def test_ignoring_masked_entries(self):
        from scipy.special import softmax
        logits = np.array([[10, 1], [10, 1], [1, 10], [10, 1]])
        labels = np.array([-100, -100, 1, -100])
        # Transformers Trainer will remove the loss from the model output
        # We need to take the first entry of the model output, which is logits
        eval_pred = ([logits], labels)

        # Only one entry should be used (the third one)
        results = compute_metrics(eval_pred)

        expected_probs = softmax([1, 10])
        expected_perplexity = np.exp(-np.log(expected_probs[1]))

        # Check if the perplexity calculation ignores masked entries correctly
        self.assertAlmostEqual(results['perplexity'], expected_perplexity, places=5,
                               msg="Masked entries were not ignored correctly.")


if __name__ == '__main__':
    unittest.main()
