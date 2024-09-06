import random
import unittest
from unittest.mock import MagicMock

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import HFTokenizationMapping


class TestHFMaskedLanguageModellingMapping(unittest.TestCase):
    def setUp(self):
        # Mock the tokenizer with necessary properties and methods
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = 100
        self.mock_tokenizer.mask_token_index = 1
        self.mock_tokenizer.unused_token_index = 99
        self.mock_tokenizer.encode.return_value = [10, 20, 30]  # Example token IDs

        # Create an instance of the mapping class with pretraining enabled
        self.mapping = HFTokenizationMapping(self.mock_tokenizer, is_pretraining=True)

    def test_transform_with_valid_indices(self):
        # Given a valid record with start and end indices
        record = {
            "concept_ids": ["c1", "c2", "c3"],
            "mlm_skip_values": [
                0,
                0,
                0,
            ],
            "concept_value_masks": [0, 0, 0],
            "concept_values": [0.0, 0.0, 0.0],
        }

        # Random seed for predictability in tests
        random.seed(42)

        # Expected masked input ids might depend on random masking logic
        # Here we assume the second token gets masked with the mask token index (1)
        expected_masked_input_ids = [10, 20, 30]
        expected_labels = [
            10,
            20,
            30,
        ]  # Only non-masked tokens are labeled with original ids

        result = self.mapping.transform(record)

        # Check if the tokenizer's encode method was called correctly
        self.mock_tokenizer.encode.assert_called_once_with(["c1", "c2", "c3"])

        # Validate the output
        self.assertEqual(result["input_ids"], expected_masked_input_ids)
        self.assertEqual(result["labels"], expected_labels)

    def test_transform_assertion(self):
        # Given a valid record with start and end indices
        record = {
            "concept_ids": ["c1", "c2", "c3", "c4"],
            "mlm_skip_values": [0, 0, 0, 1],
            "concept_value_masks": [0, 0, 0, 0],
            "concept_values": [0.0, 0.0, 0.0, 0.0],
        }
        with self.assertRaises(AssertionError):
            self.mapping.transform(record)


if __name__ == "__main__":
    unittest.main()
