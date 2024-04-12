import unittest
import random
from unittest.mock import MagicMock
from data_generators.hf_data_generator.hf_dataset_mapping import HFMaskedLanguageModellingMapping


# Assuming the module containing your classes is named 'your_module'
# from your_module import HFMaskedLanguageModellingMapping, CehrBertTokenizer

class TestHFMaskedLanguageModellingMapping(unittest.TestCase):
    def setUp(self):
        # Mock the tokenizer with necessary properties and methods
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = 100
        self.mock_tokenizer.mask_token_index = 1
        self.mock_tokenizer.unused_token_index = 99
        self.mock_tokenizer.encode.return_value = [10, 20, 30]  # Example token IDs

        # Create an instance of the mapping class with pretraining enabled
        self.mapping = HFMaskedLanguageModellingMapping(self.mock_tokenizer, is_pretraining=True)

    def test_transform_with_valid_indices(self):
        # Given a valid record with start and end indices
        record = {
            'start_index': 0,
            'end_index': 3,
            'concept_ids': ['c1', 'c2', 'c3', 'c4'],
            'mlm_skip_values': [0, 0, 0, 1]  # Last position should be skipped
        }

        # Random seed for predictability in tests
        random.seed(42)

        # Expected masked input ids might depend on random masking logic
        # Here we assume the second token gets masked with the mask token index (1)
        expected_masked_input_ids = [10, 1, 30]
        expected_labels = [-100, 20, -100]  # Only non-masked tokens are labeled with original ids

        result = self.mapping.transform(record)

        # Check if the tokenizer's encode method was called correctly
        self.mock_tokenizer.encode.assert_called_once_with(['c1', 'c2', 'c3'])

        # Validate the output
        self.assertEqual(result['input_ids'], expected_masked_input_ids)
        self.assertEqual(result['labels'], expected_labels)

    def test_missing_start_index(self):
        # Record without 'start_index'
        record = {
            'end_index': 3,
            'concept_ids': ['c1', 'c2', 'c3', 'c4']
        }
        with self.assertRaises(ValueError):
            self.mapping.transform(record)

    def test_missing_end_index(self):
        # Record without 'end_index'
        record = {
            'start_index': 1,
            'concept_ids': ['c1', 'c2', 'c3', 'c4']
        }
        with self.assertRaises(ValueError):
            self.mapping.transform(record)


if __name__ == '__main__':
    unittest.main()
