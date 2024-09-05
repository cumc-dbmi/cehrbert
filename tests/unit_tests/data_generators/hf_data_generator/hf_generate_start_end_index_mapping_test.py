import unittest
from cehrbert.data_generators.hf_data_generator.hf_dataset_collator import CehrBertDataCollator
import random
from unittest.mock import MagicMock

# Seed the random number generator for reproducibility in tests
random.seed(42)


class TestGenerateStartEndIndexMapping(unittest.TestCase):
    def setUp(self):
        # Initialize with a fixed sequence length for consistent testing
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.vocab_size = 100
        self.mock_tokenizer.mask_token_index = 1
        self.mock_tokenizer.unused_token_index = 99
        self.mock_tokenizer.encode.return_value = [10, 20, 30]  # Example token IDs
        self.mock_tokenizer._convert_token_to_id.side_effect = [2, 3, 17, 18]
        self.mock_tokenizer._convert_id_to_token.side_effect = ['year:2000', 'age:20-30']
        self.data_collator = CehrBertDataCollator(
            tokenizer=self.mock_tokenizer,
            max_length=10
        )

    def test_long_sequence(self):
        # Test with a sequence longer than max_sequence_length
        record = {
            'input_ids': [2, 4, 3, 5, 2, 6, 7, 8, 9, 10, 3, 11, 2, 12, 3, 13, 2, 14, 3]
        }
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result['input_ids'], [2, 12, 3, 13, 2, 14, 3])

    def test_short_sequence(self):
        # Test with a sequence shorter than max_sequence_length
        record = {
            'input_ids': list(range(5))  # Shorter than max_sequence_length
        }
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result['input_ids'], list(range(5)))

    def test_edge_case_sequence_length_equal_to_max(self):
        # Test with a sequence exactly equal to max_sequence_length
        record = {
            'input_ids': list(range(9))  # Exactly max_sequence_length - 1
        }
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result['input_ids'], list(range(9)))

    def test_tail_case_sequence_length_equal_to_max(self):
        from data_generators.hf_data_generator.hf_dataset_collator import TruncationType
        # Test with a sequence exactly equal to max_sequence_length
        default_val = self.data_collator.truncate_type
        self.data_collator.truncate_type = TruncationType.TAIL
        record = {
            'input_ids': [13, 14, 15, 16] + list(range(2, 8)) + list(range(2, 6)),  # Exactly max_sequence_length - 1,
            'dates': [0, 0, 0, 0] + [2000] + list(range(2052, 2061))
        }
        result = self.data_collator.generate_start_end_index(record)
        self.assertListEqual(result['input_ids'], [17, 18, 15, 16, 2, 3, 4, 5])
        self.assertListEqual(result['dates'], [0, 0, 0, 0, 2057, 2058, 2059, 2060])
        self.data_collator.truncate_type = default_val


if __name__ == '__main__':
    unittest.main()
