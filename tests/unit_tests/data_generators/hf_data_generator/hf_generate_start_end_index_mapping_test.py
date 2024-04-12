import unittest
from data_generators.hf_data_generator.hf_dataset_mapping import GenerateStartEndIndexMapping
import random

# Seed the random number generator for reproducibility in tests
random.seed(42)


class TestGenerateStartEndIndexMapping(unittest.TestCase):
    def setUp(self):
        # Initialize with a fixed sequence length for consistent testing
        self.mapper = GenerateStartEndIndexMapping(max_sequence_length=10)

    def test_long_sequence(self):
        # Test with a sequence longer than max_sequence_length
        record = {
            'concept_ids': list(range(20))  # Longer than max_sequence_length
        }
        result = self.mapper.transform(record)
        self.assertEqual(result['start_index'], 10)
        self.assertEqual(result['end_index'], 19)

    def test_short_sequence(self):
        # Test with a sequence shorter than max_sequence_length
        record = {
            'concept_ids': list(range(5))  # Shorter than max_sequence_length
        }
        expected_start = 0
        expected_end = 5
        result = self.mapper.transform(record)
        self.assertEqual(result['start_index'], expected_start)
        self.assertEqual(result['end_index'], expected_end)

    def test_edge_case_sequence_length_equal_to_max(self):
        # Test with a sequence exactly equal to max_sequence_length
        record = {
            'concept_ids': list(range(9))  # Exactly max_sequence_length - 1
        }
        expected_start = 0
        expected_end = 9
        result = self.mapper.transform(record)
        self.assertEqual(result['start_index'], expected_start)
        self.assertEqual(result['end_index'], expected_end)


if __name__ == '__main__':
    unittest.main()
