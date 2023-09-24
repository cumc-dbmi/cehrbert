import unittest

# Import the function you want to test
from data_generators.gpt_utils import random_slice_gpt_sequence


class TestRandomSliceGPTSequence(unittest.TestCase):

    def test_ve_token_not_found(self):
        # Test case inputs
        concept_ids = [
            'year:1990', 'age:30', 'Male', 'White',
            'VS', 'concept1', 'concept3', 'VS-D370-VE', 'concept4',
            'concept1', 'concept1', 'concept1', 'D10', 'concept1', 'concept1',
            'concept1', 'concept1', 'concept1', 'concept1', 'concept1', 'concept1', 'VE'
        ]
        max_seq_len = 10
        # Call the function to get the result
        starting_index, end_index, demographic_tokens = random_slice_gpt_sequence(
            concept_ids,
            max_seq_len
        )
        self.assertEqual(starting_index, end_index)

    def test_random_slice_gpt_sequence(self):
        # Test case inputs
        concept_ids = [
            'year:1990', 'age:30', 'Male', 'White',
            'VS', 'concept1', 'VE', 'D370',
            'VS', 'concept3', 'VS-D370-VE', 'concept4', 'VE',
            'VS', 'concept1', 'concept1', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10',
            'VS', 'concept1', 'VE', 'D10'
        ]
        max_seq_len = 30

        # Call the function to get the result
        starting_index, end_index, demographic_tokens = random_slice_gpt_sequence(
            concept_ids,
            max_seq_len
        )

        # [(4, 1990, 30), (8, 1991, 31), (13, 1992, 32), (19, 1992, 32)]
        starting_year = int(demographic_tokens[0].split(':')[1])
        starting_age = int(demographic_tokens[1].split(':')[1])
        self.assertTrue(starting_index in [4, 8, 13, 19])
        self.assertTrue(starting_year in [1990, 1991, 1992])
        self.assertTrue(starting_age in [30, 31, 32])

        # You can add more specific assertions based on your function's behavior
        self.assertEqual(concept_ids[starting_index], 'VS')
        self.assertEqual(concept_ids[end_index], 'VE')


if __name__ == '__main__':
    unittest.main()
