import unittest

import numpy as np
import pandas as pd

from data_generators.data_classes import RowSlicer
from data_generators.gpt_learning_objectives import SequenceGenerationLearningObjective, CosineMaskRateScheduler
from data_generators.tokenizer import ConceptTokenizer


class TestSequenceGenerationLearningObjective(unittest.TestCase):

    def setUp(self):
        self.max_seq_len = 10  # Example maximum sequence length
        self.concept_tokenizer = ConceptTokenizer()  # Mock or actual tokenizer instance
        self.concept_tokenizer.fit_on_concept_sequences(pd.Series({'concept_ids': ['101', '102', '103']}))
        self.learning_obj = SequenceGenerationLearningObjective(
            self.concept_tokenizer,
            self.max_seq_len,
            CosineMaskRateScheduler()
        )

    @staticmethod
    def create_mock_row():
        # Create a mock row that resembles the expected input
        return RowSlicer(
            row=pd.Series({
                'dates': [1, 2, 3],
                'token_ids': [101, 102, 103],  # Example token IDs
                'visit_concept_orders': [1, 2, 3],
                'orders': [1, 2, 3]  # or any other required fields
            }),
            start_index=0,
            end_index=3,
            target_index=1
        )

    def test_process_batch(self):
        mock_rows = [self.create_mock_row() for _ in range(5)]  # Create a list of mock rows

        max_length = max(list(map(lambda r: len(r.row.token_ids), mock_rows))) + 1

        np.random.seed(42)
        cosine_mask_rate_scheduler = CosineMaskRateScheduler()
        random_mask = np.random.rand(len(mock_rows), max_length) < cosine_mask_rate_scheduler.get_rate()
        mask = np.tile([[1] * max_length], [5, 1])
        expected_mask = mask & random_mask

        np.random.seed(42)
        input_dict, output_dict = self.learning_obj.process_batch(mock_rows)

        # Test the shapes of the output
        self.assertEqual(input_dict['concept_ids'].shape, (5, max_length))
        self.assertEqual(input_dict['visit_concept_orders'].shape, (5, max_length))
        self.assertEqual(output_dict['concept_predictions'].shape, (5, max_length, 2))

        # Add more tests here to validate the contents of input_dict and output_dict
        self.assertTrue(
            (input_dict['concept_ids'][0, 1:4] == np.asarray([101, 102, 103])).all()
        )
        self.assertTrue(
            (input_dict['visit_concept_orders'][0, 1:4] == np.asarray([1, 2, 3])).all()
        )
        # Test the shifted concept ids
        self.assertTrue(
            (input_dict['concept_ids'][:, 1:4] == output_dict['concept_predictions'][:, :3, 0]).all()
        )
        # Test the mask
        self.assertTrue(
            (expected_mask == output_dict['concept_predictions'][:, :, 1]).all()
        )


if __name__ == '__main__':
    unittest.main()
