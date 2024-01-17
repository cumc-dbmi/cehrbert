import unittest

import numpy as np
import pandas as pd
from data_generators.data_classes import RowSlicer
from data_generators.tokenizer import ConceptTokenizer
from data_generators.gpt_learning_objectives import SequenceGenerationLearningObjective


class TestSequenceGenerationLearningObjective(unittest.TestCase):

    def setUp(self):
        self.max_seq_len = 10  # Example maximum sequence length
        self.concept_tokenizer = ConceptTokenizer()  # Mock or actual tokenizer instance
        self.concept_tokenizer.fit_on_concept_sequences(pd.Series({'concept_ids': ['101', '102', '103']}))
        self.learning_obj = SequenceGenerationLearningObjective(self.concept_tokenizer, self.max_seq_len)

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
        input_dict, output_dict = self.learning_obj.process_batch(mock_rows)

        # Test the shapes of the output
        self.assertEqual(input_dict['concept_ids'].shape, (5, self.max_seq_len))
        self.assertEqual(input_dict['visit_concept_orders'].shape, (5, self.max_seq_len))
        self.assertEqual(output_dict['concept_predictions'].shape, (5, self.max_seq_len, 2))

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


if __name__ == '__main__':
    unittest.main()
