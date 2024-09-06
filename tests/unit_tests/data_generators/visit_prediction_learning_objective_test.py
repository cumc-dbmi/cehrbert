import unittest

import numpy as np
import pandas as pd

from cehrbert.data_generators.data_classes import RowSlicer
from cehrbert.data_generators.learning_objective import VisitPredictionLearningObjective
from cehrbert.data_generators.tokenizer import ConceptTokenizer


class TestVisitPredictionLearningObjective(unittest.TestCase):

    def setUp(self):
        self.visit_tokenizer = ConceptTokenizer()  # Use a real or mock ConceptTokenizer as needed
        self.max_seq_len = 5
        self.learning_obj = VisitPredictionLearningObjective(self.visit_tokenizer, self.max_seq_len)

    @staticmethod
    def create_mock_row():
        # Create a mock row with 5 elements in each list
        return RowSlicer(
            row=pd.Series(
                {
                    "visit_token_ids": [101, 102, 103],  # Example token IDs
                    "visit_concept_orders": [1, 2, 3],  # Example orders for sorting
                }
            ),
            start_index=0,
            end_index=3,  # Updated to include all 5 elements
            target_index=2,  # Adjusted target index for demonstration
        )

    def test_initialization(self):
        self.assertEqual(self.learning_obj._max_seq_len, self.max_seq_len)
        self.assertEqual(self.learning_obj._visit_tokenizer, self.visit_tokenizer)

    def test_get_tf_dataset_schema(self):
        input_schema, output_schema = self.learning_obj.get_tf_dataset_schema()
        self.assertIn("masked_visit_concepts", input_schema)
        self.assertIn("mask_visit", input_schema)
        self.assertIn("visit_predictions", output_schema)

    def test_process_batch(self):
        # Test the process_batch method with a mock input
        mock_rows = [self.create_mock_row() for _ in range(5)]  # Create a list of mock rows
        input_dict, output_dict = self.learning_obj.process_batch(mock_rows)

        self.assertIn("masked_visit_concepts", input_dict)
        self.assertIn("mask_visit", input_dict)
        self.assertIn("visit_predictions", output_dict)

        # Test the concept mask, where 1 indicates attention and 0 indicates mask
        self.assertTrue((input_dict["mask_visit"][0] == np.asarray([1, 1, 1, 0, 0])).all())


if __name__ == "__main__":
    unittest.main()
