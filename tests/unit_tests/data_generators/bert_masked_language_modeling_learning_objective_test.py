import unittest

import numpy as np
import pandas as pd

from cehrbert.data_generators.data_classes import RowSlicer
from cehrbert.data_generators.learning_objective import MaskedLanguageModelLearningObjective
from cehrbert.data_generators.tokenizer import ConceptTokenizer


class TestMaskedLanguageModelLearningObjective(unittest.TestCase):

    def setUp(self):
        # Setup code to run before each test, e.g., create a ConceptTokenizer instance
        self.concept_tokenizer = (
            ConceptTokenizer()
        )  # Initialize this with whatever parameters are appropriate for your implementation
        self.max_seq_len = 6
        self.is_pretraining = True
        self.learning_obj = MaskedLanguageModelLearningObjective(
            self.concept_tokenizer, self.max_seq_len, self.is_pretraining
        )

    @staticmethod
    def create_mock_row():
        # Create a mock row with 5 elements in each list
        return RowSlicer(
            row=pd.Series(
                {
                    "dates": [1, 2, 3, 4, 5],
                    "token_ids": [101, 102, 103, 104, 105],  # Example token IDs
                    "visit_segments": [1, 1, 2, 2, 1],  # Example visit segments
                    "ages": [25, 26, 27, 28, 29],  # Example ages
                    "visit_concept_orders": [
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],  # Example visit concept orders
                    "concept_values": [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.9,
                    ],  # Example concept values
                    "concept_value_masks": [
                        0,
                        0,
                        0,
                        0,
                        1,
                    ],  # Example concept value masks
                    "mlm_skip_values": [0, 0, 0, 0, 1],  # Example MLM skip values
                    "orders": [1, 2, 3, 4, 5],  # Example orders for sorting
                }
            ),
            start_index=0,
            end_index=5,  # Updated to include all 5 elements
            target_index=2,  # Adjusted target index for demonstration
        )

    def test_initialization(self):
        # Test that the object is initialized correctly
        self.assertEqual(self.learning_obj._max_seq_len, self.max_seq_len)
        self.assertEqual(self.learning_obj._concept_tokenizer, self.concept_tokenizer)
        self.assertEqual(self.learning_obj._is_pretraining, self.is_pretraining)

    def test_get_tf_dataset_schema(self):
        # Test the get_tf_dataset_schema method
        input_schema, output_schema = self.learning_obj.get_tf_dataset_schema()
        self.assertIn("masked_concept_ids", input_schema)
        self.assertIn("concept_ids", input_schema)
        self.assertIn("mask", input_schema)
        self.assertIn("concept_predictions", output_schema)

    def test_process_batch(self):
        # Test the process_batch method with a mock input
        mock_rows = [self.create_mock_row() for _ in range(5)]  # Create a list of mock rows

        input_dict, output_dict = self.learning_obj.process_batch(mock_rows)

        # Assert that the input and output dictionaries have the correct structure and values
        self.assertIn("masked_concept_ids", input_dict)
        self.assertIn("concept_ids", input_dict)
        self.assertIn("mask", input_dict)
        # Continue for all expected keys in the input and output dictionaries...
        self.assertIn("concept_predictions", output_dict)

        self.assertTrue(
            (
                input_dict["concept_ids"][0]
                == np.asarray(
                    [
                        101,
                        102,
                        103,
                        104,
                        105,
                        self.concept_tokenizer.get_unused_token_id(),
                    ]
                )
            ).all()
        )

        # Test the concept mask, where 1 indicates attention and 0 indicates mask
        self.assertTrue((input_dict["mask"][0] == np.asarray([1, 1, 1, 1, 1, 0])).all())


if __name__ == "__main__":
    unittest.main()
