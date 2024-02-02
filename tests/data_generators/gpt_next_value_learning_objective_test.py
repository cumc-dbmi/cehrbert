import unittest

import numpy as np
import pandas as pd
from data_generators.data_classes import RowSlicer
from data_generators.gpt_learning_objectives import PredictNextValueLearningObjective


class TestPredictNextValueLearningObjective(unittest.TestCase):

    def setUp(self):
        self.max_seq_len = 10  # Example max sequence length
        self.learning_obj = PredictNextValueLearningObjective(self.max_seq_len)

    @staticmethod
    def create_mock_row():
        # Create a mock row that resembles the expected input
        return RowSlicer(
            row=pd.Series(
                {
                    "dates": [1, 2, 3],
                    "concept_values": [0.5, 0.0, 0.7],
                    "concept_value_masks": [1, 0, 1],
                    "orders": [1, 2, 3],  # or any other required fields
                }
            ),
            start_index=0,
            end_index=3,
            target_index=1,
        )

    def test_process_batch(self):
        mock_rows = [
            self.create_mock_row() for _ in range(5)
        ]  # Create a list of mock rows
        input_dict, output_dict = self.learning_obj.process_batch(mock_rows)

        max_length = max(list(map(lambda r: len(r.row.concept_values), mock_rows))) + 1

        # Test the shapes of the output
        self.assertEqual(input_dict["concept_values"].shape, (5, max_length))
        self.assertEqual(input_dict["concept_value_masks"].shape, (5, max_length))
        self.assertEqual(
            output_dict["next_value_predictions"].shape, (5, max_length, 2)
        )

        self.assertTrue(
            (
                input_dict["concept_values"][:, 1:]
                == output_dict["next_value_predictions"][:, 0:-1, 0]
            ).all()
        )
        self.assertTrue(
            (
                input_dict["concept_value_masks"][:, 1:]
                == output_dict["next_value_predictions"][:, 0:-1, 1]
            ).all()
        )

        self.assertTrue(
            (
                input_dict["concept_values"][0, 1:4]
                == np.asarray([0.5, 0.0, 0.7], dtype="float32")
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
