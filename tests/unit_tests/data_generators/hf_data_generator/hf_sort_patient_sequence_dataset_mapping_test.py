import unittest

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import SortPatientSequenceMapping


class TestSortPatientSequenceMapping(unittest.TestCase):
    def test_transform_with_orders(self):
        # Create an instance of the mapping class
        mapper = SortPatientSequenceMapping()

        # Mock data with 'orders' column as integers
        record = {
            "orders": [2, 1, 3],
            "concept_ids": ["c", "b", "a"],
            "values": [30, 20, 10],
            "ages": [30, 25, 40],
            "visit_concept_orders": [5, 3, 9],
        }

        # Expected output after sorting
        expected = {
            "concept_ids": ["b", "c", "a"],
            "values": [20, 30, 10],
            "ages": [25, 30, 40],
            "visit_concept_orders": [3, 5, 9],
            "orders": [1, 2, 3],
        }

        # Perform transformation
        result = mapper.transform(record)

        # Verify the output
        self.assertDictEqual(result, expected)

    def test_transform_with_dates(self):
        # Create an instance of the mapping class
        mapper = SortPatientSequenceMapping()

        # Mock data with 'dates' column as integers
        record = {
            "dates": [20210301, 20210101, 20210201],
            "concept_ids": ["c", "b", "a"],
            "values": [30, 20, 10],
            "ages": [40, 25, 30],
            "visit_concept_orders": [5, 3, 9],
        }

        # Expected output after sorting based on dates
        expected = {
            "concept_ids": ["b", "a", "c"],
            "values": [20, 10, 30],
            "ages": [25, 30, 40],
            "visit_concept_orders": [3, 9, 5],
            "dates": [20210101, 20210201, 20210301],
        }

        # Perform transformation
        result = mapper.transform(record)

        # Verify the output
        self.assertEqual(result, expected)

    def test_transform_without_sorting_columns(self):
        # Create an instance of the mapping class
        mapper = SortPatientSequenceMapping()

        # Mock data without 'orders' or 'dates'
        record = {
            "concept_ids": ["c", "b", "a"],
            "values": [30, 20, 10],
            "ages": [30, 25, 40],
            "visit_concept_orders": [5, 3, 9],
        }

        # Expected output should be unchanged since no sorting column is provided
        expected = record

        # Perform transformation
        result = mapper.transform(record)

        # Verify the output is unchanged
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
