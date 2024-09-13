import unittest

import numpy as np

from cehrbert.utils.stat_utils import TruncatedOfflineStatistics  # Replace with the actual module name


class TestTruncatedOfflineStatistics(unittest.TestCase):

    def setUp(self):
        # Create an instance of ExcludingOutlierOnlineStatistics with default settings
        self.stats = TruncatedOfflineStatistics(capacity=10, value_outlier_std=2.0)

    def test_two_elements(self):
        # Test adding data within the capacity
        for i in range(2):
            self.stats.add(i)
        self.stats._update_filtered_data()
        self.assertEqual(0, len(self.stats.filtered_data))

    def test_add_data_within_capacity(self):
        # Test adding data within the capacity
        for i in range(10):
            self.stats.add(i)
        self.assertEqual(len(self.stats.raw_data), 10)
        self.stats.reset()

    def test_add_data_beyond_capacity(self):
        # Test adding data beyond the capacity
        for i in range(10):
            self.stats.add(i)
        with self.assertRaises(ValueError):
            self.stats.add(11)
        self.stats.reset()

    def test_remove_outliers(self):
        # Test removing outliers
        data = [10, 12, 13, 14, 99999, 15, 16, 17, 18, -1000]
        for x in data:
            self.stats.add(x)

        # Trigger outlier removal
        self.stats._update_filtered_data()

        # The expected filtered data excludes -1000 and 1000 since they are extreme values
        expected_filtered_data = [10, 12, 13, 14, 15, 16, 17, 18]
        self.assertListEqual(self.stats.filtered_data, expected_filtered_data)

    def test_mean_calculation(self):
        # Test the mean calculation after excluding outliers
        data = [10, 12, 13, 14, 1000, 15, 16, 17, 18, -1000]
        for x in data:
            self.stats.add(x)

        # Test mean after excluding outliers
        mean = self.stats.get_current_mean()
        expected_mean = np.mean([10, 12, 13, 14, 15, 16, 17, 18])
        self.assertAlmostEqual(mean, expected_mean, places=5)

    def test_get_sum_of_squared(self):
        # Test removing outliers
        data = [10, 12, 13, 14, 99999, 15, 16, 17, 18, -1000]
        for x in data:
            self.stats.add(x)

        actual_sum_of_squared = self.stats.get_sum_of_squared()
        expected_sum_of_squares = np.sum(
            (np.asarray([10, 12, 13, 14, 15, 16, 17, 18]) - np.mean([10, 12, 13, 14, 15, 16, 17, 18])) ** 2
        )
        self.assertEqual(actual_sum_of_squared, expected_sum_of_squares)
        self.stats.reset()

    def test_standard_deviation_calculation(self):
        # Test the standard deviation after excluding outliers
        data = [10, 12, 13, 14, 1000, 15, 16, 17, 18, -1000]
        for x in data:
            self.stats.add(x)

        # Test standard deviation after excluding outliers
        stddev = self.stats.get_standard_deviation()
        expected_stddev = np.std([10, 12, 13, 14, 15, 16, 17, 18], ddof=0)
        self.assertAlmostEqual(stddev, expected_stddev, places=5)
        self.stats.reset()

    def test_empty_filtered_data(self):
        # Test when no data is present
        self.assertEqual(0.0, self.stats.get_current_mean())
        self.assertEqual(0.0, self.stats.get_standard_deviation())
        self.assertEqual(0.0, self.stats.get_sum_of_squared())


if __name__ == "__main__":
    unittest.main()
