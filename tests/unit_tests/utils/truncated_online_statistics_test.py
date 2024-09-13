import unittest

import numpy as np

from cehrbert.utils.stat_utils import TruncatedOfflineStatistics, TruncatedOnlineStatistics


class TestTruncatedOnlineStatistics(unittest.TestCase):

    # def setUp(self):
    #     # Set up instances of TruncatedOnlineStatistics and TruncatedOfflineStatistics for testing
    #     self.online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
    #     self.offline_stats = TruncatedOfflineStatistics(capacity=10, value_outlier_std=2.0)

    def test_add_data_before_online_mode(self):
        # Set up instances of TruncatedOnlineStatistics and TruncatedOfflineStatistics for testing
        online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        offline_stats = TruncatedOfflineStatistics(capacity=10, value_outlier_std=2.0)
        # Test adding data before transitioning to online mode
        data = [1, 2, 3, 4, 5, 100, -100]
        for x in data:
            online_stats.add(1.0, x)

        # Before switching to online, check that offline stats are still active
        self.assertFalse(online_stats.is_online_update_started)
        self.assertEqual(offline_stats.get_count(), 0)  # Offline mode hasn't reached full capacity yet

    def test_add_data_and_switch_to_online_mode(self):
        # Set up instances of TruncatedOnlineStatistics and TruncatedOfflineStatistics for testing
        online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        # Add enough data to trigger the switch to online mode
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in data:
            online_stats.add(1.0, x)

        # Online update should have been started
        self.assertTrue(online_stats.is_online_update_started)

    def test_mean_calculation_in_online_mode(self):
        # Set up instances of TruncatedOnlineStatistics and TruncatedOfflineStatistics for testing
        online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        # Test that the mean is calculated correctly in online mode
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in data:
            online_stats.add(1.0, x)

        # Mean after adding all data points (in online mode)
        expected_mean = np.mean(data)
        self.assertAlmostEqual(online_stats.mean(), expected_mean, places=5)

    def test_standard_deviation_calculation_in_online_mode(self):
        # Set up instances of TruncatedOnlineStatistics and TruncatedOfflineStatistics for testing
        online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        # Test that the standard deviation is calculated correctly in online mode
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for x in data:
            online_stats.add(1.0, x)

        # Standard deviation after adding all data points (in online mode)
        # The truncated offline stats removes the first and the last number from the array
        expected_stddev = np.std(data[1:-1], ddof=0)
        self.assertAlmostEqual(online_stats.standard_deviation(), expected_stddev, places=5)

    def test_online_mean_with_outliers(self):
        online_stats = TruncatedOnlineStatistics(capacity=11, value_outlier_std=2.0)
        # Add data with outliers, ensuring they are excluded
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, -1000]
        for x in data:
            online_stats.add(1.0, x)
        # After excluding the outliers, calculate mean
        expected_mean = np.mean([1, 2, 3, 4, 5, 6, 7, 8, 9])  # Exclude outliers
        self.assertAlmostEqual(online_stats.mean(), expected_mean, places=5)

    def test_online_standard_deviation_with_outliers(self):
        online_stats = TruncatedOnlineStatistics(capacity=11, value_outlier_std=2.0)
        # Add data with outliers, ensuring they are excluded
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000, -1000]
        for x in data:
            online_stats.add(1.0, x)

        # After excluding the outliers, calculate standard deviation
        expected_stddev = np.std([1, 2, 3, 4, 5, 6, 7, 8, 9], ddof=0)  # Exclude outliers
        self.assertAlmostEqual(online_stats.standard_deviation(), expected_stddev, places=5)

    def test_combining_two_truncated_online_stats(self):

        online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        # Combine two TruncatedOnlineStatistics objects and check the mean and variance
        data1 = [1, 2, 3, 4, 5]
        data2 = [6, 7, 8, 9, 10]

        for x in data1:
            online_stats.add(1.0, x)

        other_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        for x in data2:
            other_stats.add(1.0, x)

        online_stats.combine(other_stats)

        # Check the combined mean and variance
        combined_data = data1[1:-1] + data2[1:-1]
        expected_mean = np.mean(combined_data)
        expected_stddev = np.std(combined_data, ddof=0)

        self.assertAlmostEqual(online_stats.mean(), expected_mean, places=5)
        self.assertAlmostEqual(online_stats.standard_deviation(), expected_stddev, places=5)

    def test_add_data_beyond_capacity(self):
        online_stats = TruncatedOnlineStatistics(capacity=10, value_outlier_std=2.0)
        # Test that exceeding the capacity raises an error
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for x in data[:-1]:  # Add data up to capacity
            online_stats.add(1.0, x)
        prev_mean = online_stats.mean()
        prev_std = online_stats.standard_deviation()
        # This falls outside the range (mean - 2 * std, mean + 2 * std)
        # Therefore the mean and std do not change
        online_stats.add(1.0, data[-1])
        self.assertEqual(prev_mean, online_stats.mean())
        self.assertEqual(prev_std, online_stats.standard_deviation())


if __name__ == "__main__":
    unittest.main()
