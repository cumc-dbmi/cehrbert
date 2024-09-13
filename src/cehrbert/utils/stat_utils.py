import numpy as np
import scipy.stats as stats
from femr.stat_utils import OnlineStatistics


class TruncatedOnlineStatistics(OnlineStatistics):

    def __init__(self, capacity=100, value_outlier_std=2.0):
        super().__init__()
        self.is_online_update_started = False
        self.value_outlier_std = value_outlier_std
        self.truncated_offline_statistics = TruncatedOfflineStatistics(
            capacity=capacity, value_outlier_std=value_outlier_std
        )

    def add(self, weight: float, value: float) -> None:
        if self.is_online_update_started:
            std = self.standard_deviation()
            if (
                self.current_mean - self.value_outlier_std * std
                <= value
                <= self.current_mean + self.value_outlier_std * std
            ):
                super().add(weight, value)
        else:
            self.truncated_offline_statistics.add(value)
            if self.truncated_offline_statistics.is_full():
                self.begin_online_stats()

    def mean(self) -> float:
        """Return the current mean."""
        if self.is_online_update_started:
            return super().mean()
        else:
            return self.truncated_offline_statistics.get_current_mean()

    def standard_deviation(self) -> float:
        """Return the current standard deviation."""
        # If the count is zero, we don't calculate the standard deviation
        if self.count == 0:
            return 0.0
        elif self.is_online_update_started:
            return super().standard_deviation()
        else:
            return self.truncated_offline_statistics.get_standard_deviation()

    def begin_online_stats(self):

        # This prevents the online stats from being started twice
        if self.is_online_update_started:
            raise RuntimeError(f"The statistics has already been brought online, you can't start the online mode twice")

        self.is_online_update_started = True
        self.current_mean = self.truncated_offline_statistics.get_current_mean()
        self.variance = self.truncated_offline_statistics.get_sum_of_squared()
        self.count = self.truncated_offline_statistics.get_count()

    def combine(self, other) -> None:
        """
        The two truncated online stats objects need to be brought to the online mode before the stats are combined.

        Args:
            other:

        Returns:
        """
        if not self.is_online_update_started:
            self.begin_online_stats()
        if not other.is_online_update_started:
            other.begin_online_stats()
        super().combine(other)


class TruncatedOfflineStatistics:
    """
    A class to compute and maintain statistics for a dataset while excluding outliers based on a.

    truncated normal distribution defined by a specified number of standard deviations.

    This class supports offline data collection (i.e., data is accumulated until capacity is reached),
    and outliers beyond the specified number of standard deviations are excluded before computing
    statistics such as mean and standard deviation.

    Attributes:
    -----------
    capacity : int
       The maximum number of data points that can be stored.
    value_outlier_std : float
       The number of standard deviations used to define outliers. Data points outside this range
       are excluded when computing statistics.
    lower_quantile : float
       The quantile corresponding to the lower bound for valid data, computed as the cumulative
       distribution function (CDF) of -`value_outlier_std`.
    upper_quantile : float
       The quantile corresponding to the upper bound for valid data, computed as the CDF of
       `value_outlier_std`.
    raw_data : list
       A list to store all incoming data points.
    filtered_data : list
       A list to store data points after removing outliers.
    updated : bool
       A flag that indicates whether the filtered data has been updated after new data points were
       added.
    """

    def __init__(self, capacity=100, value_outlier_std=2.0):
        """
        Initializes the TruncatedOfflineStatistics instance with a capacity and standard deviation threshold.

        for outlier detection.

        Parameters:
        -----------
        capacity : int, optional
            The maximum number of data points to store (default is 100).
        value_outlier_std : float, optional
            The number of standard deviations to use for outlier detection (default is 2.0).
        """
        super().__init__()
        self.lower_quantile = stats.norm.cdf(-value_outlier_std)
        self.upper_quantile = stats.norm.cdf(value_outlier_std)
        self.capacity = capacity
        self.raw_data = list()
        self.filtered_data = list()
        self.updated = False

    def is_full(self) -> bool:
        """
        Checks if the number of data points in the `raw_data` list has reached the capacity.

        Returns:
        --------
        bool
            True if the number of data points is greater than or equal to the capacity, otherwise False.
        """
        return len(self.raw_data) >= self.capacity

    def add(self, value: float) -> None:
        """
        Adds a new data point to the `raw_data` list if the capacity is not full.

        If the capacity is reached, raises a ValueError.
        Also marks the `updated` flag as False to indicate that the filtered data needs to be refreshed.

        Parameters:
        -----------
        value : float
            The new data point to be added to the dataset.

        Raises:
        -------
        ValueError:
            If the capacity of the underlying data is full.
        """
        if len(self.raw_data) < self.capacity:
            self.raw_data.append(value)
            # When new data is added to the raw_data array, we need to update the filtered_data later on
            self.updated = False
        else:
            raise ValueError(f"The capacity of the underlying data is full at {self.capacity}")

    def get_count(self) -> int:
        """
        Returns the count of data points in `filtered_data` after removing outliers.

        Returns:
        --------
        int
            The number of data points that remain after outliers are filtered out.
        """
        self._update_filtered_data()
        return len(self.filtered_data)

    def get_current_mean(self) -> float:
        """
        Computes and returns the mean of the `filtered_data` (excluding outliers).

        Returns:
        --------
        float
            The mean of the filtered data. Returns 0.0 if there are no valid data points.
        """
        self._update_filtered_data()
        if self.filtered_data:
            return np.mean(self.filtered_data)
        else:
            return 0.0

    def get_sum_of_squared(self) -> float:
        """
        Computes the sum of squared differences from the mean for the `filtered_data`.

        Returns:
        --------
        float
            The sum of squared differences from the mean for the filtered data.
            Returns 0.0 if no valid data points are present.
        """
        self._update_filtered_data()
        if self.filtered_data:
            current_mean = np.mean(self.filtered_data)
            return np.sum([(x - current_mean) ** 2 for x in self.filtered_data])
        else:
            return 0.0

    def get_standard_deviation(self) -> float:
        """
        Computes the standard deviation of the `filtered_data` (excluding outliers).

        Returns:
        --------
        float
            The standard deviation of the filtered data.
            Returns 0.0 if there are no valid data points.
        """
        self._update_filtered_data()
        if self.filtered_data:
            return np.std(self.filtered_data)
        else:
            return 0.0

    def _update_filtered_data(
        self,
    ) -> None:
        """
        Filters the `raw_data` to remove outliers based on the `value_outlier_std` threshold.

        This method is called internally before any computation of statistics to ensure
        that the data being used is current and valid.

        This method uses the `lower_quantile` and `upper_quantile` to filter the data points.
        """
        if not self.updated and len(self.raw_data) > 0:
            # Toggle the updated variable
            self.updated = True
            lower_bound = np.quantile(self.raw_data, self.lower_quantile)
            upper_bound = np.quantile(self.raw_data, self.upper_quantile)
            # Update the filtered_data
            self.filtered_data = [x for x in self.raw_data if lower_bound <= x <= upper_bound]

    def reset(self):
        """
        Resets the raw and filtered data, clearing all stored data points.

        This method also resets the `updated` flag to indicate that the data needs to be re-filtered
        when new data is added.

        A useful method for unittests
        """
        self.raw_data.clear()
        self.filtered_data.clear()
        self.updated = False
