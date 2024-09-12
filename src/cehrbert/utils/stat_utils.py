import numpy as np
from femr.stat_utils import OnlineStatistics


class RunningStatistics(OnlineStatistics):
    def __init__(self, capacity=100, lower_quantile=0.05, upper_quantile=0.950):
        super().__init__()
        self.excluding_outlier_online_statistics = ExcludingOutlierOnlineStatistics(
            capacity=capacity, lower_quantile=lower_quantile, upper_quantile=upper_quantile
        )

    def add(self, weight: float, value: float) -> None:
        if self.excluding_outlier_online_statistics.is_full():
            super().add(weight, value)
        else:
            self.excluding_outlier_online_statistics.add(value)
            if self.excluding_outlier_online_statistics.is_full():
                self.current_mean = self.excluding_outlier_online_statistics.get_current_mean()
                self.variance = self.excluding_outlier_online_statistics.get_sum_of_squared()
                self.count = self.excluding_outlier_online_statistics.get_count()

    def mean(self) -> float:
        """Return the current mean."""
        if self.excluding_outlier_online_statistics.is_full():
            return super().mean()
        else:
            self.excluding_outlier_online_statistics.get_current_mean()

    def standard_deviation(self) -> float:
        """Return the current standard devation."""
        if self.excluding_outlier_online_statistics.is_full():
            return super().standard_deviation()
        else:
            return self.excluding_outlier_online_statistics.standard_deviation()


class ExcludingOutlierOnlineStatistics:
    def __init__(self, capacity=100, lower_quantile=0.05, upper_quantile=0.950):
        super().__init__()
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.capacity = capacity
        self.raw_data = list()
        self.filtered_data = list()
        self.updated = False

    def reset(self):
        self.raw_data.clear()
        self.filtered_data.clear()
        self.updated = False

    def is_full(self) -> bool:
        return len(self.raw_data) >= self.capacity

    def add(self, value: float) -> None:
        if len(self.raw_data) < self.capacity:
            self.raw_data.append(value)
            # When new data is added to the raw_data array, we need to update the filtered_data later on
            self.updated = False
        else:
            raise ValueError(f"The capacity of the underlying data is full at {self.capacity}")

    def get_count(self) -> int:
        self.update_remove_outliers()
        return len(self.filtered_data)

    def get_current_mean(self) -> float:
        self.update_remove_outliers()
        if self.filtered_data:
            return np.mean(self.filtered_data)
        else:
            raise ValueError(f"There is no value")

    def get_sum_of_squared(self) -> float:
        self.update_remove_outliers()
        if self.filtered_data:
            current_mean = np.mean(self.filtered_data)
            return np.sum([(x - current_mean) ** 2 for x in self.filtered_data])
        else:
            raise ValueError(f"There is no value")

    def standard_deviation(self) -> float:
        self.update_remove_outliers()
        if self.filtered_data:
            return np.std(self.filtered_data)
        else:
            raise ValueError(f"There is no value")

    def update_remove_outliers(
        self,
    ) -> None:
        if not self.updated and len(self.raw_data) > 0:
            # Toggle the updated variable
            self.updated = True
            lower_bound = np.quantile(self.raw_data, self.lower_quantile)
            upper_bound = np.quantile(self.raw_data, self.upper_quantile)
            # Update the filtered_data
            self.filtered_data = [x for x in self.raw_data if lower_bound <= x <= upper_bound]
