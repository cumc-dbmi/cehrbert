from typing import List, NamedTuple

LEARNING_RATE = "LEARNING_RATE"
LSTM_DIRECTION = "LSTM_DIRECTION"
LSTM_UNIT = "LSTM_UNIT"


class GridSearchConfig(NamedTuple):
    """A data class for storing the row from the pandas data frame and the indexes for slicing the."""

    learning_rates: List[float] = [1.0e-4]
    lstm_directions: List[bool] = [True]
    lstm_units: List[int] = [128]
