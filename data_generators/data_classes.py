from typing import Tuple, NamedTuple


class RowSlicer(NamedTuple):
    """
    A data class for storing the row from the pandas data frame and the indexes for slicing the
    """
    row: Tuple
    start_index: int
    end_index: int
    target_index: int = 0


class TokenizeFieldInfo(NamedTuple):
    column_name: str
    tokenized_column_name: str = None
