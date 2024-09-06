from enum import Enum
from typing import NamedTuple, Tuple


class RecordStatus(Enum):
    """
    COMPLETE indicates the record contains the entire history of a patient, therefore we should add [START] and [END].

    tokens to the patient history. RIGHT_TRUNCATION indicates that we employ the right truncation of the patient
    history for long sequences. This means that we should not add the [END] token to the end of the record because
    this partial history is not supposed to end. TRUNCATION indicates that the record is a slice of the patient
    history, we should not add [START] token and [END] token to such a sequence because this partial history got
    truncated on both ends.
    """

    COMPLETE = 1
    RIGHT_TRUNCATION = 2
    TRUNCATION = 3


class RowSlicer(NamedTuple):
    """A data class for storing the row from the pandas data frame and the indexes for slicing the."""

    row: Tuple
    start_index: int
    end_index: int
    target_index: int = 0
    record_status: RecordStatus = RecordStatus.COMPLETE


class TokenizeFieldInfo(NamedTuple):
    column_name: str
    tokenized_column_name: str = None
