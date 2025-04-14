import collections
import copy
import datetime
import itertools
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import pandas as pd
from cehrbert_data.const.common import NA
from cehrbert_data.decorators.patient_event_decorator_base import get_att_function
from datasets.formatting.formatting import LazyBatch
from dateutil.relativedelta import relativedelta
from meds.schema import birth_code, death_code
from pandas import Series

from cehrbert.med_extension.schema_extension import Event
from cehrbert.models.hf_models.tokenization_hf_cehrbert import CehrBertTokenizer
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments

birth_codes = [birth_code, "MEDS_BIRTH"]
death_codes = [death_code, "MEDS_DEATH"]

# OMOP concept ids for inpatient related visits
INPATIENT_VISIT_TYPES = ["9201", "262", "8971", "8920", "38004311"]
INPATIENT_VISIT_TYPE_CODES = [
    "Visit/IP",
    "Visit/ERIP",
    "Visit/51",
    "Visit/61",
    "NUCC/315D00000X",
]
ED_VISIT_TYPE_CODES = ["Visit/ER"]
DISCHARGE_FACILITY_TYPES = [
    "8536",
    "8863",
    "44814650",
    "4161979",
    "38004519",
    "4216643",
    "8717",
    "8920",
    "4021968",
    "8546",
    "8971",
    "8970",
    "44814649",
    "8827",
    "8676",
    "38003619",
    "8870",
    "4146681",
]

DATE_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


@dataclass
class VisitObject:
    visit_start_datetime: datetime.datetime
    visit_end_datetime: Optional[datetime.datetime]
    visit_type: str
    discharge_facility: Optional[str]
    events: Union[List[Event], Generator[Event, None, None]]

    def __lt__(self, other):
        """Compare visits chronologically by start datetime."""
        if not isinstance(other, VisitObject):
            return NotImplemented
        return self.visit_start_datetime < other.visit_start_datetime


def get_value(obj, key, default=None):
    """
    Generic function to get a value from either a dictionary or an object attribute.

    Args:
        obj: Either a dictionary or an object with attributes
        key: The key/attribute name to retrieve
        default: Value to return if the key/attribute doesn't exist

    Returns:
        The value associated with the key/attribute or the default value
    """
    if isinstance(obj, dict):
        # If it's a dictionary, use .get() method
        return obj.get(key, default)
    else:
        # If it's an object, use getattr
        return getattr(obj, key, default)


def has_events_and_get_events(events_iterable):
    """
    Check if an events iterable has at least one event, and return a new.

    iterable that includes all events if it does.

    Returns:
        (bool, Iterable): A tuple with (has_events, events_iterable)
    """
    if events_iterable is None:
        return False, None

    try:
        events_iter = iter(events_iterable)
        first_event = next(events_iter, None)

        if first_event is None:
            return False, None
        return True, itertools.chain([first_event], events_iter)
    except (TypeError, StopIteration):
        return False, None


def convert_date_to_posix_time(index_date: Union[datetime.date, datetime.datetime]) -> float:
    """
    Convert a date or datetime object to POSIX (Unix) time in seconds.

    Parameters
    ----------
    index_date : Union[datetime.date, datetime.datetime]
        The date or datetime object to be converted to POSIX time.

    Returns
    -------
    float
        The POSIX time in seconds as a float.

    Raises
    ------
    ValueError
        If `index_date` is not an instance of `datetime.date` or `datetime.datetime`.

    Examples
    --------
    >>> convert_date_to_posix_time(datetime.date(2024, 10, 25))
    1735104000.0

    >>> convert_date_to_posix_time(datetime.datetime(2024, 10, 25, 12, 30))
    1735144200.0
    """
    if isinstance(index_date, datetime.datetime):
        return index_date.timestamp()
    elif isinstance(index_date, datetime.date):
        return datetime.datetime.combine(index_date, datetime.datetime.min.time()).timestamp()
    else:
        raise ValueError("index_date must be datetime or datetime.datetime")


def replace_escape_chars(text: str) -> str:
    return re.sub(r"\s+", "_", text)


class TruncationType(Enum):
    RANDOM_COMPLETE = "random_complete"
    RANDOM_RIGHT_TRUNCATION = "random_right_truncation"
    RANDOM_TRUNCATION = "random_truncation"
    TAIL = "tail"


class DatasetMapping(ABC):

    def batch_transform(self, records: Union[LazyBatch, Dict[str, Any]]) -> List[Dict[str, Any]]:
        if isinstance(records, LazyBatch):
            dataframe = records.pa_table.to_pandas()
        else:
            dataframe = pd.DataFrame(records)
        applied_dataframe = dataframe.apply(self.transform_pandas_series, axis=1)
        return applied_dataframe.to_dict(orient="list")

    def transform_pandas_series(self, series: Series) -> Series:
        record = self.transform(series.to_dict())
        return pd.Series(record)

    def remove_columns(self):
        return []

    @abstractmethod
    def transform(self, record: Dict[str, Any]) -> Union[Dict[str, Any], Series]:
        """
        Transform the record.

        Args
            record: The row to process, as generated by the CDM processing
        Returns
            A dictionary from names to numpy arrays to be used by pytorch.
        """

    @staticmethod
    def convert_visit_columnar_to_python(visits: Dict[str, List[Any]]) -> List[VisitObject]:

        def event_generator(columnar_events: Dict[str, List[Any]]) -> Generator[Event, None, None]:
            batched_time = columnar_events["time"]
            batched_code = columnar_events["code"]
            batched_text_value = columnar_events["text_value"]
            batched_numeric_value = columnar_events["numeric_value"]
            batched_unit = columnar_events["unit"]
            batched_properties = columnar_events["properties"]
            batched_event_tuples = zip(
                batched_time, batched_code, batched_text_value, batched_numeric_value, batched_unit, batched_properties
            )
            for time, code, text_value, numeric_value, unit, properties in batched_event_tuples:
                yield Event(
                    time=time,
                    code=code,
                    text_value=text_value,
                    numeric_value=numeric_value,
                    unit=unit,
                    properties=properties,
                )

        batched_visit_start_datetime = visits["visit_start_datetime"]
        batched_visit_end_datetime = visits["visit_end_datetime"]
        batched_visit_type = visits["visit_type"]
        batched_discharge_facility = visits["discharge_facility"]
        batched_events = visits["events"]

        batched_tuples = zip(
            batched_visit_start_datetime,
            batched_visit_end_datetime,
            batched_visit_type,
            batched_discharge_facility,
            batched_events,
        )

        python_list = []
        for visit_tuple in batched_tuples:
            visit_start_datetime, visit_end_datetime, visit_type, discharge_facility, events = visit_tuple
            python_list.append(
                VisitObject(
                    visit_type=visit_type,
                    visit_start_datetime=visit_start_datetime,
                    visit_end_datetime=visit_end_datetime,
                    discharge_facility=discharge_facility,
                    events=event_generator(events),
                )
            )
        return python_list


class MedToCehrBertDatasetMapping(DatasetMapping):
    def __init__(self, data_args: DataTrainingArguments, is_pretraining: bool = True):
        self._time_token_function = get_att_function(data_args.att_function_type)
        self._include_auxiliary_token = data_args.include_auxiliary_token
        self._inpatient_time_token_function = get_att_function(data_args.inpatient_att_function_type)
        self._include_demographic_prompt = data_args.include_demographic_prompt
        self._is_pretraining = is_pretraining

    """
    This mapping function converts the MED (https://github.com/Medical-Event-Data-Standard/meds/tree/main) extension
    to the CehrBert format. We make several assumptions
    - The first event contains the demographic information
    - From the second event onward
        - the time of the event is visit_start_datetime.
        - the first measurement contains the code indicating a standard OMOP Visit concept_id (e.g. 9201, 9202)
        - in case of inpatient visits, the last measurement is assumed to
            contain the standard OMOP concept id for discharge facilities (e.g 8536)
        - in case of inpatient visits, datetime_value of the last measurement stores visit_end_datetime
    """

    def remove_columns(self):
        if self._is_pretraining:
            return ["visits", "birth_datetime", "index_date"]
        else:
            return [
                "visits",
                "birth_datetime",
                "visit_concept_ids",
            ]

    @staticmethod
    def _update_cehrbert_record(
        cehrbert_record: Dict[str, Any],
        code: str,
        visit_segment: int = 0,
        date: int = 0,
        age: int = -1,
        visit_concept_order: int = 0,
        visit_concept_id: str = "0",
        concept_value_mask: int = 0,
        concept_value: float = -1.0,
        mlm_skip_value: int = 0,
        unit: str = NA,
    ) -> None:
        cehrbert_record["concept_ids"].append(replace_escape_chars(code))
        cehrbert_record["visit_concept_orders"].append(visit_concept_order)
        cehrbert_record["ages"].append(age)
        cehrbert_record["dates"].append(date)
        cehrbert_record["visit_segments"].append(visit_segment)
        cehrbert_record["visit_concept_ids"].append(visit_concept_id)
        cehrbert_record["concept_value_masks"].append(concept_value_mask)
        cehrbert_record["concept_values"].append(concept_value)
        cehrbert_record["units"].append(unit)
        cehrbert_record["mlm_skip_values"].append(mlm_skip_value)

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:

        cehrbert_record = {
            "person_id": record["patient_id"],
            "concept_ids": [],
            "visit_segments": [],
            "orders": [],
            "dates": [],
            "ages": [],
            "visit_concept_orders": [],
            "concept_value_masks": [],
            "concept_values": [],
            "units": [],
            "mlm_skip_values": [],
            "visit_concept_ids": [],
        }
        # Extract the demographic information
        birth_datetime = record["birth_datetime"]
        if isinstance(birth_datetime, pd.Timestamp):
            birth_datetime = birth_datetime.to_pydatetime()
        gender = record["gender"]
        race = record["race"]
        visits = record["visits"]
        # This indicates this is columnar format
        if isinstance(visits, dict):
            visits = sorted(self.convert_visit_columnar_to_python(visits))
        else:
            visits = sorted(visits, key=lambda e: get_value(e, "visit_start_datetime"))

        if self._include_demographic_prompt:
            first_visit = visits[0]
            first_visit_start_datetime: datetime.datetime = get_value(first_visit, "visit_start_datetime")
            year_str = f"year:{str(first_visit_start_datetime.year)}"
            age_str = f"age:{str(relativedelta(first_visit_start_datetime, birth_datetime).years)}"

            self._update_cehrbert_record(cehrbert_record, year_str)
            self._update_cehrbert_record(cehrbert_record, age_str)
            self._update_cehrbert_record(cehrbert_record, gender)
            self._update_cehrbert_record(cehrbert_record, race)

        # A bool indicator to toggle between 1 and 2
        visit_segment_indicator = False

        # Use a data cursor to keep track of time
        date_cursor: Optional[datetime.datetime] = None
        visit: VisitObject
        # Loop through all the visits
        for i, visit in enumerate(visits):
            events: Generator[Event, None, None] = get_value(visit, "events")
            has_events, events = has_events_and_get_events(events)
            if not has_events:
                continue

            visit_start_datetime: datetime.datetime = get_value(visit, "visit_start_datetime")
            time_delta = (visit_start_datetime - date_cursor).days if date_cursor else None
            date_cursor = visit_start_datetime

            # We assume the first measurement to be the visit type of the current visit
            visit_type = get_value(visit, "visit_type")
            is_er_or_inpatient = (
                visit_type in INPATIENT_VISIT_TYPES
                or visit_type in INPATIENT_VISIT_TYPE_CODES
                or visit_type in ED_VISIT_TYPE_CODES
            )

            # Add artificial time tokens to the patient timeline if timedelta exists
            if time_delta is not None:
                # This generates an artificial time token depending on the choice of the time token functions
                self._update_cehrbert_record(
                    cehrbert_record,
                    code=self._time_token_function(time_delta),
                    visit_concept_order=i + 1,
                )

            # Add the VS token to the patient timeline to mark the start of a visit
            age = relativedelta(visit_start_datetime, birth_datetime).years
            # Calculate the week number since the epoch time
            date = (visit_start_datetime - datetime.datetime(year=1970, month=1, day=1)).days // 7
            visit_segment = int(visit_segment_indicator) + 1

            self._update_cehrbert_record(
                cehrbert_record,
                code="[VS]",
                visit_concept_order=i + 1,
                age=age,
                date=date,
                visit_segment=visit_segment,
                visit_concept_id=visit_type,
            )

            if self._include_auxiliary_token:
                self._update_cehrbert_record(
                    cehrbert_record,
                    code=visit_type,
                    visit_concept_order=i + 1,
                    age=age,
                    date=date,
                    visit_segment=visit_segment,
                    visit_concept_id=visit_type,
                )
            # Keep track of the existing outpatient events, we don't want to add them again
            existing_outpatient_events = list()
            for e in events:
                # If the event doesn't have a time stamp, we skip it
                if not e["time"]:
                    continue

                # If numeric_value exists, this is a concept/value tuple, we indicate this using a concept_value_mask
                numeric_value = e.get("numeric_value", None)
                # The unit might be populated with a None value
                unit = e.get("unit", NA) if e.get("unit", NA) else NA
                concept_value_mask = int(numeric_value is not None)
                concept_value = numeric_value if concept_value_mask == 1 else -1.0
                code = replace_escape_chars(e["code"])

                # If the value mask is 1, this indicates a numeric value associated with the concept
                if concept_value_mask != 1:
                    # Otherwise we will try to concatenate the answer with the code if the categorical value is provide
                    text_value = e.get("text_value", None)
                    if text_value:
                        text_value_replaced = replace_escape_chars(text_value)
                        code = f"{code}//option:{text_value_replaced}"

                # Add a medical token to the patient timeline
                # If this is an inpatient visit, we use the event time stamps to calculate age and date
                # because the patient can stay in the hospital for a period of time.
                if is_er_or_inpatient:
                    # Calculate age using the event time stamp
                    age = relativedelta(e["time"], birth_datetime).years
                    # Calculate the week number since the epoch time
                    date = (e["time"] - datetime.datetime(year=1970, month=1, day=1)).days // 7
                    # Calculate the time diff in days w.r.t the previous measurement
                    meas_time_diff = (e["time"] - date_cursor).days
                    # Update the date_cursor if the time diff between two neighboring measurements is greater than and
                    # equal to 1 day
                    if meas_time_diff > 0:
                        date_cursor = e["time"]
                        if self._inpatient_time_token_function:
                            # This generates an artificial time token depending on the choice of the time token functions
                            self._update_cehrbert_record(
                                cehrbert_record,
                                code=f"i-{self._inpatient_time_token_function(meas_time_diff)}",
                                visit_concept_order=i + 1,
                                visit_segment=visit_segment,
                                visit_concept_id=visit_type,
                            )
                else:
                    # For outpatient visits, we use the visit time stamp to calculate age and time because we assume
                    # the outpatient visits start and end on the same day.
                    # We check whether the date/code/value combination already exists in the existing events
                    # If they exist, we do not add them to the patient timeline for outpatient visits.
                    if (date, code, concept_value) in existing_outpatient_events:
                        continue

                self._update_cehrbert_record(
                    cehrbert_record,
                    code=code,
                    age=age,
                    date=date,
                    visit_concept_order=i + 1,
                    visit_segment=visit_segment,
                    visit_concept_id=visit_type,
                    concept_value_mask=concept_value_mask,
                    concept_value=concept_value,
                    unit=unit,
                    mlm_skip_value=concept_value_mask,
                )
                existing_outpatient_events.append((date, code, concept_value))

            # For inpatient or ER visits, we want to discharge_facility to the end of the visit
            if is_er_or_inpatient:
                # If visit_end_datetime is populated for the inpatient visit, we update the date_cursor
                visit_end_datetime: Optional[datetime.datetime] = get_value(visit, "visit_end_datetime")
                if visit_end_datetime:
                    date_cursor = visit_end_datetime

                if self._include_auxiliary_token:
                    # Reuse the age and date calculated for the last event in the patient timeline for the discharge
                    # facility event
                    discharge_facility = get_value(visit, "discharge_facility")
                    if not discharge_facility:
                        discharge_facility = "0"

                    self._update_cehrbert_record(
                        cehrbert_record,
                        code=discharge_facility,
                        age=age,
                        date=date,
                        visit_concept_order=i + 1,
                        visit_segment=visit_segment,
                        visit_concept_id=visit_type,
                    )

            # Reuse the age and date calculated for the last event in the patient timeline
            self._update_cehrbert_record(
                cehrbert_record,
                code="[VE]",
                age=age,
                date=date,
                visit_concept_order=i + 1,
                visit_segment=visit_segment,
                visit_concept_id=visit_type,
            )

            # Toggle visit_segment_indicator
            visit_segment_indicator = not visit_segment_indicator

        # Generate the orders of the concepts that the cehrbert dataset mapping function expects
        cehrbert_record["orders"] = list(range(1, len(cehrbert_record["concept_ids"]) + 1))

        # Add some count information for this sequence
        cehrbert_record["num_of_concepts"] = len(cehrbert_record["concept_ids"])
        cehrbert_record["num_of_visits"] = len(visits)

        if "label" in record:
            cehrbert_record["label"] = record["label"]
        if "age_at_index" in record:
            cehrbert_record["age_at_index"] = record["age_at_index"]

        return cehrbert_record


class SortPatientSequenceMapping(DatasetMapping):
    """
    A mapping function to order all the features using a pre-defined orders/dates column.

    This may not be necessary since the order is feature columns should've been ordered
    correctly during the data generation process in the spark application. However,
    it's a good idea to sort them explicitly one more time
    """

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sort all the list features using a pre-defined orders/dates.

        If orders/dates columns are not provided,
        do nothing.
        """
        sorting_columns = record.get("orders", None)
        if sorting_columns is None:
            sorting_columns = record.get("dates", None)

        if sorting_columns is None:
            return record

        sorting_columns = list(map(int, sorting_columns))
        seq_length = len(record["concept_ids"])
        column_names = ["concept_ids"]
        column_values = [record["concept_ids"]]

        for k, v in record.items():
            if k in column_names:
                continue
            if isinstance(v, (list, np.ndarray)) and len(v) == seq_length:
                column_names.append(k)
                column_values.append(v)

        sorted_list = sorted(zip(sorting_columns, *column_values), key=lambda tup2: (tup2[0], tup2[1]))

        # uses a combination of zip() and unpacking (*) to transpose the list of tuples. This means converting rows
        # into columns: the first tuple formed from all the first elements of the sorted tuples, the second tuple
        # from all the second elements, and so on. Then slices the resulting list of tuples to skip the first tuple
        # (which contains the sorting criteria) and retain only the data columns.
        sorted_features = list(zip(*list(sorted_list)))[1:]
        new_record = collections.OrderedDict()
        for i, new_val in enumerate(sorted_features):
            new_record[column_names[i]] = list(new_val)
        return new_record


class HFTokenizationMapping(DatasetMapping):
    def __init__(self, concept_tokenizer: CehrBertTokenizer, is_pretraining: bool):
        self._concept_tokenizer = concept_tokenizer
        self._is_pretraining = is_pretraining
        self._lab_token_ids = self._concept_tokenizer.lab_token_ids

    @staticmethod
    def fill_na_value(values, value_to_fill):
        none_values = np.array([x is None for x in values])
        if none_values.any():
            values = values.copy()
            values[none_values] = value_to_fill
        return values

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:

        input_ids = self._concept_tokenizer.encode(record["concept_ids"])
        record["input_ids"] = input_ids
        concept_value_masks = record["concept_value_masks"]

        # These fields may not exist in the old version of the datasets
        if "units" in record:
            record["units"] = self.fill_na_value(record["units"], NA)
        if "concept_as_values" in record:
            record["concept_as_values"] = self.fill_na_value(record["concept_as_values"], NA)

        # Backward compatibility
        if "concept_values" not in record:
            record["concept_values"] = record["number_as_values"]

        concept_value_is_nan = np.isnan(record["concept_values"])
        if concept_value_is_nan.any():
            # Create a writeable copy
            concept_value_masks = concept_value_masks.copy()
            concept_value_masks[concept_value_is_nan] = 0
            record["concept_value_masks"] = concept_value_masks
            concept_values = record["concept_values"].copy()
            concept_values[concept_value_is_nan] = 0.0
            record["concept_values"] = concept_values

        assert len(input_ids) == len(
            record["concept_ids"]
        ), "the length of input_ids needs to be the same as the length of concept_ids"

        # If any concept has a value associated with it, we normalize the value
        if np.any(np.asarray(concept_value_masks) > 0):
            units = record["units"]
            normalized_concept_values = copy.deepcopy(record["concept_values"])
            for i, (
                concept_id,
                unit,
                token_id,
                concept_value_mask,
                concept_value,
            ) in enumerate(
                zip(
                    record["concept_ids"],
                    units,
                    input_ids,
                    concept_value_masks,
                    record["concept_values"],
                )
            ):
                if token_id in self._lab_token_ids:
                    normalized_concept_value = self._concept_tokenizer.normalize(concept_id, unit, concept_value)
                    normalized_concept_values[i] = normalized_concept_value
            record["concept_values"] = normalized_concept_values

        # If mlm_skip_value=1, this indicates there is a value associated with this position and
        # hence we block the MLM to randomly pick this token to be predicted
        if self._is_pretraining:
            record.update({"labels": copy.deepcopy(input_ids)})

        return record


class HFFineTuningMapping(DatasetMapping):
    """Consider removing this transformation in the future."""

    def transform(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "age_at_index": record["age"] if "age" in record else record["age_at_index"],
            "classifier_label": record["label"],
            "index_date": (convert_date_to_posix_time(record["index_date"]) if "index_date" in record else None),
        }

    def remove_columns(self):
        return ["label"]
