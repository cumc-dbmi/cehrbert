import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import meds_reader
from meds.schema import birth_code
from transformers import logging

from cehrbert.data_generators.hf_data_generator import (
    DEFAULT_ED_TO_INPATIENT_CONCEPT_ID,
    DEFAULT_INPATIENT_CONCEPT_ID,
    DEFAULT_OUTPATIENT_CONCEPT_ID,
)
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import (
    MedsToBertMimic4,
    MedsToCehrBertConversion,
    MedsToCehrbertOMOP,
)
from cehrbert.med_extension.schema_extension import Event

LOG = logging.get_logger("transformers")


def is_visit_table(table_name: str) -> bool:
    return table_name in ["visit", "visit_occurrence"]


def is_condition_table(table_name: str) -> bool:
    return table_name in ["condition", "condition_occurrence"]


def is_procedure_table(table_name: str) -> bool:
    return table_name in ["procedure", "procedure_occurrence"]


def is_drug_table(table_name: str) -> bool:
    return table_name in ["drug", "drug_exposure"]


def is_device_table(table_name: str) -> bool:
    return table_name in ["device", "device_exposure"]


def is_measurement_table(table_name: str) -> bool:
    return table_name == "measurement"


def is_observation_table(table_name: str) -> bool:
    return table_name == "observation"


@dataclass
class PatientDemographics:
    birth_datetime: datetime = None
    race: str = None
    gender: str = None
    ethnicity: str = None


class PatientBlock:
    """
    Represents a block of medical events for a single patient visit, including.

    inferred visit type and various admission and discharge statuses.

    Attributes:
        visit_id (int): The unique ID of the visit.
        events (List[meds_reader.Event]): A list of medical events associated with this visit.
        min_time (datetime): The earliest event time in the visit.
        max_time (datetime): The latest event time in the visit.
        conversion (MedsToCehrBertConversion): Conversion object for mapping event codes to CEHR-BERT.
        has_ed_admission (bool): Whether the visit includes an emergency department (ED) admission event.
        has_admission (bool): Whether the visit includes an admission event.
        has_discharge (bool): Whether the visit includes a discharge event.
        visit_type (str): The inferred type of visit, such as inpatient, ED, or outpatient.
    """

    def __init__(
        self,
        events: List[meds_reader.Event],
        visit_id: int,
        conversion: MedsToCehrBertConversion,
    ):
        """
        Initializes a PatientBlock instance, inferring the visit type based on the events and caching.

        admission and discharge status.

        Args:
            events (List[meds_reader.Event]): The medical events associated with the visit.
            visit_id (int): The unique ID of the visit.
            conversion (MedsToCehrBertConversion): Conversion object for mapping event codes to CEHR-BERT.

        Attributes are initialized to store visit metadata and calculate admission/discharge statuses.
        """
        self.visit_id = visit_id
        self.events = sorted(events, key=lambda e: [e.time, e.code])
        self.conversion = conversion

        self._admission_event, admission_event_time = self._find_admission()
        self._ed_admission_event, ed_admission_event_time = self._find_ed_admission()
        self._inferred_visit_type, inferred_visit_event_time = self._infer_visit_type()
        self._discharge_facility, discharge_event_time = self._find_discharge_facility()

        # Set the block start time depending on the visit type
        self.block_start_time: datetime
        # Admission takes precedence over ED
        if self.has_admission:
            self.visit_type = self._admission_event
            self.block_start_time = admission_event_time
        elif self.has_ed_admission:
            self.visit_type = self._ed_admission_event
            self.block_start_time = ed_admission_event_time
        else:
            self.visit_type = self._inferred_visit_type
            self.block_start_time = inferred_visit_event_time

        # Set the block end time depending on the visit type
        self.block_end_time: datetime
        if self.has_ed_or_hospital_admission and discharge_event_time:
            self.block_end_time = discharge_event_time
        else:
            self.block_end_time = self._infer_block_end_time()
            if not self.block_end_time:
                self.block_end_time = self.block_start_time.replace(hour=23, minute=59, second=59, microsecond=999999)

    @property
    def min_time(self) -> datetime:
        return self.events[0].time

    @property
    def max_time(self) -> datetime:
        return self.events[-1].time

    @property
    def has_admission(self) -> bool:
        return self._admission_event is not None

    @property
    def has_ed_admission(self) -> bool:
        return self._ed_admission_event is not None

    @property
    def has_ed_or_hospital_admission(self) -> bool:
        return self.has_admission | self.has_ed_admission

    @property
    def has_discharge(self) -> bool:
        return self.discharged_to is not None

    @property
    def discharged_to(self) -> Optional[str]:
        return self._discharge_facility

    def _infer_block_start_time(self) -> Optional[datetime]:
        for event in self.events:
            table = getattr(event, "table", None)
            if is_visit_table(table):
                return event.time
        return self.events[0].time

    def _infer_block_end_time(self) -> Optional[datetime]:
        """
        If the discharge event is missing, we try to infer the end of the visit.

        by one of the following records. We do not want to use drug because they could
        occur years away after the visit occurred. We also do not want to use condition
        because they could be problem list conditions that occurred years before the visit

        Returns datetime.datetime:
        """
        for event in self.events:
            table = getattr(event, "table", None)
            if is_visit_table(table) and event.end:
                return event.end

        for event in reversed(self.events):
            table = getattr(event, "table", None)
            can_infer = is_measurement_table(table) | is_procedure_table(table) | is_observation_table(table)
            if can_infer:
                return event.time
        return self.events[-1].time

    def _infer_visit_type(self) -> Tuple[Optional[str], Optional[datetime]]:

        inferred_visit_type: Optional[str] = None
        inferred_block_start_time: Optional[datetime] = None
        for event in self.events:
            table = getattr(event, "table", None)
            for matching_rule in self.conversion.get_other_visit_matching_rules():
                if re.match(matching_rule, event.code):
                    return event.code, event.time
            if is_visit_table(table):
                inferred_visit_type = event.code
                inferred_block_start_time = event.time
                break

        if inferred_visit_type is None:
            inferred_block_start_time = self._infer_block_start_time()
            inferred_visit_type = DEFAULT_OUTPATIENT_CONCEPT_ID

        return inferred_visit_type, inferred_block_start_time

    def _find_ed_admission(self) -> Tuple[Optional[str], Optional[datetime]]:
        """
        Determines if the visit includes an emergency department (ED) admission event.

        Returns:
            bool: True if an ED admission event is found, False otherwise.
        """
        for event in self.events:
            for matching_rule in self.conversion.get_ed_admission_matching_rules():
                if re.match(matching_rule, event.code):
                    return event.code, event.time
        return None, None

    def _find_admission(self) -> Tuple[Optional[str], Optional[datetime]]:
        """
        Determines if the visit includes a hospital admission event.

        Returns:
            bool: True if an admission event is found, False otherwise.
        """
        for event in self.events:
            for matching_rule in self.conversion.get_admission_matching_rules():
                if re.match(matching_rule, event.code):
                    return event.code, event.time
        return None, None

    def _find_discharge_facility(self) -> Tuple[Optional[str], Optional[datetime]]:
        """
        Determines if the visit includes a discharge event.

        Returns:
            bool: True if a discharge event is found, False otherwise.
        """
        for event in self.events:
            for matching_rule in self.conversion.get_discharge_matching_rules():
                if re.match(matching_rule, event.code):
                    return event.code, event.time
        return None, None

    def _convert_event(self, event) -> List[Event]:
        """
        Converts a medical event into a list of CEHR-BERT-compatible events, potentially parsing.

        numeric values from text-based events.

        Args:
            event (meds_reader.Event): The medical event to be converted.

        Returns:
            List[Event]: A list of converted events, possibly numeric, based on the original event's code and value.
        """
        code = event.code
        time = getattr(event, "time", None)
        text_value = getattr(event, "text_value", None)
        numeric_value = getattr(event, "numeric_value", None)
        unit = getattr(event, "unit", None)

        if numeric_value is None and text_value is not None:
            conversion_rule = self.conversion.get_text_event_to_numeric_events_rule(code)
            if conversion_rule:
                match = re.search(conversion_rule.parsing_pattern, text_value)
                if match:
                    if len(match.groups()) == len(conversion_rule.mapped_event_labels):
                        events = [
                            Event(
                                code=label,
                                time=time,
                                numeric_value=float(value),
                                unit=unit,
                                properties={"visit_id": self.visit_id, "table": "meds"},
                            )
                            for label, value in zip(conversion_rule.mapped_event_labels, match.groups())
                            if value.isnumeric()
                        ]
                        return events

        return [
            Event(
                code=code,
                time=time,
                numeric_value=numeric_value,
                unit=unit,
                text_value=text_value,
                properties={"visit_id": self.visit_id, "table": "meds"},
            )
        ]

    def get_visit_start_datetime(self) -> datetime:
        return self.block_start_time

    def get_visit_end_datetime(self) -> datetime:
        return self.block_end_time

    def get_meds_events(self) -> Iterable[Event]:
        """
        Retrieves all medication events for the visit, converting each raw event if necessary.

        Returns:
            Iterable[Event]: A list of CEHR-BERT-compatible medication events for the visit.
        """
        events = []
        for e in self.events:
            if self.conversion.meds_exclude_tables:
                table = getattr(e, "table", None)
                if table and (table in self.conversion.meds_exclude_tables):
                    continue
            # We only convert the events that are not visit type and discharge facility events
            if (e.code == self.visit_type) or (self.discharged_to is not None and e.code == self.discharged_to):
                continue
            events.extend(self._convert_event(e))
        return events


def generate_demographics_and_patient_blocks(
    conversion: MedsToCehrBertConversion,
    patient: meds_reader.Subject,
    prediction_time: datetime = None,
) -> Tuple[PatientDemographics, List[PatientBlock]]:
    if isinstance(conversion, MedsToBertMimic4):
        return mimic_meds_generate_demographics_and_patient_blocks(
            patient, conversion, prediction_time, conversion.default_visit_id
        )
    elif isinstance(conversion, MedsToCehrbertOMOP):
        return omop_meds_generate_demographics_and_patient_blocks(patient, conversion, prediction_time)
    else:
        raise RuntimeError(f"Unrecognized conversion: {conversion}")


def omop_meds_generate_demographics_and_patient_blocks(
    patient: meds_reader.Subject, conversion: MedsToCehrBertConversion, prediction_time: datetime = None
) -> Tuple[PatientDemographics, List[PatientBlock]]:
    disconnect_problem_list_events = getattr(conversion, "disconnect_problem_list_events", False)
    birth_datetime = None
    race = None
    gender = None
    ethnicity = None
    visit_events = defaultdict(list)
    unlinked_event_mapping = defaultdict(list)
    for e in patient.events:
        # This indicates demographics features
        event_code_uppercase = e.code.upper()
        if event_code_uppercase.startswith(birth_code):
            birth_datetime = e.time
        elif event_code_uppercase.startswith("RACE"):
            race = e.code
        elif event_code_uppercase.startswith("GENDER"):
            gender = e.code
        elif event_code_uppercase.startswith("ETHNICITY"):
            ethnicity = e.code
        elif e.time is not None:
            # Skip out of the loop if the events' time stamps are beyond the prediction time
            if prediction_time is not None:
                if e.time > prediction_time:
                    break

            if getattr(e, "visit_id", None):
                visit_id = e.visit_id
                visit_events[visit_id].append(e)
            else:
                unlinked_event_mapping[e.time.strftime("%Y-%m-%d")].append(e)

    # We create patient blocks (placeholders for visits), we need to disassociate the problem list records from the
    # corresponding visit otherwise this will mess up the patient timeline
    if disconnect_problem_list_events:
        patient_block_mapping = dict()
        for visit_id, events in visit_events.items():
            patient_block = PatientBlock(events=events, visit_id=visit_id, conversion=conversion)
            updated_events = []
            for e in patient_block.events:
                # We need to handle the problem list here, because those records could occur years before the current visit
                # we use one day as the threshold to disconnect the records from the visit.  For some drug records, they
                # could occur years after the visit, we need to disconnect such records from the visit as well.
                if (patient_block.block_start_time - e.time).days > 1:
                    unlinked_event_mapping[e.time.strftime("%Y-%m-%d")].append(e)
                elif patient_block.block_end_time and (e.time - patient_block.block_end_time).days > 1:
                    unlinked_event_mapping[e.time.strftime("%Y-%m-%d")].append(e)
                else:
                    updated_events.append(e)
            patient_block.events = updated_events
            patient_block_mapping[visit_id] = patient_block
    else:
        patient_block_mapping = {
            visit_id: PatientBlock(events=events, visit_id=visit_id, conversion=conversion)
            for visit_id, events in visit_events.items()
        }

    # Try to connect the unlinked events to existing visits
    for current_date_str in list(unlinked_event_mapping.keys()):
        current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
        for visit_id, patient_block in patient_block_mapping.items():
            if (
                patient_block.get_visit_start_datetime().date()
                <= current_date
                <= patient_block.get_visit_end_datetime().date()
            ):
                patient_block.events.extend(unlinked_event_mapping.pop(current_date_str, []))
                # Need to sort the events if we insert new events to the patient block
                patient_block.events = sorted(patient_block.events, key=lambda _: [_.time, _.code])
                break

    # For the orphan records, we create artificial visits for them
    max_visit_id = max(patient_block_mapping.keys()) + 1 if len(patient_block_mapping) > 0 else 1
    for events in unlinked_event_mapping.values():
        patient_block_mapping[max_visit_id] = PatientBlock(events, max_visit_id, conversion)
        max_visit_id += 1

    patient_blocks = list(patient_block_mapping.values())
    demographics = PatientDemographics(birth_datetime=birth_datetime, race=race, gender=gender, ethnicity=ethnicity)

    # If there are unlinked events, we have added them as new patient blocks, therefore we need to re-order the patient block
    if len(unlinked_event_mapping) > 0:
        patient_blocks = sorted(patient_block_mapping.values(), key=lambda block: block.block_start_time)

    merged_patient_blocks = merge_patient_blocks(patient, patient_blocks)
    return demographics, merged_patient_blocks


def merge_patient_blocks(patient: meds_reader.Subject, patient_blocks: List[PatientBlock]) -> List[PatientBlock]:
    """
    Merge patient blocks where one visit completely contains another visit.

    This function merges PatientBlock objects when one block's time range fully
    contains another block. When a merge occurs, events from the contained block
    are added to the containing block, and the contained block is removed from
    the final result.

    The algorithm assumes that patient_blocks are already sorted chronologically
    by their min_time.

    Args:
        patient (meds_reader.Subject): The patient subject whose blocks are being merged
        patient_blocks (List[PatientBlock]): A list of PatientBlock objects to process

    Returns:
        List[PatientBlock]: A new list with merged PatientBlock objects

    Example:
        Given these blocks:
        1. INPATIENT: Jan 1-10
        2. OUTPATIENT: Jan 3-5 (contained within block 1)
        3. EMERGENCY: Jan 15-16 (not overlapping)

        The result would be:
        1. INPATIENT: Jan 1-10 (now contains events from block 2)
        2. EMERGENCY: Jan 15-16 (unchanged)
    """
    merging_info = []
    block_indices_to_skip = []
    merged_patient_blocks = []

    for prev_index in range(len(patient_blocks)):
        # Skip blocks that have already been merged into previous blocks
        if prev_index in block_indices_to_skip:
            continue

        prev_block = patient_blocks[prev_index]

        for next_index in range(prev_index + 1, len(patient_blocks)):
            next_block = patient_blocks[next_index]

            # Check if prev_block completely contains next_block:
            # [prev_block_start] [next_block_start] [next_block_end] [prev_block_end]
            if (
                prev_block.block_start_time <= next_block.block_start_time
                and next_block.block_end_time <= prev_block.block_end_time
            ):
                # If the long visit is not an admission visit and the short visit is an admission visit, this could happen
                # when the hospitalization only lasts less than a day, and followed by an outpatient visit
                # the outpatient starts at midnight of the day simply because how it is generated by the system
                # and does not represent the true time stamps, we swap these two blocks
                if not prev_block.has_ed_or_hospital_admission and next_block.has_ed_or_hospital_admission:
                    prev_block, next_block = next_block, prev_block

                # Merge the events from next_block into prev_block
                for e in next_block.events:
                    # We don't want to take the visit type and discharge facility codes from the patient block that will be merged
                    if (e.code == next_block.visit_type) or (
                        next_block.discharged_to is not None and e.code == next_block.discharged_to
                    ):
                        continue
                    prev_block.events.append(e)

                # Sort the combined events chronologically and by code
                prev_block.events = sorted(prev_block.events, key=lambda _: [_.time, _.code])

                # Set longer visit to E-I
                if prev_block.has_admission and next_block.has_ed_admission:
                    prev_block.visit_type = DEFAULT_ED_TO_INPATIENT_CONCEPT_ID

                # Mark this block to be skipped in the outer loop
                block_indices_to_skip.append(next_index)

                # Record merge information for debugging
                merging_info.append(
                    (
                        prev_block.visit_type,
                        prev_block.block_start_time,
                        prev_block.block_end_time,
                        next_block.visit_type,
                        next_block.block_start_time,
                        next_block.block_end_time,
                    )
                )

        # Add the block (with any merged events) to the result
        merged_patient_blocks.append(prev_block)

    # Log debugging information about merges if any occurred
    if merging_info:
        debug_log = "\n"
        for prev_v, prev_min_t, prev_max_t, next_v, next_min_t, next_max_t in merging_info:
            debug_log += (
                f"{patient.subject_id}: visit {next_v} with {next_min_t} and {next_max_t} "
                f"has been merged into visit {prev_v} with {prev_min_t} and {prev_max_t}\n"
            )
        LOG.debug(debug_log)

    return merged_patient_blocks


def mimic_meds_generate_demographics_and_patient_blocks(
    patient: meds_reader.Subject,
    conversion: MedsToCehrBertConversion,
    prediction_time: datetime = None,
    default_visit_id: int = 1,
) -> Tuple[PatientDemographics, List[PatientBlock]]:
    birth_datetime = None
    race = None
    gender = None
    ethnicity = None

    visit_id = default_visit_id
    current_date = None
    events_for_current_date = []
    patient_blocks = []
    for e in patient.events:

        # Skip out of the loop if the events' time stamps are beyond the prediction time
        if prediction_time is not None and e.time is not None:
            if e.time > prediction_time:
                break

        # This indicates demographics features
        event_code_uppercase = e.code.upper()
        if event_code_uppercase.startswith(birth_code):
            birth_datetime = e.time
        elif event_code_uppercase.startswith("RACE"):
            race = e.code
        elif event_code_uppercase.startswith("GENDER"):
            gender = e.code
        elif event_code_uppercase.startswith("ETHNICITY"):
            ethnicity = e.code
        elif e.time is not None:
            if not current_date:
                current_date = e.time

            if current_date.date() == e.time.date():
                events_for_current_date.append(e)
            else:
                patient_blocks.append(PatientBlock(events_for_current_date, visit_id, conversion))
                events_for_current_date = [e]
                current_date = e.time
                visit_id += 1

    if events_for_current_date:
        patient_blocks.append(PatientBlock(events_for_current_date, visit_id, conversion))

    admit_discharge_pairs = []
    active_ed_index = None
    active_admission_index = None
    # |ED|24-hours|Admission| ... |Discharge| -> ED will be merged into the admission (within 24 hours)
    # |ED|25-hours|Admission| ... |Discharge| -> ED will NOT be merged into the admission
    # |Admission|ED| ... |Discharge| -> ED will be merged into the admission
    # |Admission|Admission|ED| ... |Discharge|
    #   -> The first admission will be ignored and turned into a separate visit
    #   -> The second Admission and ED will be merged
    for i, patient_block in enumerate(patient_blocks):
        # Keep track of the ED block when there is no on-going admission
        if patient_block.has_ed_admission and active_admission_index is None:
            active_ed_index = i
        # Keep track of the admission block
        if patient_block.has_admission:
            # If the ED event has occurred, we need to check the time difference between
            # the ED event and the subsequent hospital admission
            if active_ed_index is not None:

                hour_diff = (patient_block.min_time - patient_blocks[active_ed_index].max_time).total_seconds() / 3600
                # If the time difference between the ed and admission is leq 24 hours,
                # we consider ED to be part of the visits
                if hour_diff <= 24 or active_ed_index == i:
                    active_admission_index = active_ed_index
                    active_ed_index = None
            else:
                active_admission_index = i

        if patient_block.has_discharge:
            if active_admission_index is not None:
                admit_discharge_pairs.append((active_admission_index, i))
            # When the patient is discharged from the hospital, we assume the admission and ED should end
            active_admission_index = None
            active_ed_index = None

        # Check the last block of the patient history to see whether the admission is partial
        if i == len(patient_blocks) - 1:
            # This indicates an ongoing (incomplete) inpatient visit,
            # this is a common pattern for inpatient visit prediction problems,
            # where the data from the first 24-48 hours after the admission
            # are used to predict something about the admission
            if active_admission_index is not None and prediction_time is not None:
                admit_discharge_pairs.append((active_admission_index, i))

    # Update visit_id for the admission blocks
    for admit_index, discharge_index in admit_discharge_pairs:
        admission_block = patient_blocks[admit_index]
        discharge_block = patient_blocks[discharge_index]
        visit_id = admission_block.visit_id
        for i in range(admit_index, discharge_index + 1):
            patient_blocks[i].visit_id = visit_id
            patient_blocks[i].visit_type = DEFAULT_INPATIENT_CONCEPT_ID
        # There could be events that occur after the discharge, which are considered as part of the visit
        # we need to check if the time stamp of the next block is within 12 hours
        if discharge_index + 1 < len(patient_blocks):
            next_block = patient_blocks[discharge_index + 1]
            hour_diff = (next_block.min_time - discharge_block.max_time).total_seconds() / 3600
            assert hour_diff >= 0, (
                f"next_block.min_time: {next_block.min_time} "
                f"must be GE discharge_block.max_time: {discharge_block.max_time}"
            )
            if hour_diff <= 12:
                next_block.visit_id = visit_id
                next_block.visit_type = DEFAULT_INPATIENT_CONCEPT_ID

    demographics = PatientDemographics(birth_datetime=birth_datetime, race=race, gender=gender, ethnicity=ethnicity)
    return demographics, patient_blocks
