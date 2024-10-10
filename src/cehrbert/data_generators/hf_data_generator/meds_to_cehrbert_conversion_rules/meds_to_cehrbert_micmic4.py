import re
from datetime import datetime
from typing import List, Tuple

import meds_reader

from cehrbert.data_generators.hf_data_generator import DEFAULT_INPATIENT_CONCEPT_ID
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import birth_codes
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)
from cehrbert.data_generators.hf_data_generator.meds_utils import PatientBlock, PatientDemographics


class MedsToBertMimic4(MedsToCehrBertConversion):

    def __init__(self, default_visit_id, **kwargs):
        super().__init__(**kwargs)
        self.default_visit_id = default_visit_id

    def generate_demographics_and_patient_blocks(
        self, patient: meds_reader.Subject, prediction_time: datetime = None
    ) -> Tuple[PatientDemographics, List[PatientBlock]]:

        birth_datetime = None
        race = None
        gender = None
        ethnicity = None

        visit_id = self.default_visit_id
        current_date = None
        events_for_current_date = []
        patient_blocks = []
        for e in patient.events:

            # Skip out of the loop if the events' time stamps are beyond the prediction time
            if prediction_time is not None and e.time is not None:
                if e.time > prediction_time:
                    break

            # This indicates demographics features
            if e.code in birth_codes:
                birth_datetime = e.time
            elif e.code.upper().startswith("RACE"):
                race = e.code
            elif e.code.upper().startswith("GENDER"):
                gender = e.code
            elif e.code.upper().startswith("ETHNICITY"):
                ethnicity = e.code
            elif e.time is not None:
                if not current_date:
                    current_date = e.time

                if current_date.date() == e.time.date():
                    events_for_current_date.append(e)
                else:
                    patient_blocks.append(PatientBlock(events_for_current_date, visit_id, self))
                    events_for_current_date = [e]
                    current_date = e.time
                    visit_id += 1

        if events_for_current_date:
            patient_blocks.append(PatientBlock(events_for_current_date, visit_id, self))

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

                    hour_diff = (
                        patient_block.min_time - patient_blocks[active_ed_index].max_time
                    ).total_seconds() / 3600
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

    def _create_ed_admission_matching_rules(self) -> List[str]:
        return ["ED_REGISTRATION//", "TRANSFER_TO//ED"]

    def _create_admission_matching_rules(self) -> List[str]:
        return ["HOSPITAL_ADMISSION//"]

    def _create_discharge_matching_rules(self) -> List[str]:
        return ["HOSPITAL_DISCHARGE//"]

    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        blood_pressure_codes = [
            "Blood Pressure",
            "Blood Pressure Lying",
            "Blood Pressure Sitting",
            "Blood Pressure Standing (1 min)",
            "Blood Pressure Standing (3 mins)",
        ]
        blood_pressure_rules = [
            EventConversionRule(
                code=code,
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=[f"Systolic {code}", f"Diastolic {code}"],
            )
            for code in blood_pressure_codes
        ]
        height_weight_codes = ["Weight (Lbs)", "Height (Inches)", "BMI (kg/m2)"]
        height_weight_rules = [
            EventConversionRule(
                code=code,
                parsing_pattern=re.compile(r"(\d+)"),
                mapped_event_labels=[code],
            )
            for code in height_weight_codes
        ]
        ventilation_rate_rules = [
            EventConversionRule(
                code="LAB//50827//UNK",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["LAB//50827//UNK//1", "LAB//50827//UNK//2"],
            )
        ]
        return blood_pressure_rules + height_weight_rules + ventilation_rate_rules
