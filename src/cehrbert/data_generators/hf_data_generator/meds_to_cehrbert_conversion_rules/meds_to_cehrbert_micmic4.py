import re
from datetime import datetime
from typing import List, Tuple

import meds_reader

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import birth_codes
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)
from cehrbert.data_generators.hf_data_generator.meds_utils import PatientBlock, PatientDemographics


class MedsToBertMimic4(MedsToCehrBertConversion):

    def __init__(self, default_visit_id):
        super().__init__()
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
