from datetime import datetime
from typing import List, Tuple

import meds_reader

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import birth_codes
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    EventConversionRule,
    MedsToCehrBertConversion,
)
from cehrbert.data_generators.hf_data_generator.meds_utils import PatientBlock, PatientDemographics


class MedsToCehrbertOMOP(MedsToCehrBertConversion):

    def generate_demographics_and_patient_blocks(
        self, patient: meds_reader.Subject, prediction_time: datetime = None
    ) -> Tuple[PatientDemographics, List[PatientBlock]]:

        birth_datetime = None
        race = None
        gender = None
        ethnicity = None

        current_visit_id = None
        current_date = None
        events_for_current_date = []
        patient_blocks = []
        for e in patient.events:

            # Skip out of the loop if the events' time stamps are beyond the prediction time
            if prediction_time is not None and e.time is not None:
                if e.time > prediction_time:
                    break

            # Try to set current_visit_id
            if not current_visit_id:
                current_visit_id = e.visit_id if hasattr(e, "visit_id") else None

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
                    patient_blocks.append(PatientBlock(events_for_current_date, current_visit_id, self))
                    events_for_current_date = [e]
                    current_date = e.time

        if events_for_current_date:
            patient_blocks.append(PatientBlock(events_for_current_date, current_visit_id, self))

        demographics = PatientDemographics(birth_datetime=birth_datetime, race=race, gender=gender, ethnicity=ethnicity)
        return demographics, patient_blocks

    def _create_ed_admission_matching_rules(self) -> List[str]:
        return ["Visit/ER"]

    def _create_admission_matching_rules(self) -> List[str]:
        return ["Visit/IP", "Visit/ERIP", "CMS Place of Service/51", "CMS Place of Service/61"]

    def _create_discharge_matching_rules(self) -> List[str]:
        return [
            "PCORNet/Generic-NI",
            "CMS Place of Service/12",
            "SNOMED/371827001",
            "CMS Place of Service/21",
            "NUCC/261Q00000X",
            "CMS Place of Service/31",
            "SNOMED/397709008",
            "Medicare Specialty/A4",
            "SNOMED/225928004",
            "CMS Place of Service/34",
            "CMS Place of Service/61",
            "CMS Place of Service/51",
            "CMS Place of Service/23",
            "PCORNet/Generic-OT",
            "CMS Place of Service/27",
            "CMS Place of Service/24",
            "CMS Place of Service/09",
            "CMS Place of Service/33",
            "SNOMED/34596002",
            "CMS Place of Service/25",
            "CMS Place of Service/32",
            "CMS Place of Service/20",
        ]

    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        return []
