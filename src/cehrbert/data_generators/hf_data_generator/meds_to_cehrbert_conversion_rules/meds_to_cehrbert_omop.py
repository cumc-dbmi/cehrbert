from collections import defaultdict
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

        visit_events = defaultdict(list)
        unlinked_event_mapping = defaultdict(list)
        for e in patient.events:
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
                # Skip out of the loop if the events' time stamps are beyond the prediction time
                if prediction_time is not None:
                    if e.time > prediction_time:
                        break
                if hasattr(e, "visit_id"):
                    visit_id = e.visit_id
                    visit_events[visit_id].append(e)
                else:
                    unlinked_event_mapping[e.time.strftime("%Y-%m-%d")].append(e)

        patient_block_mapping = {
            visit_id: PatientBlock(events=events, visit_id=visit_id, conversion=self)
            for visit_id, events in visit_events.items()
        }

        # Try to connect the unlinked events to existing visits
        for current_date_str in list(unlinked_event_mapping.keys()):
            current_date = datetime.strptime(current_date_str, "%Y-%m-%d")
            for visit_id, patient_block in patient_block_mapping.items():
                if patient_block.min_time.date() <= current_date <= patient_block.max_time.date():
                    patient_block.events.extend(unlinked_event_mapping.pop(current_date_str, []))
                    # Need to sort the events if we insert new events to the patient block
                    patient_block.events = sorted(patient_block.events, key=lambda _: _.time)
                    break

        max_visit_id = max(patient_block_mapping.keys()) + 1
        for events in unlinked_event_mapping.values():
            patient_block_mapping[max_visit_id] = PatientBlock(events, max_visit_id, self)
            max_visit_id += 1

        patient_blocks = list(patient_block_mapping.values())
        demographics = PatientDemographics(birth_datetime=birth_datetime, race=race, gender=gender, ethnicity=ethnicity)

        # If there are unlinked events, we need to add them as new patient blocks, therefore we need to re-order the patient block
        if len(unlinked_event_mapping) > 0:
            patient_blocks = sorted(patient_block_mapping.values(), key=lambda block: block.min_time)

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
