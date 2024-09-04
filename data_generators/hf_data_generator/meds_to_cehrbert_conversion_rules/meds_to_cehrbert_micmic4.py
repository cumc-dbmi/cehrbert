import re
from typing import List

from data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules.meds_to_cehrbert_base import (
    MedsToCehrBertConversion, EventConversionRule
)


class MedsToBertMimic4(MedsToCehrBertConversion):

    def get_ed_admission_matching_rules(self) -> List[str]:
        return ["ED_REGISTRATION", "TRANSFER_TO//ED"]

    def get_admission_matching_rules(self) -> List[str]:
        return ["HOSPITAL_ADMISSION"]

    def get_discharge_matching_rules(self) -> List[str]:
        return ["HOSPITAL_DISCHARGE"]

    def _text_event_to_numeric_events(self) -> List[EventConversionRule]:
        return [
            EventConversionRule(
                code="Blood Pressure Standing",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["Standing SBP", "Standing DBP"]
            ),
            EventConversionRule(
                code="Blood Pressure Standing (1 min)",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["SBP Standing (1 min)", "DBP Standing (1 min)"]
            ),
            EventConversionRule(
                code="Blood Pressure Sitting",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["SBP Sitting", "DBP Sitting"]
            ),
            EventConversionRule(
                code="Blood Pressure Lying",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["SBP Lying", "DBP Lying"]
            ),
            EventConversionRule(
                code="Blood Pressure Standing (3 mins)",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["SBP Standing (3 mins)", "DBP Standing (3 mins)"]
            ),
            EventConversionRule(
                code="Blood Pressure",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["SBP", "DBP"]
            ),
            EventConversionRule(
                code="LAB//50827//UNK",
                parsing_pattern=re.compile(r"(\d+)/(\d+)"),
                mapped_event_labels=["LAB//50827//UNK//1", "LAB//50827//UNK//2"]
            )
        ]

    def get_text_value_as_text_event(self) -> List[str]:
        return ["LAB//220001//UNK"]
