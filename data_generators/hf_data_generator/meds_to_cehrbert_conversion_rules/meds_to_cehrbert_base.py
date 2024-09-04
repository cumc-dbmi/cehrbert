import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EventConversionRule:
    code: str
    parsing_pattern: re.Pattern
    mapped_event_labels: List[str]

    def __post_init__(self):
        assert self.parsing_pattern.groups == len(self.mapped_event_labels), \
            "The number the mapped event labels needs to match the number of groups in the regex"


class MedsToCehrBertConversion(ABC):
    def __init__(self):
        # Cache these variables once
        self._ed_admission_matching_rules = self._create_ed_admission_matching_rules()
        self._admission_matching_rules = self._create_admission_matching_rules()
        self._discharge_matching_rules = self._create_discharge_matching_rules()
        self._text_event_numeric_event_map = {
            r.code: r for r in self._create_text_event_to_numeric_event_rules()
        }
        self._open_ended_event_codes = self._create_open_ended_event_codes()

    @abstractmethod
    def _create_ed_admission_matching_rules(self) -> List[str]:
        raise NotImplementedError("Must implement the matching rules for identifying the ED admission")

    @abstractmethod
    def _create_admission_matching_rules(self) -> List[str]:
        raise NotImplementedError("Must implement the matching rules for identifying the admission")

    @abstractmethod
    def _create_discharge_matching_rules(self) -> List[str]:
        raise NotImplementedError("Must implement the matching rules for identifying the discharge")

    @abstractmethod
    def _create_text_event_to_numeric_event_rules(self) -> List[EventConversionRule]:
        raise NotImplementedError(
            "Must implement the event mapping rules for converting the text events to numeric events"
        )

    @abstractmethod
    def _create_open_ended_event_codes(self) -> List[str]:
        raise NotImplementedError(
            "Must implement the event mapping rules for extracting the text_values as the event codes"
        )

    def get_ed_admission_matching_rules(self) -> List[str]:
        return self._ed_admission_matching_rules

    def get_admission_matching_rules(self) -> List[str]:
        return self._admission_matching_rules

    def get_discharge_matching_rules(self) -> List[str]:
        return self._discharge_matching_rules

    def get_text_event_to_numeric_events_rule(self, code) -> Optional[EventConversionRule]:
        if code in self._text_event_numeric_event_map:
            return self._text_event_numeric_event_map[code]
        return None

    def get_open_ended_event_codes(self) -> List[str]:
        return self._open_ended_event_codes
