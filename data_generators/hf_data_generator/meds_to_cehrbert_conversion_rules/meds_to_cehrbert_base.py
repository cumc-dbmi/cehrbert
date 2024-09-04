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
        self._text_event_numeric_event_map = {
            r.code: r for r in self._text_event_to_numeric_events()
        }

    @abstractmethod
    def get_ed_admission_matching_rules(self) -> List[str]:
        raise NotImplementedError("Must implement the matching rules for identifying the ED admission")

    @abstractmethod
    def get_admission_matching_rules(self) -> List[str]:
        raise NotImplementedError("Must implement the matching rules for identifying the admission")

    @abstractmethod
    def get_discharge_matching_rules(self) -> List[str]:
        raise NotImplementedError("Must implement the matching rules for identifying the discharge")

    @abstractmethod
    def _text_event_to_numeric_events(self) -> List[EventConversionRule]:
        raise NotImplementedError(
            "Must implement the event mapping rules for converting the text events to numeric events"
        )

    def get_text_event_to_numeric_events_rule(self, code) -> Optional[EventConversionRule]:
        if code in self._text_event_numeric_event_map:
            return self._text_event_numeric_event_map[code]
        return None

    @abstractmethod
    def get_text_value_as_text_event(self) -> List[str]:
        raise NotImplementedError(
            "Must implement the event mapping rules for extracting the text_values as the events"
        )
