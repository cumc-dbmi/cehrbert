import datetime
from typing import Any, List, Mapping, Optional, TypedDict, Union

from typing_extensions import NotRequired

Event = TypedDict(
    "Event",
    {
        "time": NotRequired[datetime.datetime],
        "code": str,
        "text_value": NotRequired[Optional[str]],
        "numeric_value": NotRequired[Optional[float]],
        "unit": NotRequired[Optional[str]],
        "datetime_value": NotRequired[datetime.datetime],
        "properties": NotRequired[Optional[Mapping[str, Any]]],
    },
)


class Visit(TypedDict):
    visit_type: str
    visit_start_datetime: datetime.datetime
    visit_end_datetime: NotRequired[datetime.datetime]
    discharge_facility: NotRequired[str]
    events: List[Event]


class CehrBertPatient(TypedDict):
    patient_id: int
    birth_datetime: datetime.datetime
    gender: str
    race: str
    ethnicity: str
    visits: List[Visit]
    index_date: datetime.datetime
    age_at_index: int
    label: Union[int, float]
