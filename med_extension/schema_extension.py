from meds.schema import Event

from typing import TypedDict, List
from typing_extensions import NotRequired
import datetime


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
