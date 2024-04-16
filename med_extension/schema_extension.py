from meds.schema import Event, Measurement

from typing import TypedDict, List
from typing_extensions import NotRequired
import datetime
import itertools


class Visit(TypedDict):
    visit_type: str
    visit_start_datetime: datetime.datetime
    visit_end_datetime: NotRequired[datetime.datetime]
    discharge_facility: NotRequired[str]
    events: List[Event]


class PatientExtension(TypedDict):
    patient_id: int
    static_measurements: List[Measurement]
    birth_datetime: datetime.datetime
    gender: str
    race: str
    visits: List[Visit]


def get_measurements_from_visit(visit: Visit) -> List[Measurement]:
    measurements = []
    for event in visit['events']:
        measurements.extend(event['measurements'])
    return measurements
