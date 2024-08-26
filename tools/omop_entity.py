from abc import ABC, abstractmethod
from datetime import datetime, date


def fill_datetime(year: int):
    return datetime.strptime(str(year) + '-01' + '-01', '%Y-%m-%d').isoformat()


def fill_start_datetime(date: date):
    return datetime.combine(date, datetime.min.time()).isoformat()


def fill_end_datetime(date: date):
    return datetime.combine(date, datetime.max.time()).isoformat()


class OmopEntity(ABC):
    @abstractmethod
    def export_as_json(self):
        pass

    @classmethod
    @abstractmethod
    def get_schema(cls):
        pass

    @abstractmethod
    def get_table_name(self):
        pass


class Person(OmopEntity):
    def __init__(
            self,
            person_id,
            gender_concept_id,
            year_of_birth,
            race_concept_id
    ):
        self._person_id = person_id
        self._gender_concept_id = gender_concept_id
        self._year_of_birth = year_of_birth
        self._race_concept_id = race_concept_id

    def export_as_json(self):
        return {
            'person_id': self._person_id,
            'gender_concept_id': self._gender_concept_id,
            'year_of_birth': self._year_of_birth,
            'month_of_birth': 1,
            'day_of_birth': 1,
            'birth_datetime': fill_datetime(self._year_of_birth),
            'race_concept_id': self._race_concept_id,
            'ethnicity_concept_id': 0,
            'location_id': 0,
            'provider_id': 0,
            'care_site_id': 0,
            'person_source_value': '',
            'gender_source_value': '',
            'gender_souce_concept_id': 0,
            'race_source_value': '',
            'race_source_concept_id': 0,
            'ethnicity_source_value': '',
            'ethnicity_source_concept_id': 0
        }

    @classmethod
    def get_schema(cls):
        return {
            'person_id': int,
            'gender_concept_id': int,
            'year_of_birth': int,
            'month_of_birth': int,
            'day_of_birth': int,
            'birth_datetime': datetime,
            'race_concept_id': int,
            'ethnicity_concept_id': int,
            'location_id': int,
            'provider_id': int,
            'care_site_id': int,
            'person_source_value': str,
            'gender_source_value': str,
            'gender_souce_concept_id': int,
            'race_source_value': str,
            'race_source_concept_id': int,
            'ethnicity_source_value': str,
            'ethnicity_source_concept_id': int
        }

    def get_table_name(self):
        return 'person'


class VisitOccurrence(OmopEntity):
    def __init__(
            self,
            visit_occurrence_id: int,
            visit_concept_id: int,
            visit_start_date: date,
            person: Person,
            discharged_to_concept_id: int = 0
    ):
        self._visit_occurrence_id = visit_occurrence_id
        self._visit_concept_id = visit_concept_id
        self._visit_start_date = visit_start_date
        self._visit_start_datetime = fill_start_datetime(self._visit_start_date)
        self._visit_end_date = visit_start_date
        self._visit_end_datetime = fill_end_datetime(self._visit_end_date)
        self._person = person
        self._discharged_to_concept_id = discharged_to_concept_id

    def export_as_json(self):
        return {
            'visit_occurrence_id': self._visit_occurrence_id,
            'visit_concept_id': self._visit_concept_id,
            'person_id': self._person._person_id,
            'visit_start_date': self._visit_start_date,
            'visit_start_datetime': self._visit_start_datetime,
            'visit_end_date': self._visit_end_date,
            'visit_end_datetime': self._visit_end_datetime,
            'visit_type_concept_id': 44818702,
            'provider_id': 0,
            'care_site_id': 0,
            'visit_source_value': '',
            'visit_source_concept_id': self._visit_concept_id,
            'admitting_source_concept_id': 0,
            'admitting_source_value': '',
            'discharged_to_concept_id': self._discharged_to_concept_id,
            'discharged_to_source_value': '',
            'preceding_visit_occurrence_id': 0
        }

    @classmethod
    def get_schema(cls):
        return {
            'visit_occurrence_id': int,
            'visit_concept_id': int,
            'person_id': int,
            'visit_start_date': date,
            'visit_start_datetime': datetime,
            'visit_end_date': date,
            'visit_end_datetime': datetime,
            'visit_type_concept_id': int,
            'provider_id': int,
            'care_site_id': int,
            'visit_source_value': str,
            'visit_source_concept_id': int,
            'admitting_source_concept_id': int,
            'admitting_source_value': str,
            'discharged_to_concept_id': int,
            'discharged_to_source_value': str,
            'preceding_visit_occurrence_id': int
        }

    def get_table_name(self):
        return 'visit_occurrence'

    @property
    def person(self):
        return self._person

    @property
    def discharged_to_concept_id(self):
        return self._discharged_to_concept_id

    # a setter function
    def set_discharged_to_concept_id(self, discharged_to_concept_id):
        self._discharged_to_concept_id = discharged_to_concept_id

    def set_visit_end_date(self, visit_end_date):
        self._visit_end_date = visit_end_date


class ConditionOccurrence(OmopEntity):

    def __init__(
            self,
            condition_occurrence_id,
            condition_concept_id,
            visit_occurrence: VisitOccurrence,
            condition_date: date,
    ):
        self._condition_occurrence_id = condition_occurrence_id
        self._condition_concept_id = condition_concept_id
        self._visit_occurrence = visit_occurrence
        self._condition_start_date = condition_date
        self._condition_start_datetime = fill_start_datetime(condition_date)
        self._condition_end_date = condition_date
        self._condition_end_datetime = fill_end_datetime(condition_date)

    def export_as_json(self):
        return {
            'condition_occurrence_id': self._condition_occurrence_id,
            'person_id': self._visit_occurrence._person._person_id,
            'condition_concept_id': self._condition_concept_id,
            'condition_start_date': self._condition_start_date,
            'condition_start_datetime': self._condition_start_datetime,
            'condition_end_date': self._condition_end_date,
            'condition_end_datetime': self._condition_end_datetime,
            'condition_type_concept_id': 32817,
            'stop_reason': '',
            'provider_id': 0,
            'visit_occurrence_id': self._visit_occurrence._visit_occurrence_id,
            'visit_detail_id': 0,
            'condition_source_value': '',
            'condition_source_concept_id': self._condition_concept_id,
            'condition_status_source_value': '',
            'condition_status_concept_id': 0
        }

    @classmethod
    def get_schema(cls):
        return {
            'condition_occurrence_id': int,
            'person_id': int,
            'condition_concept_id': int,
            'condition_start_date': date,
            'condition_start_datetime': datetime,
            'condition_end_date': date,
            'condition_end_datetime': datetime,
            'condition_type_concept_id': int,
            'stop_reason': str,
            'provider_id': int,
            'visit_occurrence_id': int,
            'visit_detail_id': int,
            'condition_source_value': str,
            'condition_source_concept_id': int,
            'condition_status_source_value': str,
            'condition_status_concept_id': int
        }

    def get_table_name(self):
        return 'condition_occurrence'


class DrugExposure(OmopEntity):

    def __init__(
            self,
            drug_exposure_id,
            drug_concept_id,
            visit_occurrence: VisitOccurrence,
            drug_date: date
    ):
        self._drug_exposure_id = drug_exposure_id
        self._drug_concept_id = drug_concept_id
        self._visit_occurrence = visit_occurrence
        self._drug_exposure_start_date = drug_date
        self._drug_exposure_start_datetime = fill_start_datetime(drug_date)
        self._drug_exposure_end_date = drug_date
        self._drug_exposure_end_datetime = fill_end_datetime(drug_date)

    def export_as_json(self):
        return {
            'drug_exposure_id': self._drug_exposure_id,
            'person_id': self._visit_occurrence._person._person_id,
            'drug_concept_id': self._drug_concept_id,
            'drug_exposure_start_date': self._drug_exposure_start_date,
            'drug_exposure_start_datetime': self._drug_exposure_start_datetime,
            'drug_exposure_end_date': self._drug_exposure_end_date,
            'drug_exposure_end_datetime': self._drug_exposure_end_datetime,
            'verbatim_end_date': self._drug_exposure_end_date,
            'drug_type_concept_id': 38000177,
            'stop_reason': '',
            'refills': None,
            'quantity': None,
            'days_supply': None,
            'sig': '',
            'route_concept_id': 0,
            'lot_number': '',
            'provider_id': 0,
            'visit_occurrence_id': self._visit_occurrence._visit_occurrence_id,
            'visit_detail_id': 0,
            'drug_source_value': '',
            'drug_source_concept_id': self._drug_concept_id,
            'route_source_value': '',
            'dose_unit_source_value': ''
        }

    @classmethod
    def get_schema(cls):
        return {
            'drug_exposure_id': int,
            'person_id': int,
            'drug_concept_id': int,
            'drug_exposure_start_date': date,
            'drug_exposure_start_datetime': datetime,
            'drug_exposure_end_date': date,
            'drug_exposure_end_datetime': datetime,
            'verbatim_end_date': date,
            'drug_type_concept_id': int,
            'stop_reason': str,
            'refills': int,
            'quantity': int,
            'days_supply': int,
            'sig': str,
            'route_concept_id': int,
            'lot_number': str,
            'provider_id': int,
            'visit_occurrence_id': int,
            'visit_detail_id': int,
            'drug_source_value': str,
            'drug_source_concept_id': int,
            'route_source_value': str,
            'dose_unit_source_value': str
        }

    def get_table_name(self):
        return 'drug_exposure'


class ProcedureOccurrence(OmopEntity):

    def __init__(
            self,
            procedure_occurrence_id,
            procedure_concept_id,
            visit_occurrence: VisitOccurrence,
            procedure_date: date
    ):
        self._procedure_occurrence_id = procedure_occurrence_id
        self._procedure_concept_id = procedure_concept_id
        self._visit_occurrence = visit_occurrence
        self._procedure_date = procedure_date
        self._procedure_datetime = fill_start_datetime(procedure_date)

    def export_as_json(self):
        return {
            'procedure_occurrence_id': self._procedure_occurrence_id,
            'person_id': self._visit_occurrence._person._person_id,
            'procedure_concept_id': self._procedure_concept_id,
            'procedure_date': self._procedure_date,
            'procedure_datetime': self._procedure_datetime,
            'procedure_type_concept_id': 38000178,
            'modifier_concept_id': 0,
            'quantity': 1,
            'provider_id': 0,
            'visit_occurrence_id': self._visit_occurrence._visit_occurrence_id,
            'visit_detail_id': 0,
            'procedure_source_value': '',
            'procedure_source_concept_id': self._procedure_concept_id,
            'qualifier_source_value': ''
        }

    @classmethod
    def get_schema(cls):
        return {
            'procedure_occurrence_id': int,
            'person_id': int,
            'procedure_concept_id': int,
            'procedure_date': date,
            'procedure_datetime': datetime,
            'procedure_type_concept_id': int,
            'modifier_concept_id': int,
            'quantity': int,
            'provider_id': int,
            'visit_occurrence_id': int,
            'visit_detail_id': int,
            'procedure_source_value': str,
            'procedure_source_concept_id': int,
            'qualifier_source_value': str
        }

    def get_table_name(self):
        return 'procedure_occurrence'


class Death(OmopEntity):
    def __init__(
            self,
            person,
            death_date: date,
            death_type_concept_id=0,
    ):
        self._person = person
        self._death_date = death_date
        self._death_datetime = fill_end_datetime(death_date)
        self._death_type_concept_id = death_type_concept_id

    def export_as_json(self):
        return {
            'person_id': self._person._person_id,
            'death_date': self._death_date,
            'death_datetime': self._death_datetime,
            'death_type_concept_id': self._death_type_concept_id,
            'cause_concept_id': 0,
            'cause_source_value': '',
            'cause_source_concept_id': 0
        }

    @classmethod
    def get_schema(cls):
        return {
            'person_id': int,
            'death_date': date,
            'death_datetime': datetime,
            'death_type_concept_id': int,
            'cause_concept_id': int,
            'cause_source_value': str,
            'cause_source_concept_id': int
        }

    def get_table_name(self):
        return 'death'


class Measurement(OmopEntity):

    def __init__(
            self,
            measurement_id,
            measurement_concept_id,
            value_as_number,
            visit_occurrence: VisitOccurrence,
            measurement_date: date
    ):
        self._measurement_id = measurement_id
        self._measurement_concept_id = measurement_concept_id
        self._value_as_number = value_as_number
        self._visit_occurrence = visit_occurrence
        self._measurement_date = measurement_date
        self._measurement_datetime = fill_start_datetime(measurement_date)

    def export_as_json(self):
        return {
            'measurement_id': self._measurement_id,
            'person_id': self._visit_occurrence._person._person_id,
            'measurement_concept_id': self._measurement_concept_id,
            'measurement_date': self._measurement_date,
            'measurement_datetime': self._measurement_datetime,
            'value_as_number': self._value_as_number,
            'operator_concept_id': 4172703,
            'provider_id': 0,
            'visit_occurrence_id': self._visit_occurrence._visit_occurrence_id,
            'visit_detail_id': 0,
            'measurement_source_value': '',
            'measurement_source_concept_id': self._measurement_concept_id
        }

    @classmethod
    def get_schema(cls):
        return {
            'measurement_id': int,
            'person_id': int,
            'measurement_concept_id': int,
            'measurement_date': date,
            'measurement_datetime': datetime,
            'value_as_number': float,
            'operator_concept_id': int,
            'provider_id': int,
            'visit_occurrence_id': int,
            'visit_detail_id': int,
            'measurement_source_value': str,
            'measurement_source_concept_id': int
        }

    def get_table_name(self):
        return 'measurement'
