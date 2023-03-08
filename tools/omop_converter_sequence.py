import sys
from pathlib import Path
import math
from datetime import datetime, date, timedelta

import csv
# from pyspark.sql import SparkSession
from models.gpt_model import generate_artificial_time_tokens


from abc import ABC, abstractmethod


def fill_datetime(year: int):
    return datetime.strptime(str(year) + '-01' + '-01', '%Y-%m-%d')


def fill_start_datetime(date: date):
    return datetime.combine(date, datetime.min.time()),


def fill_end_datetime(date: date):
    return datetime.combine(date, datetime.max.time()),


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
            visit_occurrence_id,
            visit_concept_id,
            visit_start_date,
            person: Person
    ):
        self._visit_occurrence_id = visit_occurrence_id
        self._visit_concept_id = visit_concept_id
        self._visit_start_date = visit_start_date
        self._visit_start_datetime = fill_start_datetime(self._visit_start_date)
        self._visit_end_date = visit_start_date
        self._visit_end_datetime = fill_end_datetime(self._visit_end_date)
        self._person = person

    def export_as_json(self):
        return {
            'visit_occurrence_id': self._visit_occurrence_id,
            'visit_concept_id': self._visit_concept_id,
            'person_id': self._person.self._person_id,
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
            'discharge_to_concept_id': 0,
            'discharge_to_source_value': '',
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
            'discharge_to_concept_id': int,
            'discharge_to_source_value': str,
            'preceding_visit_occurrence_id': int
        }

    def get_table_name(self):
        return 'visit_occurrence'

    @property
    def person(self):
        return self._person


class ConditionOccurrence(OmopEntity):

    def __init__(
            self,
            condition_occurrence_id,
            condition_concept_id,
            visit_occurrence: VisitOccurrence
    ):
        self._condition_occurrence_id = condition_occurrence_id
        self._condition_concept_id = condition_concept_id
        self._visit_occurrence = visit_occurrence

    def export_as_json(self):
        return {
            'condition_occurrence_id': self._condition_occurrence_id,
            'person_id': self._visit_occurrence._person._person_id,
            'condition_concept_id': self._condition_concept_id,
            'condition_start_date': self._visit_occurrence._visit_start_date,
            'condition_start_datetime': self._visit_occurrence._visit_start_datetime,
            'condition_end_date': self._visit_occurrence._visit_end_date,
            'condition_end_datetime': self._visit_occurrence._visit_end_datetime,
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