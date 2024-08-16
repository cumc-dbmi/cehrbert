import os
import re
import collections
import functools
from typing import Dict, List, Optional

import meds_reader
import pandas as pd
from meds import Event

from runner.hf_runner_argument_dataclass import DataTrainingArguments
from data_generators.hf_data_generator.hf_dataset_mapping import birth_codes
from med_extension.schema_extension import CehrBertPatient, Visit

from datasets import Dataset, DatasetDict, IterableDataset


def get_patient_split(meds_reader_db_path: str) -> Dict[str, List[int]]:
    patient_split = pd.read_parquet(os.path.join(meds_reader_db_path, "metadata/patient_splits.parquet"))
    result = {
        str(group): records["patient_id"].tolist()
        for group, records in patient_split.groupby("split")
    }
    return result


class PatientBlock:
    def __init__(
            self,
            events: List[meds_reader.Event],
            visit_id: int
    ):
        self.visit_id = visit_id
        self.events = events
        self.min_time = events[0].time
        self.max_time = events[-1].time
        self.visit_type = "9202"
        self.visit_end_datetime = events[-1].time
        self.discharge_facility = None

    def has_admission(self) -> bool:
        for event in self.events:
            if 'HOSPITAL_ADMISSION' in event.code:
                return True
        return False

    def has_discharge(self) -> bool:
        for event in self.events:
            if 'HOSPITAL_DISCHARGE' in event.code:
                return True
        return False

    def get_discharge_facility(self) -> Optional[str]:
        if self.has_discharge():
            for event in self.events:
                if 'HOSPITAL_DISCHARGE' in event.code:
                    discharge_facility = event.code.replace('HOSPITAL_DISCHARGE', '')
                    discharge_facility = re.sub(r'[^a-zA-Z]', '', discharge_facility)
                    return discharge_facility
        return None

    def get_meds_events(self) -> List[Event]:
        return [
            Event(
                code=e.code.replace(' ', '_'),
                time=e.time,
                numeric_value=getattr(e, "numeric_value", None),
                text_value=getattr(e, "text_value", None),
                properties={'visit_id': self.visit_id, "table": "meds"}
            )
            for e in self.events
        ]


def convert_one_patient(patient: meds_reader.Patient, default_visit_id: int = 1) -> CehrBertPatient:
    birth_datetime = None
    race = None
    gender = None
    ethnicity = None

    visit_id = default_visit_id
    current_date = None
    events_for_current_date = []
    patient_blocks = []
    for e in patient.events:
        # This indicates demographics features
        if e.code in birth_codes:
            birth_datetime = e.time
        elif e.code.startswith('RACE'):
            race = e.code
        elif e.code.startswith('GENDER'):
            gender = e.code
        elif e.code.startswith('ETHNICITY'):
            ethnicity = e.code
        elif e.time is not None:
            if not current_date:
                current_date = e.time

            if current_date.date() == e.time.date():
                events_for_current_date.append(e)
            else:
                patient_blocks.append(PatientBlock(events_for_current_date, visit_id))
                events_for_current_date = list()
                events_for_current_date.append(e)
                current_date = e.time
                visit_id += 1

    if events_for_current_date:
        patient_blocks.append(PatientBlock(events_for_current_date, visit_id))

    admit_discharge_pairs = []
    admission_index = None
    for i, patient_block in enumerate(patient_blocks):
        if patient_block.has_admission():
            admission_index = i
        elif patient_block.has_discharge():
            if admission_index is not None:
                admit_discharge_pairs.append((admission_index, i))
            admission_index = None

    # Update visit_id for the admission blocks
    for admit_index, discharge_index in admit_discharge_pairs:
        admission_block = patient_blocks[admit_index]
        discharge_block = patient_blocks[discharge_index]
        admission_block.visit_end_datetime = discharge_block.max_time
        visit_id = admission_block.visit_id
        for i in range(admit_index, discharge_index + 1):
            patient_blocks[i].visit_id = visit_id
            patient_blocks[i].visit_type = "9201"
        # there could be events that occur after the discharge, which are considered as part of the visit
        # we need to check if the time stamp of the next block is within 12 hours
        if discharge_index + 1 < len(patient_blocks):
            next_block = patient_blocks[discharge_index + 1]
            hour_diff = (discharge_block.max_time - next_block.min_time).total_seconds() / 3600
            if hour_diff <= 12:
                next_block.visit_id = visit_id
                next_block.visit_type = "9201"
                admission_block.visit_end_datetime = next_block.max_time

    patient_block_dict = collections.defaultdict(list)
    for patient_block in patient_blocks:
        patient_block_dict[patient_block.visit_id].append(patient_block)

    visits = list()
    for visit_id, blocks in patient_block_dict.items():
        visit_type = blocks[0].visit_type
        visit_start_datetime = min([b.min_time for b in blocks])
        visit_end_datetime = min([b.max_time for b in blocks])
        discharge_facility = next(
            filter(None, [b.get_discharge_facility() for b in blocks]),
            None
        ) if visit_type == "9201" else None
        visit_events = list()
        for block in blocks:
            visit_events.extend(block.get_meds_events())

        visits.append(Visit(
            visit_type=visit_type,
            visit_start_datetime=visit_start_datetime,
            visit_end_datetime=visit_end_datetime,
            discharge_facility=discharge_facility,
            events=visit_events
        ))

    return CehrBertPatient(
        patient_id=patient.patient_id,
        birth_datetime=birth_datetime,
        visits=visits,
        race=race,
        gender=gender,
        ethnicity=ethnicity
    )


def meds_reader_to_meds_extension(path_to_db: str, split: str, default_visit_id: int = 1) -> CehrBertPatient:
    assert split in ['held_out', 'train', 'tuning']
    patient_database = meds_reader.PatientDatabase(path_to_db)
    patient_split = get_patient_split(path_to_db)
    filtered_patient_database = patient_database.filter(patient_split[split])
    for patient_id in filtered_patient_database:
        patient = filtered_patient_database[patient_id]
        yield convert_one_patient(patient, default_visit_id)


def create_dataset_from_meds_reader(
        data_args: DataTrainingArguments,
        default_visit_id: int = 1
) -> DatasetDict:
    dataset_class = IterableDataset if data_args.streaming else Dataset
    kargs = dict()
    if not data_args.streaming:
        kargs["num_proc"] = data_args.preprocessing_num_workers

    train_dataset = dataset_class.from_generator(
        functools.partial(
            meds_reader_to_meds_extension,
            path_to_db=data_args.data_folder,
            split="train",
            default_visit_id=default_visit_id
        ), **kargs
    )

    tuning_dataset = dataset_class.from_generator(
        functools.partial(
            meds_reader_to_meds_extension,
            path_to_db=data_args.data_folder,
            split="tuning",
            default_visit_id=default_visit_id
        ), **kargs
    )

    held_out_dataset = dataset_class.from_generator(
        functools.partial(
            meds_reader_to_meds_extension,
            path_to_db=data_args.data_folder,
            split="held_out",
            default_visit_id=default_visit_id
        ), **kargs
    )

    return DatasetDict({
        "train": train_dataset,
        "validation": tuning_dataset,
        "test": held_out_dataset
    })
