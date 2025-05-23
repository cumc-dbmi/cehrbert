import collections
import functools
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import meds_reader
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Sequence, Split, Value
from transformers.utils import logging

from cehrbert.data_generators.hf_data_generator import UNKNOWN_VALUE
from cehrbert.data_generators.hf_data_generator.cache_util import CacheFileCollector
from cehrbert.data_generators.hf_data_generator.hf_dataset import apply_cehrbert_dataset_mapping
from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import DatasetMapping
from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import MedsToCehrBertConversion
from cehrbert.data_generators.hf_data_generator.patient_block import generate_demographics_and_patient_blocks
from cehrbert.med_extension.schema_extension import CehrBertPatient, Visit
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments, MedsToCehrBertConversionType

MEDS_SPLIT_DATA_SPLIT_MAPPING = {
    "train": Split.TRAIN,
    "tuning": Split.VALIDATION,
    "held_out": Split.TEST,
}
NON_ALPHANUMERIC_CHARS = r"[\w\/\\:\-_]"
LOG = logging.get_logger("transformers")


def get_meds_to_cehrbert_conversion_cls(
    meds_to_cehrbert_conversion_type: Union[MedsToCehrBertConversionType, str], **kwargs
) -> MedsToCehrBertConversion:
    for cls in MedsToCehrBertConversion.__subclasses__():
        if isinstance(meds_to_cehrbert_conversion_type, MedsToCehrBertConversionType):
            if meds_to_cehrbert_conversion_type.name == cls.__name__:
                return cls(**kwargs)
        elif isinstance(meds_to_cehrbert_conversion_type, str):
            if meds_to_cehrbert_conversion_type == cls.__name__:
                return cls(**kwargs)
    raise RuntimeError(f"{meds_to_cehrbert_conversion_type} is not a valid MedsToCehrBertConversionType")


def get_subject_split(meds_reader_db_path: str) -> Dict[str, List[int]]:
    subject_split = pd.read_parquet(os.path.join(meds_reader_db_path, "metadata/subject_splits.parquet"))
    with meds_reader.SubjectDatabase(meds_reader_db_path) as patient_database:
        subject_ids = [p for p in patient_database]
    subject_split = subject_split[subject_split.subject_id.isin(subject_ids)]
    result = {str(group): records["subject_id"].tolist() for group, records in subject_split.groupby("split")}
    return result


def convert_one_patient(
    patient: meds_reader.Subject,
    conversion: MedsToCehrBertConversion,
    prediction_time: datetime = None,
    observation_window: int = None,
    label: Union[int, float] = None,
) -> CehrBertPatient:
    """
    Converts a patient's event data into a CehrBertPatient object, processing.

    their medical history, visit details, and demographic information.

    Parameters:
    ----------
    patient : meds_reader.Subject
        The patient's event data, including time-stamped medical events such as
        demographic data (race, gender, ethnicity) and clinical visits (ED admissions,
        hospital admissions, discharges).

    conversion : MedsToCehrBertConversion
        The conversion object to map and process medical event data into the format
        required by CehrBert.

    default_visit_id : int, optional (default=1)
        The starting ID for patient visits. This is incremented as new visits are
        identified in the event data.

    prediction_time : datetime, optional (default=None)
        The cutoff time for processing events. Events occurring after this time are
        ignored.

    observation_window: int, optional (default=None)
        The observation window for extracting features

    label : Union[int, float], optional (default=None)
        The prediction label associated with this patient, which could represent a
        clinical outcome (e.g., survival or treatment response).

    Returns:
    -------
    CehrBertPatient
        An object containing the patient's transformed event data, visits, demographics,
        and associated label in a structure compatible with CehrBert's input requirements.

    Description:
    -----------
    This function processes a patient's medical history, including demographic
    information (birth date, race, gender, and ethnicity) and visit details. It iterates
    through the patient's events and groups them into visits (ED, admission, discharge).
    Visits are formed based on timestamps, and certain logic is applied to merge ED visits
    into hospital admissions if they occur within 24 hours of each other.

    For each event, demographic attributes like birth date, race, gender, and ethnicity
    are extracted. If the event has a timestamp, it is compared with `prediction_time` to
    filter out events that occurred after the specified time.

    The function handles ongoing (incomplete) visits and cases where multiple visits
    should be merged (e.g., ED followed by hospital admission within 24 hours). After
    processing the events, visits are built with details such as visit type, start/end
    datetime, and events during the visit.

    The function returns a `CehrBertPatient` object that includes the patient's medical
    events, structured into visits, along with demographic information, and optionally
    a prediction label.

    Example Usage:
    -------------
    patient_data = convert_one_patient(
        patient=some_patient_object,
        conversion=some_conversion_object,
        default_visit_id=1,
        prediction_time=datetime.now(),
        label=1
    )
    """
    demographics, patient_blocks = generate_demographics_and_patient_blocks(
        conversion=conversion, patient=patient, prediction_time=prediction_time, observation_window=observation_window
    )

    patient_block_dict = collections.defaultdict(list)
    for patient_block in patient_blocks:
        patient_block_dict[patient_block.visit_id].append(patient_block)

    visits = list()
    for visit_id, blocks in patient_block_dict.items():
        visit_type = blocks[0].visit_type
        has_discharge_facility = any([b.has_ed_or_hospital_admission for b in blocks])
        visit_start_datetime = min([b.get_visit_start_datetime() for b in blocks])
        visit_end_datetime = max([b.get_visit_end_datetime() for b in blocks])
        discharge_facility = (
            next(filter(None, [b.discharged_to for b in blocks]), None) if has_discharge_facility else None
        )
        visit_events = list()
        for block in blocks:
            visit_events.extend(block.get_meds_events())

        visits.append(
            Visit(
                visit_type=visit_type,
                visit_start_datetime=visit_start_datetime,
                visit_end_datetime=visit_end_datetime,
                discharge_facility=(discharge_facility if discharge_facility else UNKNOWN_VALUE),
                events=visit_events,
            )
        )
    age_at_index = -1
    if prediction_time is not None and demographics.birth_datetime is not None:
        age_at_index = prediction_time.year - demographics.birth_datetime.year
        if (prediction_time.month, prediction_time.day) < (
            demographics.birth_datetime.month,
            demographics.birth_datetime.day,
        ):
            age_at_index -= 1

    return CehrBertPatient(
        patient_id=patient.subject_id,
        birth_datetime=demographics.birth_datetime,
        visits=visits,
        race=demographics.race if demographics.race else UNKNOWN_VALUE,
        gender=demographics.gender if demographics.gender else UNKNOWN_VALUE,
        ethnicity=demographics.ethnicity if demographics.ethnicity else UNKNOWN_VALUE,
        index_date=prediction_time,
        age_at_index=age_at_index,
        label=label,
    )


def create_dataset_from_meds_reader(
    data_args: DataTrainingArguments,
    default_visit_id: int = 1,
    dataset_mappings: Optional[List[DatasetMapping]] = None,
    cache_file_collector: Optional[CacheFileCollector] = None,
) -> DatasetDict:

    LOG.info("The meds_to_cehrbert_conversion_type: %s", data_args.meds_to_cehrbert_conversion_type)
    LOG.info("The att_function_type: %s", data_args.att_function_type)
    LOG.info("The inpatient_att_function_type: %s", data_args.inpatient_att_function_type)
    LOG.info("The include_auxiliary_token: %s", data_args.include_auxiliary_token)
    LOG.info("The include_demographic_prompt: %s", data_args.include_demographic_prompt)
    LOG.info("The meds_exclude_tables: %s", "\n".join(data_args.meds_exclude_tables))
    LOG.info("The disconnect_problem_list_events: %s", data_args.disconnect_problem_list_events)

    train_dataset = _create_cehrbert_data_from_meds(
        data_args=data_args,
        split="train",
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
    tuning_dataset = _create_cehrbert_data_from_meds(
        data_args=data_args,
        split="tuning",
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
    held_out_dataset = _create_cehrbert_data_from_meds(
        data_args=data_args,
        split="held_out",
        default_visit_id=default_visit_id,
        dataset_mappings=dataset_mappings,
        cache_file_collector=cache_file_collector,
    )
    return DatasetDict({"train": train_dataset, "validation": tuning_dataset, "test": held_out_dataset})


def _meds_to_cehrbert_generator(
    shards: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    path_to_db: str,
    default_visit_id: int,
    meds_to_cehrbert_conversion_type: MedsToCehrBertConversionType,
    meds_exclude_tables: Optional[List[str]] = None,
    disconnect_problem_list_events: bool = False,
) -> CehrBertPatient:
    conversion = get_meds_to_cehrbert_conversion_cls(
        meds_to_cehrbert_conversion_type,
        default_visit_id=default_visit_id,
        meds_exclude_tables=meds_exclude_tables,
        disconnect_problem_list_events=disconnect_problem_list_events,
    )
    with meds_reader.SubjectDatabase(path_to_db) as patient_database:
        for shard in shards:
            for patient_id, prediction_time, observation_window, label in shard:
                patient = patient_database[patient_id]
                converted_patient = convert_one_patient(patient, conversion, prediction_time, observation_window, label)
                # there are patients whose birthdate is none
                visits = converted_patient["visits"]
                if converted_patient["birth_datetime"] is None:
                    LOG.warning("patient_id: %s does not have a valid birth_datetime, therefore skipped", patient_id)
                elif visits is None or len(visits) == 0:
                    LOG.debug("patient_id: %s does not have visits, therefore skipped", patient_id)
                else:
                    yield converted_patient


def _create_cehrbert_data_from_meds(
    data_args: DataTrainingArguments,
    split: str,
    default_visit_id: int = 1,
    dataset_mappings: Optional[List[DatasetMapping]] = None,
    cache_file_collector: Optional[CacheFileCollector] = None,
):
    assert split in ["held_out", "train", "tuning"]
    batches = []
    if data_args.cohort_folder:
        # Load the entire cohort
        cohort = pd.read_parquet(os.path.expanduser(data_args.cohort_folder))
        patient_split = get_subject_split(os.path.expanduser(data_args.data_folder))
        subject_ids = patient_split[split]
        cohort_split = cohort[cohort.subject_id.isin(subject_ids)]
        for cohort_row in cohort_split.itertuples():
            subject_id = cohort_row.subject_id
            prediction_time = cohort_row.prediction_time
            label = int(cohort_row.boolean_value)
            batches.append((subject_id, prediction_time, data_args.observation_window, label))
    else:
        patient_split = get_subject_split(os.path.expanduser(data_args.data_folder))
        for subject_id in patient_split[split]:
            batches.append((subject_id, None, None, None))

    features = Features(
        {
            "patient_id": Value(dtype="int32"),
            "birth_datetime": Value(dtype="timestamp[us]"),  # Use timestamp with microsecond precision
            "gender": Value(dtype="string"),
            "race": Value(dtype="string"),
            "ethnicity": Value(dtype="string"),
            "index_date": Value(dtype="timestamp[us]"),
            "age_at_index": Value(dtype="int16"),
            "label": Value(dtype="float32"),  # Using float to accommodate both int and float labels
            "visits": Sequence(
                feature={
                    "visit_type": Value(dtype="string"),
                    "visit_start_datetime": Value(dtype="timestamp[us]"),
                    "visit_end_datetime": Value(dtype="timestamp[us]"),
                    "discharge_facility": Value(dtype="string"),
                    "events": Sequence(
                        feature={
                            "time": Value(dtype="timestamp[us]"),
                            "code": Value(dtype="string"),
                            "text_value": Value(dtype="string"),
                            "numeric_value": Value(dtype="float32"),
                            "unit": Value(dtype="string"),
                            "properties": {
                                "visit_id": Value(dtype="int32"),
                                "table": Value(dtype="string"),
                            },
                        }
                    ),
                }
            ),
        }
    )

    split_batches = np.array_split(np.asarray(batches), data_args.preprocessing_num_workers)
    batch_func = functools.partial(
        _meds_to_cehrbert_generator,
        path_to_db=os.path.expanduser(data_args.data_folder),
        default_visit_id=default_visit_id,
        meds_to_cehrbert_conversion_type=data_args.meds_to_cehrbert_conversion_type,
        meds_exclude_tables=data_args.meds_exclude_tables,
        disconnect_problem_list_events=data_args.disconnect_problem_list_events,
    )
    dataset = Dataset.from_generator(
        batch_func,
        gen_kwargs={
            "shards": split_batches,
        },
        num_proc=(data_args.preprocessing_num_workers if not data_args.streaming else None),
        writer_batch_size=data_args.preprocessing_batch_size,
        streaming=data_args.streaming,
        features=features,
    )
    # These cached files need to be deleted manually
    if cache_file_collector:
        cache_file_collector.add_cache_files(dataset)

    if dataset_mappings:
        for dataset_mapping in dataset_mappings:
            # Convert the CehrBertPatient to CehrBert data inputs
            dataset = apply_cehrbert_dataset_mapping(
                dataset,
                dataset_mapping,
                num_proc=data_args.preprocessing_num_workers,
                batch_size=data_args.preprocessing_batch_size,
                streaming=data_args.streaming,
                cache_file_collector=cache_file_collector,
            )
    return dataset
