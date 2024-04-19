from datasets import load_from_disk, DatasetDict
from meds import Patient, Event, Measurement, birth_code

from med_extension.schema_extension import CehrBertPatient


def convert_med_extension_to_med(
        patient_extension: CehrBertPatient
) -> Patient:
    birth_date = patient_extension['birth_datetime']
    gender = patient_extension['gender']
    race = patient_extension['race']
    demographic_event = Event(
        time=birth_date,
        measurements=[
            Measurement(
                code=birth_code
            ),
            Measurement(
                code=gender
            ),
            Measurement(
                code=race
            )
        ]
    )
    events = [demographic_event]
    for visit in sorted(patient_extension['visits'], key=lambda v: v['visit_start_datetime']):
        events.extend(sorted(visit['events'], key=lambda e: e['time']))
    return Patient(
        patient_id=patient_extension['patient_id'],
        static_measurements=patient_extension['static_measurements'],
        events=events
    )


def main(args):
    med_extension_dataset = load_from_disk(args.med_extension_dataset)
    if isinstance(med_extension_dataset, DatasetDict):
        column_names = med_extension_dataset['train'].column_names
    else:
        column_names = med_extension_dataset.column_names
    converted_dataset = med_extension_dataset.map(
        convert_med_extension_to_med,
        remove_columns=column_names,
        num_proc=args.num_proc
    )
    converted_dataset.save_to_disk(args.med_output_folder)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Arguments for converting data from the MED extension to MED'
    )
    parser.add_argument(
        '--med_extension_dataset',
        dest='med_extension_dataset',
        action='store',
        help='The path for the med_extension_dataset',
        required=True
    )
    parser.add_argument(
        '--med_output_folder',
        dest='med_output_folder',
        action='store',
        help='The path to the original MED output folder',
        required=True
    )
    parser.add_argument(
        '--num_proc',
        dest='num_proc',
        action='store',
        required=False,
        type=int,
        default=4
    )
    main(parser.parse_args())
