import unittest
from meds.schema import Event, Measurement
from datetime import datetime
from datasets import Dataset, DatasetDict

from tools.convert_med_extension_to_med import convert_med_extension_to_med
from med_extension.schema_extension import CehrBertPatient, Visit


# Actual test class
class TestMedToCehrBertDatasetMapping(unittest.TestCase):

    def setUp(self):
        outpatient_visit = Visit(
            visit_type='9202',
            visit_start_datetime=datetime(2024, 4, 14, 0, 0),
            events=[
                Event(
                    time=datetime(2024, 4, 14, 0, 0),
                    measurements=[
                        Measurement(code='320128', datetime_value=datetime(2024, 4, 14, 1, 0))
                    ]
                )
            ]
        )

        inpatient_visit = Visit(
            visit_type='9201',
            visit_start_datetime=datetime(2024, 4, 21, 0, 0),
            visit_end_datetime=datetime(2024, 4, 22, 0, 0),
            discharge_facility='8536',
            events=[
                Event(
                    time=datetime(2024, 4, 21, 0, 0),
                    measurements=[
                        Measurement(code='320128', datetime_value=datetime(2024, 4, 21, 0, 0))
                    ]
                ),
                Event(
                    time=datetime(2024, 4, 22, 0, 0),
                    measurements=[
                        Measurement(
                            code='4134120',
                            datetime_value=datetime(2024, 4, 22, 0, 0),
                            numeric_value=0.5
                        )
                    ]
                )
            ]
        )

        # Intentionally perturb the chronological order of visits by putting outpatient_visit after inpatient_visit,
        # the mapping function should be able to re-order the events based on their time stamps first
        patient = CehrBertPatient(
            patient_id=0,
            visits=[inpatient_visit, outpatient_visit],
            birth_datetime=datetime(1980, 4, 14, 0, 0),
            gender='8507',
            race='0',
            static_measurements=[]
        )

        self.dataset = Dataset.from_list([patient])

    def test_med_extension_to_med(self):
        if isinstance(self.dataset, DatasetDict):
            column_names = self.dataset['train'].column_names
        else:
            column_names = self.dataset.column_names
        converted_dataset = self.dataset.map(convert_med_extension_to_med, remove_columns=column_names)
        if isinstance(converted_dataset, DatasetDict):
            converted_column_names = converted_dataset['train'].column_names
        else:
            converted_column_names = converted_dataset.column_names
        self.assertListEqual(converted_column_names, ['patient_id', 'static_measurements', 'events'])


if __name__ == '__main__':
    unittest.main()
