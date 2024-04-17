import unittest
from meds.schema import Event, Measurement
from med_extension.schema_extension import PatientExtension, Visit
from datetime import datetime

from runner.hf_runner_argument_dataclass import DataTrainingArguments
from data_generators.hf_data_generator.hf_dataset_mapping import MedToCehrBertDatasetMapping
from spark_apps.decorators.patient_event_decorator import time_token_func, time_day_token, AttType


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
        self.patient = PatientExtension(
            patient_id=0,
            visits=[inpatient_visit, outpatient_visit],
            birth_datetime=datetime(1980, 4, 14, 0, 0),
            gender='8507',
            race='0',
            static_measurements=[]
        )

    def test_transform_cehrbert_with_auxiliary_token(self):
        # Setup
        data_args = DataTrainingArguments(
            data_folder=None,  # required field set to None
            dataset_prepared_path=None,  # required field set to None
            att_function_type=AttType.CEHR_BERT.value,
            include_auxiliary_token=True
        )
        # Create an instance of the mapping class
        mapper = MedToCehrBertDatasetMapping(
            data_args
        )

        transformed_record = mapper.transform(self.patient)

        # Assert
        self.assertEqual(transformed_record['person_id'], 0)
        self.assertEqual(transformed_record['gender'], '8507')
        self.assertEqual(transformed_record['race'], '0')

        # Test concept_ids
        self.assertListEqual(
            transformed_record['concept_ids'],
            ['[VS]', '9202', '320128', '[VE]', 'W1', '[VS]', '9201', '320128', '4134120', '8536', '[VE]']
        )

        # Test ages, age=-1 used for the ATT tokens
        self.assertListEqual(
            transformed_record['ages'],
            [44, 44, 44, 44, -1, 44, 44, 44, 44, 44, 44]
        )

        # Test dates, dates=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['dates'],
            [2832, 2832, 2832, 2832, 0, 2833, 2833, 2833, 2833, 2833, 2833]
        )

        # Test visit_segments, visit_segment=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_segments'],
            [1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2]
        )

        # Test visit_concept_orders, we visit_concept_order to be same as next visit for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_concept_orders'],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        )

        # Test concept_value_masks
        self.assertListEqual(
            transformed_record['concept_value_masks'],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

        # Test concept_values, concept_value=-1 is a default value associated with non-numeric measurements
        self.assertListEqual(
            transformed_record['concept_values'],
            [-1, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1]
        )

        # Test mlm_skip_values
        self.assertListEqual(
            transformed_record['mlm_skip_values'],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

    def test_transform_basic(self):
        data_args = DataTrainingArguments(
            data_folder=None,  # required field set to None
            dataset_prepared_path=None,  # required field set to None
            att_function_type=AttType.CEHR_BERT.value,
            include_auxiliary_token=False
        )
        # Create an instance of the mapping class
        mapper = MedToCehrBertDatasetMapping(
            data_args
        )

        transformed_record = mapper.transform(self.patient)

        # Assert
        self.assertEqual(transformed_record['person_id'], 0)
        self.assertEqual(transformed_record['gender'], '8507')
        self.assertEqual(transformed_record['race'], '0')

        # Test concept_ids
        self.assertListEqual(
            transformed_record['concept_ids'],
            ['[VS]', '320128', '[VE]', 'W1', '[VS]', '320128', '4134120', '[VE]']
        )

        # Test ages, age=-1 used for the ATT tokens
        self.assertListEqual(
            transformed_record['ages'],
            [44, 44, 44, -1, 44, 44, 44, 44]
        )

        # Test dates, dates=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['dates'],
            [2832, 2832, 2832, 0, 2833, 2833, 2833, 2833]
        )

        # Test visit_segments, visit_segment=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_segments'],
            [1, 1, 1, 0, 2, 2, 2, 2]
        )

        # Test visit_concept_orders, we visit_concept_order to be same as next visit for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_concept_orders'],
            [1, 1, 1, 2, 2, 2, 2, 2]
        )

        # Test concept_value_masks
        self.assertListEqual(
            transformed_record['concept_value_masks'],
            [0, 0, 0, 0, 0, 0, 1, 0]
        )

        # Test concept_values, concept_value=-1 is a default value associated with non-numeric measurements
        self.assertListEqual(
            transformed_record['concept_values'],
            [-1, -1, -1, -1, -1, -1, 0.5, -1]
        )

        # Test mlm_skip_values
        self.assertListEqual(
            transformed_record['mlm_skip_values'],
            [0, 0, 0, 0, 0, 0, 1, 0]
        )

    def test_cehrgpt_transform(self):
        data_args = DataTrainingArguments(
            data_folder=None,  # required field set to None
            dataset_prepared_path=None,  # required field set to None
            att_function_type=AttType.DAY.value,
            inpatient_att_function_type=AttType.DAY.value,
            include_auxiliary_token=True,
            include_demographic_prompt=True
        )
        # Create an instance of the mapping class
        mapper = MedToCehrBertDatasetMapping(
            data_args
        )
        transformed_record = mapper.transform(self.patient)

        # Test concept_ids
        self.assertListEqual(
            transformed_record['concept_ids'],
            ['year:2024', 'age:44', '8507', '0', '[VS]', '9202', '320128', '[VE]',
             'D7', '[VS]', '9201', '320128', 'i-D7', '4134120', '8536', '[VE]']
        )

        # Test ages, age=-1 used for the ATT tokens
        self.assertListEqual(
            transformed_record['ages'],
            [-1, -1, -1, -1, 44, 44, 44, 44, -1, 44, 44, 44, -1, 44, 44, 44]
        )

        # Test dates, dates=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['dates'],
            [0, 0, 0, 0, 2832, 2832, 2832, 2832, 0, 2833, 2833, 2833, 0, 2833, 2833, 2833]
        )

        # Test visit_segments, visit_segment=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_segments'],
            [0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2]
        )

        # Test visit_concept_orders, we visit_concept_order to be same as next visit for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_concept_orders'],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        )

        # Test concept_value_masks
        self.assertListEqual(
            transformed_record['concept_value_masks'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

        # Test concept_values, concept_value=-1 is a default value associated with non-numeric measurements
        self.assertListEqual(
            transformed_record['concept_values'],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1]
        )

        # Test mlm_skip_values
        self.assertListEqual(
            transformed_record['mlm_skip_values'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

    def test_inpatient_att_transform(self):
        data_args = DataTrainingArguments(
            data_folder=None,  # required field set to None
            dataset_prepared_path=None,  # required field set to None
            att_function_type=AttType.CEHR_BERT.value,
            inpatient_att_function_type=AttType.DAY.value,
            include_auxiliary_token=True,
            include_demographic_prompt=False
        )

        # Create an instance of the mapping class
        mapper = MedToCehrBertDatasetMapping(
            data_args
        )

        transformed_record = mapper.transform(self.patient)

        # Assert
        self.assertEqual(transformed_record['person_id'], 0)
        self.assertEqual(transformed_record['gender'], '8507')
        self.assertEqual(transformed_record['race'], '0')

        # Test concept_ids
        self.assertListEqual(
            transformed_record['concept_ids'],
            ['[VS]', '9202', '320128', '[VE]', 'W1', '[VS]', '9201', '320128', 'i-D7', '4134120', '8536', '[VE]']
        )

        # Test ages, age=-1 used for the ATT tokens
        self.assertListEqual(
            transformed_record['ages'],
            [44, 44, 44, 44, -1, 44, 44, 44, -1, 44, 44, 44]
        )

        # Test dates, dates=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['dates'],
            [2832, 2832, 2832, 2832, 0, 2833, 2833, 2833, 0, 2833, 2833, 2833]
        )

        # Test visit_segments, visit_segment=0 used for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_segments'],
            [1, 1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 2]
        )

        # Test visit_concept_orders, we visit_concept_order to be same as next visit for the ATT tokens
        self.assertListEqual(
            transformed_record['visit_concept_orders'],
            [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]
        )

        # Test concept_value_masks
        self.assertListEqual(
            transformed_record['concept_value_masks'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )

        # Test concept_values, concept_value=-1 is a default value associated with non-numeric measurements
        self.assertListEqual(
            transformed_record['concept_values'],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0.5, -1, -1]
        )

        # Test mlm_skip_values
        self.assertListEqual(
            transformed_record['mlm_skip_values'],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
        )


if __name__ == '__main__':
    unittest.main()