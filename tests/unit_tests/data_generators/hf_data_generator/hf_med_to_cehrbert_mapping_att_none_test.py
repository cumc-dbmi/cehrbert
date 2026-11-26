import unittest
from datetime import datetime

from cehrbert_data.decorators.patient_event_decorator_base import AttType

from cehrbert.data_generators.hf_data_generator.hf_dataset_mapping import MedToCehrBertDatasetMapping
from cehrbert.med_extension.schema_extension import CehrBertPatient, Event, Visit
from cehrbert.runners.hf_runner_argument_dataclass import DataTrainingArguments


class TestMedToCehrBertDatasetMappingWithoutAtt(unittest.TestCase):
    def test_transform_without_att_tokens(self):
        patient = CehrBertPatient(
            patient_id=0,
            birth_datetime=datetime(1980, 4, 14, 0, 0),
            gender="Gender/F",
            race="Race/unknown",
            ethnicity="Ethnicity/unknown",
            visits=[
                Visit(
                    visit_type="9202",
                    visit_start_datetime=datetime(2024, 4, 14, 0, 0),
                    events=[Event(time=datetime(2024, 4, 14, 0, 0), code="320128")],
                ),
                Visit(
                    visit_type="9202",
                    visit_start_datetime=datetime(2024, 4, 21, 0, 0),
                    events=[Event(time=datetime(2024, 4, 21, 0, 0), code="4134120")],
                ),
            ],
        )
        data_args = DataTrainingArguments(
            data_folder=None,
            dataset_prepared_path=None,
            att_function_type=AttType.NONE.value,
            include_auxiliary_token=False,
        )
        transformed_record = MedToCehrBertDatasetMapping(data_args).transform(patient)

        self.assertListEqual(
            transformed_record["concept_ids"],
            ["[VS]", "320128", "[VE]", "[VS]", "4134120", "[VE]"],
        )
        self.assertListEqual(transformed_record["visit_segments"], [1, 1, 1, 2, 2, 2])
        self.assertListEqual(transformed_record["visit_concept_orders"], [1, 1, 1, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
