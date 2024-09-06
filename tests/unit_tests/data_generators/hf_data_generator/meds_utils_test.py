import unittest

from cehrbert.data_generators.hf_data_generator.meds_to_cehrbert_conversion_rules import MedsToBertMimic4
from cehrbert.data_generators.hf_data_generator.meds_utils import get_meds_to_cehrbert_conversion_cls
from cehrbert.runners.hf_runner_argument_dataclass import AttType, MedsToCehrBertConversionType


class TestGetMedsToCehrBertConversionCls(unittest.TestCase):

    def test_conversion(self):
        conversion_type = MedsToCehrBertConversionType["MedsToBertMimic4"]
        result = get_meds_to_cehrbert_conversion_cls(conversion_type)
        self.assertIsInstance(result, MedsToBertMimic4)

    def test_invalid_conversion(self):
        # Test for an invalid conversion type
        with self.assertRaises(RuntimeError) as context:
            get_meds_to_cehrbert_conversion_cls(AttType.CEHR_BERT)
        self.assertIn("is not a valid MedsToCehrBertConversionType", str(context.exception))


if __name__ == "__main__":
    unittest.main()
