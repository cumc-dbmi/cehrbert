import unittest
from analyses.gpt.generate_cooccurrence import next_visit_concept_pair, temporal_concept_pair


class TestNextVisitConceptPair(unittest.TestCase):

    def test_single_visit(self):
        self.assertEqual(next_visit_concept_pair(['VS', '1', '2', 'VE']), [('1', '2')])

    def test_multiple_visits(self):
        sequence = ['VS', '1', '2', 'VE', 'D1', 'VS', '2', '3', 'VE']
        expected_result = [('1', '2'), ('1', '3'), ('2', '3')]
        result = next_visit_concept_pair(sequence)
        self.assertSetEqual(set(result), set(expected_result))

    def test_no_visits(self):
        self.assertEqual(next_visit_concept_pair([]), [])

    def test_non_numeric(self):
        sequence = ['VS', 'A', '1', 'VE', 'D1', 'VS', '2', 'B', 'VE']
        expected_result = [('1', '2')]
        result = next_visit_concept_pair(sequence)
        self.assertSetEqual(set(result), set(expected_result))


class TestTemporalConceptPair(unittest.TestCase):

    def test_empty_sequence(self):
        self.assertEqual(temporal_concept_pair([]), [])

    def test_sequence_with_no_numeric(self):
        self.assertEqual(temporal_concept_pair(['VS', 'A', 'B']), [])

    def test_multiple_visits(self):
        sequence = ['VS', '1', '2', 'VE', 'D1', 'VS', '2', '3', 'VE']
        expected_result = [('1', '2'), ('1', '3'), ('2', '3')]
        result = temporal_concept_pair(sequence)
        self.assertSetEqual(set(result), set(expected_result))
