import math
import unittest

from data_generators.gpt_learning_objectives import CosineMaskRateScheduler


class TestCosineMaskRateScheduler(unittest.TestCase):

    def test_initialization(self):
        scheduler = CosineMaskRateScheduler()
        self.assertEqual(scheduler._low_rate, 0.5)
        self.assertEqual(scheduler._high_rate, 1.0)
        self.assertEqual(scheduler._low_rate_mult, 1.1)
        self.assertEqual(scheduler._period, 1000)
        self.assertEqual(scheduler._total, math.inf)

    def test_low_high_rate_assertion(self):
        with self.assertRaises(AssertionError):
            CosineMaskRateScheduler(low_rate=1.0, high_rate=0.5)

    def test_rate_adjustment(self):
        scheduler = CosineMaskRateScheduler(low_rate=0.5, high_rate=1.0, period=10, total=50)
        for _ in range(11):
            scheduler.get_rate()
        self.assertEqual(scheduler._low_rate, 0.5 * scheduler._low_rate_mult)

        # Eventually low_rate will converge to high_rate
        for _ in range(1000):
            scheduler.get_rate()

        self.assertEqual(scheduler._low_rate, scheduler._high_rate)
        self.assertEqual(scheduler.get_rate(), scheduler._high_rate)

    def test_rate_periodicity(self):
        scheduler = CosineMaskRateScheduler(period=10)
        first_rate = scheduler.get_rate()
        for _ in range(9):
            scheduler.get_rate()
        self.assertTrue(first_rate < scheduler.get_rate())

    def test_stopped(self):
        scheduler = CosineMaskRateScheduler(total=5)
        for _ in range(6):
            scheduler.get_rate()
        self.assertTrue(scheduler.stopped())


if __name__ == '__main__':
    unittest.main()
