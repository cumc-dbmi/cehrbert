import unittest
from importlib.metadata import entry_points


class ConsoleScriptsTest(unittest.TestCase):
    def test_console_scripts_load(self):
        scripts = [ep for ep in entry_points(group="console_scripts") if ep.dist.name == "cehrbert"]
        self.assertEqual({ep.name for ep in scripts}, {"cehrbert-pretraining", "cehrbert-finetuning"})
        for ep in scripts:
            with self.subTest(script=ep.name):
                self.assertTrue(callable(ep.load()))


if __name__ == "__main__":
    unittest.main()
