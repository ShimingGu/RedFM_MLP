import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / "notebooks" / "iotfm_mlp.py"
spec = importlib.util.spec_from_file_location("iotfm_mlp", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TestColumns(unittest.TestCase):
    def test_default_exclusions(self):
        names = ["ID", "RA", "DEC", "tract", "patch", "FLUX", "ZPHOT", "ZPHOT_NIR", "Likelihood-Log", "Likelihood-Log_star"]
        included, excluded = module.select_input_columns(names)
        self.assertEqual(included, ["FLUX", "Likelihood-Log_star"])
        self.assertEqual(excluded, ["ID", "RA", "DEC", "tract", "patch", "ZPHOT", "ZPHOT_NIR", "Likelihood-Log"])

    def test_independent_switches(self):
        names = ["ID", "RA", "tract", "FLUX"]
        self.assertEqual(module.select_input_columns(names, include_id=True)[0], ["ID", "FLUX"])
        self.assertEqual(module.select_input_columns(names, include_location=True)[0], ["RA", "tract", "FLUX"])


if __name__ == "__main__": unittest.main()
