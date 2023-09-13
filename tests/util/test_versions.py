import unittest
from importlib import import_module

from roboflow.util.versions import get_wrong_dependencies_versions


class TestVersions(unittest.TestCase):
    def test_wrong_dependencies_versions(self):
        module_path = "tests.util.dummy_module"
        module_version = import_module(module_path).__version__
        tests = [
            ("tests.util.dummy_module", "==", module_version),
            ("tests.util.dummy_module", "<=", "0.2.0"),
            ("tests.util.dummy_module", "<=", "1.0.0"),
            ("tests.util.dummy_module", ">=", "0.1.0"),
            ("tests.util.dummy_module", ">=", "0.6.0"),
            ("tests.util.dummy_module", ">=", "0.1.34"),
        ]
        # true if dep is correc
        expected_results = [True, False, True, True, False, True]

        for test, expected_result in zip(tests, expected_results):
            wrong_dependencies_versions = get_wrong_dependencies_versions([test])
            is_correct_dep = len(wrong_dependencies_versions) == 0
            self.assertEqual(is_correct_dep, expected_result)
