import os
import sys
import unittest
from unittest.mock import patch
import importlib.util

# Get the path to config.py directly
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'roboflow', 'config.py')


class TestClassSorting(unittest.TestCase):
    """Test DISABLE_CLASS_SORTING configuration for class ordering."""

    def test_disable_class_sorting_default_value(self):
        """Test that DISABLE_CLASS_SORTING defaults to False."""
        # Load config module directly without triggering __init__.py
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        self.assertFalse(config.DISABLE_CLASS_SORTING)

    def test_disable_class_sorting_env_var_true(self):
        """Test that ROBOFLOW_DISABLE_CLASS_SORTING=true sets config to True."""
        with patch.dict(os.environ, {"ROBOFLOW_DISABLE_CLASS_SORTING": "true"}):
            # Reload config to pick up env var
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            # After reload, should be True
            self.assertTrue(config.DISABLE_CLASS_SORTING)

    def test_disable_class_sorting_env_var_false(self):
        """Test that ROBOFLOW_DISABLE_CLASS_SORTING=false keeps config as False."""
        with patch.dict(os.environ, {"ROBOFLOW_DISABLE_CLASS_SORTING": "false"}):
            spec = importlib.util.spec_from_file_location("config", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            
            self.assertFalse(config.DISABLE_CLASS_SORTING)

    def test_disable_class_sorting_env_var_case_insensitive(self):
        """Test that env var is case-insensitive."""
        for value in ["True", "TRUE", "tRuE"]:
            with patch.dict(os.environ, {"ROBOFLOW_DISABLE_CLASS_SORTING": value}):
                spec = importlib.util.spec_from_file_location("config", config_path)
                config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(config)
                
                self.assertTrue(config.DISABLE_CLASS_SORTING)

    def test_config_import(self):
        """Test that DISABLE_CLASS_SORTING exists and is a boolean."""
        spec = importlib.util.spec_from_file_location("config", config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        
        config_value = config.DISABLE_CLASS_SORTING
        self.assertIsNotNone(config_value)
        self.assertIsInstance(config_value, bool)


if __name__ == "__main__":
    unittest.main()
