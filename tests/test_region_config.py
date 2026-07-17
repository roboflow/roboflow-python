"""Tests for region-aware Roboflow URL configuration."""

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

import roboflow.config as config_module

URL_DEFAULTS = {
    "API_URL": "https://api.roboflow.com",
    "APP_URL": "https://app.roboflow.com",
    "UNIVERSE_URL": "https://universe.roboflow.com",
    "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.com",
    "SEMANTIC_SEGMENTATION_URL": "https://segment.roboflow.com",
    "OBJECT_DETECTION_URL": "https://serverless.roboflow.com",
    "CLIP_FEATURIZE_URL": "CLIP FEATURIZE URL NOT IN ENV",
    "OCR_URL": "OCR URL NOT IN ENV",
    "DEDICATED_DEPLOYMENT_URL": "https://roboflow.cloud",
}

REGION_ENVIRONMENT_KEYS = ("ROBOFLOW_CONFIG_DIR", "ROBOFLOW_REGION", *URL_DEFAULTS)


class TestRegionConfiguration(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.config_path = Path(self.temp_directory.name) / "config.json"
        self.saved_environment = {key: os.environ[key] for key in REGION_ENVIRONMENT_KEYS if key in os.environ}
        for key in REGION_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ["ROBOFLOW_CONFIG_DIR"] = str(self.config_path)
        self.config = importlib.reload(config_module)

    def tearDown(self) -> None:
        for key in REGION_ENVIRONMENT_KEYS:
            os.environ.pop(key, None)
        os.environ.update(self.saved_environment)
        importlib.reload(config_module)
        self.temp_directory.cleanup()

    def _write_config(self, config: dict) -> None:
        self.config_path.write_text(json.dumps(config))

    def _reload_config(self):
        self.config = importlib.reload(config_module)
        return self.config

    def test_existing_us_url_defaults_are_unchanged(self) -> None:
        self.assertEqual(self.config.get_effective_region(), "us")
        for key, expected_url in URL_DEFAULTS.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(self.config, key), expected_url)
                self.assertEqual(self.config.resolve_url(key), expected_url)

    def test_region_and_explicit_url_precedence(self) -> None:
        self._write_config({"ROBOFLOW_REGION": "us"})
        os.environ["ROBOFLOW_REGION"] = "EU"
        config = self._reload_config()
        self.assertEqual(config.get_effective_region(), "eu")
        self.assertEqual(config.API_URL, "https://api.roboflow.eu")

        os.environ.pop("ROBOFLOW_REGION")
        self._write_config({"ROBOFLOW_REGION": "eU"})
        config = self._reload_config()
        self.assertEqual(config.get_effective_region(), "eu")
        self.assertEqual(config.API_URL, "https://api.roboflow.eu")

        os.environ["API_URL"] = "https://api.env.example"
        config = self._reload_config()
        self.assertEqual(config.API_URL, "https://api.env.example")
        self.assertEqual(config.resolve_url("API_URL"), "https://api.env.example")

        os.environ.pop("API_URL")
        self._write_config(
            {
                "ROBOFLOW_REGION": "eu",
                "API_URL": "https://api.config.example",
            }
        )
        config = self._reload_config()
        self.assertEqual(config.API_URL, "https://api.config.example")
        self.assertEqual(config.resolve_url("API_URL"), "https://api.config.example")

    def test_eu_region_url_map(self) -> None:
        self._write_config({"ROBOFLOW_REGION": "eu"})
        config = self._reload_config()
        expected_urls = {
            "API_URL": "https://api.roboflow.eu",
            "APP_URL": "https://app.roboflow.eu",
            "OBJECT_DETECTION_URL": "https://serverless.roboflow.eu",
            "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.eu",
            "DEDICATED_DEPLOYMENT_URL": "https://eu.roboflow.cloud",
            "UNIVERSE_URL": "https://universe.roboflow.com",
            "SEMANTIC_SEGMENTATION_URL": "https://segment.roboflow.com",
        }
        for key, expected_url in expected_urls.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(config, key), expected_url)
                self.assertEqual(config.resolve_url(key), expected_url)

        os.environ["ROBOFLOW_REGION"] = "us"
        self.assertEqual(
            config.resolve_url("API_URL", region="EU"),
            "https://api.roboflow.eu",
        )

    def test_unknown_region_warns_once_and_falls_back_to_us(self) -> None:
        os.environ["ROBOFLOW_REGION"] = "bogus"
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            config = self._reload_config()
            self.assertEqual(config.get_effective_region(), "us")
            self.assertEqual(config.resolve_url("API_URL"), URL_DEFAULTS["API_URL"])

        warning_lines = stderr.getvalue().splitlines()
        self.assertEqual(len(warning_lines), 1)
        self.assertIn("unknown Roboflow region 'bogus'", warning_lines[0])
        self.assertIn("falling back to 'us'", warning_lines[0])


if __name__ == "__main__":
    unittest.main()
