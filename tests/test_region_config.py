"""Tests for region-aware Roboflow URL configuration."""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import roboflow.config as config_module

URL_DEFAULTS = {
    "API_URL": "https://api.roboflow.com",
    "APP_URL": "https://app.roboflow.com",
    "UNIVERSE_URL": "https://universe.roboflow.com",
    "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.com",
    "SEMANTIC_SEGMENTATION_URL": "https://segment.roboflow.com",
    "OBJECT_DETECTION_URL": "https://serverless.roboflow.com",
    "SERVERLESS_URL": "https://serverless.roboflow.com",
    "CLIP_FEATURIZE_URL": "CLIP FEATURIZE URL NOT IN ENV",
    "OCR_URL": "OCR URL NOT IN ENV",
    "DEDICATED_DEPLOYMENT_URL": "https://roboflow.cloud",
}

REGION_ENVIRONMENT_KEYS = ("ROBOFLOW_CONFIG_DIR", "ROBOFLOW_REGION", "ROBOFLOW_ENVIRONMENT", *URL_DEFAULTS)


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
            "SERVERLESS_URL": "https://serverless.roboflow.eu",
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

    def test_region_environment_url_matrix(self) -> None:
        expected = {
            ("us", "staging"): {
                "API_URL": "https://api.roboflow.one",
                "APP_URL": "https://app.roboflow.one",
                "UNIVERSE_URL": "https://universe.roboflow.one",
                "OBJECT_DETECTION_URL": "https://serverless.roboflow.one",
                "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow.one",
                "SERVERLESS_URL": "https://serverless.roboflow.one",
                "SEMANTIC_SEGMENTATION_URL": "https://lambda-semantic-segmentation.staging.roboflow.com",
                "DEDICATED_DEPLOYMENT_URL": "https://staging.roboflow.cloud",
            },
            ("eu", "staging"): {
                "API_URL": "https://api.roboflow-eu.one",
                "APP_URL": "https://app.roboflow-eu.one",
                "UNIVERSE_URL": "https://universe.roboflow.one",
                "OBJECT_DETECTION_URL": "https://serverless.roboflow-eu.one",
                "INSTANCE_SEGMENTATION_URL": "https://serverless.roboflow-eu.one",
                "SERVERLESS_URL": "https://serverless.roboflow-eu.one",
                "DEDICATED_DEPLOYMENT_URL": "https://eu.staging.roboflow.cloud",
            },
        }
        for (region, environment), urls in expected.items():
            os.environ["ROBOFLOW_REGION"] = region
            os.environ["ROBOFLOW_ENVIRONMENT"] = environment
            config = self._reload_config()
            self.assertEqual(config.get_effective_environment(), environment)
            for key, expected_url in urls.items():
                with self.subTest(region=region, environment=environment, key=key):
                    self.assertEqual(getattr(config, key), expected_url)

    def test_environment_is_read_from_config_file(self) -> None:
        self._write_config({"ROBOFLOW_ENVIRONMENT": "Staging"})
        config = self._reload_config()
        self.assertEqual(config.get_effective_environment(), "staging")
        self.assertEqual(config.API_URL, "https://api.roboflow.one")

    def test_explicit_url_beats_environment(self) -> None:
        os.environ["ROBOFLOW_ENVIRONMENT"] = "staging"
        os.environ["API_URL"] = "https://localapi.roboflow.one"
        config = self._reload_config()
        self.assertEqual(config.API_URL, "https://localapi.roboflow.one")
        self.assertEqual(config.APP_URL, "https://app.roboflow.one")

    def test_unknown_environment_warns_and_falls_back_to_prod(self) -> None:
        os.environ["ROBOFLOW_ENVIRONMENT"] = "production"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._reload_config()
            self.assertEqual(config.get_effective_environment(), "prod")
            self.assertEqual(config.API_URL, URL_DEFAULTS["API_URL"])

        messages = [str(w.message) for w in caught if issubclass(w.category, config.RegionWarning)]
        self.assertEqual(messages, ["unknown Roboflow environment 'production'; falling back to 'prod'."])
        self.assertEqual(config.get_region_warning(), messages[0])

    def test_semantic_segmentation_availability_by_region_and_environment(self) -> None:
        os.environ["ROBOFLOW_ENVIRONMENT"] = "staging"
        config = self._reload_config()
        config.ensure_url_available_in_region("SEMANTIC_SEGMENTATION_URL")

        os.environ["ROBOFLOW_REGION"] = "eu"
        with self.assertRaisesRegex(RuntimeError, "EU staging"):
            config.ensure_url_available_in_region("SEMANTIC_SEGMENTATION_URL")

    def test_unknown_region_warns_once_and_falls_back_to_us(self) -> None:
        os.environ["ROBOFLOW_REGION"] = "bogus"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            config = self._reload_config()
            self.assertEqual(config.get_effective_region(), "us")
            self.assertEqual(config.resolve_url("API_URL"), URL_DEFAULTS["API_URL"])

        region_warnings = [w for w in caught if issubclass(w.category, config.RegionWarning)]
        self.assertEqual(len(region_warnings), 1)
        self.assertIn("unknown Roboflow region 'bogus'", str(region_warnings[0].message))
        self.assertIn("falling back to 'us'", str(region_warnings[0].message))
        self.assertEqual(config.get_region_warning(), "unknown Roboflow region 'bogus'; falling back to 'us'.")

    def test_valid_region_has_no_region_warning(self) -> None:
        self._write_config({"ROBOFLOW_REGION": "EU"})
        config = self._reload_config()
        self.assertIsNone(config.get_region_warning())

    def test_unknown_region_is_not_printed_in_cli_json_mode(self) -> None:
        env = {key: value for key, value in os.environ.items() if key != "ROBOFLOW_API_KEY"}
        env["ROBOFLOW_REGION"] = "bogus"
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run(
            [sys.executable, "-m", "roboflow.roboflowpy", "--json", "auth", "status"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )

        self.assertEqual(result.returncode, 2, result.stderr)
        payload = json.loads(result.stderr)
        self.assertEqual(payload["error"]["message"], "Not logged in.")
        self.assertEqual(payload["region"], "us")
        self.assertIn("unknown Roboflow region 'bogus'", payload["region_warning"])


if __name__ == "__main__":
    unittest.main()
