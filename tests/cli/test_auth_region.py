"""Region-specific tests for the auth CLI handler."""

import json
import os
import tempfile
import unittest
from unittest import mock

import responses
from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestAuthRegion(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.tempdir.name, "config.json")
        self.env_patch = mock.patch.dict(
            os.environ,
            {"ROBOFLOW_CONFIG_DIR": self.config_path},
            clear=False,
        )
        self.env_patch.start()
        for key in ("ROBOFLOW_REGION", "API_URL", "APP_URL", "ROBOFLOW_API_KEY"):
            os.environ.pop(key, None)

    def tearDown(self) -> None:
        self.env_patch.stop()
        self.tempdir.cleanup()

    def _write_config(self, config: dict) -> None:
        with open(self.config_path, "w") as config_file:
            json.dump(config, config_file)

    def _read_config(self) -> dict:
        with open(self.config_path) as config_file:
            return json.load(config_file)

    def _write_logged_in_config(self) -> None:
        self._write_config(
            {
                "workspaces": {
                    "eu-workspace": {
                        "url": "eu-workspace",
                        "name": "EU Workspace",
                        "apiKey": "eu-secret-key",
                    }
                },
                "RF_WORKSPACE": "eu-workspace",
            }
        )

    def test_login_and_alias_help_include_region(self) -> None:
        auth_result = runner.invoke(app, ["auth", "login", "--help"])
        alias_result = runner.invoke(app, ["login", "--help"])

        self.assertEqual(auth_result.exit_code, 0)
        self.assertEqual(alias_result.exit_code, 0)
        self.assertIn("--region", auth_result.output)
        self.assertIn("--region", alias_result.output)

    def test_interactive_login_passes_normalized_region(self) -> None:
        with mock.patch("roboflow.login") as login:
            result = runner.invoke(app, ["auth", "login", "--region", "EU"])

        self.assertEqual(result.exit_code, 0, result.output)
        login.assert_called_once_with(workspace=None, force=False, region="eu")

    def test_login_alias_passes_normalized_region(self) -> None:
        with mock.patch("roboflow.login") as login:
            result = runner.invoke(app, ["login", "--region", "EU"])

        self.assertEqual(result.exit_code, 0, result.output)
        login.assert_called_once_with(workspace=None, force=False, region="eu")

    @responses.activate
    def test_api_key_login_uses_eu_api_and_persists_region(self) -> None:
        responses.add(
            responses.POST,
            "https://api.roboflow.eu/?api_key=eu-key",
            json={"workspace": "eu-workspace"},
            status=200,
        )
        responses.add(
            responses.GET,
            "https://api.roboflow.eu/eu-workspace?api_key=eu-key",
            json={"workspace": {"name": "EU Workspace"}},
            status=200,
        )

        result = runner.invoke(
            app,
            ["auth", "login", "--api-key", "eu-key", "--region", "eu"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            [call.request.url for call in responses.calls],
            [
                "https://api.roboflow.eu/?api_key=eu-key",
                "https://api.roboflow.eu/eu-workspace?api_key=eu-key",
            ],
        )
        config = self._read_config()
        self.assertEqual(config["ROBOFLOW_REGION"], "eu")
        self.assertEqual(config["RF_WORKSPACE"], "eu-workspace")
        self.assertEqual(config["workspaces"]["eu-workspace"]["apiKey"], "eu-key")

    def test_set_region_then_status_shows_eu_endpoints(self) -> None:
        self._write_logged_in_config()

        set_result = runner.invoke(app, ["auth", "set-region", "eu"])
        status_result = runner.invoke(app, ["auth", "status"])

        self.assertEqual(set_result.exit_code, 0, set_result.output)
        self.assertIn("Region set to: eu", set_result.output)
        self.assertIn("API URL: https://api.roboflow.eu", set_result.output)
        self.assertIn("App URL: https://app.roboflow.eu", set_result.output)
        self.assertIn("separate authentication backends and API keys", set_result.output)
        self.assertIn("roboflow auth login --force", set_result.output)
        self.assertEqual(status_result.exit_code, 0, status_result.output)
        self.assertIn("Region: eu", status_result.output)
        self.assertIn("API URL: https://api.roboflow.eu", status_result.output)
        self.assertIn("App URL: https://app.roboflow.eu", status_result.output)
        self.assertEqual(self._read_config()["ROBOFLOW_REGION"], "eu")

    def test_status_json_includes_region_and_urls(self) -> None:
        self._write_logged_in_config()
        runner.invoke(app, ["auth", "set-region", "eu"])

        result = runner.invoke(app, ["--json", "auth", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["region"], "eu")
        self.assertEqual(payload["api_url"], "https://api.roboflow.eu")
        self.assertEqual(payload["app_url"], "https://app.roboflow.eu")

    def test_set_region_reports_environment_override_as_effective(self) -> None:
        os.environ["ROBOFLOW_REGION"] = "us"

        result = runner.invoke(app, ["--json", "auth", "set-region", "eu"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["configured_region"], "eu")
        self.assertEqual(payload["region"], "us")
        self.assertEqual(payload["api_url"], "https://api.roboflow.com")
        self.assertEqual(payload["app_url"], "https://app.roboflow.com")
        self.assertEqual(self._read_config()["ROBOFLOW_REGION"], "eu")

    def test_region_only_status_shows_endpoints_and_remains_not_logged_in(self) -> None:
        set_result = runner.invoke(app, ["auth", "set-region", "eu"])
        status_result = runner.invoke(app, ["--json", "auth", "status"])

        self.assertEqual(set_result.exit_code, 0, set_result.output)
        self.assertEqual(status_result.exit_code, 2, status_result.output)
        payload = json.loads(status_result.stderr)
        self.assertEqual(payload["error"]["message"], "Not logged in.")
        self.assertEqual(payload["region"], "eu")
        self.assertEqual(payload["api_url"], "https://api.roboflow.eu")
        self.assertEqual(payload["app_url"], "https://app.roboflow.eu")

    def test_set_region_rejects_invalid_value_without_mutating_config(self) -> None:
        original = {"ROBOFLOW_REGION": "us", "preserved": True}
        self._write_config(original)

        result = runner.invoke(app, ["auth", "set-region", "bogus"])

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Invalid region 'bogus'", result.output)
        self.assertIn("must be 'us' or 'eu'", result.output)
        self.assertEqual(self._read_config(), original)


if __name__ == "__main__":
    unittest.main()
