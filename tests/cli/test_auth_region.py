"""Region-specific tests for the auth CLI handler."""

import json
import os
import re
import tempfile
import unittest
from unittest import mock

import responses
from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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
        for key in ("ROBOFLOW_REGION", "ROBOFLOW_ENVIRONMENT", "API_URL", "APP_URL", "ROBOFLOW_API_KEY"):
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
        # Rich styles "--region" per segment when colors are forced (as in CI).
        self.assertIn("--region", _strip_ansi(auth_result.output))
        self.assertIn("--region", _strip_ansi(alias_result.output))

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

    def test_login_with_new_region_reauthenticates_existing_user(self) -> None:
        self._write_logged_in_config()

        with mock.patch("roboflow.login") as login:
            result = runner.invoke(app, ["auth", "login", "--region", "eu"])

        self.assertEqual(result.exit_code, 0, result.output)
        login.assert_called_once_with(workspace=None, force=True, region="eu")

    def test_login_with_current_region_keeps_existing_session(self) -> None:
        self._write_logged_in_config()

        with mock.patch("roboflow.login") as login:
            result = runner.invoke(app, ["auth", "login", "--region", "us"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Already logged in", result.output)
        login.assert_not_called()

    def test_login_after_set_region_on_fresh_install_authenticates(self) -> None:
        set_result = runner.invoke(app, ["auth", "set-region", "eu"])
        for command in (["auth", "login", "--region", "eu"], ["auth", "login"]):
            with self.subTest(command=command), mock.patch("roboflow.login") as login:
                result = runner.invoke(app, command)

                self.assertEqual(result.exit_code, 0, result.output)
                self.assertNotIn("Already logged in", result.output)
                login.assert_called_once()
        self.assertEqual(set_result.exit_code, 0, set_result.output)

    def test_set_region_does_not_relabel_existing_credentials(self) -> None:
        # US credentials, then a switch of the routing preference only.
        self._write_logged_in_config()
        runner.invoke(app, ["auth", "set-region", "eu"])

        for command, region in ((["auth", "login", "--region", "eu"], "eu"), (["auth", "login"], None)):
            with self.subTest(command=command), mock.patch("roboflow.login") as login:
                result = runner.invoke(app, command)

                self.assertEqual(result.exit_code, 0, result.output)
                login.assert_called_once_with(workspace=None, force=True, region=region)

    def test_login_region_conflicting_with_environment_is_refused(self) -> None:
        os.environ["ROBOFLOW_REGION"] = "us"

        with mock.patch("roboflow.login") as login, responses.RequestsMock() as mocked:
            for command in (
                ["auth", "login", "--region", "eu"],
                ["auth", "login", "--api-key", "eu-key", "--region", "eu"],
            ):
                with self.subTest(command=command):
                    result = runner.invoke(app, command)

                    self.assertEqual(result.exit_code, 2, result.output)
                    self.assertIn("ROBOFLOW_REGION=us", result.output)
            self.assertEqual(len(mocked.calls), 0)
        login.assert_not_called()
        self.assertFalse(os.path.exists(self.config_path))

    @responses.activate
    def test_api_key_login_on_other_platform_replaces_workspaces(self) -> None:
        self._write_logged_in_config()
        responses.add(
            responses.POST, "https://api.roboflow.eu/?api_key=eu-key", json={"workspace": "new-eu"}, status=200
        )
        responses.add(
            responses.GET,
            "https://api.roboflow.eu/new-eu?api_key=eu-key",
            json={"workspace": {"name": "New EU"}},
            status=200,
        )

        result = runner.invoke(app, ["auth", "login", "--api-key", "eu-key", "--region", "eu"])

        self.assertEqual(result.exit_code, 0, result.output)
        config = self._read_config()
        self.assertEqual(list(config["workspaces"]), ["new-eu"])
        self.assertEqual(config["ROBOFLOW_CREDENTIALS_REGION"], "eu")

    def test_set_region_without_credentials_has_no_warning(self) -> None:
        result = runner.invoke(app, ["--json", "auth", "set-region", "eu"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(json.loads(result.stdout)["warning"])

    def test_set_region_to_current_region_has_no_warning(self) -> None:
        self._write_logged_in_config()

        result = runner.invoke(app, ["auth", "set-region", "us"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertNotIn("Warning", result.output)

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

    def test_status_json_includes_staging_environment(self) -> None:
        self._write_logged_in_config()
        os.environ["ROBOFLOW_REGION"] = "eu"
        os.environ["ROBOFLOW_ENVIRONMENT"] = "staging"

        result = runner.invoke(app, ["--json", "auth", "status"])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["region"], "eu")
        self.assertEqual(payload["environment"], "staging")
        self.assertEqual(payload["api_url"], "https://api.roboflow-eu.one")
        self.assertEqual(payload["app_url"], "https://app.roboflow-eu.one")

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
