"""Tests for region-aware interactive login."""

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import responses

import roboflow
from roboflow.adapters import rfapi
from roboflow.config import refresh_region_urls


class TestLoginRegion(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temporary_directory.name, "config.json")
        self.environment = mock.patch.dict(
            os.environ,
            {
                "HOME": self.temporary_directory.name,
                # config.py builds its default path from USERPROFILE on Windows.
                "USERPROFILE": self.temporary_directory.name,
                "ROBOFLOW_CONFIG_DIR": self.config_path,
            },
            clear=True,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()
        # A region login rebinds package-wide URL constants; restore them.
        refresh_region_urls()

    @responses.activate
    def test_eu_login_uses_eu_app_and_persists_region(self) -> None:
        token = "auth-token"
        workspaces = {
            "workspace-id": {
                "url": "example-workspace",
                "apiKey": "example-api-key",
            }
        }
        responses.get(
            f"https://app.roboflow.eu/query/cliAuthToken/{token}",
            json=workspaces,
            status=200,
        )

        output = io.StringIO()
        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(output):
            roboflow.login(region="EU")

        self.assertIn("https://app.roboflow.eu/auth-cli", output.getvalue())
        self.assertEqual(responses.calls[0].request.url, f"https://app.roboflow.eu/query/cliAuthToken/{token}")
        with open(self.config_path) as config_file:
            config = json.load(config_file)
        self.assertEqual(config["ROBOFLOW_REGION"], "eu")
        self.assertEqual(config["workspaces"], workspaces)
        self.assertEqual(config["RF_WORKSPACE"], "example-workspace")

    @responses.activate
    def test_eu_login_switches_import_time_url_constants(self) -> None:
        self.assertEqual(roboflow.API_URL, "https://api.roboflow.com")
        token = "auth-token"
        responses.get(
            f"https://app.roboflow.eu/query/cliAuthToken/{token}",
            json={"workspace-id": {"url": "example-workspace", "apiKey": "example-api-key"}},
            status=200,
        )

        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(io.StringIO()):
            roboflow.login(region="eu")

        self.assertEqual(roboflow.API_URL, "https://api.roboflow.eu")
        self.assertEqual(roboflow.APP_URL, "https://app.roboflow.eu")
        self.assertEqual(rfapi.API_URL, "https://api.roboflow.eu")
        self.assertEqual(roboflow.config.API_URL, "https://api.roboflow.eu")

    @responses.activate
    def test_forced_login_preserves_existing_region_and_other_config(self) -> None:
        existing_config = {
            "ROBOFLOW_REGION": "eu",
            "API_URL": "https://custom-api.example.com",
            "workspaces": {"old": {"url": "old-workspace", "apiKey": "old-key"}},
            "RF_WORKSPACE": "old-workspace",
        }
        with open(self.config_path, "w") as config_file:
            json.dump(existing_config, config_file)

        token = "replacement-token"
        workspaces = {
            "new": {
                "url": "new-workspace",
                "apiKey": "new-key",
            }
        }
        responses.get(
            f"https://app.roboflow.eu/query/cliAuthToken/{token}",
            json=workspaces,
            status=200,
        )

        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(io.StringIO()):
            roboflow.login(force=True)

        with open(self.config_path) as config_file:
            config = json.load(config_file)
        self.assertEqual(config["ROBOFLOW_REGION"], "eu")
        self.assertEqual(config["API_URL"], "https://custom-api.example.com")
        self.assertEqual(config["workspaces"], workspaces)
        self.assertEqual(config["RF_WORKSPACE"], "new-workspace")

    def _write_config(self, config: dict) -> None:
        with open(self.config_path, "w") as config_file:
            json.dump(config, config_file)

    def _read_config(self) -> dict:
        with open(self.config_path) as config_file:
            return json.load(config_file)

    @responses.activate
    def test_region_only_config_is_not_a_session(self) -> None:
        # What `auth set-region eu` writes on a fresh install.
        self._write_config({"ROBOFLOW_REGION": "eu"})
        token = "auth-token"
        workspaces = {"workspace-id": {"url": "example-workspace", "apiKey": "example-api-key"}}
        responses.get(f"https://app.roboflow.eu/query/cliAuthToken/{token}", json=workspaces, status=200)

        output = io.StringIO()
        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(output):
            roboflow.login()

        self.assertNotIn("already logged in", output.getvalue())
        config = self._read_config()
        self.assertEqual(config["ROBOFLOW_REGION"], "eu")
        self.assertEqual(config["ROBOFLOW_CREDENTIALS_REGION"], "eu")
        self.assertEqual(config["workspaces"], workspaces)

    @responses.activate
    def test_credentials_from_other_platform_are_replaced_without_force(self) -> None:
        # US credentials, then `auth set-region eu`.
        self._write_config(
            {
                "ROBOFLOW_REGION": "eu",
                "workspaces": {"old": {"url": "us-workspace", "apiKey": "us-key"}},
                "RF_WORKSPACE": "us-workspace",
            }
        )
        token = "auth-token"
        workspaces = {"new": {"url": "eu-workspace", "apiKey": "eu-key"}}
        responses.get(f"https://app.roboflow.eu/query/cliAuthToken/{token}", json=workspaces, status=200)

        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(io.StringIO()):
            roboflow.login()

        config = self._read_config()
        self.assertEqual(config["workspaces"], workspaces)
        self.assertEqual(config["ROBOFLOW_CREDENTIALS_REGION"], "eu")

    def test_credentials_from_same_platform_keep_session(self) -> None:
        original = {
            "ROBOFLOW_CREDENTIALS_REGION": "us",
            "workspaces": {"old": {"url": "us-workspace", "apiKey": "us-key"}},
            "RF_WORKSPACE": "us-workspace",
        }
        self._write_config(original)

        output = io.StringIO()
        with mock.patch.object(roboflow, "getpass") as getpass, redirect_stdout(output):
            roboflow.login()

        getpass.assert_not_called()
        self.assertIn("already logged in", output.getvalue())
        self.assertEqual(self._read_config(), original)

    @responses.activate
    def test_environment_region_is_recorded_as_credentials_region(self) -> None:
        os.environ["ROBOFLOW_REGION"] = "eu"
        refresh_region_urls()
        token = "auth-token"
        responses.get(
            f"https://app.roboflow.eu/query/cliAuthToken/{token}",
            json={"workspace-id": {"url": "example-workspace", "apiKey": "example-api-key"}},
            status=200,
        )

        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(io.StringIO()):
            roboflow.login()

        config = self._read_config()
        self.assertEqual(config["ROBOFLOW_CREDENTIALS_REGION"], "eu")
        self.assertNotIn("ROBOFLOW_REGION", config)

    def test_explicit_region_conflicting_with_environment_is_refused(self) -> None:
        os.environ["ROBOFLOW_REGION"] = "us"

        with mock.patch.object(roboflow, "getpass") as getpass, responses.RequestsMock() as mocked:
            with self.assertRaisesRegex(ValueError, "ROBOFLOW_REGION=us.*region='eu'"):
                roboflow.login(region="eu")
            self.assertEqual(len(mocked.calls), 0)

        getpass.assert_not_called()
        self.assertFalse(os.path.exists(self.config_path))

    @responses.activate
    def test_failed_forced_login_keeps_existing_credentials(self) -> None:
        original = {
            "workspaces": {"old": {"url": "old-workspace", "apiKey": "old-key"}},
            "RF_WORKSPACE": "old-workspace",
        }
        self._write_config(original)
        token = "bad-token"
        responses.get(f"https://app.roboflow.com/query/cliAuthToken/{token}", status=500)

        with mock.patch.object(roboflow, "getpass", return_value=token), redirect_stdout(io.StringIO()):
            with self.assertRaises(Exception):
                roboflow.login(force=True)

        self.assertEqual(self._read_config(), original)

    def test_invalid_region_does_not_mutate_config(self) -> None:
        original_config = {"ROBOFLOW_REGION": "eu", "marker": "unchanged"}
        with open(self.config_path, "w") as config_file:
            json.dump(original_config, config_file)

        with self.assertRaisesRegex(ValueError, "Invalid region 'bogus'.*us, eu"):
            roboflow.login(force=True, region="bogus")

        with open(self.config_path) as config_file:
            self.assertEqual(json.load(config_file), original_config)


class TestEuAppUrls(unittest.TestCase):
    def test_download_dataset_accepts_eu_app_url(self) -> None:
        with mock.patch.object(roboflow, "initialize_roboflow") as initialize:
            workspace = initialize.return_value
            roboflow.download_dataset("https://app.roboflow.eu/eu-ws/eu-project/3", "coco", location="/tmp/x")

        initialize.assert_called_once_with(the_workspace="eu-ws")
        workspace.project.assert_called_once_with("eu-project")
        workspace.project.return_value.version.assert_called_once_with(3)

    def test_load_model_accepts_eu_app_url(self) -> None:
        with mock.patch.object(roboflow, "initialize_roboflow") as initialize:
            roboflow.load_model("https://app.roboflow.eu/eu-ws/eu-project/2")

        initialize.return_value.project.assert_called_once_with("eu-project")
        initialize.return_value.project.return_value.version.assert_called_once_with(2)

    def test_load_model_accepts_staging_app_urls(self) -> None:
        for url in (
            "https://app.roboflow.one/ws/project/2",
            "https://app.roboflow-eu.one/ws/project/2",
            "https://universe.roboflow.one/ws/project/2",
        ):
            with self.subTest(url=url), mock.patch.object(roboflow, "initialize_roboflow") as initialize:
                roboflow.load_model(url)
                initialize.return_value.project.assert_called_once_with("project")

    def test_unknown_host_is_still_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "app.roboflow.eu"):
            roboflow.download_dataset("https://example.com/ws/project/1", "coco")


if __name__ == "__main__":
    unittest.main()
