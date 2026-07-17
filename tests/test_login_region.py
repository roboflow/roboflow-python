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


class TestLoginRegion(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temporary_directory.name, "config.json")
        self.environment = mock.patch.dict(
            os.environ,
            {
                "HOME": self.temporary_directory.name,
                "ROBOFLOW_CONFIG_DIR": self.config_path,
            },
            clear=True,
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

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

    def test_invalid_region_does_not_mutate_config(self) -> None:
        original_config = {"ROBOFLOW_REGION": "eu", "marker": "unchanged"}
        with open(self.config_path, "w") as config_file:
            json.dump(original_config, config_file)

        with self.assertRaisesRegex(ValueError, "Invalid region 'bogus'.*us, eu"):
            roboflow.login(force=True, region="bogus")

        with open(self.config_path) as config_file:
            self.assertEqual(json.load(config_file), original_config)


if __name__ == "__main__":
    unittest.main()
