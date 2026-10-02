"""Tests for the project CLI handler."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import responses
from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestProjectHandlerRegistration(unittest.TestCase):
    """Verify that the project handler registers correctly."""

    def test_project_list_exists(self) -> None:
        result = runner.invoke(app, ["project", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_project_list_help_shows_type(self) -> None:
        result = runner.invoke(app, ["project", "list", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("type", result.output.lower())

    def test_project_get_exists(self) -> None:
        result = runner.invoke(app, ["project", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_project_create_exists(self) -> None:
        result = runner.invoke(app, ["project", "create", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("type", result.output.lower())

    def test_project_delete_exists(self) -> None:
        result = runner.invoke(app, ["project", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())

    def test_project_restore_exists(self) -> None:
        result = runner.invoke(app, ["project", "restore", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())

    def test_subcommands_visible(self) -> None:
        result = runner.invoke(app, ["project", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("get", result.output)
        self.assertIn("create", result.output)
        self.assertIn("delete", result.output)
        self.assertIn("restore", result.output)


class TestProjectCreateHandler(unittest.TestCase):
    """project create sends the chosen type and reports server errors."""

    def _mock_create_flow(self, api_key: str, project_type: str) -> None:
        from roboflow.config import API_URL

        responses.add(responses.POST, f"{API_URL}/?api_key={api_key}", json={"workspace": "target-ws"})
        responses.add(
            responses.GET,
            f"{API_URL}/target-ws?api_key={api_key}",
            json={"workspace": {"name": "Target", "url": "target-ws", "projects": []}},
        )
        responses.add(
            responses.POST,
            f"{API_URL}/target-ws/projects?api_key={api_key}",
            json={
                "id": "target-ws/clips",
                "name": "Clips",
                "type": project_type,
                "annotation": "Clips",
                "classes": {},
                "colors": {},
                "created": 0,
                "updated": 0,
                "images": 0,
                "public": False,
                "splits": {},
                "unannotated": 0,
            },
        )

    def _assert_create_flow(self, result, api_key: str, project_type: str) -> None:
        from roboflow.config import API_URL

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            json.loads(result.stdout),
            {"id": "target-ws/clips", "name": "Clips", "type": project_type},
        )
        self.assertEqual(
            [(call.request.method, call.request.url) for call in responses.calls],
            [
                ("POST", f"{API_URL}/?api_key={api_key}"),
                ("GET", f"{API_URL}/target-ws?api_key={api_key}"),
                ("POST", f"{API_URL}/target-ws/projects?api_key={api_key}"),
            ],
        )
        self.assertEqual(
            json.loads(responses.calls[2].request.body),
            {"name": "Clips", "type": project_type, "license": "Private", "annotation": "Clips"},
        )

    @responses.activate
    def test_create_explicit_api_key_overrides_environment_key(self) -> None:
        self._mock_create_flow("override-key", "object-detection")

        with patch.dict(os.environ, {"ROBOFLOW_API_KEY": "environment-key"}):
            result = runner.invoke(
                app,
                [
                    "--json",
                    "--api-key",
                    "override-key",
                    "--workspace",
                    "target-ws",
                    "project",
                    "create",
                    "Clips",
                    "--type",
                    "object-detection",
                ],
            )

        self._assert_create_flow(result, "override-key", "object-detection")

    @responses.activate
    def test_create_uses_selected_workspaces_stored_key(self) -> None:
        self._mock_create_flow("target-key", "action-recognition")

        with tempfile.TemporaryDirectory() as config_dir:
            config_path = Path(config_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "RF_WORKSPACE": "default-ws",
                        "workspaces": {
                            "default": {"url": "default-ws", "apiKey": "default-key"},
                            "target": {"url": "target-ws", "apiKey": "target-key"},
                        },
                    }
                )
            )
            with patch.dict(
                os.environ,
                {"HOME": config_dir, "USERPROFILE": config_dir, "ROBOFLOW_CONFIG_DIR": str(config_path)},
                clear=True,
            ):
                result = runner.invoke(
                    app,
                    [
                        "--json",
                        "--workspace",
                        "target-ws",
                        "project",
                        "create",
                        "Clips",
                        "--type",
                        "action-recognition",
                    ],
                )

        self._assert_create_flow(result, "target-key", "action-recognition")

    @responses.activate
    def test_create_preserves_server_error_message(self) -> None:
        from roboflow.config import API_URL
        from roboflow.core.workspace import Workspace

        workspace = Workspace(
            {"workspace": {"name": "My WS", "url": "my-ws", "projects": []}},
            api_key="fake-key",
            default_workspace="my-ws",
            model_format="yolov8",
        )
        responses.add(
            responses.POST,
            f"{API_URL}/my-ws/projects",
            json={"error": {"message": "Invalid project type."}},
            status=422,
        )

        with patch("roboflow.Roboflow") as mock_rf:
            mock_rf.return_value.workspace.return_value = workspace
            result = runner.invoke(app, ["--json", "project", "create", "Clips", "--type", "action-recognition"])

        self.assertEqual(result.exit_code, 1)
        hint = json.loads(result.stderr)["error"]["hint"]
        self.assertEqual(hint, "Invalid project type.")


class TestProjectDeleteHandler(unittest.TestCase):
    """project delete calls rfapi.delete_project and honors --yes."""

    def _args(self, project_id="my-ws/my-proj"):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace=None,
            api_key="fake-key",
            quiet=False,
            project_id=project_id,
            yes=True,
        )

    def test_delete_calls_rfapi(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.project import _delete_project

        with patch("roboflow.adapters.rfapi.delete_project", return_value={"deleted": True}) as mock_del:
            _delete_project(self._args())
            mock_del.assert_called_once_with("fake-key", "my-ws", "my-proj")


class TestProjectRestoreHandler(unittest.TestCase):
    """project restore looks up the item in Trash by URL, then restores."""

    def _args(self, project_id="my-ws/my-proj"):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace=None,
            api_key="fake-key",
            quiet=False,
            project_id=project_id,
        )

    def test_restore_found(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.project import _restore_project

        trash = {"sections": {"projects": [{"id": "abc123", "url": "my-proj", "name": "My Project"}]}}
        with (
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash),
            patch(
                "roboflow.adapters.rfapi.restore_trash_item",
                return_value={"restored": True, "type": "project", "id": "abc123"},
            ) as mock_restore,
        ):
            _restore_project(self._args())
            mock_restore.assert_called_once_with("fake-key", "my-ws", "project", "abc123")

    def test_restore_not_in_trash(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.project import _restore_project

        # Trash doesn't contain this project — handler should error without
        # calling restore_trash_item.
        with (
            patch(
                "roboflow.adapters.rfapi.list_trash",
                return_value={"sections": {"projects": []}},
            ),
            patch("roboflow.adapters.rfapi.restore_trash_item") as mock_restore,
            patch("sys.exit"),
        ):
            _restore_project(self._args())
            mock_restore.assert_not_called()


class TestProjectHealthHandler(unittest.TestCase):
    """project health calls project.health() via SDK."""

    def _args(self, project_id="my-project", regenerate=False):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace="my-ws",
            api_key="fake-key",
            quiet=False,
            project_id=project_id,
            regenerate=regenerate,
        )

    def test_health_exists(self) -> None:
        result = runner.invoke(app, ["project", "health", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("regenerate", result.output.lower())

    def test_health_calls_sdk(self) -> None:
        from unittest.mock import MagicMock, patch

        from roboflow.cli.handlers.project import _health_project

        mock_project = MagicMock()
        mock_project.health.return_value = {"images": 100, "classes": {"cat": 50, "dog": 50}}

        mock_rf = MagicMock()
        mock_rf.workspace.return_value.project.return_value = mock_project

        with patch("roboflow.Roboflow", return_value=mock_rf):
            _health_project(self._args())
            mock_project.health.assert_called_once_with(regenerate=False)

    def test_health_regenerate(self) -> None:
        from unittest.mock import MagicMock, patch

        from roboflow.cli.handlers.project import _health_project

        mock_project = MagicMock()
        mock_project.health.return_value = {"images": 100}

        mock_rf = MagicMock()
        mock_rf.workspace.return_value.project.return_value = mock_project

        with patch("roboflow.Roboflow", return_value=mock_rf):
            _health_project(self._args(regenerate=True))
            mock_project.health.assert_called_once_with(regenerate=True)


if __name__ == "__main__":
    unittest.main()
