"""Tests for the version CLI handler."""

import json
import tempfile
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestVersionHandlerRegistration(unittest.TestCase):
    """Verify that the version handler registers correctly."""

    def test_version_list_exists(self) -> None:
        result = runner.invoke(app, ["version", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_version_get_exists(self) -> None:
        result = runner.invoke(app, ["version", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_version_download_exists(self) -> None:
        result = runner.invoke(app, ["version", "download", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_version_export_exists(self) -> None:
        result = runner.invoke(app, ["version", "export", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_version_create_exists(self) -> None:
        result = runner.invoke(app, ["version", "create", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_subcommands_visible(self) -> None:
        result = runner.invoke(app, ["version", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("get", result.output)
        self.assertIn("download", result.output)
        self.assertIn("export", result.output)
        self.assertIn("create", result.output)


class TestVersionCreate(unittest.TestCase):
    """Test version create handler."""

    def test_create_missing_settings_file(self) -> None:
        result = runner.invoke(
            app,
            ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", "/nonexistent/file.json"],
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_create_invalid_json_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json")
            f.flush()
            result = runner.invoke(
                app,
                ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", f.name],
            )
            self.assertNotEqual(result.exit_code, 0)

    def test_create_no_api_key(self) -> None:
        settings = {"augmentation": {}, "preprocessing": {}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(settings, f)
            f.flush()
            with patch("roboflow.config.load_roboflow_api_key", return_value=None):
                result = runner.invoke(
                    app,
                    ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", f.name],
                )
                self.assertNotEqual(result.exit_code, 0)

    def test_create_json_error_output(self) -> None:
        result = runner.invoke(
            app,
            ["--json", "version", "create", "-p", "my-ws/my-project", "--settings", "/nonexistent/file.json"],
        )
        self.assertNotEqual(result.exit_code, 0)


class TestParseUrl(unittest.TestCase):
    """Test the _parse_url helper."""

    def test_shorthand(self) -> None:
        from roboflow.cli.handlers.version import _parse_url

        w, p, v = _parse_url("my-ws/my-project/3")
        self.assertEqual(w, "my-ws")
        self.assertEqual(p, "my-project")
        self.assertEqual(v, "3")

    def test_full_url(self) -> None:
        from roboflow.cli.handlers.version import _parse_url

        w, p, v = _parse_url("https://universe.roboflow.com/my-ws/my-project/3")
        self.assertEqual(w, "my-ws")
        self.assertEqual(p, "my-project")
        self.assertEqual(v, "3")

    def test_no_version(self) -> None:
        from roboflow.cli.handlers.version import _parse_url

        w, p, v = _parse_url("my-ws/my-project")
        self.assertEqual(w, "my-ws")
        self.assertEqual(p, "my-project")
        self.assertIsNone(v)


class TestVersionDeleteRestoreRegistration(unittest.TestCase):
    """Verify delete/restore commands register under `version`."""

    def test_version_delete_exists(self) -> None:
        result = runner.invoke(app, ["version", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())

    def test_version_restore_exists(self) -> None:
        result = runner.invoke(app, ["version", "restore", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("trash", result.output.lower())


class TestVersionDeleteHandler(unittest.TestCase):
    """version delete calls rfapi.delete_version and honors --yes."""

    def _args(self, version_ref="my-ws/my-proj/3"):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace=None,
            api_key="fake-key",
            quiet=False,
            version_ref=version_ref,
            yes=True,
        )

    def test_delete_calls_rfapi(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.version import _delete_version

        with patch(
            "roboflow.adapters.rfapi.delete_version", return_value={"deleted": True}
        ) as mock_del:
            _delete_version(self._args())
            mock_del.assert_called_once_with("fake-key", "my-ws", "my-proj", 3)


class TestVersionRestoreHandler(unittest.TestCase):
    """version restore looks up by (parentUrl, version id) in Trash."""

    def _args(self, version_ref="my-ws/my-proj/3"):
        from argparse import Namespace

        return Namespace(
            json=False,
            workspace=None,
            api_key="fake-key",
            quiet=False,
            version_ref=version_ref,
        )

    def test_restore_found(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.version import _restore_version

        trash = {
            "sections": {
                "versions": [
                    {
                        "id": "3",
                        "parentId": "proj-id-123",
                        "parentUrl": "my-proj",
                        "name": "v3",
                    }
                ]
            }
        }
        with (
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash),
            patch(
                "roboflow.adapters.rfapi.restore_trash_item",
                return_value={"restored": True, "type": "version", "id": "3"},
            ) as mock_restore,
        ):
            _restore_version(self._args())
            mock_restore.assert_called_once_with(
                "fake-key", "my-ws", "version", "3", parent_id="proj-id-123"
            )

    def test_restore_wrong_project_not_found(self) -> None:
        from unittest.mock import patch

        from roboflow.cli.handlers.version import _restore_version

        # version id matches but parentUrl doesn't — must not restore.
        trash = {
            "sections": {
                "versions": [
                    {
                        "id": "3",
                        "parentId": "other-id",
                        "parentUrl": "other-proj",
                        "name": "v3",
                    }
                ]
            }
        }
        with (
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash),
            patch("roboflow.adapters.rfapi.restore_trash_item") as mock_restore,
            patch("sys.exit"),
        ):
            _restore_version(self._args())
            mock_restore.assert_not_called()


if __name__ == "__main__":
    unittest.main()
