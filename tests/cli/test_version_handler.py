"""Tests for the version CLI handler."""

import io
import json
import sys
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


if __name__ == "__main__":
    unittest.main()
