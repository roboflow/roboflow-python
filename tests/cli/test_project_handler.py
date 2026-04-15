"""Tests for the project CLI handler."""

import unittest

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

    def test_subcommands_visible(self) -> None:
        result = runner.invoke(app, ["project", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("list", result.output)
        self.assertIn("get", result.output)
        self.assertIn("create", result.output)


if __name__ == "__main__":
    unittest.main()
