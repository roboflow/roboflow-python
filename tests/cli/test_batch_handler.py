"""Tests for the batch CLI handler."""

import unittest

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestBatchRegistration(unittest.TestCase):
    """Verify batch handler registers expected subcommands."""

    def test_batch_app_exists(self) -> None:
        from roboflow.cli.handlers.batch import batch_app

        self.assertIsNotNone(batch_app)

    def test_batch_create_exists(self) -> None:
        result = runner.invoke(app, ["batch", "create", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_batch_status_exists(self) -> None:
        result = runner.invoke(app, ["batch", "status", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_batch_list_exists(self) -> None:
        result = runner.invoke(app, ["batch", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_batch_results_exists(self) -> None:
        result = runner.invoke(app, ["batch", "results", "--help"])
        self.assertEqual(result.exit_code, 0)


if __name__ == "__main__":
    unittest.main()
