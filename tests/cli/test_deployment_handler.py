"""Tests for the deployment CLI handler."""

import argparse
import io
import json
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestDeploymentRegistration(unittest.TestCase):
    """Verify deployment handler registers expected subcommands."""

    def test_deployment_app_exists(self) -> None:
        from roboflow.cli.handlers.deployment import deployment_app

        self.assertIsNotNone(deployment_app)

    def test_deployment_subcommand_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_create_canonical(self) -> None:
        result = runner.invoke(app, ["deployment", "create", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_machine_type_canonical(self) -> None:
        result = runner.invoke(app, ["deployment", "machine-type", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_get_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_delete_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_usage_canonical(self) -> None:
        result = runner.invoke(app, ["deployment", "usage", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestDeploymentErrorWrapping(unittest.TestCase):
    """Verify deployment errors produce structured output."""

    def test_wrapped_error_uses_structured_output(self) -> None:
        """Deployment errors should go through output_error, not bare print."""
        from roboflow.cli.handlers.deployment import _wrap

        def _fake_handler(args: object) -> None:
            print("401: Unauthorized (invalid api_key)")
            raise SystemExit(401)

        ns = argparse.Namespace(json=True, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap(_fake_handler)
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as ctx:
                wrapped(ns)
            self.assertLessEqual(ctx.exception.code, 3)
        err_output = stderr.getvalue().strip()
        parsed = json.loads(err_output)
        self.assertIn("error", parsed)

    def test_wrapped_success_prints_output(self) -> None:
        """On success, wrapped func should replay captured stdout."""
        from roboflow.cli.handlers.deployment import _wrap

        def _fake_handler(args: object) -> None:
            print('{"machines": []}')

        ns = argparse.Namespace(json=False, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap(_fake_handler)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            wrapped(ns)
        self.assertIn('{"machines": []}', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
