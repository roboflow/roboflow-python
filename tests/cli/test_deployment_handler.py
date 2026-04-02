"""Tests for the deployment CLI handler."""

import io
import sys
import unittest
from unittest.mock import patch


class TestDeploymentRegistration(unittest.TestCase):
    """Verify deployment handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.deployment import register

        self.assertTrue(callable(register))

    def test_deployment_subcommand_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "list"])
        self.assertIsNotNone(args.func)

    def test_deployment_add_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["deployment", "add", "mydepl", "-m", "gpu-small", "-e", "test@example.com"]
        )
        self.assertIsNotNone(args.func)

    def test_deployment_create_alias(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["deployment", "create", "mydepl", "-m", "gpu-small", "-e", "test@example.com"]
        )
        self.assertIsNotNone(args.func)

    def test_deployment_machine_type_alias(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "machine-type"])
        self.assertIsNotNone(args.func)

    def test_deployment_get_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "get", "mydepl"])
        self.assertIsNotNone(args.func)

    def test_deployment_delete_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "delete", "mydepl"])
        self.assertIsNotNone(args.func)

    def test_deployment_no_subcommand_shows_own_help(self) -> None:
        """Running 'roboflow deployment' should show deployment help, not top-level help."""
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment"])
        self.assertIsNotNone(args.func)
        # Calling func should print deployment help (containing 'deployment subcommands')
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            args.func(args)
        self.assertIn("deployment subcommands", captured.getvalue())


class TestDeploymentErrorWrapping(unittest.TestCase):
    """Verify deployment errors produce structured output."""

    def test_wrapped_error_uses_structured_output(self) -> None:
        """Deployment errors should go through output_error, not bare print."""
        from roboflow.cli.handlers.deployment import _wrap_deployment_func

        def _fake_handler(args: object) -> None:
            print("401: Unauthorized (invalid api_key)")
            raise SystemExit(401)

        import argparse

        ns = argparse.Namespace(json=True, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap_deployment_func(_fake_handler)
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as ctx:
                wrapped(ns)
            # Exit code should be normalised (<=3)
            self.assertLessEqual(ctx.exception.code, 3)
        # stderr should contain JSON with "error" key
        import json

        err_output = stderr.getvalue().strip()
        parsed = json.loads(err_output)
        self.assertIn("error", parsed)
        self.assertIn("401", parsed["error"]["message"])

    def test_wrapped_success_prints_output(self) -> None:
        """On success, wrapped func should replay captured stdout."""
        from roboflow.cli.handlers.deployment import _wrap_deployment_func

        def _fake_handler(args: object) -> None:
            print('{"machines": []}')

        import argparse

        ns = argparse.Namespace(json=False, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap_deployment_func(_fake_handler)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            wrapped(ns)
        self.assertIn('{"machines": []}', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
