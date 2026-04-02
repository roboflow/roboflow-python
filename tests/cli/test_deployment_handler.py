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

    def test_deployment_add_hidden_alias(self) -> None:
        """Legacy 'add' alias should still work (hidden from help)."""
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["deployment", "add", "mydepl", "-m", "gpu-small", "-e", "test@example.com"]
        )
        self.assertIsNotNone(args.func)

    def test_deployment_create_canonical(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(
            ["deployment", "create", "mydepl", "-m", "gpu-small", "-e", "test@example.com"]
        )
        self.assertIsNotNone(args.func)

    def test_deployment_machine_type_canonical(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "machine-type"])
        self.assertIsNotNone(args.func)

    def test_deployment_machine_type_legacy_alias(self) -> None:
        """Legacy 'machine_type' alias should still work."""
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "machine_type"])
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

    def test_deployment_subparser_registered(self) -> None:
        """The 'deployment' subparser should be registered on the root parser."""
        from roboflow.cli import build_parser

        parser = build_parser()
        # Find the subparsers action
        for action in parser._actions:
            if isinstance(action, type(parser._subparsers._group_actions[0])):
                self.assertIn("deployment", action.choices)
                return
        self.fail("No subparsers action found")

    def test_deployment_usage_canonical(self) -> None:
        """The new 'usage' command accepts optional deployment name."""
        from roboflow.cli import build_parser

        parser = build_parser()
        # Workspace-wide usage (no deployment name)
        args = parser.parse_args(["deployment", "usage"])
        self.assertIsNotNone(args.func)
        self.assertIsNone(args.deployment_name)

        # Deployment-specific usage
        args = parser.parse_args(["deployment", "usage", "mydepl"])
        self.assertEqual(args.deployment_name, "mydepl")

    def test_deployment_usage_legacy_aliases(self) -> None:
        """Legacy usage_workspace and usage_deployment aliases should still work."""
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["deployment", "usage_workspace"])
        self.assertIsNotNone(args.func)

        args = parser.parse_args(["deployment", "usage_deployment", "mydepl"])
        self.assertIsNotNone(args.func)


class TestDeploymentErrorWrapping(unittest.TestCase):
    """Verify deployment errors produce structured output."""

    def test_wrapped_error_uses_structured_output(self) -> None:
        """Deployment errors should go through output_error, not bare print."""
        from roboflow.cli.handlers.deployment import _wrap

        def _fake_handler(args: object) -> None:
            print("401: Unauthorized (invalid api_key)")
            raise SystemExit(401)

        import argparse

        ns = argparse.Namespace(json=True, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap(_fake_handler)
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as ctx:
                wrapped(ns)
            self.assertLessEqual(ctx.exception.code, 3)
        import json

        err_output = stderr.getvalue().strip()
        parsed = json.loads(err_output)
        self.assertIn("error", parsed)

    def test_wrapped_success_prints_output(self) -> None:
        """On success, wrapped func should replay captured stdout."""
        from roboflow.cli.handlers.deployment import _wrap

        def _fake_handler(args: object) -> None:
            print('{"machines": []}')

        import argparse

        ns = argparse.Namespace(json=False, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap(_fake_handler)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            wrapped(ns)
        self.assertIn('{"machines": []}', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
