"""Tests for the deployment CLI handler."""

import unittest


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


if __name__ == "__main__":
    unittest.main()
