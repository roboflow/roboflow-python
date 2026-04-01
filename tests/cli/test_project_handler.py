"""Tests for the project CLI handler."""

import argparse
import unittest


def _make_parser() -> argparse.ArgumentParser:
    """Build a minimal parser with just the project handler."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", default=False)
    parser.add_argument("--api-key", dest="api_key", default=None)
    parser.add_argument("--workspace", "-w", dest="workspace", default=None)
    subs = parser.add_subparsers(dest="command")

    from roboflow.cli.handlers.project import register

    register(subs)
    return parser


class TestProjectHandlerRegistration(unittest.TestCase):
    """Verify that the project handler registers correctly."""

    def test_register_creates_project_subcommand(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["project", "list"])
        self.assertIsNotNone(args.func)

    def test_project_list_defaults(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["project", "list"])
        self.assertIsNone(args.type)

    def test_project_list_with_type_filter(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["project", "list", "--type", "classification"])
        self.assertEqual(args.type, "classification")

    def test_project_get_requires_id(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["project", "get"])

    def test_project_get_parses_id(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["project", "get", "my-project"])
        self.assertEqual(args.project_id, "my-project")

    def test_project_create_requires_name_and_type(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["project", "create"])

    def test_project_create_parses_args(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["project", "create", "My Project", "--type", "object-detection"])
        self.assertEqual(args.name, "My Project")
        self.assertEqual(args.type, "object-detection")

    def test_project_create_rejects_invalid_type(self) -> None:
        parser = _make_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["project", "create", "My Project", "--type", "invalid-type"])

    def test_project_create_default_license(self) -> None:
        parser = _make_parser()
        args = parser.parse_args(["project", "create", "Test", "--type", "classification"])
        self.assertEqual(args.license, "Private")

    def test_subcommands_have_func(self) -> None:
        parser = _make_parser()
        for subcmd in ["list", "get my-proj", "create Foo --type classification"]:
            args = parser.parse_args(["project"] + subcmd.split())
            self.assertIsNotNone(args.func, f"project {subcmd} has no func")


if __name__ == "__main__":
    unittest.main()
