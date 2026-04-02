"""Tests for the completion CLI handler."""

import unittest


class TestCompletionRegistration(unittest.TestCase):
    """Verify completion handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.completion import register

        self.assertTrue(callable(register))

    def test_completion_bash_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["completion", "bash"])
        self.assertIsNotNone(args.func)

    def test_completion_zsh_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["completion", "zsh"])
        self.assertIsNotNone(args.func)

    def test_completion_fish_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["completion", "fish"])
        self.assertIsNotNone(args.func)


if __name__ == "__main__":
    unittest.main()
