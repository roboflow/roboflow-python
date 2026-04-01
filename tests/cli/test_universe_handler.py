"""Tests for the universe CLI handler."""

import unittest


class TestUniverseRegistration(unittest.TestCase):
    """Verify universe handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.universe import register

        self.assertTrue(callable(register))

    def test_universe_search_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "cats"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.query, "cats")

    def test_universe_search_with_flags(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["universe", "search", "dogs", "--type", "model", "--limit", "5"])
        self.assertEqual(args.type, "model")
        self.assertEqual(args.limit, 5)


if __name__ == "__main__":
    unittest.main()
