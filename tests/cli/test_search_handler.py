"""Tests for the search CLI handler."""

import unittest


class TestSearchRegistration(unittest.TestCase):
    """Verify search handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.search import register

        self.assertTrue(callable(register))

    def test_search_subcommand_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["search", "tag:review"])
        self.assertIsNotNone(args.func)

    def test_search_defaults(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["search", "tag:review"])
        self.assertEqual(args.query, "tag:review")
        self.assertEqual(args.limit, 50)
        self.assertIsNone(args.cursor)
        self.assertIsNone(args.fields)
        self.assertFalse(args.export)
        self.assertEqual(args.format, "coco")
        self.assertIsNone(args.location)
        self.assertIsNone(args.dataset)
        self.assertIsNone(args.name)
        self.assertFalse(args.no_extract)

    def test_search_with_export_flag(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["search", "*", "--export", "-f", "yolov8", "--no-extract"])
        self.assertTrue(args.export)
        self.assertEqual(args.format, "yolov8")
        self.assertTrue(args.no_extract)

    def test_search_with_pagination(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["search", "class:car", "--limit", "10", "--cursor", "abc123"])
        self.assertEqual(args.limit, 10)
        self.assertEqual(args.cursor, "abc123")


if __name__ == "__main__":
    unittest.main()
