"""Tests for the batch CLI handler."""

import unittest


class TestBatchRegistration(unittest.TestCase):
    """Verify batch handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.batch import register

        self.assertTrue(callable(register))

    def test_batch_create_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["batch", "create", "--workflow", "wf-1", "--input", "/tmp/imgs"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.workflow, "wf-1")
        self.assertEqual(args.input, "/tmp/imgs")

    def test_batch_status_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["batch", "status", "job-abc"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.job_id, "job-abc")

    def test_batch_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["batch", "list"])
        self.assertIsNotNone(args.func)

    def test_batch_results_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["batch", "results", "job-abc"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.job_id, "job-abc")


if __name__ == "__main__":
    unittest.main()
