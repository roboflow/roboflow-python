"""Unit tests for roboflow.cli.handlers.annotation."""

import argparse
import io
import sys
import types
import unittest


def _build_annotation_parser():
    """Build a minimal parser with just the annotation handler registered."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", "-j", action="store_true", default=False)
    parser.add_argument("--api-key", "-k", dest="api_key", default=None)
    parser.add_argument("--workspace", "-w", dest="workspace", default=None)
    parser.add_argument("--quiet", "-q", action="store_true", default=False)
    sub = parser.add_subparsers(title="commands", dest="command")

    from roboflow.cli.handlers.annotation import register

    register(sub)
    return parser


class TestAnnotationParserRegistration(unittest.TestCase):
    """Verify the annotation handler registers its subcommands."""

    def test_annotation_subcommand_exists(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "list", "-p", "proj"])
        self.assertEqual(args.project, "proj")
        self.assertTrue(callable(args.func))

    def test_annotation_batch_get(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "batch", "get", "batch-1", "-p", "proj"])
        self.assertEqual(args.batch_id, "batch-1")
        self.assertEqual(args.project, "proj")

    def test_annotation_job_list(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "job", "list", "-p", "proj"])
        self.assertEqual(args.project, "proj")

    def test_annotation_job_get(self):
        parser = _build_annotation_parser()
        args = parser.parse_args(["annotation", "job", "get", "job-1", "-p", "proj"])
        self.assertEqual(args.job_id, "job-1")

    def test_annotation_job_create(self):
        parser = _build_annotation_parser()
        args = parser.parse_args([
            "annotation", "job", "create",
            "-p", "proj",
            "--name", "my-job",
            "--batch", "batch-1",
            "--assignees", "a@b.com,c@d.com",
        ])
        self.assertEqual(args.name, "my-job")
        self.assertEqual(args.batch, "batch-1")
        self.assertEqual(args.assignees, "a@b.com,c@d.com")


class TestAnnotationStub(unittest.TestCase):
    """Verify stub handlers print not-yet-implemented."""

    def test_stub_prints_message(self):
        from roboflow.cli.handlers.annotation import _stub

        args = types.SimpleNamespace(json=False)

        buf = io.StringIO()
        old = sys.stderr
        sys.stderr = buf
        try:
            _stub(args)
        finally:
            sys.stderr = old

        self.assertIn("not yet implemented", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
