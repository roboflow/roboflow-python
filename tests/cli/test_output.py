"""Unit tests for roboflow.cli._output."""

import io
import json
import sys
import types
import unittest


class TestOutput(unittest.TestCase):
    """Tests for the output() helper."""

    def _make_args(self, *, json_mode: bool = False) -> types.SimpleNamespace:
        return types.SimpleNamespace(json=json_mode)

    def test_json_mode_prints_json(self) -> None:
        from roboflow.cli._output import output

        args = self._make_args(json_mode=True)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            output(args, data={"key": "value"}, text="human text")
        finally:
            sys.stdout = old_stdout
        result = json.loads(buf.getvalue())
        self.assertEqual(result, {"key": "value"})

    def test_text_mode_prints_text(self) -> None:
        from roboflow.cli._output import output

        args = self._make_args(json_mode=False)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            output(args, data={"key": "value"}, text="human text")
        finally:
            sys.stdout = old_stdout
        self.assertEqual(buf.getvalue().strip(), "human text")

    def test_text_mode_falls_back_to_json_when_no_text(self) -> None:
        from roboflow.cli._output import output

        args = self._make_args(json_mode=False)
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            output(args, data={"fallback": True})
        finally:
            sys.stdout = old_stdout
        result = json.loads(buf.getvalue())
        self.assertTrue(result["fallback"])

    def test_output_error_json_mode(self) -> None:
        from roboflow.cli._output import output_error

        args = self._make_args(json_mode=True)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                output_error(args, "something broke", hint="try again", exit_code=1)
        finally:
            sys.stderr = old_stderr
        self.assertEqual(ctx.exception.code, 1)
        result = json.loads(buf.getvalue())
        self.assertEqual(result["error"]["message"], "something broke")
        self.assertEqual(result["error"]["hint"], "try again")

    def test_output_error_text_mode(self) -> None:
        from roboflow.cli._output import output_error

        args = self._make_args(json_mode=False)
        buf = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = buf
        try:
            with self.assertRaises(SystemExit) as ctx:
                output_error(args, "not found", exit_code=3)
        finally:
            sys.stderr = old_stderr
        self.assertEqual(ctx.exception.code, 3)
        self.assertIn("not found", buf.getvalue())


class TestTable(unittest.TestCase):
    """Tests for the format_table() helper."""

    def test_empty_rows(self) -> None:
        from roboflow.cli._table import format_table

        result = format_table([], ["a", "b"])
        self.assertEqual(result, "(no results)")

    def test_basic_table(self) -> None:
        from roboflow.cli._table import format_table

        rows = [
            {"name": "proj-a", "type": "object-detection"},
            {"name": "proj-b", "type": "classification"},
        ]
        result = format_table(rows, ["name", "type"])
        lines = result.split("\n")
        self.assertEqual(len(lines), 4)  # header + separator + 2 rows
        self.assertIn("NAME", lines[0])
        self.assertIn("TYPE", lines[0])
        self.assertIn("proj-a", lines[2])


class TestResolver(unittest.TestCase):
    """Tests for the resource shorthand resolver."""

    def test_single_segment(self) -> None:
        from roboflow.cli._resolver import resolve_resource

        ws, proj, ver = resolve_resource("my-project", workspace_override="default-ws")
        self.assertEqual(ws, "default-ws")
        self.assertEqual(proj, "my-project")
        self.assertIsNone(ver)

    def test_workspace_project(self) -> None:
        from roboflow.cli._resolver import resolve_resource

        ws, proj, ver = resolve_resource("my-ws/my-project")
        self.assertEqual(ws, "my-ws")
        self.assertEqual(proj, "my-project")
        self.assertIsNone(ver)

    def test_project_version(self) -> None:
        from roboflow.cli._resolver import resolve_resource

        ws, proj, ver = resolve_resource("my-project/3", workspace_override="default-ws")
        self.assertEqual(ws, "default-ws")
        self.assertEqual(proj, "my-project")
        self.assertEqual(ver, 3)

    def test_full_triple(self) -> None:
        from roboflow.cli._resolver import resolve_resource

        ws, proj, ver = resolve_resource("my-ws/my-project/42")
        self.assertEqual(ws, "my-ws")
        self.assertEqual(proj, "my-project")
        self.assertEqual(ver, 42)

    def test_no_workspace_raises(self) -> None:
        from unittest.mock import patch

        from roboflow.cli._resolver import resolve_resource

        with patch("roboflow.cli._resolver.get_conditional_configuration_variable", return_value=None):
            with self.assertRaises(ValueError):
                resolve_resource("my-project")  # no override, no default

    def test_too_many_segments_raises(self) -> None:
        from roboflow.cli._resolver import resolve_resource

        with self.assertRaises(ValueError):
            resolve_resource("a/b/c/d")


if __name__ == "__main__":
    unittest.main()
