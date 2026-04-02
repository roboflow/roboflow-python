"""Tests for _reorder_argv — global flag reordering."""

import unittest


class TestReorderArgv(unittest.TestCase):
    """Verify _reorder_argv moves global flags before subcommands."""

    def _reorder(self, argv: list[str]) -> list[str]:
        from roboflow.cli import _reorder_argv

        return _reorder_argv(argv)

    def test_no_flags(self) -> None:
        self.assertEqual(self._reorder(["project", "list"]), ["project", "list"])

    def test_empty(self) -> None:
        self.assertEqual(self._reorder([]), [])

    def test_bool_flag_after_subcommand(self) -> None:
        result = self._reorder(["project", "list", "--json"])
        self.assertEqual(result, ["--json", "project", "list"])

    def test_bool_flag_already_first(self) -> None:
        result = self._reorder(["--json", "project", "list"])
        self.assertEqual(result, ["--json", "project", "list"])

    def test_short_bool_flag(self) -> None:
        result = self._reorder(["project", "list", "-j"])
        self.assertEqual(result, ["-j", "project", "list"])

    def test_value_flag_after_subcommand(self) -> None:
        result = self._reorder(["project", "list", "--api-key", "abc123"])
        self.assertEqual(result, ["--api-key", "abc123", "project", "list"])

    def test_short_value_flag(self) -> None:
        result = self._reorder(["project", "list", "-k", "abc123"])
        self.assertEqual(result, ["-k", "abc123", "project", "list"])

    def test_multiple_flags_mixed(self) -> None:
        result = self._reorder(["project", "list", "--json", "-w", "my-ws"])
        self.assertEqual(result, ["--json", "-w", "my-ws", "project", "list"])

    def test_value_flag_at_end_without_value(self) -> None:
        """A value flag at the very end with no following arg should still be moved."""
        result = self._reorder(["project", "list", "--api-key"])
        self.assertEqual(result, ["--api-key", "project", "list"])

    def test_non_global_flags_preserved(self) -> None:
        """Flags not in the global set stay in place."""
        result = self._reorder(["image", "upload", "--project", "my-proj", "--json"])
        self.assertEqual(result, ["--json", "image", "upload", "--project", "my-proj"])

    def test_quiet_and_version_flags(self) -> None:
        result = self._reorder(["project", "list", "--quiet", "--version"])
        self.assertEqual(result, ["--quiet", "--version", "project", "list"])

    def test_workspace_flag(self) -> None:
        result = self._reorder(["project", "list", "--workspace", "ws-1"])
        self.assertEqual(result, ["--workspace", "ws-1", "project", "list"])

    def test_preserves_subcommand_positional_args(self) -> None:
        result = self._reorder(["version", "download", "ws/proj/3", "--json", "-f", "yolov8"])
        self.assertEqual(result, ["--json", "version", "download", "ws/proj/3", "-f", "yolov8"])


if __name__ == "__main__":
    unittest.main()
