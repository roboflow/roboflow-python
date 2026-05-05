"""Tests for the completion CLI handler.

Covers script generation and the install flow (which delegates to
``typer.completion.install``).
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import click
from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestCompletionRegistration(unittest.TestCase):
    """Verify completion handler registers expected subcommands."""

    def test_completion_app_exists(self) -> None:
        from roboflow.cli.handlers.completion import completion_app

        self.assertIsNotNone(completion_app)

    def test_completion_bash_exists(self) -> None:
        result = runner.invoke(app, ["completion", "bash", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_completion_zsh_exists(self) -> None:
        result = runner.invoke(app, ["completion", "zsh", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_completion_fish_exists(self) -> None:
        result = runner.invoke(app, ["completion", "fish", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_completion_install_exists(self) -> None:
        result = runner.invoke(app, ["completion", "install", "--help"])
        self.assertEqual(result.exit_code, 0)


class TestCompletionScriptGeneration(unittest.TestCase):
    """Raw script generation paths (`completion bash|zsh|fish`)."""

    def test_bash_script_contains_marker(self) -> None:
        result = runner.invoke(app, ["completion", "bash"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("_ROBOFLOW_COMPLETE", result.output)

    def test_zsh_script_contains_marker(self) -> None:
        result = runner.invoke(app, ["completion", "zsh"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("_ROBOFLOW_COMPLETE", result.output)
        self.assertIn("complete_zsh", result.output)
        self.assertNotIn("zsh_complete", result.output)
        self.assertIn("compdef", result.output)

    def test_fish_script_contains_marker(self) -> None:
        result = runner.invoke(app, ["completion", "fish"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("_ROBOFLOW_COMPLETE", result.output)

    def test_hidden_commands_filtered_from_completion(self) -> None:
        """Click filters hidden commands from completion at iteration time.

        Anything registered with ``hidden=True`` (legacy aliases, snake_case
        shims, stubbed groups) must not be visible. We replicate Click's
        filter without depending on a private symbol.
        """
        import typer

        from roboflow.cli import app as rf_app

        click_app = typer.main.get_command(rf_app)
        ctx = click.Context(click_app, info_name="roboflow")
        visible = {
            name
            for name in click_app.list_commands(ctx)
            if (cmd := click_app.get_command(ctx, name)) is not None and not cmd.hidden
        }
        hidden_examples = {
            "download",
            "login",
            "whoami",
            "upload",
            "import",
            "search-export",
            "upload_model",
            "get_workspace_info",
            "run_video_inference_api",
            "help",
            "batch",
        }
        leaked = hidden_examples & visible
        self.assertFalse(leaked, f"Hidden commands leaked into completion: {leaked}")

    def test_bad_completion_invocation_exits_without_traceback(self) -> None:
        from roboflow.cli import main

        with mock.patch.dict(os.environ, {"_ROBOFLOW_COMPLETE": "bash_complete"}, clear=False):
            os.environ.pop("COMP_WORDS", None)
            os.environ.pop("COMP_CWORD", None)
            with mock.patch.object(sys, "argv", ["roboflow", "im"]):
                with self.assertRaises(SystemExit) as exc:
                    main()
        self.assertEqual(exc.exception.code, 0)


class _IsolatedHomeMixin:
    """Mixin: isolated $HOME, $SHELL=zsh, and roboflow-on-PATH stub."""

    def setUp(self) -> None:  # type: ignore[override]
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.home = Path(self.tmpdir.name)

        # Stub `shutil.which("roboflow")` only — Click's BashComplete also
        # calls shutil.which("bash") for version detection; don't intercept
        # that.
        import shutil as _shutil

        real_which = _shutil.which

        def _fake_which(cmd, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            if cmd == "roboflow":
                return "/usr/local/bin/roboflow"
            return real_which(cmd, *args, **kwargs)

        self._which_patch = mock.patch.object(_shutil, "which", side_effect=_fake_which)
        self._which_patch.start()
        self.addCleanup(self._which_patch.stop)

        self._env_patch = mock.patch.dict(
            os.environ,
            {"HOME": str(self.home), "SHELL": "/bin/zsh"},
            clear=False,
        )
        self._env_patch.start()
        self.addCleanup(self._env_patch.stop)

        # typer.completion.install reads Path.home() to pick rc/script paths.
        self._home_patch = mock.patch.object(Path, "home", return_value=self.home)
        self._home_patch.start()
        self.addCleanup(self._home_patch.stop)


class TestInstall(_IsolatedHomeMixin, unittest.TestCase):
    def test_install_zsh_writes_file(self) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "zsh"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        target = self.home / ".zfunc" / "_roboflow"
        self.assertTrue(target.exists())
        self.assertIn("_ROBOFLOW_COMPLETE", target.read_text())

    def test_install_bash_writes_file(self) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "bash"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        target = self.home / ".bash_completions" / "roboflow.sh"
        self.assertTrue(target.exists())

    def test_install_bash_appends_source_line_to_bashrc(self) -> None:
        runner.invoke(app, ["completion", "install", "--shell", "bash"])
        rc = (self.home / ".bashrc").read_text()
        target = self.home / ".bash_completions" / "roboflow.sh"
        self.assertTrue(
            any(line.lstrip().startswith("source ") and str(target) in line for line in rc.splitlines()),
            msg=f"no source line for {target} in {rc!r}",
        )

    def test_install_fish_writes_file(self) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "fish"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        target = self.home / ".config" / "fish" / "completions" / "roboflow.fish"
        self.assertTrue(target.exists())

    def test_install_unsupported_shell_errors(self) -> None:
        result = runner.invoke(app, ["completion", "install", "--shell", "csh"])
        self.assertEqual(result.exit_code, 3, msg=result.output)

    def test_install_missing_binary_hard_errors(self) -> None:
        import shutil as _shutil

        with mock.patch.object(_shutil, "which", return_value=None):
            result = runner.invoke(app, ["completion", "install", "--shell", "zsh"])
        self.assertEqual(result.exit_code, 1, msg=result.output)
        combined = result.output + (result.stderr or "")
        self.assertIn("PATH", combined)

    def test_install_idempotent(self) -> None:
        first = runner.invoke(app, ["completion", "install", "--shell", "zsh"])
        second = runner.invoke(app, ["completion", "install", "--shell", "zsh"])
        self.assertEqual(first.exit_code, 0)
        self.assertEqual(second.exit_code, 0)
        self.assertTrue((self.home / ".zfunc" / "_roboflow").exists())

    def test_install_bash_idempotent_does_not_duplicate_source_line(self) -> None:
        runner.invoke(app, ["completion", "install", "--shell", "bash"])
        runner.invoke(app, ["completion", "install", "--shell", "bash"])
        rc = (self.home / ".bashrc").read_text()
        target = self.home / ".bash_completions" / "roboflow.sh"
        source_lines = [line for line in rc.splitlines() if line.lstrip().startswith("source ") and str(target) in line]
        self.assertEqual(len(source_lines), 1, msg=f"unexpected rc: {rc!r}")

    def test_install_json_schema(self) -> None:
        result = runner.invoke(app, ["--json", "completion", "install", "--shell", "fish"])
        self.assertEqual(result.exit_code, 0, msg=result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["shell"], "fish")
        self.assertIn("path", payload)


if __name__ == "__main__":
    unittest.main()
