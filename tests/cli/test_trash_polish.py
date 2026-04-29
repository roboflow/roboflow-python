"""Behavioral tests for the 1.3.8 CLI polish fixes.

Covers the three findings from the prod CLI shake-down:
  1. Auth errors (HTTP 401) should exit with code 2, not 3.
  2. Destructive commands without ``--yes`` and no TTY should bail with a
     hint instead of either hanging on a closed stdin or silently
     proceeding when ``--json`` is set.
  3. Re-running ``project|version|workflow delete`` on something already in
     Trash should be a no-op success rather than surfacing a misleading
     "missing scope" 404 message.
"""

import io
import sys
import unittest
from argparse import Namespace
from unittest.mock import patch

from roboflow.adapters import rfapi


def _args(**overrides):
    base = {
        "json": False,
        "workspace": "test-ws",
        "api_key": "fake-key",
        "quiet": False,
        "yes": False,
    }
    base.update(overrides)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# 1. Auth errors → exit code 2
# ---------------------------------------------------------------------------


class TestAuthExitCode(unittest.TestCase):
    """``output_api_error`` must map ``status_code=401`` to exit code 2.

    Before this change every ``rfapi.RoboflowError`` from the trash
    endpoints was caught with ``exit_code=3`` (not-found), so scripts /
    agents couldn't tell "your key is bad, get a new one" apart from
    "this resource doesn't exist."
    """

    def test_401_maps_to_exit_2(self) -> None:
        from roboflow.cli._output import output_api_error

        exc = rfapi.RoboflowError("This API key does not exist", status_code=401)
        with self.assertRaises(SystemExit) as ctx:
            output_api_error(_args(), exc, hint="ignored when 401")
        self.assertEqual(ctx.exception.code, 2)

    def test_404_maps_to_exit_3(self) -> None:
        from roboflow.cli._output import output_api_error

        exc = rfapi.RoboflowError("Not found", status_code=404)
        with self.assertRaises(SystemExit) as ctx:
            output_api_error(_args(), exc)
        self.assertEqual(ctx.exception.code, 3)

    def test_other_status_maps_to_exit_1(self) -> None:
        from roboflow.cli._output import output_api_error

        exc = rfapi.RoboflowError("Server died", status_code=500)
        with self.assertRaises(SystemExit) as ctx:
            output_api_error(_args(), exc)
        self.assertEqual(ctx.exception.code, 1)

    def test_no_status_code_maps_to_exit_1(self) -> None:
        # Older `raise RoboflowError(text)` call sites don't set status_code;
        # they should default to a generic exit 1, NOT to 3 (which would
        # impersonate "not found") and NOT to 2 (which would impersonate
        # "auth error").
        from roboflow.cli._output import output_api_error

        exc = rfapi.RoboflowError("ambiguous")
        with self.assertRaises(SystemExit) as ctx:
            output_api_error(_args(), exc)
        self.assertEqual(ctx.exception.code, 1)

    def test_trash_response_attaches_status_code(self) -> None:
        # rfapi._raise_for_trash_response is the funnel for every 4xx/5xx
        # from the soft-delete endpoints — verify it stamps status_code.
        class FakeResponse:
            status_code = 401
            text = '{"error": "Unauthorized"}'

            def json(self):
                return {"error": "Unauthorized"}

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi._raise_for_trash_response(FakeResponse())
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(str(ctx.exception), "Unauthorized")


# ---------------------------------------------------------------------------
# 2. Destructive commands gate on --yes OR a TTY (not on --json)
# ---------------------------------------------------------------------------


class TestDestructiveConfirm(unittest.TestCase):
    """``confirm_destructive`` should:

    * return True when ``--yes`` is set (regardless of ``--json`` or TTY);
    * exit cleanly with code 1 when no TTY AND no ``--yes`` (the regression
      scenario: ``roboflow project delete X --json < /dev/null`` previously
      went through silently because ``--json`` was treated as an implicit
      "skip prompt" signal);
    * prompt via typer.confirm when on a TTY without ``--yes`` and respect
      the user's choice.
    """

    def test_yes_flag_short_circuits(self) -> None:
        from roboflow.cli._output import confirm_destructive

        # No TTY, no prompt — but --yes is set, so it should still proceed.
        with patch.object(sys.stdin, "isatty", return_value=False):
            self.assertTrue(confirm_destructive(_args(yes=True), "destroy?"))

    def test_no_tty_no_yes_bails_with_exit_1(self) -> None:
        from roboflow.cli._output import confirm_destructive

        with patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                confirm_destructive(_args(yes=False), "destroy?")
            self.assertEqual(ctx.exception.code, 1)

    def test_json_alone_does_not_bypass(self) -> None:
        # Regression guard for the original bug: --json without --yes
        # should NOT short-circuit the destructive guard.
        from roboflow.cli._output import confirm_destructive

        with patch.object(sys.stdin, "isatty", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                confirm_destructive(_args(yes=False, json=True), "destroy?")
            self.assertEqual(ctx.exception.code, 1)

    def test_tty_prompts_and_respects_decline(self) -> None:
        from roboflow.cli._output import confirm_destructive

        captured = io.StringIO()
        with (
            patch.object(sys.stdin, "isatty", return_value=True),
            patch("typer.confirm", return_value=False),
            patch("sys.stdout", captured),
        ):
            self.assertFalse(confirm_destructive(_args(yes=False), "destroy?"))
        # Caller doesn't need to re-emit "Cancelled." — confirm_destructive
        # already calls output() with the cancelled marker.
        self.assertIn("Cancelled", captured.getvalue())

    def test_tty_prompts_and_respects_accept(self) -> None:
        from roboflow.cli._output import confirm_destructive

        with patch.object(sys.stdin, "isatty", return_value=True), patch("typer.confirm", return_value=True):
            self.assertTrue(confirm_destructive(_args(yes=False), "destroy?"))


# ---------------------------------------------------------------------------
# 3. Idempotent re-delete on project/version/workflow
# ---------------------------------------------------------------------------


class TestIdempotentDelete(unittest.TestCase):
    """When the DELETE call returns 404 because the resource is already in
    Trash (the public API's URL filter excludes trashed items), the handler
    should probe ``list_trash`` and emit a synthetic success payload with
    ``alreadyInTrash: True``. Previously we surfaced the raw 404 with a
    misleading "missing scope" hint."""

    def _trash_payload_with_project(self, slug: str, project_id: str = "p_id"):
        return {
            "items": [],
            "sections": {
                "projects": [{"id": project_id, "url": slug, "name": slug}],
                "versions": [],
                "workflows": [],
            },
        }

    def test_project_already_in_trash_returns_success(self) -> None:
        from roboflow.cli.handlers.project import _delete_project

        not_found = rfapi.RoboflowError("Endpoint does not exist", status_code=404)
        captured = io.StringIO()

        # Resolver works on "ws/slug" shorthand — pass an explicit workspace
        # via args.workspace and a bare slug via args.project_id.
        args = _args(project_id="my-proj", yes=True, json=True)
        with (
            patch("roboflow.adapters.rfapi.delete_project", side_effect=not_found),
            patch(
                "roboflow.adapters.rfapi.list_trash",
                return_value=self._trash_payload_with_project("my-proj"),
            ),
            patch("sys.stdout", captured),
        ):
            _delete_project(args)

        out = captured.getvalue()
        self.assertIn('"alreadyInTrash": true', out)
        self.assertIn('"deleted": true', out)
        self.assertIn('"trash": true', out)

    def test_project_404_not_in_trash_propagates_error(self) -> None:
        # If the slug really doesn't exist (not active, not trashed),
        # we should NOT swallow the 404 — propagate as exit 3.
        from roboflow.cli.handlers.project import _delete_project

        not_found = rfapi.RoboflowError("Endpoint does not exist", status_code=404)
        empty_trash = {"items": [], "sections": {"projects": [], "versions": [], "workflows": []}}

        args = _args(project_id="ghost-proj", yes=True, json=True)
        with (
            patch("roboflow.adapters.rfapi.delete_project", side_effect=not_found),
            patch("roboflow.adapters.rfapi.list_trash", return_value=empty_trash),
            patch("sys.stderr", io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as ctx:
                _delete_project(args)
        self.assertEqual(ctx.exception.code, 3)

    def test_workflow_already_in_trash_returns_success(self) -> None:
        from roboflow.cli.handlers.workflow import _delete_workflow

        not_found = rfapi.RoboflowError("Endpoint does not exist", status_code=404)
        trash = {
            "items": [],
            "sections": {
                "projects": [],
                "versions": [],
                "workflows": [{"id": "wf_id", "url": "my-wf", "name": "My WF"}],
            },
        }
        captured = io.StringIO()
        args = _args(workflow_url="my-wf", yes=True, json=True)
        with (
            patch("roboflow.adapters.rfapi.delete_workflow", side_effect=not_found),
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash),
            patch("sys.stdout", captured),
        ):
            _delete_workflow(args)
        out = captured.getvalue()
        self.assertIn('"alreadyInTrash": true', out)
        self.assertIn('"workflowId": "wf_id"', out)

    def test_version_already_in_trash_returns_success(self) -> None:
        from roboflow.cli.handlers.version import _delete_version

        not_found = rfapi.RoboflowError("Endpoint does not exist", status_code=404)
        trash = {
            "items": [],
            "sections": {
                "projects": [],
                "versions": [
                    {
                        "id": "1",
                        "parentUrl": "my-proj",
                        "parentId": "p_id",
                        "name": "v1",
                    }
                ],
                "workflows": [],
            },
        }
        captured = io.StringIO()
        args = _args(version_ref="my-proj/1", yes=True, json=True)
        with (
            patch("roboflow.adapters.rfapi.delete_version", side_effect=not_found),
            patch("roboflow.adapters.rfapi.list_trash", return_value=trash),
            patch("sys.stdout", captured),
        ):
            _delete_version(args)
        out = captured.getvalue()
        self.assertIn('"alreadyInTrash": true', out)
        self.assertIn('"version": "1"', out)


if __name__ == "__main__":
    unittest.main()
