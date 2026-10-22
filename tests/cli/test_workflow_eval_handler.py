"""Tests for the Workflow Evals CLI handler (`roboflow workflow-eval ...`)."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import unittest
from typing import Any, Callable, Dict, List, Tuple
from unittest.mock import ANY, MagicMock, patch

import responses
from typer.testing import CliRunner

from roboflow.adapters.workflowevalsapi import WorkflowEvalError
from roboflow.cli import app
from roboflow.core.workflow_eval_local_runner import (
    PREFLIGHT_HINT,
    RESUME_HINT,
    ExecutionTally,
    InferenceServerUnreachable,
    LocalRunnerError,
    LocalRunnerInterrupted,
    LocalRunResult,
)

runner = CliRunner()

COMMANDS = {
    "": [
        "list",
        "get",
        "create",
        "update",
        "delete",
        "capabilities",
        "openapi",
        "operation",
        "compare",
    ],
    "spec": ["list", "get", "create", "update", "validate", "compatible-datasets", "ai-draft"],
    "dataset": ["list", "get", "create", "update", "update-contract", "compatible-specs"],
    "case": ["list", "get", "add", "replace", "delete", "import", "import-status", "upload"],
    "binding": ["suggest", "validate"],
    "preparation": ["start", "current", "get", "resume", "cases"],
    "run": ["list", "get", "start", "local", "cancel", "retry", "replay", "rescore", "delete", "config"],
    "execution": ["list", "get", "config", "input-cases", "capture", "overview", "results", "result"],
    "export": ["start", "status", "download-url"],
    "embedding": ["start", "get", "cancel"],
    "evaluator": ["list", "get"],
    "schema": ["get"],
    "agent": ["manifest", "skills", "skill", "bundles", "bundle", "asset"],
}

CLIENT = "roboflow.core.workflow_evals.WorkflowEvals"


def _invoke(*argv: str, input: str | None = None):
    return runner.invoke(app, ["--workspace", "ws", "--api-key", "key", *argv], input=input)


class TestWorkflowEvalRegistration(unittest.TestCase):
    def test_every_subcommand_has_help(self) -> None:
        for group, commands in COMMANDS.items():
            for command in commands:
                argv = ["workflow-eval", *([group] if group else []), command, "--help"]
                with self.subTest(command=" ".join(argv)):
                    result = runner.invoke(app, argv)
                    self.assertEqual(result.exit_code, 0, result.output)


class TestWorkflowEvalCommands(unittest.TestCase):
    @patch(f"{CLIENT}.list")
    def test_list_renders_table_and_cursor(self, mock_list) -> None:
        mock_list.return_value = {
            "items": [{"id": "e1", "name": "First", "state": "ready", "runCount": 2, "lastRunAt": None}],
            "nextCursor": "next",
        }
        result = _invoke("workflow-eval", "list", "--limit", "5")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("First", result.output)
        self.assertIn("--cursor next", result.output)
        mock_list.assert_called_once_with(limit=5, cursor=None, search=None, include_spec_summary=False)

    @patch(f"{CLIENT}.list")
    def test_list_json_is_raw_page(self, mock_list) -> None:
        page = {"items": [{"id": "e1"}]}
        mock_list.return_value = page
        result = _invoke("--json", "workflow-eval", "list")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), page)

    @patch(f"{CLIENT}.create")
    def test_create_merges_body_and_flags(self, mock_create) -> None:
        mock_create.return_value = {"id": "e1"}
        body = json.dumps({"name": "from body", "subject": {"kind": "savedWorkflow"}})
        result = _invoke("--json", "workflow-eval", "create", "--body", body, "--name", "flag", "--spec", "s1")
        self.assertEqual(result.exit_code, 0, result.output)
        payload = mock_create.call_args.args[0]
        self.assertEqual(payload, {"name": "flag", "subject": {"kind": "savedWorkflow"}, "specId": "s1"})

    @patch(f"{CLIENT}.update")
    def test_update_sends_revision(self, mock_update) -> None:
        mock_update.return_value = {"id": "e1", "revision": 4}
        result = _invoke("--json", "workflow-eval", "update", "e1", "--revision", "3", "--dataset", "d1")
        self.assertEqual(result.exit_code, 0, result.output)
        mock_update.assert_called_once_with("e1", {"evalDatasetId": "d1"}, revision="3")

    def test_update_requires_revision(self) -> None:
        result = _invoke("workflow-eval", "update", "e1", "--name", "x")
        self.assertNotEqual(result.exit_code, 0)

    @patch(f"{CLIENT}.create_spec")
    def test_spec_create_reads_body_from_file(self, mock_create) -> None:
        mock_create.return_value = {"id": "s1"}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "spec.json")
            with open(path, "w") as handle:
                json.dump({"spec": {"checks": []}}, handle)
            result = _invoke("--json", "workflow-eval", "spec", "create", "--body", f"@{path}", "--name", "S")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_create.call_args.args[0], {"spec": {"checks": []}, "name": "S"})

    @patch(f"{CLIENT}.validate_spec")
    def test_spec_validate_reads_stdin(self, mock_validate) -> None:
        mock_validate.return_value = {"valid": True}
        result = _invoke("--json", "workflow-eval", "spec", "validate", "--body", "-", input='{"checks": []}')
        self.assertEqual(result.exit_code, 0, result.output)
        mock_validate.assert_called_once_with({"checks": []})

    def test_invalid_json_body_is_a_structured_error(self) -> None:
        result = _invoke("--json", "workflow-eval", "spec", "validate", "--body", "{not json")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("--body is not valid JSON", result.output)

    @patch(f"{CLIENT}.get")
    def test_api_errors_map_to_exit_codes_and_json(self, mock_get) -> None:
        mock_get.side_effect = WorkflowEvalError(
            "Workflow Eval not found", status_code=404, code="NOT_FOUND", request_id="req-1", retryable=False
        )
        result = _invoke("--json", "workflow-eval", "get", "missing")
        self.assertEqual(result.exit_code, 3)
        error = json.loads(result.output.strip().splitlines()[-1])["error"]
        self.assertEqual(error["code"], "NOT_FOUND")
        self.assertEqual(error["requestId"], "req-1")
        self.assertEqual(error["message"], "Workflow Eval not found")

    @patch(f"{CLIENT}.list")
    def test_scope_error_exits_with_auth_code_and_hint(self, mock_list) -> None:
        mock_list.side_effect = WorkflowEvalError("Missing scope", status_code=403, code="FORBIDDEN")
        result = _invoke("workflow-eval", "list")
        self.assertEqual(result.exit_code, 2)
        self.assertIn("workflow-evals", result.output)

    @patch(f"{CLIENT}.get_by_url")
    def test_get_by_url(self, mock_get) -> None:
        mock_get.return_value = {"id": "e1"}
        result = _invoke("--json", "workflow-eval", "get", "my-eval", "--by-url")
        self.assertEqual(result.exit_code, 0, result.output)
        mock_get.assert_called_once_with("my-eval")

    @patch(f"{CLIENT}.delete")
    def test_delete_confirms_with_deletion_key(self, mock_delete) -> None:
        mock_delete.side_effect = [
            WorkflowEvalError(
                "Deletion confirmation is required",
                status_code=409,
                code="deletion_confirmation_required",
                details={"impact": {"runs": 1}, "deletionKey": "dk"},
            ),
            {"asyncTaskId": "t1", "statusUrl": "/ops/t1"},
        ]
        result = _invoke("--json", "workflow-eval", "delete", "e1", "--yes")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["asyncTaskId"], "t1")
        self.assertEqual(mock_delete.call_args_list[0].kwargs, {"deletion_key": None})
        self.assertEqual(mock_delete.call_args_list[1].kwargs, {"deletion_key": "dk"})

    @patch(f"{CLIENT}.delete_run")
    def test_run_delete_without_yes_refuses_when_not_a_tty(self, mock_delete) -> None:
        mock_delete.side_effect = WorkflowEvalError(
            "Deletion confirmation is required",
            status_code=409,
            code="deletion_confirmation_required",
            details={"deletionKey": "dk"},
        )
        result = _invoke("workflow-eval", "run", "delete", "r1", "--eval", "e1")
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(mock_delete.call_count, 1)

    @patch(f"{CLIENT}.import_cases")
    def test_case_import_builds_items(self, mock_import) -> None:
        mock_import.return_value = {"intakeId": "i1"}
        result = _invoke(
            "--json",
            "workflow-eval",
            "case",
            "import",
            "--dataset",
            "d1",
            "--input-field",
            "image",
            "--source",
            "src-1",
            "--from-dataset",
            "proj:valid",
            "--from-dataset",
            "proj2",
            "--artifact",
            "art-1",
            "--slice",
            "night",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        dataset_id, payload = mock_import.call_args.args
        self.assertEqual(dataset_id, "d1")
        self.assertEqual(
            payload,
            {
                "inputField": "image",
                "slices": ["night"],
                "items": [
                    {"kind": "source", "sourceId": "src-1"},
                    {"kind": "dataset", "datasetId": "proj", "split": "valid"},
                    {"kind": "dataset", "datasetId": "proj2"},
                    {"kind": "artifact", "artifactId": "art-1"},
                ],
            },
        )

    @patch(f"{CLIENT}.delete_case")
    def test_case_delete_with_yes(self, mock_delete) -> None:
        result = _invoke("--json", "workflow-eval", "case", "delete", "c1", "-d", "d1", "--revision", "2", "--yes")
        self.assertEqual(result.exit_code, 0, result.output)
        mock_delete.assert_called_once_with("d1", "c1", revision="2")
        self.assertEqual(json.loads(result.output), {"deleted": True, "caseId": "c1"})

    @patch(f"{CLIENT}.suggest_bindings")
    def test_binding_suggest_builds_saved_workflow_version_subject(self, mock_suggest) -> None:
        mock_suggest.return_value = {"bindingSet": {}}
        result = _invoke(
            "--json",
            "workflow-eval",
            "binding",
            "suggest",
            "--spec",
            "s1",
            "--dataset",
            "d1",
            "--workflow",
            "wf",
            "--workflow-version",
            "v2",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        mock_suggest.assert_called_once_with(
            "s1",
            "d1",
            {"kind": "savedWorkflowVersion", "workflowId": "wf", "subjectKey": "wf", "workflowVersionId": "v2"},
        )

    def test_binding_suggest_requires_a_subject(self) -> None:
        result = _invoke("workflow-eval", "binding", "suggest", "--spec", "s1", "--dataset", "d1")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("--workflow", result.output)

    @patch(f"{CLIENT}.start_case_preparation")
    def test_preparation_start_defaults_and_explicit_cases(self, mock_start) -> None:
        mock_start.return_value = {"id": "p1"}
        result = _invoke("--json", "workflow-eval", "preparation", "start", "-e", "e1", "--case", "c1", "--case", "c2")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            mock_start.call_args.args[1],
            {"runtime": {"kind": "serverless"}, "caseSelection": {"kind": "explicit", "caseIds": ["c1", "c2"]}},
        )

    @patch(f"{CLIENT}.wait_for_run")
    @patch(f"{CLIENT}.start_run")
    def test_run_start_with_wait_polls_the_admitted_run(self, mock_start, mock_wait) -> None:
        mock_start.return_value = {"run": {"id": "r1", "state": "queued"}, "executions": []}
        mock_wait.return_value = {"id": "r1", "state": "completed"}
        body = json.dumps({"executions": [{"subject": {}, "bindingSet": {}}]})
        result = _invoke("--json", "workflow-eval", "run", "start", "-e", "e1", "--body", body, "--wait")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["state"], "completed")
        mock_wait.assert_called_once_with("e1", "r1", timeout=0, interval=5, on_poll=ANY)

    @patch(f"{CLIENT}.list_case_results")
    def test_execution_results_filters(self, mock_results) -> None:
        mock_results.return_value = {"items": []}
        result = _invoke(
            "workflow-eval",
            "execution",
            "results",
            "x1",
            "-e",
            "e1",
            "-r",
            "r1",
            "--slice",
            "a",
            "--failed-check",
            "--state",
            "failed",
        )
        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = mock_results.call_args.kwargs
        self.assertEqual(kwargs["slices"], ["a"])
        self.assertTrue(kwargs["failed_check"])
        self.assertEqual(kwargs["state"], "failed")

    @patch(f"{CLIENT}.compare")
    def test_compare_collects_execution_ids(self, mock_compare) -> None:
        mock_compare.return_value = {"checks": []}
        result = _invoke("--json", "workflow-eval", "compare", "-e", "e1", "-x", "a", "-x", "b")
        self.assertEqual(result.exit_code, 0, result.output)
        mock_compare.assert_called_once_with("e1", {"executionIds": ["a", "b"]})

    @patch(f"{CLIENT}.wait_for_export")
    @patch(f"{CLIENT}.start_export")
    def test_export_start_with_wait(self, mock_start, mock_wait) -> None:
        mock_start.return_value = {"asyncTaskId": "t1", "pollUrl": "/p"}
        mock_wait.return_value = {"asyncTaskId": "t1", "state": "completed"}
        result = _invoke("--json", "workflow-eval", "export", "start", "-e", "e1", "-r", "r1", "-f", "csv", "--wait")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_start.call_args.args[2], {"format": "csv"})
        mock_wait.assert_called_once_with("e1", "r1", "t1", timeout=1800)

    @patch(f"{CLIENT}.start_export")
    def test_export_start_keeps_body_format_and_defaults_to_json(self, mock_start) -> None:
        mock_start.return_value = {"asyncTaskId": "t1"}
        result = _invoke(
            "--json", "workflow-eval", "export", "start", "-e", "e1", "-r", "r1", "--body", '{"format": "xlsx"}'
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_start.call_args.args[2], {"format": "xlsx"})
        _invoke("--json", "workflow-eval", "export", "start", "-e", "e1", "-r", "r1")
        self.assertEqual(mock_start.call_args.args[2], {"format": "json"})

    @patch(f"{CLIENT}.get_agent_skill")
    def test_agent_skill_prints_markdown_in_text_mode(self, mock_skill) -> None:
        mock_skill.return_value = {"engineVersion": "1", "id": "skill:x", "path": "x.md", "content": "# Skill"}
        result = _invoke("workflow-eval", "agent", "skill", "skill:x")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.output.strip(), "# Skill")

    @patch(f"{CLIENT}.ai_draft")
    def test_spec_ai_draft(self, mock_draft) -> None:
        mock_draft.return_value = {"spec": {}, "workflowBindings": {}}
        result = _invoke("--json", "workflow-eval", "spec", "ai-draft", "-e", "e1", "-i", "be strict")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_draft.call_args.args, ("e1", "be strict"))


class TestWorkflowEvalRunWait(unittest.TestCase):
    """``run get/start --wait`` with a fake clock: ``sleep`` advances ``monotonic``."""

    def _wait(self, *argv: str, runs: list, interval: str = "3600", timeout: str | None = None, admission=None):
        clock = [0.0]

        def sleep(seconds: float) -> None:
            clock[0] += seconds

        with (
            patch(f"{CLIENT}.get_run", side_effect=runs) as get_run,
            patch("time.monotonic", side_effect=lambda: clock[0]),
            patch("time.sleep", side_effect=sleep),
            patch(f"{CLIENT}.start_run", return_value=admission),
        ):
            if admission is None:
                command = ["workflow-eval", "run", "get", "r1", "-e", "e1"]
            else:
                command = ["workflow-eval", "run", "start", "-e", "e1", "--body", '{"executions": []}']
            command += ["--wait", "--interval", interval, *(["--timeout", timeout] if timeout else [])]
            result = _invoke(*argv, *command)
        return result, get_run.call_count

    @staticmethod
    def _runs(mode: str, polls: int, **extra) -> list:
        running = {"id": "r1", "state": "running", "executionMode": mode, "selectedCaseCount": 100, **extra}
        return [running] * polls + [{"id": "r1", "state": "completed", "executionMode": mode}]

    def test_browser_run_without_timeout_waits_past_the_old_limit(self) -> None:
        result, polls = self._wait("--json", runs=self._runs("browser", 4))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.stdout)["state"], "completed")
        self.assertEqual(polls, 5)

    def test_server_run_without_timeout_keeps_3600(self) -> None:
        result, polls = self._wait("--json", runs=self._runs("server", 4))
        self.assertEqual(result.exit_code, 1)
        self.assertIn("still running after 3600s", result.stderr)
        self.assertEqual(polls, 2)

    def test_timeout_zero_waits_without_limit(self) -> None:
        result, polls = self._wait("--json", runs=self._runs("server", 4), timeout="0")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(polls, 5)

    def test_explicit_timeout_applies_to_browser_runs(self) -> None:
        result, _ = self._wait("--json", runs=self._runs("browser", 4), interval="30", timeout="60")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("still running after 60s", result.stderr)

    def test_run_start_uses_the_admission_execution_mode(self) -> None:
        admission = {"run": {"id": "r1", "state": "queued"}, "executionMode": "browser"}
        runs = self._runs("", 4)  # polled Runs lack executionMode: the admission decides
        result, polls = self._wait("--json", runs=runs, admission=admission)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(polls, 5)

    def test_progress_is_throttled_to_one_line_every_10s(self) -> None:
        runs = self._runs("browser", 25, childStateCounts={"running": 1}, lastActivityAt="2026-10-22T10:00:00Z")
        result, _ = self._wait(runs=runs, interval="1")
        self.assertEqual(result.exit_code, 0, result.output)
        lines = [line for line in result.stderr.splitlines() if line.startswith("Run r1")]
        self.assertEqual(len(lines), 3)  # t = 0, 10, 20
        self.assertEqual(
            lines[0], "Run r1 · running · 100 cases · executions running 1 · last activity 2026-10-22T10:00:00Z"
        )

    def test_progress_is_suppressed_with_json_and_quiet(self) -> None:
        for flag in ("--json", "--quiet"):
            with self.subTest(flag=flag):
                result, _ = self._wait(flag, runs=self._runs("browser", 25), interval="1")
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertNotIn("Run r1 ·", result.stderr)
                if flag == "--json":
                    self.assertEqual(json.loads(result.stdout)["state"], "completed")


RUNNER = "roboflow.core.workflow_eval_local_runner.LocalRunner"
LOCAL = ["workflow-eval", "run", "local", "-e", "e1", "-r", "r1"]


def _result(state: str = "completed") -> LocalRunResult:
    tally = ExecutionTally("x1", submitted=9, already_done=2, skipped=0, failed=1)
    return LocalRunResult(run_id="r1", state=state, executions=[tally], wall_clock_seconds=12.5, concurrency=4)


class _FakeRunner:
    """Stands in for ``LocalRunner``: ``behaviour(self)`` is what ``run()`` does."""

    instances: List["_FakeRunner"] = []
    behaviour: Callable[["_FakeRunner"], LocalRunResult] = staticmethod(lambda runner: _result())

    def __init__(self, client: Any, eval_id: str, run_id: str, **options: Any) -> None:
        self.eval_id, self.run_id, self.options = eval_id, run_id, options
        self.state_dir = "/tmp/rf-state/r1"
        self.stop_requests = 0
        _FakeRunner.instances.append(self)

    def request_stop(self) -> None:
        self.stop_requests += 1

    @property
    def aborting(self) -> bool:
        return self.stop_requests > 1

    def run(self) -> LocalRunResult:
        return type(self).behaviour(self)


def _fake(behaviour: Callable[[_FakeRunner], Any]) -> type:
    _FakeRunner.instances = []
    return type("FakeRunner", (_FakeRunner,), {"behaviour": staticmethod(behaviour)})


def _raise(exc: Exception) -> Callable[[_FakeRunner], Any]:
    def behaviour(runner: _FakeRunner) -> Any:
        raise exc

    return behaviour


class TestWorkflowEvalRunLocal(unittest.TestCase):
    def _run(self, globals_: tuple = (), extra: tuple = (), behaviour=lambda runner: _result()):
        with patch(RUNNER, _fake(behaviour)):
            return _invoke(*globals_, *LOCAL, *extra)

    def test_json_output_shape_and_runner_options(self) -> None:
        result = self._run(
            ("--json",),
            ("--inference-url", "http://gpu:9001", "-x", "x1", "-x", "x2", "--concurrency", "8", "--case-timeout", "0"),
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(
            json.loads(result.stdout),
            {
                "runId": "r1",
                "state": "completed",
                "executions": [{"id": "x1", "submitted": 9, "alreadyDone": 2, "skipped": 0, "failed": 1, "pending": 0}],
                "wallClockSeconds": 12.5,
                "concurrency": 4,
                "warnings": [],
            },
        )
        runner = _FakeRunner.instances[0]
        self.assertEqual((runner.eval_id, runner.run_id), ("e1", "r1"))
        self.assertEqual(runner.options["inference_url"], "http://gpu:9001")
        self.assertEqual(runner.options["execution_ids"], ["x1", "x2"])
        self.assertEqual(runner.options["concurrency"], 8)
        self.assertEqual(runner.options["case_timeout"], 0)
        self.assertFalse(runner.options["retry"])

    def test_text_output_is_a_short_table(self) -> None:
        result = self._run()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Run r1 · completed · 12.5s · concurrency 4", result.stdout)
        self.assertIn("ALREADY DONE", result.stdout)
        self.assertIn("PENDING", result.stdout)

    def test_defaults(self) -> None:
        result = self._run(("--json",))
        self.assertEqual(result.exit_code, 0, result.output)
        options = _FakeRunner.instances[0].options
        self.assertEqual((options["concurrency"], options["case_timeout"]), (4, 300))
        self.assertIsNone(options["inference_url"])
        self.assertIsNone(options["state_dir"])

    def test_concurrency_out_of_range_is_rejected(self) -> None:
        for value in ("0", "17"):
            with self.subTest(concurrency=value):
                result = self._run(("--json",), ("--concurrency", value))
                self.assertEqual(result.exit_code, 1)
                error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
                self.assertIn("--concurrency must be between 1 and 16", error["message"])
                self.assertEqual(_FakeRunner.instances, [])

    def test_negative_case_timeout_is_rejected(self) -> None:
        result = self._run((), ("--case-timeout", "-1"))
        self.assertEqual(result.exit_code, 1)
        self.assertIn("--case-timeout", result.stderr)

    def test_runner_error_exit_codes(self) -> None:
        cases = {
            1: LocalRunnerError("Run r1 was cancelled.", hint="Start a new Run.", result=_result("cancelled")),
            2: LocalRunnerError("Missing scope", hint="needs workflow-evals:read and run", exit_code=2),
            3: LocalRunnerError("Run not found", exit_code=3),
        }
        for code, exc in cases.items():
            with self.subTest(exit_code=code):
                result = self._run(("--json",), behaviour=_raise(exc))
                self.assertEqual(result.exit_code, code)
                self.assertEqual(result.stdout, "")
                error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
                self.assertEqual(error["message"], exc.message)
                if exc.hint:
                    self.assertEqual(error["hint"], exc.hint)
        cancelled = self._run(("--json",), behaviour=_raise(cases[1]))
        error = json.loads(cancelled.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(error["result"]["state"], "cancelled")

    def test_interrupted_exits_130_with_resume_hint(self) -> None:
        result = self._run(behaviour=_raise(LocalRunnerInterrupted()))
        self.assertEqual(result.exit_code, 130)
        self.assertIn("Run the same command again to resume", result.stderr)
        self.assertIn("/tmp/rf-state/r1", result.stderr)

    def test_signal_handlers_are_installed_and_restored(self) -> None:
        before = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
        seen: Dict[str, Any] = {}

        def behaviour(runner: _FakeRunner) -> Any:
            handler = signal.getsignal(signal.SIGTERM)
            seen["installed"] = handler is not before[signal.SIGTERM] and callable(handler)
            handler(signal.SIGINT, None)  # first signal: notice + graceful stop
            seen["stops"] = runner.stop_requests
            handler(signal.SIGINT, None)  # second: abort, raised from the handler
            seen["returned"] = True

        result = self._run(behaviour=behaviour)
        self.assertEqual(result.exit_code, 130)
        self.assertTrue(seen["installed"])
        self.assertEqual(seen["stops"], 1)
        self.assertNotIn("returned", seen)
        self.assertEqual(_FakeRunner.instances[0].stop_requests, 2)
        self.assertIn("/tmp/rf-state/r1", result.stderr)  # resume hint
        self.assertEqual(result.stderr.count("Stopping after in-flight cases"), 1)
        self.assertEqual({signum: signal.getsignal(signum) for signum in before}, before)

    def test_progress_is_throttled_on_stderr(self) -> None:
        clock = [0.0]

        def behaviour(runner: _FakeRunner) -> Any:
            for done in range(25):
                runner.options["on_progress"](
                    {
                        "executionId": "x1",
                        "execution": 1,
                        "executions": 2,
                        "ok": done,
                        "failed": 1,
                        "alreadyDone": 0,
                        "skipped": 0,
                        "remaining": 30 - done,
                        "inFlight": 4,
                        "lastLatencyMs": 1234.4,
                    }
                )
                clock[0] += 1
            return _result()

        with patch("time.monotonic", side_effect=lambda: clock[0]):
            result = self._run(behaviour=behaviour)
        self.assertEqual(result.exit_code, 0, result.output)
        lines = [line for line in result.stderr.splitlines() if line.startswith("Execution")]
        self.assertEqual(len(lines), 3)  # t = 0, 10, 20
        self.assertEqual(
            lines[0], "Execution 1/2 · cases 0 ok / 1 failed / 30 remaining · in flight 4 · last latency 1234 ms"
        )

    def test_progress_and_warnings_with_json_and_quiet(self) -> None:
        def behaviour(runner: _FakeRunner) -> Any:
            runner.options["warn"]("Using the inference server A instead of B")
            progress = {"execution": 1, "executions": 1, "ok": 1, "failed": 0, "remaining": 0, "inFlight": 0}
            runner.options["on_progress"](progress)
            return _result()

        for flag in ("--json", "--quiet"):
            with self.subTest(flag=flag):
                result = self._run((flag,), behaviour=behaviour)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertNotIn("Execution 1/1", result.stderr)
                if flag == "--json":
                    self.assertEqual(result.stderr, "")
                    summary = json.loads(result.stdout)
                    self.assertEqual(summary["state"], "completed")
                    self.assertEqual(summary["warnings"], ["Using the inference server A instead of B"])
                else:
                    self.assertIn("Warning: Using the inference server A instead of B", result.stderr)

    def test_unreachable_inference_server_exits_1_with_resume_hint(self) -> None:
        error = InferenceServerUnreachable(
            "2 Cases were not run because the inference server at http://localhost:9001 was unreachable.",
            hint="Start the inference server and run the same command again.",
            result=_result(),
        )
        result = self._run(("--json",), behaviour=_raise(error))
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        payload = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertIn("were not run", payload["message"])
        self.assertEqual(
            payload["hint"],
            "Start the inference server. Run the same command again to resume; the journal is in /tmp/rf-state/r1.",
        )
        self.assertIn("pending", payload["result"]["executions"][0])

    @patch(f"{CLIENT}.get_run")
    def test_non_terminal_state_is_read_once_more(self, mock_get_run) -> None:
        mock_get_run.return_value = {"id": "r1", "state": "completed"}
        result = self._run(("--json",), behaviour=lambda runner: _result("running"))
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.stdout)["state"], "completed")
        mock_get_run.assert_called_once_with("e1", "r1")

    @patch(f"{CLIENT}.get_run")
    def test_run_still_finalizing_its_cancellation_exits_1(self, mock_get_run) -> None:
        mock_get_run.return_value = {"run": {"id": "r1", "state": "running", "cancellationRequestedAt": "2026-10-22"}}
        result = self._run(("--json",), behaviour=lambda runner: _result("running"))
        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(error["message"], "Run r1 was cancelled.")

    def _run_with_fake_clock(self) -> Tuple[Any, List[float]]:
        clock = [1000.0]
        sleeps: List[float] = []

        def sleep(seconds: float) -> None:
            sleeps.append(seconds)
            clock[0] += seconds

        with patch("time.time", side_effect=lambda: clock[0]), patch("time.sleep", side_effect=sleep):
            result = self._run(("--json",), behaviour=lambda runner: _result("running"))
        return result, sleeps

    @patch(f"{CLIENT}.get_run")
    def test_transient_error_on_the_final_read_is_retried(self, mock_get_run) -> None:
        unavailable = WorkflowEvalError("Service Unavailable", status_code=503)
        mock_get_run.side_effect = [unavailable, {"id": "r1", "state": "completed"}]
        result, sleeps = self._run_with_fake_clock()
        self.assertEqual(result.exit_code, 0, result.output)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["warnings"], [])
        self.assertEqual(sleeps, [1.0])

    @patch(f"{CLIENT}.get_run")
    def test_final_read_failing_past_the_retry_budget_exits_1(self, mock_get_run) -> None:
        mock_get_run.side_effect = WorkflowEvalError("Service Unavailable", status_code=503)
        result, sleeps = self._run_with_fake_clock()
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.stdout, "")
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertIn("kept failing for 10 minutes", error["message"])
        self.assertEqual(max(sleeps), 60)
        self.assertLessEqual(sum(sleeps), 600)

    def test_warnings_go_to_the_json_error(self) -> None:
        def behaviour(runner: _FakeRunner) -> Any:
            runner.options["warn"]("Using the inference server A instead of B")
            raise LocalRunnerError("Boom", hint="Fix it.")

        result = self._run(("--json",), behaviour=behaviour)
        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(error["message"], "Boom")
        self.assertEqual(error["warnings"], ["Using the inference server A instead of B"])

    def test_warnings_go_to_the_json_error_of_other_exceptions(self) -> None:
        def behaviour(runner: _FakeRunner) -> Any:
            runner.options["warn"]("Using the inference server A instead of B")
            raise RuntimeError("Boom")

        result = self._run(("--json",), behaviour=behaviour)
        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(error["message"], "Boom")
        self.assertEqual(error["warnings"], ["Using the inference server A instead of B"])

    @patch(f"{CLIENT}.get_run")
    def test_server_run_is_rejected_with_hint(self, mock_get_run) -> None:
        mock_get_run.return_value = {"id": "r1", "state": "running", "executionMode": "server"}
        with tempfile.TemporaryDirectory() as state_dir:
            result = _invoke("--json", *LOCAL, "--state-dir", state_dir)
        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(error["hint"], "This Run executes on Roboflow; use `run get --wait`.")

    @responses.activate
    @patch(f"{CLIENT}.list_executions", return_value={"items": [{"id": "x1"}]})
    @patch(f"{CLIENT}.get_run_configuration")
    @patch(f"{CLIENT}.get_run", return_value={"id": "r1", "state": "running", "executionMode": "browser"})
    def test_preflight_failure_exits_1_with_hint(self, _get_run, mock_config, _executions) -> None:
        mock_config.return_value = {"sharedConfiguration": {"inferenceServer": {"url": "http://localhost:9001"}}}
        # No inference server is registered with ``responses``: the preflight POST fails to connect.
        with tempfile.TemporaryDirectory() as state_dir:
            result = _invoke("--json", *LOCAL, "--state-dir", state_dir)
        self.assertEqual(result.exit_code, 1)
        error = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertIn("http://localhost:9001", error["message"])
        self.assertEqual(error["hint"], PREFLIGHT_HINT)


class TestWorkflowEvalRunStartLocal(unittest.TestCase):
    BODY = json.dumps({"executions": [{"subject": {}, "bindingSet": {}}]})

    def _start(
        self,
        *extra: str,
        behaviour=lambda runner: _result(),
        json_flag: bool = True,
        admission: Any = None,
        get_run: Any = None,
    ):
        calls: List[str] = []

        def tracked(runner: _FakeRunner) -> Any:
            calls.append("runner")
            return behaviour(runner)

        def wait(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            calls.append("wait")
            return {"id": "r1", "state": "completed"}

        if admission is None:
            admission = {"run": {"id": "r1", "state": "queued"}, "executionMode": "browser"}
        start = MagicMock(return_value=admission)
        with (
            patch(RUNNER, _fake(tracked)),
            patch(f"{CLIENT}.start_run", start),
            patch(f"{CLIENT}.wait_for_run", side_effect=wait) as mock_wait,
            patch(f"{CLIENT}.get_run", get_run or MagicMock(side_effect=AssertionError("unexpected get_run"))),
        ):
            argv = ["--json"] if json_flag else []
            result = _invoke(*argv, "workflow-eval", "run", "start", "-e", "e1", "--body", self.BODY, *extra)
        return result, calls, start, mock_wait

    def test_wait_with_inference_url_runs_locally_then_waits(self) -> None:
        result, calls, start, mock_wait = self._start(
            "--wait", "--inference-url", "http://localhost:9001", "--concurrency", "2"
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, ["runner", "wait"])
        self.assertEqual(json.loads(result.stdout), {"id": "r1", "state": "completed"})
        options = _FakeRunner.instances[0].options
        self.assertEqual(options["inference_url"], "http://localhost:9001")
        self.assertEqual(options["concurrency"], 2)
        self.assertNotIn("case_timeout", options)  # the runner default (300) applies
        mock_wait.assert_called_once_with("e1", "r1", timeout=0, interval=5, on_poll=ANY)

    def test_wait_without_inference_url_only_waits(self) -> None:
        result, calls, _, _ = self._start("--wait")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, ["wait"])

    def test_runner_summary_goes_to_stderr(self) -> None:
        result, _, _, _ = self._start("--wait", "--inference-url", "http://localhost:9001", json_flag=False)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("Local run finished in 12.5s: 9 submitted (1 failed), 2 already done.", result.stderr)
        self.assertEqual(json.loads(result.stdout)["state"], "completed")

    def test_inference_url_without_wait_is_a_usage_error(self) -> None:
        result, calls, start, _ = self._start("--inference-url", "http://localhost:9001")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("--inference-url requires --wait", result.stderr)
        start.assert_not_called()
        self.assertEqual(calls, [])

    def test_local_options_require_inference_url(self) -> None:
        result, _, start, _ = self._start("--wait", "--concurrency", "2")
        self.assertEqual(result.exit_code, 1)
        self.assertIn("require --inference-url", result.stderr)
        start.assert_not_called()

    def test_concurrency_out_of_range_is_rejected(self) -> None:
        result, _, start, _ = self._start("--wait", "--inference-url", "http://x", "--concurrency", "17")
        self.assertEqual(result.exit_code, 1)
        start.assert_not_called()

    def test_server_run_with_inference_url_warns_and_only_waits(self) -> None:
        admission = {"run": {"id": "r1", "state": "queued"}, "executionMode": "server"}
        warning = "--inference-url has no effect: Run r1 executes on Roboflow; waiting for it."
        for json_flag in (True, False):
            with self.subTest(json=json_flag):
                result, calls, _, mock_wait = self._start(
                    "--wait", "--inference-url", "http://x", admission=admission, json_flag=json_flag
                )
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(calls, ["wait"])  # no runner: no preflight, no inference
                self.assertEqual(_FakeRunner.instances, [])
                final = json.loads(result.stdout)
                self.assertEqual(final["state"], "completed")
                if json_flag:
                    self.assertEqual(final["warnings"], [warning])
                    self.assertEqual(result.stderr, "")
                else:
                    self.assertNotIn("warnings", final)
                    self.assertIn(f"Warning: {warning}", result.stderr)
                mock_wait.assert_called_once_with("e1", "r1", timeout=0, interval=5, on_poll=ANY)

    def test_execution_mode_missing_from_admission_is_read_once(self) -> None:
        get_run = MagicMock(return_value={"id": "r1", "state": "queued", "executionMode": "server"})
        result, calls, _, _ = self._start(
            "--wait", "--inference-url", "http://x", admission={"run": {"id": "r1"}}, get_run=get_run
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, ["wait"])
        get_run.assert_called_once_with("e1", "r1")
        self.assertEqual(len(json.loads(result.stdout)["warnings"]), 1)

    def test_runner_warnings_go_to_the_final_json(self) -> None:
        def behaviour(runner: _FakeRunner) -> Any:
            runner.options["warn"]("Using the inference server http://x instead of http://y")
            return _result()

        result, calls, _, _ = self._start("--wait", "--inference-url", "http://x", behaviour=behaviour)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, ["runner", "wait"])
        self.assertEqual(result.stderr, "")
        self.assertEqual(
            json.loads(result.stdout),
            {"id": "r1", "state": "completed", "warnings": ["Using the inference server http://x instead of http://y"]},
        )

    def test_without_warnings_the_final_json_has_no_warnings_field(self) -> None:
        result, _, _, _ = self._start("--wait", "--inference-url", "http://x")
        self.assertEqual(json.loads(result.stdout), {"id": "r1", "state": "completed"})

    def test_unreachable_inference_server_hints_run_local(self) -> None:
        error = InferenceServerUnreachable(
            "1 Case was not run because the inference server at http://x was unreachable."
        )
        result, calls, _, _ = self._start("--wait", "--inference-url", "http://x", behaviour=_raise(error))
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(calls, ["runner"])
        hint = json.loads(result.stderr.strip().splitlines()[-1])["error"]["hint"]
        self.assertTrue(hint.startswith("Start the inference server. "))
        self.assertIn("run local -e e1 -r r1 --inference-url http://x", hint)

    def test_any_runner_error_hints_run_local_and_carries_the_run_id(self) -> None:
        error = LocalRunnerError("Could not reach the inference server at http://x.", hint=PREFLIGHT_HINT)
        result, calls, _, _ = self._start("--wait", "--inference-url", "http://x", behaviour=_raise(error))
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(calls, ["runner"])
        payload = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertTrue(payload["hint"].startswith(PREFLIGHT_HINT + " "))
        self.assertIn("run local -e e1 -r r1 --inference-url http://x", payload["hint"])
        self.assertEqual(payload["details"]["runId"], "r1")

    def test_interrupt_hints_run_local_to_resume(self) -> None:
        result, calls, _, _ = self._start(
            "--wait", "--inference-url", "http://x", behaviour=_raise(LocalRunnerInterrupted())
        )
        self.assertEqual(result.exit_code, 130)
        self.assertEqual(calls, ["runner"])
        hint = json.loads(result.stderr.strip().splitlines()[-1])["error"]["hint"]
        self.assertIn("run local -e e1 -r r1 --inference-url http://x", hint)

    def test_runner_resume_hint_is_replaced_by_run_local(self) -> None:
        error = LocalRunnerError(
            "Stopped after 10 connection failures in a row to the inference server at http://x.",
            hint=f"{PREFLIGHT_HINT} {RESUME_HINT}",
        )
        result, _, _, _ = self._start("--wait", "--inference-url", "http://x", behaviour=_raise(error))
        self.assertEqual(result.exit_code, 1)
        payload = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertNotIn("run the command again", payload["hint"])
        self.assertEqual(
            payload["hint"],
            f"{PREFLIGHT_HINT} Resume with `roboflow workflow-eval run local -e e1 -r r1 --inference-url http://x`; "
            "the journal is in /tmp/rf-state/r1.",
        )
        self.assertEqual(payload["details"]["runId"], "r1")

    def test_cancelled_run_has_no_resume_hint(self) -> None:
        error = LocalRunnerError(
            "Run r1 was cancelled.",
            hint="Start a new Run to evaluate again.",
            result=_result("cancelled"),
            resumable=False,
        )
        result, _, _, _ = self._start("--wait", "--inference-url", "http://x", behaviour=_raise(error))
        self.assertEqual(result.exit_code, 1)
        payload = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertEqual(payload["hint"], "Start a new Run to evaluate again.")
        self.assertEqual(payload["details"]["runId"], "r1")

    def test_other_errors_keep_the_run_id_and_resume_hint(self) -> None:
        error = PermissionError(13, "Permission denied", "/tmp/rf-state/r1")
        result, calls, _, _ = self._start("--wait", "--inference-url", "http://x", behaviour=_raise(error))
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(calls, ["runner"])
        payload = json.loads(result.stderr.strip().splitlines()[-1])["error"]
        self.assertIn("Permission denied", payload["message"])
        self.assertIn("run local -e e1 -r r1 --inference-url http://x", payload["hint"])
        self.assertEqual(payload["details"]["runId"], "r1")


if __name__ == "__main__":
    unittest.main()
