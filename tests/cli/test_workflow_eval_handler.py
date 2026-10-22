"""Tests for the Workflow Evals CLI handler (`roboflow workflow-eval ...`)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest.mock import ANY, patch

from typer.testing import CliRunner

from roboflow.adapters.workflowevalsapi import WorkflowEvalError
from roboflow.cli import app

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
    "run": ["list", "get", "start", "cancel", "retry", "replay", "rescore", "delete", "config"],
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


if __name__ == "__main__":
    unittest.main()
