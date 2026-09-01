"""Tests for the durable Batch Processing CLI."""

from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from requests.exceptions import ConnectionError, Timeout
from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()
BASE = ["--workspace", "workspace-1", "--api-key", "private-key"]


class TestBatchRegistration(unittest.TestCase):
    """Batch commands are public and documented by Typer."""

    def test_batch_is_visible_in_root_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("batch", result.output)
        self.assertIn("Run and manage Batch Processing jobs", result.output)

    def test_batch_subcommands(self) -> None:
        for verb in ("create", "status", "list", "abort", "restart"):
            with self.subTest(verb=verb):
                result = runner.invoke(app, ["batch", verb, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)


class TestBatchCreate(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.create_asset_library_batch_job")
    def test_create_exact_selection_has_stable_machine_output(self, mock_create) -> None:
        mock_create.return_value = {
            "status": "queued",
            "taskId": "task-1",
            "jobId": "al-123",
            "batchId": "asset-library-123",
            "displayName": "Night defects",
        }

        result = runner.invoke(
            app,
            [
                *BASE,
                "--json",
                "batch",
                "create",
                "--workflow",
                "workflow-1",
                "--image-ids",
                "image-1,image-1,image-2",
                "--request-id",
                "request-123",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.output)
        self.assertEqual(payload["requestId"], "request-123")
        mock_create.assert_called_once_with(
            "private-key",
            "workspace-1",
            workflow_id="workflow-1",
            idempotency_key="request-123",
            image_ids=["image-1", "image-2"],
            query=None,
            machine_type="cpu",
            display_name=None,
        )

    @patch("roboflow.adapters.rfapi.create_asset_library_batch_job")
    def test_all_is_explicit_empty_query_not_an_omitted_selection(self, mock_create) -> None:
        mock_create.return_value = {"taskId": "task-1", "jobId": "al-123"}

        result = runner.invoke(
            app,
            [*BASE, "batch", "create", "--workflow", "workflow-1", "--all"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_create.call_args.kwargs["query"], "")
        self.assertIsNone(mock_create.call_args.kwargs["image_ids"])

    @patch("roboflow.adapters.rfapi.create_asset_library_batch_job")
    def test_ambiguous_selection_fails_closed(self, mock_create) -> None:
        result = runner.invoke(
            app,
            [*BASE, "batch", "create", "--workflow", "workflow-1"],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("exactly one selection", result.output)
        mock_create.assert_not_called()

    @patch("roboflow.adapters.rfapi.create_asset_library_batch_job")
    def test_empty_query_does_not_implicitly_select_every_image(self, mock_create) -> None:
        result = runner.invoke(
            app,
            [*BASE, "batch", "create", "--workflow", "workflow-1", "--query", ""],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("--query cannot be empty", result.output)
        self.assertIn("Use --all", result.output)
        mock_create.assert_not_called()

    @patch("roboflow.adapters.rfapi.create_asset_library_batch_job")
    def test_unsafe_request_id_fails_before_network(self, mock_create) -> None:
        result = runner.invoke(
            app,
            [
                *BASE,
                "batch",
                "create",
                "--workflow",
                "workflow-1",
                "--all",
                "--request-id",
                "unsafe/key",
            ],
        )

        self.assertEqual(result.exit_code, 1)
        self.assertIn("--request-id must be", result.output)
        mock_create.assert_not_called()

    def test_ambiguous_transport_failure_preserves_generated_request_id(self) -> None:
        request_id = "generated-request-123"
        for transport_error in (ConnectionError("connection reset"), Timeout("request timed out")):
            with self.subTest(transport_error=type(transport_error).__name__):
                with (
                    patch("roboflow.cli.handlers.batch.uuid.uuid4", return_value=request_id),
                    patch("roboflow.adapters.rfapi.requests.post", side_effect=transport_error),
                ):
                    result = runner.invoke(
                        app,
                        [*BASE, "batch", "create", "--workflow", "workflow-1", "--all"],
                    )

                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn(str(transport_error), result.output)
                self.assertIn(f"--request-id {request_id}", result.output)


class TestBatchLifecycle(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_batch_processing_job")
    def test_status_rejects_malformed_job_id_before_network(self, mock_get) -> None:
        result = runner.invoke(app, [*BASE, "batch", "status", "unsafe/job"])

        self.assertEqual(result.exit_code, 1)
        self.assertIn("job ID must be", result.output)
        mock_get.assert_not_called()

    @patch("roboflow.adapters.rfapi.get_batch_processing_job")
    def test_status_json_is_api_faithful(self, mock_get) -> None:
        api_result = {
            "status": "ok",
            "job": {"jobId": "al-123", "currentStage": "inference", "isTerminal": False, "error": False},
        }
        mock_get.return_value = api_result

        result = runner.invoke(app, [*BASE, "--json", "batch", "status", "al-123"])

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), api_result)

    @patch("roboflow.adapters.rfapi.list_batch_processing_jobs")
    def test_list_passes_pagination_and_search(self, mock_list) -> None:
        mock_list.return_value = {"status": "ok", "jobs": [], "nextPageToken": None}

        result = runner.invoke(
            app,
            [*BASE, "batch", "list", "--page-size", "25", "--next-page-token", "next-1", "--search", "night"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        mock_list.assert_called_once_with(
            "private-key",
            "workspace-1",
            page_size=25,
            next_page_token="next-1",
            search="night",
        )

    @patch("roboflow.adapters.rfapi.abort_batch_processing_job")
    def test_abort_requires_and_honors_explicit_confirmation(self, mock_abort) -> None:
        mock_abort.return_value = {"status": "ok", "jobId": "al-123"}

        result = runner.invoke(app, [*BASE, "batch", "abort", "al-123", "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_abort.assert_called_once_with("private-key", "workspace-1", "al-123")

    @patch("roboflow.adapters.rfapi.restart_batch_processing_job")
    def test_restart_requires_and_honors_credit_confirmation(self, mock_restart) -> None:
        mock_restart.return_value = {"status": "ok", "jobId": "al-123"}

        result = runner.invoke(app, [*BASE, "batch", "restart", "al-123", "--yes"])

        self.assertEqual(result.exit_code, 0, result.output)
        mock_restart.assert_called_once_with("private-key", "workspace-1", "al-123")


if __name__ == "__main__":
    unittest.main()
