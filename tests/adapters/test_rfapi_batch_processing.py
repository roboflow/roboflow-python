"""HTTP contract tests for Asset Library Batch Processing adapters."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from requests.exceptions import ConnectionError, Timeout

from roboflow.adapters import rfapi


class TestBatchProcessingAdapter(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_create_uses_bearer_auth_and_idempotent_payload(self, mock_post) -> None:
        response = Mock(status_code=202)
        response.json.return_value = {"status": "queued", "jobId": "al-123"}
        mock_post.return_value = response

        result = rfapi.create_asset_library_batch_job(
            "private-key",
            "workspace-1",
            workflow_id="workflow-1",
            idempotency_key="request-123",
            image_ids=["image-1"],
        )

        self.assertEqual(result["jobId"], "al-123")
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer private-key"})
        self.assertNotIn("private-key", mock_post.call_args.args[0])
        self.assertEqual(
            mock_post.call_args.args[0],
            f"{rfapi.API_URL}/batch-processing/v1/external/workspace-1/asset-library/jobs",
        )
        self.assertEqual(kwargs["json"]["idempotencyKey"], "request-123")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_list_uses_canonical_jobs_endpoint(self, mock_get) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok", "jobs": []}
        mock_get.return_value = response

        rfapi.list_batch_processing_jobs("private-key", "workspace-1")

        self.assertEqual(
            mock_get.call_args.args[0],
            f"{rfapi.API_URL}/batch-processing/v1/external/workspace-1/jobs",
        )

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_status_encodes_untrusted_job_id(self, mock_get) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok", "job": {"jobId": "bad/id"}}
        mock_get.return_value = response

        rfapi.get_batch_processing_job("private-key", "workspace-1", "bad/id")

        self.assertEqual(
            mock_get.call_args.args[0],
            f"{rfapi.API_URL}/batch-processing/v1/external/workspace-1/jobs/bad%2Fid",
        )

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error_preserves_http_status_for_cli_exit_codes(self, mock_get) -> None:
        response = Mock(status_code=404, text='{"error":{"message":"Job not found"}}')
        response.json.return_value = {"error": {"message": "Job not found"}}
        mock_get.return_value = response

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_batch_processing_job("private-key", "workspace-1", "missing")

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(str(ctx.exception), "Job not found")

    def test_transport_errors_are_translated_for_cli_recovery(self) -> None:
        for transport_error in (ConnectionError("connection reset"), Timeout("request timed out")):
            with self.subTest(transport_error=type(transport_error).__name__):
                with patch(
                    "roboflow.adapters.rfapi.requests.post",
                    side_effect=transport_error,
                ):
                    with self.assertRaises(rfapi.RoboflowError) as ctx:
                        rfapi.create_asset_library_batch_job(
                            "private-key",
                            "workspace-1",
                            workflow_id="workflow-1",
                            idempotency_key="request-123",
                            image_ids=["image-1"],
                        )

                self.assertIs(ctx.exception.__cause__, transport_error)
                self.assertEqual(str(ctx.exception), str(transport_error))


if __name__ == "__main__":
    unittest.main()
