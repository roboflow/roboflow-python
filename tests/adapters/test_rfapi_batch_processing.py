"""HTTP contract tests for Asset Library Batch Processing adapters."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

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
        self.assertEqual(kwargs["json"]["idempotencyKey"], "request-123")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_status_encodes_untrusted_job_id(self, mock_get) -> None:
        response = Mock(status_code=200)
        response.json.return_value = {"status": "ok", "job": {"jobId": "bad/id"}}
        mock_get.return_value = response

        rfapi.get_batch_processing_job("private-key", "workspace-1", "bad/id")

        self.assertTrue(mock_get.call_args.args[0].endswith("/bad%2Fid"))

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error_preserves_http_status_for_cli_exit_codes(self, mock_get) -> None:
        response = Mock(status_code=404, text='{"error":{"message":"Job not found"}}')
        response.json.return_value = {"error": {"message": "Job not found"}}
        mock_get.return_value = response

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_batch_processing_job("private-key", "workspace-1", "missing")

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(str(ctx.exception), "Job not found")


if __name__ == "__main__":
    unittest.main()
