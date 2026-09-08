"""HTTP contract tests for hosted auto-label adapters."""

import unittest
from unittest.mock import MagicMock, patch

from roboflow.adapters import rfapi


def _response(payload=None, status_code=200, text="error"):
    return MagicMock(status_code=status_code, text=text, json=lambda: payload or {"success": True})


class TestAutolabelAdapters(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_list_models_contract(self, mock_get):
        mock_get.return_value = _response({"models": []})

        self.assertEqual(rfapi.list_autolabel_models("key", "ws"), {"models": []})
        self.assertTrue(mock_get.call_args.args[0].endswith("/ws/autolabel/models"))
        self.assertEqual(mock_get.call_args.kwargs["params"], {"api_key": "key"})

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_get_job_contract(self, mock_get):
        mock_get.return_value = _response({"status": "running"})

        self.assertEqual(rfapi.get_autolabel_job("key", "ws", "job-1"), {"status": "running"})
        self.assertTrue(mock_get.call_args.args[0].endswith("/ws/autolabel/jobs/job-1"))
        self.assertEqual(mock_get.call_args.kwargs["params"], {"api_key": "key"})

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_preview_contract(self, mock_post):
        mock_post.return_value = _response({"predictions": []})
        image = {"type": "url", "value": "https://example.com/cat.jpg"}

        rfapi.preview_autolabel("key", "ws", "proj", model_type="sam3-rle", image=image)
        self.assertTrue(mock_post.call_args.args[0].endswith("/ws/proj/autolabel/preview"))
        self.assertEqual(mock_post.call_args.kwargs["params"], {"api_key": "key"})
        self.assertEqual(mock_post.call_args.kwargs["json"], {"modelType": "sam3-rle", "image": image})

        rfapi.preview_autolabel(
            "key",
            "ws",
            "proj",
            model_type="sam3-rle",
            image=image,
            ontology={"cat": "cat"},
            confidence_threshold=0.4,
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {"modelType": "sam3-rle", "image": image, "ontology": {"cat": "cat"}, "confidenceThreshold": 0.4},
        )

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_start_job_contract_omits_unset_fields(self, mock_post):
        mock_post.return_value = _response({"jobId": "job-1"})

        result = rfapi.start_autolabel_job("key", "ws", "proj", batch_id="batch-1", model_type="gpt-6-astra-boxes")
        self.assertEqual(result, {"jobId": "job-1"})
        self.assertTrue(mock_post.call_args.args[0].endswith("/ws/proj/autolabel"))
        self.assertEqual(mock_post.call_args.kwargs["params"], {"api_key": "key"})
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {"batchId": "batch-1", "modelType": "gpt-6-astra-boxes"},
        )

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_start_job_contract_full_payload(self, mock_post):
        mock_post.return_value = _response({"jobId": "job-1"})

        rfapi.start_autolabel_job(
            "key",
            "ws",
            "proj",
            batch_id="batch-1",
            model_type="custom_roboflow",
            ontology={"a cat": "cat"},
            num_images_to_label=10,
            default_confidence=0.5,
            confidence_thresholds={"cat": 0.6},
            run_nms=False,
            reviewer_email="reviewer@example.com",
            model_options={"modelId": "proj/3"},
            preserve_existing_annotations=True,
        )
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {
                "batchId": "batch-1",
                "modelType": "custom_roboflow",
                "ontology": {"a cat": "cat"},
                "numImagesToLabel": 10,
                "defaultConfidence": 0.5,
                "confidenceThresholds": {"cat": 0.6},
                "runNMS": False,
                "reviewerEmail": "reviewer@example.com",
                "modelOptions": {"modelId": "proj/3"},
                "preserveExistingAnnotations": True,
            },
        )

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_errors_raise_roboflow_error_with_status(self, mock_post):
        mock_post.return_value = _response({"error": {"message": "batch not found"}}, status_code=404)

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.start_autolabel_job("key", "ws", "proj", batch_id="missing", model_type="sam3-rle")
        self.assertEqual(str(ctx.exception), "batch not found")
        self.assertEqual(ctx.exception.status_code, 404)
