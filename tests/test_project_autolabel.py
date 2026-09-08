"""Public Project and Workspace wrapper coverage for hosted auto-label."""

from unittest.mock import patch

from tests import PROJECT_NAME, WORKSPACE_NAME, RoboflowTest


class TestProjectAutolabel(RoboflowTest):
    @patch("roboflow.adapters.rfapi.start_autolabel_job", return_value={"jobId": "job-1"})
    def test_autolabel_foundational_passes_model_as_is(self, mock_start):
        result = self.project.autolabel(
            "batch-1",
            "gpt-6-astra-boxes",
            ontology={"a cat": "cat"},
            num_images=5,
            confidence=0.4,
            reviewer_email="reviewer@example.com",
        )

        self.assertEqual(result, {"jobId": "job-1"})
        mock_start.assert_called_once_with(
            self.rf.api_key,
            WORKSPACE_NAME,
            PROJECT_NAME,
            batch_id="batch-1",
            model_type="gpt-6-astra-boxes",
            ontology={"a cat": "cat"},
            num_images_to_label=5,
            default_confidence=0.4,
            confidence_thresholds=None,
            run_nms=None,
            reviewer_email="reviewer@example.com",
            model_options=None,
            preserve_existing_annotations=None,
        )

    @patch("roboflow.adapters.rfapi.start_autolabel_job", return_value={"jobId": "job-1"})
    def test_autolabel_forwards_preserve_existing_annotations(self, mock_start):
        self.project.autolabel("batch-1", "sam3-rle", preserve_existing_annotations=True)

        self.assertIs(mock_start.call_args.kwargs["preserve_existing_annotations"], True)

    @patch("roboflow.adapters.rfapi.start_autolabel_job", return_value={"jobId": "job-1"})
    def test_autolabel_roboflow_model_is_sent_as_custom_roboflow(self, mock_start):
        self.project.autolabel("batch-1", "my-project/3", model_type="roboflow", model_options={"outputFormat": "rle"})

        kwargs = mock_start.call_args.kwargs
        self.assertEqual(kwargs["model_type"], "custom_roboflow")
        self.assertEqual(kwargs["model_options"], {"outputFormat": "rle", "modelId": "my-project/3"})

    def test_autolabel_rejects_unknown_model_type(self):
        with self.assertRaises(ValueError):
            self.project.autolabel("batch-1", "sam3-rle", model_type="hosted")

    @patch("roboflow.adapters.rfapi.preview_autolabel", return_value={"predictions": []})
    def test_autolabel_preview_builds_image_payload(self, mock_preview):
        result = self.project.autolabel_preview(
            "sam3-rle",
            "https://example.com/cat.jpg",
            ontology=["cat"],
            confidence_threshold=0.3,
        )

        self.assertEqual(result, {"predictions": []})
        mock_preview.assert_called_once_with(
            self.rf.api_key,
            WORKSPACE_NAME,
            PROJECT_NAME,
            model_type="sam3-rle",
            image={"type": "url", "value": "https://example.com/cat.jpg"},
            ontology={"cat": "cat"},
            confidence_threshold=0.3,
        )

    @patch("roboflow.adapters.rfapi.get_autolabel_job", return_value={"status": "done"})
    def test_autolabel_job_wrappers_delegate(self, mock_get):
        self.assertEqual(self.project.autolabel_job("job-1"), {"status": "done"})
        mock_get.assert_called_with(self.rf.api_key, WORKSPACE_NAME, "job-1")

        self.assertEqual(self.workspace.autolabel_job("job-2"), {"status": "done"})
        mock_get.assert_called_with(self.rf.api_key, WORKSPACE_NAME, "job-2")

    @patch("roboflow.adapters.rfapi.list_autolabel_models", return_value={"models": []})
    def test_workspace_autolabel_models_delegates(self, mock_list):
        self.assertEqual(self.workspace.autolabel_models(), {"models": []})
        mock_list.assert_called_once_with(self.rf.api_key, WORKSPACE_NAME)
