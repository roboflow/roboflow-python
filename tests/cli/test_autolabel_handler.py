"""Unit tests for roboflow.cli.handlers.autolabel."""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.adapters.rfapi import RoboflowError
from roboflow.cli import app

runner = CliRunner()

_RESOLVE_PROJECT = "roboflow.cli.handlers.autolabel._resolve_project"
_RESOLVE_WORKSPACE = "roboflow.cli.handlers.autolabel._resolve_workspace"


class TestAutolabelRegistration(unittest.TestCase):
    def test_subcommands_have_help(self):
        for name in ["models", "preview", "start", "job"]:
            with self.subTest(command=name):
                result = runner.invoke(app, ["autolabel", name, "--help"])
                self.assertEqual(result.exit_code, 0, result.output)


class TestAutolabelModels(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_autolabel_models")
    @patch(_RESOLVE_WORKSPACE, return_value=("key", "ws"))
    def test_text_output_is_a_table(self, _resolve, mock_api):
        mock_api.return_value = {
            "models": [
                {"id": "gpt-6-astra-boxes", "name": "GPT-6 Astra", "available": True, "isDefault": True},
                {"id": "gemini-boxes", "name": "Gemini", "available": False, "unavailableReason": "plan"},
            ]
        }
        result = runner.invoke(app, ["autolabel", "models"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("gpt-6-astra-boxes", result.output)
        self.assertIn("gemini-boxes", result.output)
        mock_api.assert_called_once_with("key", "ws")

    @patch("roboflow.adapters.rfapi.list_autolabel_models", return_value={"models": [{"id": "sam3-rle"}]})
    @patch(_RESOLVE_WORKSPACE, return_value=("key", "ws"))
    def test_json_output(self, _resolve, _mock_api):
        result = runner.invoke(app, ["--json", "autolabel", "models"])
        self.assertEqual(json.loads(result.output), {"models": [{"id": "sam3-rle"}]})


class TestAutolabelPreview(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.preview_autolabel", return_value={"summary": {"totalDetections": 2}})
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_classes_become_identity_ontology(self, _resolve, mock_api):
        result = runner.invoke(
            app,
            [
                "--json",
                "autolabel",
                "preview",
                "-p",
                "ws/proj",
                "-m",
                "sam3-rle",
                "--image",
                "https://example.com/cat.jpg",
                "--class",
                "cat",
                "--class",
                "dog",
                "--confidence",
                "0.4",
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), {"summary": {"totalDetections": 2}})
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            model_type="sam3-rle",
            image={"type": "url", "value": "https://example.com/cat.jpg"},
            ontology={"cat": "cat", "dog": "dog"},
            confidence_threshold=0.4,
        )

    @patch("roboflow.adapters.rfapi.preview_autolabel", return_value={})
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_ontology_json_takes_precedence_over_classes(self, _resolve, mock_api):
        runner.invoke(
            app,
            ["autolabel", "preview", "-p", "ws/proj", "-m", "sam3-rle", "--image", "https://x/y.jpg"]
            + ["--class", "cat", "--ontology", '{"cat": "a tabby cat"}'],
        )
        self.assertEqual(mock_api.call_args.kwargs["ontology"], {"cat": "a tabby cat"})

    @patch("roboflow.adapters.rfapi.preview_autolabel", return_value={})
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_ontology_can_be_read_from_a_file(self, _resolve, mock_api):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump({"cat": "a tabby cat"}, handle)
            path = handle.name
        try:
            result = runner.invoke(
                app,
                ["autolabel", "preview", "-p", "ws/proj", "-m", "sam3-rle", "--image", "https://x/y.jpg"]
                + ["--ontology", f"@{path}"],
            )
        finally:
            os.unlink(path)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(mock_api.call_args.kwargs["ontology"], {"cat": "a tabby cat"})

    @patch("roboflow.adapters.rfapi.preview_autolabel")
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_invalid_ontology_json_fails_without_calling_api(self, _resolve, mock_api):
        result = runner.invoke(
            app,
            ["autolabel", "preview", "-p", "ws/proj", "-m", "sam3-rle", "--image", "https://x/y.jpg"]
            + ["--ontology", "not-json"],
        )
        self.assertNotEqual(result.exit_code, 0)
        mock_api.assert_not_called()


class TestAutolabelStart(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.start_autolabel_job", return_value={"jobId": "job-1"})
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_foundational_start(self, _resolve, mock_api):
        result = runner.invoke(
            app,
            ["--json", "autolabel", "start", "-p", "ws/proj", "--batch-id", "batch-1", "-m", "gpt-6-astra-boxes"]
            + ["--class", "cat", "--num-images", "10", "--confidence", "0.5", "--no-nms"]
            + ["--reviewer", "r@example.com", "--model-options", '{"outputFormat": "polygon"}'],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output), {"jobId": "job-1"})
        mock_api.assert_called_once_with(
            "key",
            "ws",
            "proj",
            batch_id="batch-1",
            model_type="gpt-6-astra-boxes",
            ontology={"cat": "cat"},
            num_images_to_label=10,
            default_confidence=0.5,
            confidence_thresholds=None,
            run_nms=False,
            reviewer_email="r@example.com",
            model_options={"outputFormat": "polygon"},
        )

    @patch("roboflow.adapters.rfapi.start_autolabel_job", return_value={"jobId": "job-1"})
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_roboflow_model_type(self, _resolve, mock_api):
        result = runner.invoke(
            app,
            ["autolabel", "start", "-p", "ws/proj", "--batch-id", "batch-1", "-m", "proj/3"]
            + ["--model-type", "roboflow", "--confidence-thresholds", '{"cat": 0.6}'],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        kwargs = mock_api.call_args.kwargs
        self.assertEqual(kwargs["model_type"], "custom_roboflow")
        self.assertEqual(kwargs["model_options"], {"modelId": "proj/3"})
        self.assertEqual(kwargs["confidence_thresholds"], {"cat": 0.6})
        self.assertIsNone(kwargs["run_nms"])

    @patch("roboflow.adapters.rfapi.start_autolabel_job")
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_unknown_model_type_is_an_error(self, _resolve, mock_api):
        result = runner.invoke(
            app,
            ["autolabel", "start", "-p", "ws/proj", "--batch-id", "batch-1", "-m", "x", "--model-type", "hosted"],
        )
        self.assertNotEqual(result.exit_code, 0)
        mock_api.assert_not_called()

    @patch("roboflow.adapters.rfapi.start_autolabel_job", side_effect=RoboflowError("batch not found", status_code=404))
    @patch(_RESOLVE_PROJECT, return_value=("key", "ws", "proj"))
    def test_api_error_maps_to_not_found_exit_code(self, _resolve, _mock_api):
        result = runner.invoke(app, ["autolabel", "start", "-p", "ws/proj", "--batch-id", "missing", "-m", "sam3-rle"])
        self.assertEqual(result.exit_code, 3, result.output)


class TestAutolabelJob(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_autolabel_job", return_value={"status": "running", "progress": 0.5})
    @patch(_RESOLVE_WORKSPACE, return_value=("key", "ws"))
    def test_json_output(self, _resolve, mock_api):
        result = runner.invoke(app, ["--json", "autolabel", "job", "job-1"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.output)["status"], "running")
        mock_api.assert_called_once_with("key", "ws", "job-1")
