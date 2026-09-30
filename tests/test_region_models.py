"""Hosted inference models must stay inside the selected region."""

import os
import tempfile
import unittest
from unittest import mock

import responses

from roboflow.config import refresh_region_urls
from roboflow.core.training import TrainedModel
from roboflow.models.classification import ClassificationModel
from roboflow.models.keypoint_detection import KeypointDetectionModel
from roboflow.models.semantic_segmentation import SemanticSegmentationModel
from roboflow.models.vlm import VLMModel


class TestRegionModels(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ,
            {
                "HOME": self.temporary_directory.name,
                "USERPROFILE": self.temporary_directory.name,
                "ROBOFLOW_CONFIG_DIR": os.path.join(self.temporary_directory.name, "config.json"),
                "ROBOFLOW_REGION": "eu",
            },
            clear=True,
        )
        self.environment.start()
        refresh_region_urls()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()
        refresh_region_urls()

    def test_serverless_models_use_eu_host(self) -> None:
        models = {
            "classification": ClassificationModel("key", "ws/proj/1", version="1"),
            "keypoint": KeypointDetectionModel("key", "ws/proj/1", version="1"),
            "vlm": VLMModel("key", "ws/proj/1", version="1"),
        }
        for name, model in models.items():
            with self.subTest(model=name):
                self.assertEqual(model.base_url, "https://serverless.roboflow.eu/")

    def test_semantic_segmentation_refuses_to_leave_eu(self) -> None:
        model = SemanticSegmentationModel("key", "ws/proj/1")
        with responses.RequestsMock() as mocked:
            with self.assertRaisesRegex(RuntimeError, "SEMANTIC_SEGMENTATION_URL"):
                model.predict("tests/images/rabbit.JPG")
            self.assertEqual(len(mocked.calls), 0)

    def test_trained_semantic_segmentation_refuses_to_leave_eu(self) -> None:
        model = TrainedModel("key", "ws", "proj", "ws/model-slug", model_type="yolo26-sem")
        with self.assertRaisesRegex(RuntimeError, "SEMANTIC_SEGMENTATION_URL"):
            model.predict("tests/images/rabbit.JPG")

    def test_explicit_semantic_segmentation_url_is_allowed(self) -> None:
        os.environ["SEMANTIC_SEGMENTATION_URL"] = "https://segment.example"
        model = SemanticSegmentationModel("key", "ws/proj/1")
        with mock.patch("roboflow.models.inference.InferenceModel.predict", return_value="ok") as predict:
            self.assertEqual(model.predict("tests/images/rabbit.JPG"), "ok")
        predict.assert_called_once()


if __name__ == "__main__":
    unittest.main()
