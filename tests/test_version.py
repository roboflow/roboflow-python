import os
import tempfile
import unittest
from unittest.mock import patch

import responses
import yaml

from roboflow.adapters import rfapi
from roboflow.config import (
    TYPE_CLASSICATION,
    TYPE_INSTANCE_SEGMENTATION,
    TYPE_KEYPOINT_DETECTION,
    TYPE_OBJECT_DETECTION,
    TYPE_SEMANTIC_SEGMENTATION,
)
from roboflow.core.version import Version, unwrap_version_id
from tests.helpers import get_version


def mock_generating_url_response(generating_url):
    """Helper function to mock the generating URL response that's repeated across tests."""
    responses.add(
        responses.GET,
        generating_url,
        json={"version": {"generating": False, "progress": 1.0, "images": 10}},
    )


class TestDownload(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.api_url = "https://api.roboflow.com/test-workspace/test-project/4/coco"
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="4",
        )

        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_download_raises_exception_on_bad_request(self):
        responses.add(responses.GET, self.api_url, status=404, json={"error": "Broken"})
        mock_generating_url_response(self.generating_url)
        with self.assertRaises(rfapi.RoboflowError):
            self.version.download("coco")

    @responses.activate
    def test_download_raises_exception_on_api_failure(self):
        responses.add(responses.GET, self.api_url, status=500)
        mock_generating_url_response(self.generating_url)
        with self.assertRaises(rfapi.RoboflowError):
            self.version.download("coco")

    @responses.activate
    @patch.object(Version, "_Version__download_zip")
    @patch("roboflow.core.version.extract_zip")
    @patch.object(Version, "_Version__reformat_yaml")
    def test_download_returns_dataset(self, *_):
        responses.add(responses.GET, self.api_url, json={"export": {"link": None}})
        mock_generating_url_response(self.generating_url)
        dataset = self.version.download("coco", location="/my-spot")
        self.assertEqual(dataset.name, self.version.name)
        self.assertEqual(dataset.version, self.version.version)
        self.assertEqual(dataset.model_format, "coco")
        self.assertEqual(dataset.location, os.path.abspath("/my-spot"))


class TestExport(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.api_url = "https://api.roboflow.com/test-workspace/test-project/4/test-format"
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="4",
        )

        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_export_returns_true_on_api_success(self):
        responses.add(
            responses.GET,
            self.api_url,
            status=200,
            json={"export": {"link": "https://api.roboflow.com/test-workspace/test-project/4/test-format"}},
        )
        mock_generating_url_response(self.generating_url)
        export = self.version.export("test-format")
        request = responses.calls[0].request

        self.assertTrue(export)
        self.assertEqual(request.method, "GET")
        self.assertDictEqual(request.params, {"nocache": "true", "api_key": "test-api-key"})

    @responses.activate
    def test_export_raises_error_on_bad_request(self):
        responses.add(responses.GET, self.api_url, status=400, json={"error": "BROKEN!!"})
        mock_generating_url_response(self.generating_url)
        with self.assertRaises(rfapi.RoboflowError):
            self.version.export("test-format")

    @responses.activate
    def test_export_raises_error_on_api_failure(self):
        responses.add(responses.GET, self.api_url, status=500)
        mock_generating_url_response(self.generating_url)
        with self.assertRaises(rfapi.RoboflowError):
            self.version.export("test-format")


@patch.object(os, "makedirs")
class TestGetDownloadLocation(unittest.TestCase):
    def setUp(self, *_):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="3",
        )

        # This is a weird python thing to get access to the private function for testing
        self.get_download_location = self.version._Version__get_download_location
        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_get_download_location_with_env_variable(self, *_):
        mock_generating_url_response(self.generating_url)
        with patch.dict(os.environ, {"DATASET_DIRECTORY": "/my/exports"}, clear=True):
            self.assertEqual(self.get_download_location(), "/my/exports/Test-Dataset-3")

    @responses.activate
    def test_get_download_location_without_env_variable(self, *_):
        mock_generating_url_response(self.generating_url)
        self.assertEqual(self.get_download_location(), "Test-Dataset-3")


class TestGetDownloadURL(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="3",
        )

        # This is a weird python thing to get access to the private function for testing
        self.get_download_url = self.version._Version__get_download_url
        self.generating_url = "https://api.roboflow.com/Test Workspace Name/Test Dataset/4"

    @responses.activate
    def test_get_download_url(self):
        mock_generating_url_response(self.generating_url)
        url = self.get_download_url("yolo1337")
        self.assertEqual(url, "https://api.roboflow.com/test-workspace/test-project/3/yolo1337")


class TestGetFormatIdentifier(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="3",
        )

        # This is a weird python thing to get access to the private function for testing
        self.get_format_identifier = self.version._Version__get_format_identifier

    def test_returns_simple_format(self):
        self.assertEqual(self.get_format_identifier("coco"), "coco")

    def test_returns_friendly_names_for_supported_formats(self):
        formats = [("yolov5", "yolov5pytorch"), ("yolov7", "yolov7pytorch")]
        for external_format, internal_format in formats:
            self.assertEqual(self.get_format_identifier(external_format), internal_format)

    def test_falls_back_to_instance_variable_if_model_format_is_none(self):
        self.version.model_format = "fallback"
        self.assertEqual(self.get_format_identifier(None), "fallback")

    def test_falls_back_to_instance_variable_if_model_format_is_none_and_converts_human_readable_format_to_identifier(  # noqa: E501
        self,
    ):
        self.version.model_format = "yolov5"
        self.assertEqual(self.get_format_identifier(None), "yolov5pytorch")

    def test_raises_runtime_error_if_model_format_is_none(self):
        self.version.model_format = None
        with self.assertRaises(RuntimeError):
            self.get_format_identifier(None)


def test_unwrap_version_id_when_full_identifier_is_given() -> None:
    # when
    result = unwrap_version_id(version_id="some-workspace/some-project/3")

    # then
    assert result == "3"


def test_unwrap_version_id_when_only_version_id_is_given() -> None:
    # when
    result = unwrap_version_id(version_id="3")

    # then
    assert result == "3"


class TestValidateAgainstProjectType(unittest.TestCase):
    def _version(self, project_type):
        return get_version(type=project_type)

    def test_detection_project_accepts_plain_yolo(self):
        self._version(TYPE_OBJECT_DETECTION)._validate_against_project_type("yolov11")

    def test_detection_project_accepts_rfdetr_detection(self):
        self._version(TYPE_OBJECT_DETECTION)._validate_against_project_type("rfdetr-medium")

    def test_detection_project_rejects_seg_model(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_OBJECT_DETECTION)._validate_against_project_type("yolov11-seg")

    def test_detection_project_rejects_rfdetr_seg(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_OBJECT_DETECTION)._validate_against_project_type("rfdetr-seg-medium")

    def test_instance_seg_project_accepts_seg_model(self):
        self._version(TYPE_INSTANCE_SEGMENTATION)._validate_against_project_type("yolov11-seg")

    def test_instance_seg_project_accepts_rfdetr_seg(self):
        self._version(TYPE_INSTANCE_SEGMENTATION)._validate_against_project_type("rfdetr-seg-medium")

    def test_instance_seg_project_rejects_detection(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_INSTANCE_SEGMENTATION)._validate_against_project_type("yolov11")

    def test_keypoint_project_accepts_pose_model(self):
        self._version(TYPE_KEYPOINT_DETECTION)._validate_against_project_type("yolov11-pose")

    def test_keypoint_project_rejects_detection(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_KEYPOINT_DETECTION)._validate_against_project_type("yolov11")

    def test_classification_project_accepts_cls(self):
        self._version(TYPE_CLASSICATION)._validate_against_project_type("yolov11-cls")

    def test_semantic_seg_project_accepts_sem_model(self):
        self._version(TYPE_SEMANTIC_SEGMENTATION)._validate_against_project_type("yolo26-sem")

    def test_semantic_seg_project_rejects_detection(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_SEMANTIC_SEGMENTATION)._validate_against_project_type("yolov11")

    def test_semantic_seg_project_rejects_instance_seg(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_SEMANTIC_SEGMENTATION)._validate_against_project_type("yolov11-seg")

    def test_instance_seg_project_rejects_sem_model(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_INSTANCE_SEGMENTATION)._validate_against_project_type("yolo26-sem")

    def test_detection_project_rejects_sem_model(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_OBJECT_DETECTION)._validate_against_project_type("yolo26-sem")

    def test_classification_project_rejects_detection(self):
        with self.assertRaises(ValueError):
            self._version(TYPE_CLASSICATION)._validate_against_project_type("yolov11")


# ---------------------------------------------------------------------------
# __reformat_yaml — fixing issue #240 (Incorrect Data Path in YOLOv8 Dataset)
# ---------------------------------------------------------------------------

_INITIAL_YAML = {
    "train": "../train/images",
    "val": "../valid/images",
    "test": "../test/images",
    "nc": 3,
    "names": ["cat", "dog", "bird"],
    "roboflow": {"license": "MIT", "project": "test-project", "version": 1},
}


def _write_data_yaml(directory: str, content: dict) -> str:
    """Write a ``data.yaml`` file in *directory* and return its path."""
    path = os.path.join(directory, "data.yaml")
    with open(path, "w") as fh:
        yaml.dump(content, fh)
    return path


def _read_data_yaml(directory: str) -> dict:
    """Read the ``data.yaml`` file from *directory*."""
    with open(os.path.join(directory, "data.yaml")) as fh:
        return yaml.safe_load(fh)


class TestReformatYaml(unittest.TestCase):
    """Tests for ``Version.__reformat_yaml`` (issue #240)."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _version_with_test(self):
        return get_version(splits={"train": 210, "valid": 20, "test": 10})

    def _version_without_test(self):
        return get_version(splits={"train": 210, "valid": 20, "test": 0})

    def _reformat(self, version, location, fmt):
        """Invoke the name-mangled private method."""
        version._Version__reformat_yaml(location, fmt)

    # ------------------------------------------------------------------
    # yolov8 — relative paths
    # ------------------------------------------------------------------

    def test_yolov8_paths_are_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "yolov8")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "train/images")
            self.assertEqual(content["val"], "valid/images")
            self.assertEqual(content["test"], "test/images")

    def test_yolov8_without_test_split_removes_test_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_without_test(), tmp, "yolov8")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "train/images")
            self.assertEqual(content["val"], "valid/images")
            self.assertNotIn("test", content)

    def test_yolov8_preserves_other_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "yolov8")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["nc"], 3)
            self.assertEqual(content["names"], ["cat", "dog", "bird"])
            self.assertEqual(content["roboflow"]["license"], "MIT")

    # ------------------------------------------------------------------
    # yolov9 — relative paths (was completely missing before fix)
    # ------------------------------------------------------------------

    def test_yolov9_paths_are_relative(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "yolov9")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "train/images")
            self.assertEqual(content["val"], "valid/images")
            self.assertEqual(content["test"], "test/images")

    def test_yolov9_without_test_split_removes_test_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_without_test(), tmp, "yolov9")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "train/images")
            self.assertEqual(content["val"], "valid/images")
            self.assertNotIn("test", content)

    # ------------------------------------------------------------------
    # Legacy formats — absolute paths
    # ------------------------------------------------------------------

    def test_yolov5pytorch_uses_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "yolov5pytorch")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], os.path.join(tmp, "train/images"))
            self.assertEqual(content["val"], os.path.join(tmp, "valid/images"))
            self.assertEqual(content["test"], os.path.join(tmp, "test/images"))

    def test_yolov5pytorch_without_test_split_removes_test_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_without_test(), tmp, "yolov5pytorch")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], os.path.join(tmp, "train/images"))
            self.assertEqual(content["val"], os.path.join(tmp, "valid/images"))
            self.assertNotIn("test", content)

    def test_yolov7pytorch_uses_absolute_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "yolov7pytorch")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], os.path.join(tmp, "train/images"))
            self.assertEqual(content["val"], os.path.join(tmp, "valid/images"))
            self.assertEqual(content["test"], os.path.join(tmp, "test/images"))

    def test_mt_yolov6_uses_absolute_paths(self):
        initial = dict(_INITIAL_YAML)
        initial["train"] = "./train/images"
        initial["val"] = "./valid/images"
        initial["test"] = "./test/images"
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, initial)
            self._reformat(self._version_with_test(), tmp, "mt-yolov6")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], os.path.join(tmp, "train/images"))
            self.assertEqual(content["val"], os.path.join(tmp, "valid/images"))
            self.assertEqual(content["test"], os.path.join(tmp, "test/images"))

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_strips_double_parent_prefix(self):
        """``../../train/images`` should become ``<location>/train/images``."""
        initial = dict(_INITIAL_YAML)
        initial["train"] = "../../train/images"
        initial["val"] = "../../valid/images"
        initial["test"] = "../../test/images"
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, initial)
            self._reformat(self._version_with_test(), tmp, "yolov5pytorch")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], os.path.join(tmp, "train/images"))
            self.assertEqual(content["val"], os.path.join(tmp, "valid/images"))

    def test_non_yolo_format_not_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "coco")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "../train/images")
            self.assertEqual(content["val"], "../valid/images")

    # ------------------------------------------------------------------
    # Independence from ultralytics (the root cause of issue #240)
    # ------------------------------------------------------------------

    def test_yolov8_works_without_ultralytics(self):
        """Paths must be fixed even when ultralytics is not installed."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            with patch(
                "roboflow.util.versions.import_module",
                side_effect=ModuleNotFoundError,
            ):
                self._reformat(self._version_with_test(), tmp, "yolov8")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "train/images")
            self.assertEqual(content["val"], "valid/images")
            self.assertEqual(content["test"], "test/images")

    def test_yolov8_works_with_any_ultralytics_version(self):
        """Paths must be fixed regardless of the installed ultralytics version."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_data_yaml(tmp, dict(_INITIAL_YAML))
            self._reformat(self._version_with_test(), tmp, "yolov8")
            content = _read_data_yaml(tmp)
            self.assertEqual(content["train"], "train/images")
            self.assertEqual(content["val"], "valid/images")
            self.assertEqual(content["test"], "test/images")
