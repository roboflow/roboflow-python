import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import requests
import responses

from roboflow.adapters import rfapi
from roboflow.config import (
    TYPE_CLASSICATION,
    TYPE_INSTANCE_SEGMENTATION,
    TYPE_KEYPOINT_DETECTION,
    TYPE_OBJECT_DETECTION,
    TYPE_SEMANTIC_SEGMENTATION,
)
from roboflow.core.version import Version, unwrap_version_id
from roboflow.models.object_detection import ObjectDetectionModel
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


class TestConstructionDoesNotProbeNetwork(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_version", side_effect=AssertionError("get_version should not be called"))
    def test_construction_makes_no_request_when_payload_has_no_model(self, _mock_get_version: MagicMock):
        version = get_version()
        self.assertIsNone(version._model)

    @patch(
        "roboflow.adapters.rfapi.get_version",
        side_effect=requests.exceptions.ConnectionError("network down"),
    )
    def test_construction_survives_request_layer_failure(self, _mock_get_version: MagicMock):
        # A transient/mocked request failure must not break basic version retrieval.
        version = get_version()
        self.assertIsNone(version._model)

    @patch("roboflow.adapters.rfapi.get_version", side_effect=AssertionError("get_version should not be called"))
    def test_legacy_model_is_derived_from_payload(self, _mock_get_version: MagicMock):
        version = get_version(type=TYPE_OBJECT_DETECTION, model={"id": "test-workspace/test-project/2"})
        self.assertIsInstance(version._model, ObjectDetectionModel)


class TestMMPVCompatibility(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_version", return_value={"version": {}})
    def test_model_property_is_deprecated_and_does_not_enumerate_models(self, _mock_get_version: MagicMock):
        version = get_version()
        with patch.object(Version, "models", side_effect=AssertionError("models should not be called")):
            with self.assertWarns(DeprecationWarning):
                self.assertIsNone(version.model)

    @patch("roboflow.adapters.rfapi.get_version", return_value={"version": {}})
    def test_models_returns_union_across_trainings(self, _mock_get_version: MagicMock):
        version = get_version()
        a, b, c = object(), object(), object()
        training_one = SimpleNamespace(models=[a, b])
        training_two = SimpleNamespace(models=[c])
        with patch.object(Version, "trainings", return_value=[training_one, training_two]):
            self.assertEqual(version.models(), [a, b, c])

    @patch.object(Version, "_Version__wait_if_generating")
    @patch("roboflow.adapters.rfapi.create_training_v2")
    @patch("roboflow.adapters.rfapi.get_version", return_value={"version": {}})
    def test_create_training_returns_v2_training(
        self,
        _mock_get_version: MagicMock,
        mock_create_training: MagicMock,
        _mock_wait_if_generating: MagicMock,
    ):
        mock_create_training.return_value = {
            "trainingId": "training-1",
            "status": "running",
            "modelType": "yolov11",
        }
        version = get_version(version_number="4")

        training = version.create_training(speed="fast", model_type=None, checkpoint="ckpt", epochs=10)

        mock_create_training.assert_called_once_with(
            api_key="test-api-key",
            workspace_url="test-workspace",
            project_url="test-project",
            version="4",
            speed="fast",
            checkpoint="ckpt",
            model_type=None,
            epochs=10,
            train_recipe=None,
            business_context=None,
        )
        self.assertEqual(training.training_id, "training-1")
        self.assertEqual(training.status, "running")
        self.assertEqual(training.model_type, "yolov11")


class V2TrainingRecipeTestCase(unittest.TestCase):
    """Base fixture for v2 train-recipe tests: an offline Version fixture."""

    RECIPE_RESPONSE = {
        "modelType": "rfdetr-medium",
        "family": "rf-detr",
        "taskType": "object-detection",
        "schema": {"hyperparameters": [{"key": "lr", "type": "float"}]},
        "template": {
            "schema_version": 1,
            "input": {},
            "online_preprocessing": [],
            "online_augmentation": {"splits": ["train"], "steps": []},
            "source_version": {},
            "hyperparameters": {},
        },
        "usage": "...",
    }

    def setUp(self):
        super().setUp()
        self.version = get_version(
            project_name="Test Dataset",
            id="test-workspace/test-project/2",
            version_number="4",
        )


class TestDescribeTrainRecipe(V2TrainingRecipeTestCase):
    def test_describe_train_recipe_passes_through(self):
        with patch.object(rfapi, "get_train_recipe", return_value=self.RECIPE_RESPONSE) as mock_recipe:
            result = self.version.describe_train_recipe("rfdetr-medium")

        self.assertEqual(result, self.RECIPE_RESPONSE)
        mock_recipe.assert_called_once_with(
            api_key="test-api-key",
            workspace_url="test-workspace",
            project_url="test-project",
            version="4",
            model_type="rfdetr-medium",
        )


class TestCreateTrainingWithRecipe(V2TrainingRecipeTestCase):
    """The train_recipe/business_context extension of Version.create_training."""

    CREATE_RESPONSE = {"trainingId": "t-1", "status": "queued", "jobId": "job-1"}
    NOT_GENERATING = {"version": {"generating": False, "progress": 1.0, "images": 10}}

    def _create(self, **kwargs):
        with (
            patch.object(rfapi, "get_version", return_value=self.NOT_GENERATING),
            patch.object(rfapi, "get_train_recipe", return_value=self.RECIPE_RESPONSE) as mock_recipe,
            patch.object(rfapi, "create_training_v2", return_value=self.CREATE_RESPONSE) as mock_create,
            patch.object(Version, "export", return_value=True) as mock_export,
        ):
            result = self.version.create_training(**kwargs)
        return result, mock_recipe, mock_create, mock_export

    def test_create_training_requires_model_type_for_train_recipe(self):
        # Raised before any network call: neither create nor generation polling runs.
        with patch.object(rfapi, "create_training_v2") as mock_create:
            with self.assertRaises(ValueError):
                self.version.create_training(train_recipe={"schema_version": 1})
        mock_create.assert_not_called()

    def test_recipe_kwargs_pass_through_to_canonical_create(self):
        result, mock_recipe, mock_create, mock_export = self._create(
            model_type="rfdetr-medium",
            epochs=10,
            speed="fast",
            checkpoint="ckpt",
            business_context="baseline",
        )

        self.assertEqual(result.training_id, "t-1")
        self.assertEqual(result.status, "queued")
        mock_recipe.assert_not_called()
        mock_export.assert_called_once_with("coco")
        mock_create.assert_called_once_with(
            api_key="test-api-key",
            workspace_url="test-workspace",
            project_url="test-project",
            version="4",
            speed="fast",
            checkpoint="ckpt",
            model_type="rfdetr-medium",
            epochs=10,
            train_recipe=None,
            business_context="baseline",
        )

    def test_explicit_recipe_submitted_as_is_without_describe(self):
        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.5}}
        _, mock_recipe, mock_create, mock_export = self._create(model_type="rfdetr-medium", train_recipe=recipe)

        mock_recipe.assert_not_called()  # no describe fetch on the explicit-recipe path
        mock_export.assert_called_once_with("coco")  # model_type is required, so export is ensured
        self.assertEqual(mock_create.call_args.kwargs["train_recipe"], recipe)
        self.assertEqual(mock_create.call_args.kwargs["model_type"], "rfdetr-medium")

    def test_epochs_folded_into_explicit_recipe(self):
        recipe = {"schema_version": 1, "hyperparameters": {"lr": 0.5}}
        _, mock_recipe, mock_create, _ = self._create(model_type="rfdetr-medium", train_recipe=recipe, epochs=50)

        mock_recipe.assert_not_called()
        submitted = mock_create.call_args.kwargs["train_recipe"]
        self.assertEqual(submitted["hyperparameters"], {"lr": 0.5, "epochs": 50})
        self.assertEqual(mock_create.call_args.kwargs["epochs"], 50)
        # The caller's recipe dict is not mutated by the fold.
        self.assertEqual(recipe["hyperparameters"], {"lr": 0.5})

    def test_explicit_recipe_epochs_wins_over_argument(self):
        recipe = {"schema_version": 1, "hyperparameters": {"epochs": 25}}
        _, _, mock_create, _ = self._create(model_type="rfdetr-medium", train_recipe=recipe, epochs=50)

        submitted = mock_create.call_args.kwargs["train_recipe"]
        self.assertEqual(submitted["hyperparameters"]["epochs"], 25)

    def test_epochs_fold_creates_hyperparameters_in_explicit_recipe(self):
        _, _, mock_create, _ = self._create(model_type="rfdetr-medium", train_recipe={"schema_version": 1}, epochs=50)

        submitted = mock_create.call_args.kwargs["train_recipe"]
        self.assertEqual(submitted["hyperparameters"], {"epochs": 50})

    def test_export_skipped_when_format_already_present(self):
        self.version.exports = ["coco"]
        _, _, _, mock_export = self._create(model_type="rfdetr-medium")
        mock_export.assert_not_called()
