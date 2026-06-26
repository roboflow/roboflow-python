import unittest
from unittest.mock import patch

from roboflow.config import (
    CLASSIFICATION_MODEL,
    INSTANCE_SEGMENTATION_MODEL,
    KEYPOINT_DETECTION_MODEL,
    OBJECT_DETECTION_MODEL,
    SEMANTIC_SEGMENTATION_MODEL,
)
from roboflow.core.training import TrainedModel, Training
from roboflow.models.classification import ClassificationModel
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.models.keypoint_detection import KeypointDetectionModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.models.semantic_segmentation import SemanticSegmentationModel


class TestTrainedModelPredict(unittest.TestCase):
    def test_predict_routes_through_shared_inference_model_with_task_prediction_type(self):
        cases = [
            ("yolov11", OBJECT_DETECTION_MODEL, "https://serverless.roboflow.com/ws/model-slug"),
            ("yolov11-cls", CLASSIFICATION_MODEL, "https://serverless.roboflow.com/ws/model-slug"),
            ("yolov11-seg", INSTANCE_SEGMENTATION_MODEL, "https://serverless.roboflow.com/ws/model-slug"),
            ("yolov11-pose", KEYPOINT_DETECTION_MODEL, "https://serverless.roboflow.com/ws/model-slug"),
            ("yolo26-sem", SEMANTIC_SEGMENTATION_MODEL, "https://segment.roboflow.com/ws/model-slug"),
        ]

        for model_type, prediction_type, api_url in cases:
            with self.subTest(model_type=model_type):
                model = TrainedModel("key", "ws", "proj", "ws/model-slug", model_type=model_type)
                with patch(
                    "roboflow.core.training.InferenceModel.predict",
                    autospec=True,
                    return_value="ok",
                ) as predict:
                    result = model.predict("image.jpg", confidence=17, overlap=9, format="json")

                inference_model = predict.call_args.args[0]
                self.assertEqual(result, "ok")
                self.assertEqual(inference_model.api_url, api_url)
                self.assertEqual(predict.call_args.kwargs["prediction_type"], prediction_type)
                self.assertEqual(predict.call_args.kwargs["confidence"], 17)
                self.assertEqual(predict.call_args.kwargs["overlap"], 9)
                self.assertEqual(predict.call_args.kwargs["format"], "json")


class TestTrainedModelVideo(unittest.TestCase):
    def test_predict_video_routes_through_task_appropriate_legacy_model(self):
        cases = [
            ("yolov11", ObjectDetectionModel),
            ("yolov11-cls", ClassificationModel),
            ("yolov11-seg", InstanceSegmentationModel),
            ("yolov11-pose", KeypointDetectionModel),
            ("yolo26-sem", SemanticSegmentationModel),
        ]

        for model_type, legacy_class in cases:
            with self.subTest(model_type=model_type):
                model = TrainedModel("key", "ws", "proj", "ws/model-slug", model_type=model_type)
                with patch.object(
                    legacy_class,
                    "predict_video",
                    autospec=True,
                    return_value=("job-1", "signed-url", None),
                ) as predict_video:
                    result = model.predict_video("video.mp4", fps=9)

                legacy_model = predict_video.call_args.args[0]
                self.assertIsInstance(legacy_model, legacy_class)
                self.assertEqual(legacy_model.id, "ws/proj/model-slug")
                self.assertEqual(result, ("job-1", "signed-url", None))
                self.assertEqual(predict_video.call_args.kwargs["fps"], 9)

    def test_poll_reuses_the_predict_video_legacy_model(self):
        model = TrainedModel("key", "ws", "proj", "ws/model-slug", model_type="yolov11")

        with (
            patch.object(ObjectDetectionModel, "predict_video", autospec=True, return_value=("job-1", "url", None)),
            patch.object(
                ObjectDetectionModel, "poll_until_video_results", autospec=True, return_value={"frames": []}
            ) as poll,
        ):
            model.predict_video("video.mp4")
            result = model.poll_until_video_results("job-1")

        self.assertEqual(result, {"frames": []})
        self.assertIs(poll.call_args.args[0], model._video_model())


class TestTrainingModels(unittest.TestCase):
    def test_models_are_cached_until_refresh(self):
        training = Training("key", "ws", "proj", "1", {"trainingId": "training-1"})
        bundle = {
            "status": "finished",
            "modelType": "yolov11-cls",
            "modelGroup": "group-1",
            "modelIds": ["ws/model-slug"],
            "models": [{"modelId": "ws/model-slug"}],
        }

        with patch("roboflow.core.training.rfapi.get_training", return_value=bundle) as get_training:
            first = training.models
            second = training.models
            training.refresh()
            third = training.models

        self.assertIs(first, second)
        self.assertEqual(first[0].model_id, "ws/model-slug")
        self.assertEqual(first[0].model_type, "yolov11-cls")
        self.assertEqual(third[0].model_id, "ws/model-slug")
        self.assertEqual(third[0].model_type, "yolov11-cls")
        self.assertEqual(training.status, "finished")
        self.assertEqual(training.model_type, "yolov11-cls")
        self.assertEqual(training.model_group, "group-1")
        self.assertEqual(training.model_ids, ["ws/model-slug"])
        self.assertEqual(get_training.call_count, 3)


if __name__ == "__main__":
    unittest.main()
