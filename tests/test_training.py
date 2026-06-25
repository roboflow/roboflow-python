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


class TestTrainingModels(unittest.TestCase):
    def test_models_are_cached_until_refresh(self):
        training = Training("key", "ws", "proj", "1", {"trainingId": "training-1"})
        bundle = {
            "status": "finished",
            "models": [{"modelId": "ws/model-slug", "modelType": "yolov11"}],
        }

        with patch("roboflow.core.training.rfapi.get_training", return_value=bundle) as get_training:
            first = training.models
            second = training.models
            training.refresh()
            third = training.models

        self.assertIs(first, second)
        self.assertEqual(first[0].model_id, "ws/model-slug")
        self.assertEqual(third[0].model_id, "ws/model-slug")
        self.assertEqual(get_training.call_count, 3)


if __name__ == "__main__":
    unittest.main()
