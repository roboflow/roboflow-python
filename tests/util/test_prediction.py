import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from roboflow.config import (
    CLASSIFICATION_MODEL,
    KEYPOINT_DETECTION_MODEL,
    OBJECT_DETECTION_MODEL,
    SEMANTIC_SEGMENTATION_MODEL,
)
from roboflow.util.prediction import Prediction

IMAGE_PATH = "tests/images/rabbit.JPG"
BOX = {"x": 300.5, "y": 200.5, "width": 120.0, "height": 90.0, "class": "rabbit", "confidence": 0.91}


class TestPredictionSave(unittest.TestCase):
    def _save(self, prediction):
        """Save the prediction as a PNG, so pixels aren't altered by compression, and read it back."""
        import cv2

        with tempfile.TemporaryDirectory() as tmp:
            output_path = os.path.join(tmp, "prediction.png")
            prediction.save(output_path)
            image = cv2.imread(output_path)
        self.assertIsNotNone(image)
        return image

    def test_object_detection_draws_the_label_on_its_background(self):
        import cv2
        import numpy as np

        image = self._save(Prediction(dict(BOX), IMAGE_PATH, prediction_type=OBJECT_DETECTION_MODEL))

        # The label background is filled with the stroke color, so white pixels in it can only be the label text.
        text_width, text_height = cv2.getTextSize("rabbit", cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        left = int(BOX["x"] - BOX["width"] / 2)
        top = int(BOX["y"] - BOX["height"] / 2)
        label = image[top + 1 : top + int(1.5 * text_height), left : left + text_width + 1]
        self.assertTrue(np.all(label[0, -1] == (255, 0, 0)))
        self.assertTrue(np.any(np.all(label == 255, axis=-1)))

    def test_keypoint_detection(self):
        image = self._save(Prediction(dict(BOX, keypoints=[]), IMAGE_PATH, prediction_type=KEYPOINT_DETECTION_MODEL))
        self.assertEqual(image.shape, (992, 1024, 3))

    def test_classification(self):
        prediction = Prediction(
            {"top": "rabbit", "confidence": 0.91, "predictions": [{"class": "rabbit", "confidence": 0.91}]},
            IMAGE_PATH,
            prediction_type=CLASSIFICATION_MODEL,
        )
        image = self._save(prediction)
        # The label goes on a border around the image.
        self.assertEqual(image.shape, (992 + 200, 1024 + 200, 3))

    def test_semantic_segmentation(self):
        import base64

        import cv2
        import numpy as np

        mask = np.zeros((992, 1024), np.uint8)
        mask[100:300, 100:300] = 255
        encoded_mask = base64.b64encode(cv2.imencode(".png", mask)[1].tobytes()).decode()
        prediction = Prediction(
            {"segmentation_mask": encoded_mask, "class_map": {"1": "rabbit"}},
            IMAGE_PATH,
            prediction_type=SEMANTIC_SEGMENTATION_MODEL,
        )

        image = self._save(prediction)

        self.assertEqual(image.shape, (992, 1024, 3))

    @mock.patch("urllib.request.urlopen")
    def test_hosted_https_image(self, urlopen):
        urlopen.return_value.read.return_value = Path(IMAGE_PATH).read_bytes()

        image = self._save(Prediction(dict(BOX), "https://example.com/rabbit.jpg"))

        urlopen.assert_called_once()
        self.assertEqual(image.shape, (992, 1024, 3))


if __name__ == "__main__":
    unittest.main()
