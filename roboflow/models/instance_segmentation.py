import io
import urllib

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

from roboflow.config import INSTANCE_SEGMENTATION_MODEL, INSTANCE_SEGMENTATION_URL
from roboflow.util.image_utils import validate_image_path
from roboflow.util.prediction import PredictionGroup


class InstanceSegmentationModel:
    def __init__(
        self,
        api_key,
        version_id,
    ):
        """
        :param api_key: Your API key (obtained via your workspace API settings page)
        :param version_id: The ID of the dataset version to use for predicting
        """
        self.__api_key = api_key
        self.id = version_id

        version_info = self.id.rsplit("/")
        dataset_id = version_info[1]
        version = version_info[2]

        self.api_url = f"{INSTANCE_SEGMENTATION_URL}/{dataset_id}/{version}"

    def predict(self, image_path, confidence=40):
        """
        Infers detections based on image from specified model and image path

        :param image_path: Path to image (can be local path or hosted URL)
        :param confidence: A threshold for the returned predictions on a scale of 0-100. A lower number will return more predictions. A higher number will return fewer, high-certainty predictions.

        :return: PredictionGroup - a group of predictions based on Roboflow JSON response
        """
        validate_image_path(image_path)

        params = {
            "api_key": self.__api_key,
            "confidence": confidence,
        }
        request_kwargs = {}

        hosted_image = urllib.parse.urlparse(image_path).scheme in (
            "http",
            "https",
        )

        if hosted_image:
            params["image"] = image_path
        else:
            image = Image.open(image_path)
            buffered = io.BytesIO()
            image.save(buffered, quality=90, format="JPEG")
            data = MultipartEncoder(
                fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")}
            )
            request_kwargs = {
                "data": data,
                "headers": {"Content-Type": data.content_type},
            }

        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        response = requests.post(url, **request_kwargs)
        response.raise_for_status()

        return PredictionGroup.create_prediction_group(
            response.json(),
            image_path=image_path,
            prediction_type=INSTANCE_SEGMENTATION_MODEL,
        )

    def __str__(self):
        return f"<{type(self).__name__} id={self.id}, api_url={self.api_url}>"
