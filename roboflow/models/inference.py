import io
import urllib

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder

from roboflow.util.image_utils import validate_image_path
from roboflow.util.prediction import PredictionGroup


class InferenceModel:
    def __init__(self, api_key, version_id, *args, **kwargs):
        """
        :param api_key: Your API key (obtained via your workspace API settings page)
        :param version_id: The ID of the dataset version to use for predicting
        """
        self.__api_key = api_key
        self.id = version_id

        version_info = self.id.rsplit("/")
        self.dataset_id = version_info[1]
        self.version = version_info[2]

    def __get_image_params(self, image_path):
        """
        :param image_path: Path to image (can be local path or hosted URL)

        :return: Tuple containing a dict of querystring params and a dict of requests kwargs
        :raises Exception: Image path is not valid
        """
        validate_image_path(image_path)

        hosted_image = urllib.parse.urlparse(image_path).scheme in (
            "http",
            "https",
        )

        if hosted_image:
            return {"image": image_path}, {}

        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")
        data = MultipartEncoder(
            fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")}
        )
        return {}, {
            "data": data,
            "headers": {"Content-Type": data.content_type},
        }

    def predict(self, image_path, prediction_type=None, **kwargs):
        """
        Infers detections based on image from a specified model and image path

        :param image_path: Path to image (can be local path or hosted URL)
        :param **kwargs: Any additional kwargs will be turned into querystring params

        :return: PredictionGroup - a group of predictions based on Roboflow JSON response
        :raises Exception: Image path is not valid
        """
        params, request_kwargs = self.__get_image_params(image_path)

        params["api_key"] = self.__api_key

        params.update(**kwargs)

        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
        response = requests.post(url, **request_kwargs)
        response.raise_for_status()

        return PredictionGroup.create_prediction_group(
            response.json(),
            image_path=image_path,
            prediction_type=prediction_type,
        )
