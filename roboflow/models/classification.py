import base64
import io
import os

import requests
from PIL import Image

from roboflow.util.prediction import PredictionGroup
from roboflow.config import CLASSIFICATION_MODEL


class ClassificationModel:
    def __init__(self, api_key, dataset_slug=None, version=None):
        """

        :param api_key:
        :param dataset_slug:
        :param version:
        """
        # Instantiate different API URL parameters
        self.api_key = api_key
        self.dataset_slug = dataset_slug
        self.version = version
        self.base_url = "https://classify.roboflow.com/"

        if dataset_slug is not None and version is not None:
            self.__generate_url()

    def predict(self, image_path, hosted=False):
        """

        :param image_path:
        :param hosted:
        :return:
        """
        self.__exception_check(image_path_check=image_path)
        if not hosted:
            # Load Image with PIL
            image = Image.open(image_path).convert("RGB")

            # Convert to JPEG Buffer
            buffered = io.BytesIO()
            image.save(buffered, quality=90, format="JPEG")

            # Base 64 Encode
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode("ascii")

            # POST to the API
            resp = requests.post(self.api_url, data=img_str, headers={
                "Content-Type": "application/x-www-form-urlencoded"
            })

            return PredictionGroup.create_prediction_group(resp.json(),
                                                           image_path=image_path,
                                                           prediction_type=CLASSIFICATION_MODEL)

    def load_model(self, dataset_slug, version):
        """

        :param dataset_slug:
        :param version:
        :return:
        """
        self.dataset_slug = dataset_slug
        self.version = version
        self.__generate_url()

    def __generate_url(self):
        """

        :return:
        """
        self.api_url = "".join([
            self.base_url + self.dataset_slug + '/' + self.version,
            "?api_key=" + self.api_key,
            "&name=YOUR_IMAGE.jpg"])

    def __exception_check(self, image_path_check=None):
        """

        :param image_path_check:
        :return:
        """
        if image_path_check is not None:
            if not os.path.exists(image_path_check):
                raise Exception("Image does not exist at " + image_path_check + "!")
