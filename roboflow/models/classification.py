import base64
import io
import json
import os
import urllib

import requests
from PIL import Image

from roboflow.config import CLASSIFICATION_MODEL
from roboflow.util.image_utils import check_image_url
from roboflow.util.prediction import PredictionGroup


class ClassificationModel:
    def __init__(self, api_key, id, name=None, version=None, local=False):
        """
        :param api_key: private roboflow api key
        :param id: the workspace/project id
        :param name: is the name of thep project
        :param version: version number
        :return ClassificationModel Object
        """
        # Instantiate different API URL parameters
        self.__api_key = api_key
        self.id = id
        self.name = name
        self.version = version
        self.base_url = "https://classify.roboflow.com/"

        if self.name is not None and version is not None:
            self.__generate_url()

    def predict(self, image_path, hosted=False):
        """

        :param image_path: path to the image you'd like to perform prediction on
        :param hosted: whether the image you're providing is hosted online
        :return: PredictionGroup object
        """
        self.__generate_url()
        self.__exception_check(image_path_check=image_path)
        # If image is local image
        if not hosted:
            # Open Image in RGB Format
            image = Image.open(image_path).convert("RGB")
            # Create buffer
            buffered = io.BytesIO()
            image.save(buffered, quality=90, format="JPEG")
            # Base64 encode image
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode("ascii")
            # Post to API and return response
            resp = requests.post(
                self.api_url,
                data=img_str,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        else:
            # Create API URL for hosted image (slightly different)
            self.api_url += "&image=" + urllib.parse.quote_plus(image_path)
            # POST to the API
            resp = requests.get(self.api_url)

        if resp.status_code != 200:
            raise Exception(resp.text)

        return PredictionGroup.create_prediction_group(
            resp.json(), image_path=image_path, prediction_type=CLASSIFICATION_MODEL
        )

    def load_model(self, name, version):
        """
        :param name: is the name of the model you'd like to load
        :param version: version number
        """
        # Load model based on user defined characteristics
        self.name = name
        self.version = version
        self.__generate_url()

    def __generate_url(self):
        """
        :return: roboflow API url
        """

        # Generates URL based on all parameters
        splitted = self.id.rsplit("/")
        without_workspace = splitted[1]

        self.api_url = "".join(
            [
                self.base_url + without_workspace + "/" + str(self.version),
                "?api_key=" + self.__api_key,
                "&name=YOUR_IMAGE.jpg",
            ]
        )

    def __exception_check(self, image_path_check=None):
        """
        :param image_path_check: checks to see if the image exists.
        """
        # Checks if image exists
        if image_path_check is not None:
            if not os.path.exists(image_path_check) and not check_image_url(
                image_path_check
            ):
                raise Exception("Image does not exist at " + image_path_check + "!")

    def __str__(self):
        """
        String representation of classification object
        """
        json_value = {
            "name": self.name,
            "version": self.version,
            "base_url": self.base_url,
        }

        return json.dumps(json_value, indent=2)
