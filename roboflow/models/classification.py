import base64
import io
import json
import os
import urllib
from typing import Optional

import requests
from PIL import Image

from roboflow.config import CLASSIFICATION_MODEL
from roboflow.models.inference import InferenceModel
from roboflow.util.image_utils import check_image_url
from roboflow.util.prediction import PredictionGroup


class ClassificationModel(InferenceModel):
    """
    Run inference on a classification model hosted on Roboflow or served through
        Roboflow Inference.
    """

    def __init__(
        self,
        api_key: str,
        id: str,
        name: Optional[str] = None,
        version: Optional[str] = None,
        local: Optional[str] = None,
        colors: Optional[dict] = None,
        preprocessing: Optional[dict] = None,
    ):
        """
        Create a ClassificationModel object through which you can run inference.

        Args:
            api_key (str): private roboflow api key
            id (str): the workspace/project id
            name (str): is the name of the project
            version (str): version number
            local (str): localhost address and port if pointing towards local inference engine
            colors (dict): colors to use for the image
            preprocessing (dict): preprocessing to use for the image

        Returns:
            ClassificationModel Object
        """
        # Instantiate different API URL parameters
        super().__init__(api_key, id, version=version)
        self.__api_key = api_key
        self.id = id
        self.name = name
        self.version = version
        self.base_url = "https://classify.roboflow.com/"

        if self.name is not None and version is not None:
            self.__generate_url()

        self.colors = {} if colors is None else colors
        self.preprocessing = {} if preprocessing is None else preprocessing

        if local:
            print(f"initalizing local classification model hosted at : {local}")
            self.base_url = local

    def predict(self, image_path, hosted=False):  # type: ignore[override]
        """
        Run inference on an image.

        Args:
            image_path (str): path to the image you'd like to perform prediction on
            hosted (bool): whether the image you're providing is hosted on Roboflow

        Returns:
            PredictionGroup Object

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("YOUR_IMAGE.jpg")
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
            img_dims = image.size
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
            resp = requests.post(self.api_url)
            img_dims = {"width": "0", "height": "0"}

        if resp.status_code != 200:
            raise Exception(resp.text)

        return PredictionGroup.create_prediction_group(
            resp.json(),
            image_dims=img_dims,
            image_path=image_path,
            prediction_type=CLASSIFICATION_MODEL,
            colors=self.colors,
        )

    def load_model(self, name, version):
        """
        Load a model.

        Args:
            name (str): is the name of the model you'd like to load
            version (int): version number
        """
        # Load model based on user defined characteristics
        self.name = name
        self.version = version
        self.__generate_url()

    def __generate_url(self):
        """
        Generate a Roboflow API URL on which to run inference.

        Returns:
            url (str): the url on which to run inference
        """

        # Generates URL based on all parameters
        splitted = self.id.rsplit("/")
        without_workspace = splitted[1]
        version = self.version
        if not version and len(splitted) > 2:
            version = splitted[2]

        self.api_url = "".join(
            [
                self.base_url + without_workspace + "/" + str(version),
                "?api_key=" + self.__api_key,
                "&name=YOUR_IMAGE.jpg",
            ]
        )

    def __exception_check(self, image_path_check=None):
        """
        Check to see if an image exists.

        Args:
            image_path_check (str): path to the image to check

        Raises:
            Exception: if image does not exist
        """
        # Checks if image exists
        if image_path_check is not None:
            if not os.path.exists(image_path_check) and not check_image_url(image_path_check):
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
