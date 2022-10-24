import base64
import io
import json
import os
import urllib

import cv2
import requests
from PIL import Image

from roboflow.config import OBJECT_DETECTION_MODEL
from roboflow.util.image_utils import check_image_url
from roboflow.util.prediction import PredictionGroup


class ObjectDetectionModel:
    def __init__(
        self,
        api_key,
        id,
        name=None,
        version=None,
        local=None,
        classes=None,
        overlap=30,
        confidence=40,
        stroke=1,
        labels=False,
        format="json",
    ):
        """
        From Roboflow Docs:

        :param api_key: Your API key (obtained via your workspace API settings page)
        :param name: The url-safe version of the dataset name.  You can find it in the web UI by looking at
        the URL on the main project view or by clicking the "Get curl command" button in the train results section of
        your dataset version after training your model.
        :param local: Address of the local server address if running a local Roboflow deployment server. ex. http://localhost:9001/
        :param version: The version number identifying the version of of your dataset
        :param classes: Restrict the predictions to only those of certain classes. Provide as a comma-separated string.
        :param overlap: The maximum percentage (on a scale of 0-100) that bounding box predictions of the same class are
        allowed to overlap before being combined into a single box.
        :param confidence: A threshold for the returned predictions on a scale of 0-100. A lower number will return
        more predictions. A higher number will return fewer high-certainty predictions
        :param stroke: The width (in pixels) of the bounding box displayed around predictions (only has an effect when
        format is image)
        :param labels: Whether or not to display text labels on the predictions (only has an effect when format is
        image).
        :param format: json - returns an array of JSON predictions. (See response format tab).
                       image - returns an image with annotated predictions as a binary blob with a Content-Type
                       of image/jpeg.
        """
        # Instantiate different API URL parameters
        # To be moved to predict
        self.__api_key = api_key
        self.id = id
        self.name = name
        self.version = version
        self.classes = classes
        self.overlap = overlap
        self.confidence = confidence
        self.stroke = stroke
        self.labels = labels
        self.format = format

        # local needs to be passed from Project
        if local is None:
            self.base_url = "https://detect.roboflow.com/"
        else:
            print("initalizing local object detection model hosted at :" + local)
            self.base_url = local

        # If dataset slug not none, instantiate API URL
        if name is not None and version is not None:
            self.__generate_url()

    def load_model(
        self,
        name,
        version,
        local=None,
        classes=None,
        overlap=None,
        confidence=None,
        stroke=None,
        labels=None,
        format=None,
    ):
        """
        Loads a Model based on a Model Endpoint

        :param model_endpoint: This is the endpoint that is loaded into the api_url
        """
        # To load a model manually, they must specify a dataset slug
        self.name = name
        self.version = version
        # Generate URL based on parameters
        self.__generate_url(
            local=local,
            classes=classes,
            overlap=overlap,
            confidence=confidence,
            stroke=stroke,
            labels=labels,
            format=format,
        )

    def predict(
        self,
        image_path,
        hosted=False,
        format=None,
        classes=None,
        overlap=30,
        confidence=40,
        stroke=1,
        labels=False,
    ):
        """
        Infers detections based on image from specified model and image path

        :param image_path: Path to image or image array (can be local or hosted)
        :param hosted: If image located on a hosted server, hosted should be True
        :param format: output format from this method
        :return: PredictionGroup --> a group of predictions based on Roboflow JSON response
        """
        # Generate url before predicting
        self.__generate_url(
            format=format,
            classes=classes,
            overlap=overlap,
            confidence=confidence,
            stroke=stroke,
            labels=labels,
        )

        # Check if image exists at specified path or URL or is an array
        if hasattr(image_path, "__len__") == True:
            pass
        else:
            self.__exception_check(image_path_check=image_path)

        # If image is local image
        if not hosted:
            if ".jpg" in image_path or ".png" in image_path:  # Open Image in RGB Format
                image = Image.open(image_path).convert("RGB")

                # Create buffer
                buffered = io.BytesIO()
                image.save(buffered, format="PNG")
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
                # Performing inference on a OpenCV2 frame
                retval, buffer = cv2.imencode(".jpg", image_path)
                img_str = base64.b64encode(buffer)
                # print(img_str)
                img_str = img_str.decode("ascii")
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
        # Return a prediction group if JSON data
        if self.format == "json":
            return PredictionGroup.create_prediction_group(
                resp.json(),
                image_path=image_path,
                prediction_type=OBJECT_DETECTION_MODEL,
            )
        # Returns base64 encoded Data
        elif self.format == "image":
            return resp.content

    def __exception_check(self, image_path_check=None):
        # Check if Image path exists exception check (for both hosted URL and local image)
        if image_path_check is not None:
            if not os.path.exists(image_path_check) and not check_image_url(
                image_path_check
            ):
                raise Exception("Image does not exist at " + image_path_check + "!")

    def __generate_url(
        self,
        local=None,
        classes=None,
        overlap=None,
        confidence=None,
        stroke=None,
        labels=None,
        format=None,
    ):

        # Reassign parameters if any parameters are changed
        if local is not None:
            if not local:
                self.base_url = "https://detect.roboflow.com/"
            else:
                self.base_url = "http://localhost:9001/"

        # Change any variables that the user wants to change
        if classes is not None:
            self.classes = classes
        if overlap is not None:
            self.overlap = overlap
        if confidence is not None:
            self.confidence = confidence
        if stroke is not None:
            self.stroke = stroke
        if labels is not None:
            self.labels = labels
        if format is not None:
            self.format = format

        # Create the new API URL
        splitted = self.id.rsplit("/")
        without_workspace = splitted[1]

        self.api_url = "".join(
            [
                self.base_url + without_workspace + "/" + str(self.version),
                "?api_key=" + self.__api_key,
                "&name=YOUR_IMAGE.jpg",
                "&overlap=" + str(self.overlap),
                "&confidence=" + str(self.confidence),
                "&stroke=" + str(self.stroke),
                "&labels=" + str(self.labels).lower(),
                "&format=" + self.format,
            ]
        )
        # add classes parameter to api
        if self.classes is not None:
            self.api_url += "&classes=" + self.classes

    def __str__(self):
        # Create the new API URL
        splitted = self.id.rsplit("/")
        without_workspace = splitted[1]

        json_value = {
            "id": without_workspace + "/" + str(self.version),
            "name": self.name,
            "version": self.version,
            "classes": self.classes,
            "overlap": self.overlap,
            "confidence": self.confidence,
            "stroke": self.stroke,
            "labels": self.labels,
            "format": self.format,
            "base_url": self.base_url,
        }

        return json.dumps(json_value, indent=2)
