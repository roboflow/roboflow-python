import io
import json
import os
import urllib.request
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib import patches
from PIL import Image

from roboflow.config import (
    CLASSIFICATION_MODEL,
    OBJECT_DETECTION_MODEL,
    PREDICTION_OBJECT,
)
from roboflow.util.image_utils import check_image_url


def exception_check(image_path_check=None):
    # Check if Image path exists exception check (for both hosted URL and local image)
    if image_path_check is not None:
        if not os.path.exists(image_path_check) and not check_image_url(
            image_path_check
        ):
            raise Exception("Image does not exist at " + image_path_check + "!")


def plot_image(image_path):
    """
    Helper method to plot image

    :param image_path: path of image to be plotted (can be hosted or local)
    :return:
    """
    # Exception to check if image path exists
    exception_check(image_path_check=image_path)
    # Try opening local image
    try:
        img = Image.open(image_path)
    except OSError:
        # Try opening Hosted image
        response = requests.get(image_path)
        img = Image.open(io.BytesIO(response.content))
    # Plot image axes
    figure, axes = plt.subplots()
    axes.imshow(img)
    return figure, axes


def plot_annotation(axes, prediction=None, stroke=1):
    """
    Helper method to plot annotations

    :param axes:
    :param prediction:
    :return:
    """
    # Object Detection annotation
    if prediction["prediction_type"] == OBJECT_DETECTION_MODEL:
        # Get height, width, and center coordinates of prediction
        if prediction is not None:
            height = prediction["height"]
            width = prediction["width"]
            x = prediction["x"]
            y = prediction["y"]
            rect = patches.Rectangle(
                (x - width / 2, y - height / 2),
                width,
                height,
                linewidth=stroke,
                edgecolor="r",
                facecolor="none",
            )
            # Plot Rectangle
            axes.add_patch(rect)
    elif prediction["prediction_type"] == CLASSIFICATION_MODEL:
        axes.set_title(
            "Class: "
            + prediction["top"]
            + " | Confidence: "
            + str(prediction["confidence"])
        )


class Prediction:
    def __init__(
        self, json_prediction, image_path, prediction_type=OBJECT_DETECTION_MODEL
    ):
        """
        Generalized Prediction for both Object Detection and Classification Models

        :param json_prediction:
        :param image_path:
        """
        # Set image path in JSON prediction
        json_prediction["image_path"] = image_path
        json_prediction["prediction_type"] = prediction_type
        self.image_path = image_path
        self.json_prediction = json_prediction

    def json(self):
        return self.json_prediction

    def __load_image(self):
        if "http://" in self.image_path:
            req = urllib.request.urlopen(self.image_path)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, -1)  # 'Load it as it is'

            return image

        return cv2.imread(self.image_path)

    def plot(self, stroke=1):
        # Exception to check if image path exists
        exception_check(image_path_check=self["image_path"])
        figure, axes = plot_image(self["image_path"])

        plot_annotation(axes, self, stroke)
        plt.show()

    # saves a single box or classification on the image
    def save(self, output_path="predictions.jpg", stroke=2):
        image = self.__load_image()
        if self["prediction_type"] == OBJECT_DETECTION_MODEL:
            # Get different dimensions/coordinates
            x = self["x"]
            y = self["y"]
            width = self["width"]
            height = self["height"]
            class_name = self["class"]
            # Draw bounding boxes for object detection prediction
            cv2.rectangle(
                image,
                (int(x - width / 2), int(y + height / 2)),
                (int(x + width / 2), int(y - height / 2)),
                (255, 0, 0),
                stroke,
            )
            # Get size of text
            text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x - width / 2, y - height / 2 + 1),
                (
                    x - width / 2 + text_size[0] + 1,
                    y - height / 2 + int(1.5 * text_size[1]),
                ),
                (255, 0, 0),
                -1,
            )
            # Write text onto image
            cv2.putText(
                image,
                class_name,
                (int(x - width / 2), y + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                thickness=1,
            )

        elif self["prediction_type"] == CLASSIFICATION_MODEL:
            # Get image dimensions
            height, width = image.shape[:2]
            # Get bottom amount for image
            bottom = image[height - 2 : height, 0:width]
            # Get mean of bottom amount
            mean = cv2.mean(bottom)[0]
            border_size = 100
            # Apply Border
            image = cv2.copyMakeBorder(
                image,
                top=border_size,
                bottom=border_size,
                left=border_size,
                right=border_size,
                borderType=cv2.BORDER_CONSTANT,
                value=[mean, mean, mean],
            )
            # Add text and relax
            cv2.putText(
                image,
                (self["top"] + " | " + "Confidence: " + self["confidence"]),
                (int(width / 2), 5),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Write image path
        cv2.imwrite(output_path, image)

    def __str__(self) -> str:
        """
        :return: JSON formatted string of prediction
        """
        # Pretty print the JSON prediction as a String
        prediction_string = json.dumps(self.json_prediction, indent=2)
        return prediction_string

    def __getitem__(self, key):
        """

        :param key:
        :return:
        """
        # Allows the prediction to be accessed like a dictionary
        return self.json_prediction[key]

    # Make representation equal to string value
    __repr__ = __str__


class PredictionGroup:
    def __init__(self, image_dims, image_path, *args):
        """
        :param args: The prediction(s) to be added to the prediction group
        """
        # List of predictions (core of the PredictionGroup)
        self.predictions = []
        # Base image path (path of image of first prediction in prediction group)
        self.base_image_path = image_path
        # Base prediction type (prediction type of image of first prediction in prediction group)
        self.base_prediction_type = ""

        self.image_dims = image_dims
        # Iterate through the arguments
        for index, prediction in enumerate(args):
            # Set base image path based on first prediction
            if index == 0:
                self.base_image_path = prediction["image_path"]
                self.base_prediction_type = prediction["prediction_type"]
            # If not a Prediction object then do not allow into the prediction group
            self.__exception_check(is_prediction_check=prediction)
            # Add prediction to prediction group otherwise
            self.predictions.append(prediction)

    def add_prediction(self, prediction=None):
        """

        :param prediction: Prediction to add to the prediction group
        """
        # If not a Prediction object then do not allow into the prediction group
        # Also checks if prediction types are the same (i.e. object detection predictions in object detection groups)
        self.__exception_check(
            is_prediction_check=prediction,
            prediction_type_check=prediction["prediction_type"],
        )
        # If there is more than one prediction and the prediction image path is
        # not the group image path then warn user
        if self.__len__() > 0:
            self.__exception_check(image_path_check=prediction["image_path"])
        # If the prediction group is empty, make the base image path of the prediction
        elif self.__len__() == 0:
            self.base_image_path = prediction["image_path"]
        # Append prediction to group
        self.predictions.append(prediction)

    def plot(self, stroke=1):
        if len(self) > 0:
            # Check if image path exists
            exception_check(image_path_check=self.base_image_path)
            # Plot image if image path exists
            figure, axes = plot_image(self.base_image_path)
            # Plot annotations in prediction group
            for single_prediction in self:
                plot_annotation(axes, single_prediction, stroke)
        # Show the plot to the user
        plt.show()

    def __load_image(self):
        # Check if it is a hosted image and open image as needed
        if "http://" in self.base_image_path or "https://" in self.base_image_path:
            req = urllib.request.urlopen(self.base_image_path)
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            image = cv2.imdecode(arr, -1)  # 'Load it as it is'
            # Return array with image info
            return image
        # Return array with image info of local image
        return cv2.imread(self.base_image_path)

    def save(self, output_path="predictions.jpg", stroke=2):
        # Load image based on image path as an array
        image = self.__load_image()
        # Iterate through predictions and add prediction to image
        for prediction in self.predictions:
            # Check what type of prediction it is
            if self.base_prediction_type == OBJECT_DETECTION_MODEL:
                # Get different dimensions/coordinates
                x = prediction["x"]
                y = prediction["y"]
                width = prediction["width"]
                height = prediction["height"]
                class_name = prediction["class"]
                # Draw bounding boxes for object detection prediction
                cv2.rectangle(
                    image,
                    (int(x - width / 2), int(y + height / 2)),
                    (int(x + width / 2), int(y - height / 2)),
                    (255, 0, 0),
                    stroke,
                )
                # Get size of text
                text_size = cv2.getTextSize(
                    class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )[0]
                # Draw background rectangle for text
                cv2.rectangle(
                    image,
                    (int(x - width / 2), int(y - height / 2 + 1)),
                    (
                        int(x - width / 2 + text_size[0] + 1),
                        int(y - height / 2 + int(1.5 * text_size[1])),
                    ),
                    (255, 0, 0),
                    -1,
                )
                # Write text onto image
                cv2.putText(
                    image,
                    class_name,
                    (int(x - width / 2), int(y - height / 2 + text_size[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    thickness=1,
                )
            # Plot for classification model
            elif self.base_prediction_type == CLASSIFICATION_MODEL:
                # Get image dimensions
                height, width = image.shape[:2]

                border_size = 100
                text = (
                    "Class: "
                    + prediction["top"]
                    + " | "
                    + "Confidence: "
                    + str(prediction["confidence"])
                )
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)[0]
                # Apply Border
                image = cv2.copyMakeBorder(
                    image,
                    top=border_size,
                    bottom=border_size,
                    left=border_size,
                    right=border_size,
                    borderType=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                )
                # get coords
                text_x = (image.shape[1] - text_size[0]) / 2
                # Add text and relax
                cv2.putText(
                    image,
                    text,
                    (int(text_x), int(border_size / 2)),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (0, 0, 0),
                    1,
                )
        # Write image path
        cv2.imwrite(output_path, image)

    def __str__(self):
        """

        :return:
        """
        # final string to be returned for the prediction group
        prediction_group_string = ""
        # Iterate through the predictions and convert each prediction into a string format
        for prediction in self.predictions:
            prediction_group_string += str(prediction) + "\n\n"
        # return the prediction group string
        return prediction_group_string

    def __getitem__(self, index):
        # Allows prediction group to be accessed via an index
        return self.predictions[index]

    def __len__(self):
        # Length of prediction based off of number of predictions
        return len(self.predictions)

    def __exception_check(
        self,
        is_prediction_check=None,
        image_path_check=None,
        prediction_type_check=None,
    ):
        # Ensures only predictions can be added to a prediction group
        if is_prediction_check is not None:
            if type(is_prediction_check).__name__ is not PREDICTION_OBJECT:
                raise Exception(
                    "Cannot add type "
                    + type(is_prediction_check).__name__
                    + " to PredictionGroup"
                )

        # Warns user if predictions have different prediction types
        if prediction_type_check is not None:
            if (
                self.__len__() > 0
                and prediction_type_check != self.base_prediction_type
            ):
                warnings.warn(
                    "This prediction is a different type ("
                    + prediction_type_check
                    + ") than the prediction group base type ("
                    + self.base_prediction_type
                    + ")"
                )

        # Gives user warning that base path is not equal to image path
        if image_path_check is not None:
            if self.base_image_path != image_path_check:
                warnings.warn(
                    "This prediction has a different image path ("
                    + image_path_check
                    + ") than the prediction group base image path ("
                    + self.base_image_path
                    + ")"
                )

    def json(self):
        prediction_group_json = {"predictions": []}
        for prediction in self.predictions:
            prediction_group_json["predictions"].append(prediction.json())

        prediction_group_json["image"] = self.image_dims
        return prediction_group_json

    @staticmethod
    def create_prediction_group(json_response, image_path, prediction_type):
        """
        Method to create a prediction group based on the JSON Response

        :param prediction_type:
        :param json_response: Based on Roboflow JSON Response from Inference API
        :param model:
        :param image_path:
        :return:
        """
        # List of predictions
        prediction_list = []
        # For object detection model
        if prediction_type == OBJECT_DETECTION_MODEL:
            # get all predicted bounding boxes for image
            for prediction in json_response["predictions"]:
                # Create prediction for bbox
                prediction = Prediction(
                    prediction, image_path, prediction_type=prediction_type
                )
                # Add to prediction list
                prediction_list.append(prediction)
            img_dims = json_response["image"]
        # For classification model
        elif prediction_type == CLASSIFICATION_MODEL:
            # Create prediction for predicted class
            prediction = Prediction(json_response, image_path, prediction_type)
            # add to prediction list
            prediction_list.append(prediction)

            img_dims = {}
        # Seperate list and return as a prediction group
        return PredictionGroup(img_dims, image_path, *prediction_list)
