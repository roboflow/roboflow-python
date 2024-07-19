import base64
import copy
import io
import json
import os
import random
import urllib

import requests
from PIL import Image

from roboflow.config import OBJECT_DETECTION_MODEL, OBJECT_DETECTION_URL
from roboflow.models.inference import InferenceModel
from roboflow.util.image_utils import check_image_url
from roboflow.util.prediction import PredictionGroup
from roboflow.util.versions import print_warn_for_wrong_dependencies_versions


class ObjectDetectionModel(InferenceModel):
    """
    Run inference on an object detection model hosted on Roboflow or served through Roboflow Inference.
    """  # noqa: E501 // docs

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
        colors=None,
        preprocessing=None,
    ):
        """
        Create a ObjectDetectionModel object through which you can run inference.

        Args:
            api_key (str): Your API key (obtained via your workspace API settings page).
            name (str): The url-safe version of the dataset name. You can find it in the web UI by looking at
                        the URL on the main project view or by clicking the "Get curl command" button in the train
                        results section of your dataset version after training your model.
            local (str): Address of the local server address if running a local Roboflow deployment server.
                        Ex. http://localhost:9001/
            version (str): The version number identifying the version of your dataset.
            classes (str): Restrict the predictions to only those of certain classes. Provide as a comma-separated string.
            overlap (int): The maximum percentage (on a scale of 0-100) that bounding box predictions of the same class are
                        allowed to overlap before being combined into a single box.
            confidence (int): A threshold for the returned predictions on a scale of 0-100. A lower number will return
                            more predictions. A higher number will return fewer high-certainty predictions.
            stroke (int): The width (in pixels) of the bounding box displayed around predictions (only has an effect when
                        format is image).
            labels (bool): Whether or not to display text labels on the predictions (only has an effect when format is
                        image).
            format (str): The format of the output.
                        - 'json': returns an array of JSON predictions (See response format tab).
                        - 'image': returns an image with annotated predictions as a binary blob with a Content-Type
                                    of image/jpeg.
        """  # noqa: E501 // docs
        # Instantiate different API URL parameters
        # To be moved to predict
        super().__init__(api_key, id)
        self.__api_key = api_key
        self.id = id
        self.name = name
        self.version = version or self.version
        self.classes = classes
        self.overlap = overlap
        self.confidence = confidence
        self.stroke = stroke
        self.labels = labels
        self.format = format
        self.colors = {} if colors is None else colors
        self.preprocessing = {} if preprocessing is None else preprocessing

        # local needs to be passed from Project
        if local is None:
            self.base_url = OBJECT_DETECTION_URL + "/"
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
        Loads a Model from on a model endpoint.

        Args:
            name (str): The url-safe version of the dataset name
            version (str): The version number identifying the version of your dataset.
            local (bool): Whether the model is hosted locally or on Roboflow
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

    def predict(  # type: ignore[override]
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
        Infers detections based on image from specified model and image path.

        Args:
            image_path (str): path to the image you'd like to perform prediction on
            hosted (bool): whether the image you're providing is hosted on Roboflow
            format (str): The format of the output.

        Returns:
            PredictionGroup Object

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("YOUR_IMAGE.jpg")
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
        if hasattr(image_path, "__len__") is True:
            pass
        else:
            self.__exception_check(image_path_check=image_path)

        resize = False
        original_dimensions = None
        # If image is local image
        if not hosted:
            import cv2
            import numpy as np

            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
                dimensions = image.size
                original_dimensions = copy.deepcopy(dimensions)

                # Here we resize the image to the preprocessing settings
                # before sending it over the wire
                if "resize" in self.preprocessing.keys():
                    if dimensions[0] > int(self.preprocessing["resize"]["width"]) or dimensions[1] > int(
                        self.preprocessing["resize"]["height"]
                    ):
                        image = image.resize(
                            (
                                int(self.preprocessing["resize"]["width"]),
                                int(self.preprocessing["resize"]["height"]),
                            )
                        )
                        dimensions = image.size
                        resize = True

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

                image_dims = {
                    "width": str(original_dimensions[0]),
                    "height": str(original_dimensions[1]),
                }
            elif isinstance(image_path, np.ndarray):
                # Performing inference on a OpenCV2 frame
                retval, buffer = cv2.imencode(".jpg", image_path)
                # Currently cv2.imencode does not properly return shape
                dimensions = buffer.shape
                img_str = base64.b64encode(buffer)  # type: ignore[arg-type]
                img_str = img_str.decode("ascii")
                resp = requests.post(
                    self.api_url,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                # Replace with dimensions variable once
                # cv2.imencode shape solution is found
                image_dims = {"width": "0", "height": "0"}
            else:
                raise ValueError("image_path must be a string or a numpy array.")
        else:
            # Create API URL for hosted image (slightly different)
            self.api_url += "&image=" + urllib.parse.quote_plus(image_path)
            image_dims = {"width": "0", "height": "0"}
            # POST to the API
            resp = requests.post(self.api_url)

        resp.raise_for_status()
        # Return a prediction group if JSON data
        if self.format == "json":
            resp_json = resp.json()

            if resize and original_dimensions is not None:
                new_preds = []
                for p in resp_json["predictions"]:
                    p["x"] = int(p["x"] * (int(original_dimensions[0]) / int(self.preprocessing["resize"]["width"])))
                    p["y"] = int(p["y"] * (int(original_dimensions[1]) / int(self.preprocessing["resize"]["height"])))
                    p["width"] = int(
                        p["width"] * (int(original_dimensions[0]) / int(self.preprocessing["resize"]["width"]))
                    )
                    p["height"] = int(
                        p["height"] * (int(original_dimensions[1]) / int(self.preprocessing["resize"]["height"]))
                    )

                    new_preds.append(p)

                resp_json["predictions"] = new_preds

            return PredictionGroup.create_prediction_group(
                resp_json,
                image_path=image_path,
                prediction_type=OBJECT_DETECTION_MODEL,
                image_dims=image_dims,
                colors=self.colors,
            )
        # Returns base64 encoded Data
        elif self.format == "image":
            return resp.content

    def webcam(
        self,
        webcam_id=0,
        inference_engine_url="https://detect.roboflow.com/",
        within_jupyter=False,
        confidence=40,
        overlap=30,
        stroke=1,
        labels=False,
        web_cam_res=(416, 416),
    ):
        """
        Infers detections based on webcam feed from specified model.

        Args:
            webcam_id (int): Webcam ID (default 0)
            inference_engine_url (str): Inference engine address to use (default https://detect.roboflow.com)
            within_jupyter (bool): Whether or not to display the webcam within Jupyter notebook (default True)
            confidence (int): Confidence threshold for detections
            overlap (int): Overlap threshold for detections
            stroke (int): Stroke width for bounding box
            labels (bool): Whether to show labels on bounding box
        """  # noqa: E501 // docs
        import cv2

        os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

        # Generate url before predicting
        self.__generate_url(
            confidence=confidence,
            overlap=overlap,
            stroke=stroke,
            labels=labels,
            inference_engine_url=inference_engine_url,
        )

        def plot_one_box(x, img, color=None, label=None, line_thickness=None, colors=None):
            # Plots one bounding box on image img

            self.colors = {} if colors is None else colors

            if label in self.colors and label is not None:
                color = self.colors[label]
                color = color.lstrip("#")
                color = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
            else:
                color = [random.randint(0, 255) for _ in range(3)]

            tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness

            c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

            cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            if label:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(
                    img,
                    label,
                    (c1[0], c1[1] - 2),
                    0,
                    tl / 3,
                    [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA,
                )

        cap = cv2.VideoCapture(webcam_id)

        if cap is None or not cap.isOpened():
            raise (Exception("No webcam available at webcam_id " + str(webcam_id)))

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, web_cam_res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, web_cam_res[1])

        if within_jupyter:
            os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
            print_warn_for_wrong_dependencies_versions([("IPython", ">=", "7.0.0")])
            print_warn_for_wrong_dependencies_versions([("ipywidgets", ">=", "7.0.0")])

            import threading

            import ipywidgets as widgets
            from IPython.display import Image as IPythonImage
            from IPython.display import display

            display_handle = display("loading Roboflow model...", display_id=True)

            # Stop button
            # ================
            stopButton = widgets.ToggleButton(
                value=False,
                description="Stop Inference",
                disabled=False,
                button_style="danger",  # 'success', 'info', 'warning', 'danger' or ''
                tooltip="Description",
                icon="square",  # (FontAwesome names without the `fa-` prefix)
            )

        else:
            cv2.namedWindow("Roboflow Webcam Inference", cv2.WINDOW_NORMAL)
            cv2.startWindowThread()

            stopButton = None

        def view(button):
            while True:
                if stopButton is not None:
                    if stopButton.value is True:
                        break
                else:
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # quit when 'q' is pressed
                        break

                _, frame = cap.read()
                frame = cv2.resize(frame, web_cam_res)

                frame = cv2.flip(frame, 1)  # if your camera reverses your image

                _, frame_upload = cv2.imencode(".jpeg", frame)
                img_str = base64.b64encode(frame_upload)  # type: ignore[arg-type]
                img_str = img_str.decode("ascii")

                # post frame to the Roboflow API
                r = requests.post(
                    self.api_url,
                    data=img_str,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )

                json = r.json()

                predictions = json["predictions"]

                formatted_predictions = []
                classes = []

                for pred in predictions:
                    formatted_pred = [
                        pred["x"],
                        pred["y"],
                        pred["x"],
                        pred["y"],
                        pred["confidence"],
                    ]

                    # convert to top-left x/y from center
                    formatted_pred[0] = int(formatted_pred[0] - pred["width"] / 2)
                    formatted_pred[1] = int(formatted_pred[1] - pred["height"] / 2)
                    formatted_pred[2] = int(formatted_pred[2] + pred["width"] / 2)
                    formatted_pred[3] = int(formatted_pred[3] + pred["height"] / 2)

                    formatted_predictions.append(formatted_pred)
                    classes.append(pred["class"])

                    plot_one_box(
                        formatted_pred,
                        frame,
                        label=pred["class"],
                        line_thickness=2,
                        colors=self.colors,
                    )

                _, frame_display = cv2.imencode(".jpeg", frame)

                if within_jupyter:
                    display_handle.update(IPythonImage(data=frame_display.tobytes()))
                else:
                    cv2.imshow("Roboflow Webcam Inference", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):  # quit when 'q' is pressed
                        cap.release()
                        break

            cap.release()
            if not within_jupyter:
                cv2.destroyWindow("Roboflow Webcam Inference")
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            return

        if within_jupyter:
            display(stopButton)
            thread = threading.Thread(target=view, args=(stopButton,))
            thread.start()
        else:
            view(stopButton)

    def __exception_check(self, image_path_check=None):
        # Check if Image path exists exception check
        # (for both hosted URL and local image)
        if image_path_check is not None:
            if not os.path.exists(image_path_check) and not check_image_url(image_path_check):
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
        inference_engine_url=None,
    ):
        """
        Generate the URL to run inference on.
        """
        # Reassign parameters if any parameters are changed
        if local is not None:
            if not local:
                self.base_url = OBJECT_DETECTION_URL + "/"
            else:
                self.base_url = "http://localhost:9001/"

        if inference_engine_url is not None:
            self.base_url = inference_engine_url

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
