import io
import json
import os
import time
import urllib
from typing import Optional, Tuple
from urllib.parse import urljoin

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
from tqdm import tqdm

from roboflow.config import API_URL
from roboflow.util.image_utils import validate_image_path
from roboflow.util.prediction import PredictionGroup

SUPPORTED_ROBOFLOW_MODELS = ["batch-video"]

SUPPORTED_ADDITIONAL_MODELS = {
    "clip": {
        "model_id": "clip",
        "model_version": "1",
        "inference_type": "clip-embed-image",
    },
    "gaze": {
        "model_id": "gaze",
        "model_version": "1",
        "inference_type": "gaze-detection",
    },
}


class InferenceModel:
    def __init__(
        self,
        api_key,
        version_id,
        colors=None,
        *args,
        **kwargs,
    ):
        """
        Create an InferenceModel object through which you can run inference.

        Args:
            api_key (str): private roboflow api key
            version_id (str): the ID of the dataset version to use for inference
        """

        self.__api_key = api_key
        self.id = version_id

        if version_id != "BASE_MODEL":
            version_info = self.id.rsplit("/")
            self.dataset_id = version_info[1]
            self.version = version_info[2]
            self.colors = {} if colors is None else colors

    def __get_image_params(self, image_path):
        """
        Get parameters about an image (i.e. dimensions) for use in an inference request.

        Args:
            image_path (str): path to the image you'd like to perform prediction on

        Returns:
            Tuple containing a dict of querystring params and a dict of requests kwargs

        Raises:
            Exception: Image path is not valid
        """
        validate_image_path(image_path)

        hosted_image = urllib.parse.urlparse(image_path).scheme in ("http", "https")

        if hosted_image:
            image_dims = {"width": "Undefined", "height": "Undefined"}
            return {"image": image_path}, {}, image_dims

        image = Image.open(image_path)
        dimensions = image.size
        image_dims = {"width": str(dimensions[0]), "height": str(dimensions[1])}
        buffered = io.BytesIO()
        image.save(buffered, quality=90, format="JPEG")
        data = MultipartEncoder(fields={"file": ("imageToUpload", buffered.getvalue(), "image/jpeg")})
        return (
            {},
            {"data": data, "headers": {"Content-Type": data.content_type}},
            image_dims,
        )

    def predict(self, image_path, prediction_type=None, **kwargs):
        """
        Infers detections based on image from a specified model and image path.

        Args:
            image_path (str): path to the image you'd like to perform prediction on
            prediction_type (str): type of prediction to perform
            **kwargs: Any additional kwargs will be turned into querystring params

        Returns:
            PredictionGroup Object

        Raises:
            Exception: Image path is not valid

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("YOUR_IMAGE.jpg")
        """
        params, request_kwargs, image_dims = self.__get_image_params(image_path)

        params["api_key"] = self.__api_key

        params.update(**kwargs)
        url = f"{self.api_url}?{urllib.parse.urlencode(params)}"  # type: ignore[attr-defined]
        response = requests.post(url, **request_kwargs)
        response.raise_for_status()

        return PredictionGroup.create_prediction_group(
            response.json(),
            image_path=image_path,
            prediction_type=prediction_type,
            image_dims=image_dims,
            colors=self.colors,
        )

    def predict_video(
        self,
        video_path: str,
        fps: int = 5,
        additional_models: list = [],
        prediction_type: str = "batch-video",
    ) -> Tuple[str, str, Optional[str]]:
        """
        Infers detections based on image from specified model and image path.

        Args:
            video_path (str): path to the video you'd like to perform prediction on
            prediction_type (str): type of the model to run
            fps (int): frames per second to run inference

        Returns:
            A list of the signed url and job id

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> job_id,signed_url,signed_url_expires = model.predict_video("video.mp4"
                ,fps=5, inference_type="object-detection")
        """

        signed_url_expires = None

        url = urljoin(API_URL, "/video_upload_signed_url?api_key=" + self.__api_key)
        if fps > 120:
            raise Exception("FPS must be less than or equal to 120.")

        for model in additional_models:
            if model not in SUPPORTED_ADDITIONAL_MODELS:
                raise Exception(f"Model {model} is not supported for video inference.")

        if prediction_type not in SUPPORTED_ROBOFLOW_MODELS:
            raise Exception(f"{prediction_type} is not supported for video inference.")

        model_class = self.__class__.__name__

        if model_class == "ObjectDetectionModel":
            self.type = "object-detection"
        elif model_class == "ClassificationModel":
            self.type = "classification"
        elif model_class == "InstanceSegmentationModel":
            self.type = "instance-segmentation"
        elif model_class == "GazeModel":
            self.type = "gaze-detection"
        elif model_class == "CLIPModel":
            self.type = "clip-embed-image"
        elif model_class == "KeypointDetectionModel":
            self.type = "keypoint-detection"
        else:
            raise Exception("Model type not supported for video inference.")

        payload = json.dumps(
            {
                "file_name": os.path.basename(video_path),
            }
        )

        if not video_path.startswith(("http://", "https://")):
            headers = {"Content-Type": "application/json"}

            try:
                response = requests.request("POST", url, headers=headers, data=payload)
            except Exception as e:
                raise Exception(f"Error uploading video: {e}")

            if not response.ok:
                raise Exception(f"Error uploading video: {response.text}")

            signed_url = response.json()["signed_url"]

            signed_url_expires = signed_url.split("&X-Goog-Expires")[1].split("&")[0].strip("=")

            # make a POST request to the signed URL
            headers = {"Content-Type": "application/octet-stream"}

            try:
                with open(video_path, "rb") as f:
                    video_data = f.read()
            except Exception as e:
                raise Exception(f"Error reading video: {e}")

            try:
                result = requests.put(signed_url, data=video_data, headers=headers)
            except Exception as e:
                raise Exception(f"There was an error uploading the video: {e}")

            if not result.ok:
                raise Exception(f"There was an error uploading the video: {result.text}")
        else:
            signed_url = video_path

        url = urljoin(API_URL, "/videoinfer/?api_key=" + self.__api_key)
        if model_class in ("CLIPModel", "GazeModel"):
            if model_class == "CLIPModel":
                model = "clip"
            else:
                model = "gaze"

            models = [
                {
                    "model_id": SUPPORTED_ADDITIONAL_MODELS[model]["model_id"],
                    "model_version": SUPPORTED_ADDITIONAL_MODELS[model]["model_version"],
                    "inference_type": SUPPORTED_ADDITIONAL_MODELS[model]["inference_type"],
                }
            ]
        else:
            models = [
                {
                    "model_id": self.dataset_id,
                    "model_version": self.version,
                    "inference_type": self.type,
                }
            ]

        for model in additional_models:
            models.append(SUPPORTED_ADDITIONAL_MODELS[model])

        payload = json.dumps({"input_url": signed_url, "infer_fps": fps, "models": models})

        headers = {"Content-Type": "application/json"}

        try:
            response = requests.request("POST", url, headers=headers, data=payload)
        except Exception as e:
            raise Exception(f"Error starting video inference: {e}")

        if not response.ok:
            raise Exception(f"Error starting video inference: {response.text}")

        job_id = response.json()["job_id"]

        self.job_id = job_id

        return job_id, signed_url, signed_url_expires

    def poll_for_video_results(self, job_id: Optional[str] = None) -> dict:
        """
        Polls the Roboflow API to check if video inference is complete.

        Returns:
            Inference results as a dict

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("video.mp4")

            >>> results = model.poll_for_video_results()
        """

        if job_id is None:
            job_id = self.job_id

        url = urljoin(API_URL, "/videoinfer/?api_key=" + self.__api_key + "&job_id=" + job_id)
        try:
            response = requests.get(url, headers={"Content-Type": "application/json"})
        except Exception as e:
            raise Exception(f"Error getting video inference results: {e}")

        if not response.ok:
            raise Exception(f"Error getting video inference results: {response.text}")
        data = response.json()
        if "status" not in data:
            return {}  # No status available
        if data.get("status") > 1:
            return data  # Error
        elif data.get("status") == 1:
            return {}  # Still running
        else:  # done
            output_signed_url = data["output_signed_url"]
            inference_data = requests.get(output_signed_url, headers={"Content-Type": "application/json"})

            # frame_offset and model name are top-level keys
            return inference_data.json()

    def poll_until_video_results(self, job_id) -> dict:
        """
        Polls the Roboflow API to check if video inference is complete.

        When inference is complete, the results are returned.

        Returns:
            Inference results as a dict

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("video.mp4")

            >>> results = model.poll_until_results()
        """
        if job_id is None:
            job_id = self.job_id

        attempts = 0
        print(f"Checking for video inference results for job {job_id} every 60s")
        while True:
            time.sleep(60)
            print(f"({attempts * 60}s): Checking for inference results")
            response = self.poll_for_video_results(job_id)
            attempts += 1

            if response != {}:
                return response

    def download(self, format="pt", location="."):
        """
        Download the weights associated with a model.

        Args:
            format (str): The format of the output.
                        - 'pt': returns a PyTorch weights file
            location (str): The location to save the weights file to
        """
        supported_formats = ["pt"]
        if format not in supported_formats:
            raise Exception(f"Unsupported format {format}. Must be one of {supported_formats}")

        workspace, project, version = self.id.rsplit("/")

        # get pt url
        pt_api_url = f"{API_URL}/{workspace}/{project}/{self.version}/ptFile"

        r = requests.get(pt_api_url, params={"api_key": self.__api_key})

        r.raise_for_status()

        pt_weights_url = r.json()["weightsUrl"]

        response = requests.get(pt_weights_url, stream=True)

        # write the zip file to the desired location
        with open(location + "/weights.pt", "wb") as f:
            total_length = int(response.headers.get("content-length"))  # type: ignore[arg-type]
            for chunk in tqdm(
                response.iter_content(chunk_size=1024),
                desc=f"Downloading weights to {location}/weights.pt",
                total=int(total_length / 1024) + 1,
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()

        return
