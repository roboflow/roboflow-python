import json
import time
from typing import Optional, Tuple
from urllib.parse import urljoin

import filetype
import requests

from roboflow.config import API_URL
from roboflow.models.inference import InferenceModel

SUPPORTED_ROBOFLOW_MODELS = ["object-detection", "classification", "instance-segmentation", "keypoint-detection"]

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

ACCEPTED_VIDEO_FORMATS = {
    "video/mp4",
    "video/x-msvideo",  # AVI
    "video/webm",
}


def is_valid_mime(filename):
    kind = filetype.guess(filename)

    if kind is None:
        return False

    return kind.mime in ACCEPTED_VIDEO_FORMATS


def is_valid_video(filename):
    # check file type
    if not is_valid_mime(filename):
        return False

    return True


class VideoInferenceModel(InferenceModel):
    """
    Run inference on an object detection model hosted on Roboflow or served through Roboflow Inference.
    """  # noqa: E501 // docs

    def __init__(
        self,
        api_key,
    ):
        """
        Create a VideoDetectionModel object through which you can run inference on videos.

        Args:
            api_key (str): Your API key (obtained via your workspace API settings page).
        """  # noqa: E501 // docs
        self.__api_key = api_key

    def predict(  # type: ignore[override]
        self,
        video_path: str,
        inference_type: str,
        fps: int = 5,
        additional_models: Optional[list] = None,
    ) -> Tuple[str, str]:
        """
        Infers detections based on image from specified model and image path.

        Args:
            video_path (str): path to the video you'd like to perform prediction on
            inference_type (str): type of the model to run
            fps (int): frames per second to run inference

        Returns:
            A list of the signed url and job id

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> model = project.version("1").model

            >>> prediction = model.predict("video.mp4", fps=5, inference_type="object-detection")
        """  # noqa: E501 // docs

        url = urljoin(API_URL, f"/video_upload_signed_url/?api_key={self.__api_key}")

        if fps > 120:
            raise Exception("FPS must be less than or equal to 120.")

        if additional_models is None:
            additional_models = []

        for model in additional_models:
            if model not in SUPPORTED_ADDITIONAL_MODELS:
                raise Exception(f"Model {model} is not supported for video inference.")

        if inference_type not in SUPPORTED_ROBOFLOW_MODELS:
            raise Exception(f"Model {inference_type} is not supported for video inference.")

        if not is_valid_video(video_path):
            raise Exception("Video path is not valid")

        payload = json.dumps(
            {
                "file_name": video_path,
            }
        )

        headers = {"Content-Type": "application/json"}

        response = requests.request("POST", url, headers=headers, data=payload)

        signed_url = response.json()["signed_url"]

        print("Uploaded video to signed url: " + signed_url)

        url = urljoin(API_URL, f"/videoinfer/?api_key={self.__api_key}")

        models = [
            {
                "model_id": self.dataset_id,
                "model_version": self.version,
                "inference_type": inference_type,
            }
        ]

        for model in additional_models:
            models.append(SUPPORTED_ADDITIONAL_MODELS[model])

        payload = json.dumps({"input_url": signed_url, "infer_fps": fps, "models": models})

        response = requests.request("POST", url, headers=headers, data=payload)

        job_id = response.json()["job_id"]

        self.job_id = job_id

        return job_id, signed_url

    def poll_for_results(self, job_id: Optional[str] = None) -> dict:
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

            >>> results = model.poll_for_results()
        """  # noqa: E501 // docs

        if job_id is None:
            job_id = self.job_id

        url = urljoin(API_URL, f"/videoinfer/?api_key={self.__api_key}&job_id={self.job_id}")

        try:
            response = requests.get(url, headers={"Content-Type": "application/json"})
        except Exception as e:
            print(e)
            raise Exception("Error polling for results.")

        if not response.ok:
            raise Exception("Error polling for results.")

        data = response.json()

        if data["success"] == 0:
            output_signed_url = data["output_signed_url"]

            inference_data = requests.get(output_signed_url, headers={"Content-Type": "application/json"})

            # frame_offset and model name are top-level keys
            return inference_data.json()
        elif data["success"] == 1:
            print("Job not complete yet. Check back in a minute.")
            return {}
        else:
            raise Exception("Job failed.")

    def poll_until_results(self, job_id) -> dict:
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
        """  # noqa: E501 // docs
        if job_id is None:
            job_id = self.job_id

        attempts = 0

        while True:
            response = self.poll_for_results()

            attempts += 1

            if response != {}:
                return response

            print(f"({attempts * 60}s): Checking for inference results")

            time.sleep(60)
