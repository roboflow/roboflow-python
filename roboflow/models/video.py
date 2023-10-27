import json
from urllib.parse import urljoin
import magic

import requests
import time
from typing import List

from roboflow.config import API_URL
from roboflow.models.inference import InferenceModel

VALID_VIDEO_EXTENSIONS = [".mp4"]


def is_mp4(filename):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(filename)
    return file_type == "video/mp4"


def is_valid_video(filename):
    return is_mp4(filename)


class VideoInferenceModel(InferenceModel):
    """
    Run inference on an object detection model hosted on Roboflow or served through Roboflow Inference.
    """

    def __init__(
        self,
        api_key,
    ):
        """
        Create a VideoDetectionModel object through which you can run inference on videos.

        Args:
            api_key (str): Your API key (obtained via your workspace API settings page).
        """
        self.__api_key = api_key

    def predict(
        self,
        video_path,
    ) -> List[str, str]:
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

            >>> prediction = model.predict("video.mp4")
        """

        url = urljoin(API_URL, "/video_upload_signed_url/?api_key=", self.__api_key)

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

        url = urljoin(API_URL, "/videoinfer/?api_key=", self.__api_key)

        payload = json.dumps(
            {
                "input_url": signed_url,
                "infer_fps": 5,
                "models": [
                    {
                        "model_id": self.dataset_id,
                        "model_version": self.version,
                        "inference_type": "object-detection",
                    }
                ],
            }
        )

        response = requests.request("POST", url, headers=headers, data=payload)

        job_id = response.json()["job_id"]

        self.job_id = job_id

        return job_id, signed_url

    def poll_for_results(self, job_id: str = None) -> dict:
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
        """

        if job_id is None:
            job_id = self.job_id

        url = urljoin(
            API_URL, "/videoinfer/?api_key=", self.__api_key, "&job_id=", self.job_id
        )

        response = requests.get(url, headers={"Content-Type": "application/json"})

        data = response.json()

        if data["success"] != 0 or data["status_info"] != "success":
            print("Job not complete yet. Check back in a minute.")
            return {}

        output_signed_url = data["output_signed_url"]

        inference_data = requests.get(
            output_signed_url, headers={"Content-Type": "application/json"}
        )

        # frame_offset and model name are top-level keys
        return inference_data.json()

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
        """
        if job_id is None:
            job_id = self.job_id

        attempts = 0

        while True:
            response = self.poll_for_response()

            time.sleep(60)

            print(f"({attempts * 60}s): Checking for inference results")

            attempts += 1

            if response != {}:
                return response
