import json
import os
import sys
import zipfile

import requests
import wget
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from roboflow.config import (
    API_URL,
    DEMO_KEYS,
    TYPE_CLASSICATION,
    TYPE_INSTANCE_SEGMENTATION,
    TYPE_OBJECT_DETECTION,
    TYPE_SEMANTIC_SEGMENTATION,
)
from roboflow.core.dataset import Dataset
from roboflow.models.classification import ClassificationModel
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.models.semantic_segmentation import SemanticSegmentationModel

load_dotenv()


class Version:
    def __init__(self, version_dict, type, api_key, name, version, model_format, local):
        if api_key in DEMO_KEYS:
            if api_key == "coco-128-sample":
                self.__api_key = api_key
                self.model_format = model_format
                self.name = "coco-128"
                self.version = "1"
            else:
                self.__api_key = api_key
                self.model_format = model_format
                self.name = "chess-pieces-new"
                self.version = "23"
                self.id = "joseph-nelson/chess-pieces-new"
        else:
            self.__api_key = api_key
            self.name = name

            # FIXME: the version argument is inconsistently passed into this object.
            # Sometimes it is passed as: test-workspace/test-project/2
            # Other times, it is passed as: 2
            self.version = version
            self.type = type
            self.augmentation = version_dict["augmentation"]
            self.created = version_dict["created"]
            self.id = version_dict["id"]
            self.images = version_dict["images"]
            self.preprocessing = version_dict["preprocessing"]
            self.splits = version_dict["splits"]
            self.model_format = model_format

            version_without_workspace = os.path.basename(str(version))

            if self.type == TYPE_OBJECT_DETECTION:
                self.model = ObjectDetectionModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    version_without_workspace,
                    local=local,
                )
            elif self.type == TYPE_CLASSICATION:
                self.model = ClassificationModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    version_without_workspace,
                    local=local,
                )
            elif self.type == TYPE_INSTANCE_SEGMENTATION:
                self.model = InstanceSegmentationModel(
                    self.__api_key,
                    self.id,
                )
            elif self.type == TYPE_SEMANTIC_SEGMENTATION:
                self.model = SemanticSegmentationModel(
                    self.__api_key,
                    self.id,
                )
            else:
                self.model = None

    def download(self, model_format=None, location=None):
        """
        Download and extract a ZIP of a version's dataset in a given format

        Args:
            model_format: A format to use for downloading
            location: An optional path for saving the file

        Returns:
            Dataset

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """
        if location is None:
            location = self.__get_download_location()

        model_format = self.__get_format_identifier(model_format)

        if self.__api_key == "coco-128-sample":
            link = "https://app.roboflow.com/ds/n9QwXwUK42?key=NnVCe2yMxP"
        else:
            url = self.__get_download_url(model_format)
            response = requests.get(url, params={"api_key": self.__api_key})
            if response.status_code == 200:
                link = response.json()["export"]["link"]
            else:
                try:
                    raise RuntimeError(response.json())
                except requests.exceptions.JSONDecodeError:
                    response.raise_for_status()

        self.__download_zip(link, location, model_format)
        self.__extract_zip(location, model_format)
        self.__reformat_yaml(location, model_format)

        return Dataset(self.name, self.version, model_format, os.path.abspath(location))

    def export(self, model_format: str) -> bool:
        """
        Ask the Roboflow API to generate a version's dataset in a given format so that it can be downloaded via the `download()` method.
        The export will be asynchronously generated and available for download after some amount of seconds - depending on dataset size.

        Args:
            model_format: A format to use for downloading

        Returns:
            True

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """
        url = self.__get_download_url(model_format)
        response = requests.post(url, params={"api_key": self.__api_key})
        if not response.ok:
            try:
                raise RuntimeError(response.json())
            except requests.exceptions.JSONDecodeError:
                response.raise_for_status()

        return True

    def train(self, speed: str = None, checkpoint: str = None) -> bool:
        """
        Ask the Roboflow API to train a previously exported version's dataset.

        Args:
            speed: Whether to train quickly or accurately. Note: accurate training is a paid feature. Default speed is `fast`.
            checkpoint: A string representing the checkpoint to use while training

        Returns:
            True

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """
        workspace, project, *_ = self.id.rsplit("/")
        url = f"{API_URL}/{workspace}/{project}/{self.version}/train"

        data = {}
        if speed:
            data["speed"] = speed

        if checkpoint:
            data["checkpoint"] = checkpoint

        response = requests.post(url, json=data, params={"api_key": self.__api_key})
        if not response.ok:
            try:
                raise RuntimeError(response.json())
            except requests.exceptions.JSONDecodeError:
                response.raise_for_status()

        return True

    def __download_zip(self, link, location, format):
        """
        Download a dataset's zip file from the given URL and save it in the desired location

        Args:
            link: the URL of the remote zip file
            location: filepath of the data directory to save the zip file to
            format: the format identifier string

        Returns:
            None
        """
        if not os.path.exists(location):
            os.makedirs(location)

        def bar_progress(current, total, width=80):
            progress_message = (
                "Downloading Dataset Version Zip in "
                + location
                + " to "
                + format
                + ": %d%% [%d / %d] bytes" % (current / total * 100, current, total)
            )
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        wget.download(link, out=location + "/roboflow.zip", bar=bar_progress)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __extract_zip(self, location, format):
        """
        This simply extracts the contents of a downloaded zip file and then deletes the zip

        Args:
            location: filepath of the data directory that contains the zip file
            format: the format identifier string

        Returns:
            None

        Raises:
            RuntimeError: If there's an issue extracting the zip file
        """
        with zipfile.ZipFile(location + "/roboflow.zip", "r") as zip_ref:
            for member in tqdm(
                zip_ref.infolist(),
                desc=f"Extracting Dataset Version Zip to {location} in {format}:",
            ):
                try:
                    zip_ref.extract(member, location)
                except zipfile.error:
                    raise RuntimeError("Error unzipping download")

        os.remove(location + "/roboflow.zip")

    def __get_download_location(self):
        """
        Get the local path to save a downloaded dataset to

        Returns:
            local path string
        """
        version_slug = self.name.replace(" ", "-")
        filename = f"{version_slug}-{self.version}"

        directory = os.environ.get("DATASET_DIRECTORY")
        if directory:
            return f"{directory}/{filename}"

        return filename

    def __get_download_url(self, format):
        """
        Get the Roboflow API URL for downloading (and exporting downloadable zips)

        Args:
            format: the format identifier string

        Returns:
            Roboflow API URL string
        """
        workspace, project, *_ = self.id.rsplit("/")
        return f"{API_URL}/{workspace}/{project}/{self.version}/{format}"

    def __get_format_identifier(self, format):
        """
        If `format` is none, fall back to the instance's `model_format` value.
        If a human readable format name was passed, return the identifier that should be used for Roboflow API calls
        Otherwise, assume that the passed in format is also the identifier

        Args:
            format: a human readable format string

        Returns:
            format identifier string

        Raises:
            RuntimeError: If a format was not provided
        """
        if not format:
            format = self.model_format

        if not format:
            raise RuntimeError(
                "You must pass a format argument to version.download() or define a model in your Roboflow object"
            )

        friendly_formats = {
            "yolov5": "yolov5pytorch",
            "yolov7": "yolov7pytorch",
        }
        return friendly_formats.get(format, format)

    def __reformat_yaml(self, location, format):
        """
        Certain formats seem to require reformatting the downloaded YAML.
        It'd be nice if the API did this, but we're doing it in python for now.

        Args:
            location: filepath of the data directory that contains the yaml file
            format: the format identifier string

        Returns:
            None
        """
        if format in ["yolov5pytorch", "yolov7pytorch"]:
            with open(location + "/data.yaml") as file:
                new_yaml = yaml.safe_load(file)
            new_yaml["train"] = location + new_yaml["train"].lstrip("..")
            new_yaml["val"] = location + new_yaml["val"].lstrip("..")

            os.remove(location + "/data.yaml")

            with open(location + "/data.yaml", "w") as outfile:
                yaml.dump(new_yaml, outfile)

        if format == "mt-yolov6":
            with open(location + "/data.yaml") as file:
                new_yaml = yaml.safe_load(file)
            new_yaml["train"] = location + new_yaml["train"].lstrip(".")
            new_yaml["val"] = location + new_yaml["val"].lstrip(".")
            new_yaml["test"] = location + new_yaml["test"].lstrip(".")

            os.remove(location + "/data.yaml")

            with open(location + "/data.yaml", "w") as outfile:
                yaml.dump(new_yaml, outfile)

    def __str__(self):
        """string representation of version object."""
        json_value = {
            "name": self.name,
            "type": self.type,
            "version": self.version,
            "augmentation": self.augmentation,
            "created": self.created,
            "preprocessing": self.preprocessing,
            "splits": self.splits,
        }
        return json.dumps(json_value, indent=2)
