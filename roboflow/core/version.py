import json
import os
import sys
import zipfile

import requests
import wget
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from roboflow.config import API_URL, DEMO_KEYS
from roboflow.core.dataset import Dataset
from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel

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
            self.version = version
            self.type = type
            self.augmentation = version_dict["augmentation"]
            self.created = version_dict["created"]
            self.id = version_dict["id"]
            self.images = version_dict["images"]
            self.preprocessing = version_dict["preprocessing"]
            self.splits = version_dict["splits"]
            self.model_format = model_format

            version_without_workspace = os.path.basename(version)

            if self.type == "object-detection":
                self.model = ObjectDetectionModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    version_without_workspace,
                    local=local,
                )
            elif self.type == "classification":
                self.model = ClassificationModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    version_without_workspace,
                    local=local,
                )
            else:
                self.model = None

    def download(self, model_format=None, location=None):

        if location is None:
            if "DATASET_DIRECTORY" in os.environ:
                location = (
                    os.environ["DATASET_DIRECTORY"]
                    + "/"
                    + self.name.replace(" ", "-")
                    + "-"
                    + self.version
                )
            else:
                location = self.name.replace(" ", "-") + "-" + self.version

        if not os.path.exists(location):
            os.makedirs(location)

        if model_format is None:
            if self.model_format == "yolov5":
                model_format = "yolov5pytorch"
            elif self.model_format == "yolov7":
                model_format = "yolov7pytorch"
            else:
                RuntimeError(
                    "You must pass a download_type to version.download() or define model in your Roboflow object"
                )

        if model_format == "yolov5":
            model_format = "yolov5pytorch"

        if model_format == "yolov7":
            model_format = "yolov7pytorch"

        if self.__api_key == "coco-128-sample":
            link = "https://app.roboflow.com/ds/n9QwXwUK42?key=NnVCe2yMxP"
        else:
            url = self.__get_download_url(model_format)
            resp = requests.get(url)
            if resp.status_code == 200:
                link = resp.json()["export"]["link"]
            else:
                raise RuntimeError(resp.json())

        def bar_progress(current, total, width=80):
            progress_message = (
                "Downloading Dataset Version Zip in "
                + location
                + " to "
                + model_format
                + ": %d%% [%d / %d] bytes" % (current / total * 100, current, total)
            )
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()

        wget.download(link, out=location + "/roboflow.zip", bar=bar_progress)
        sys.stdout.write("\n")
        sys.stdout.flush()

        with zipfile.ZipFile(location + "/roboflow.zip", "r") as zip_ref:
            for member in tqdm(
                zip_ref.infolist(),
                desc="Extracting Dataset Version Zip to "
                + location
                + " in "
                + model_format
                + ":",
            ):
                try:
                    zip_ref.extract(member, location)
                except zipfile.error:
                    # [TODO] sure we want to pass here?
                    pass

        if (
            (self.model_format == "yolov5")
            or (model_format == "yolov5pytorch")
            or (model_format == "yolov7")
            or (model_format == "yolov7pytorch")
        ):
            with open(location + "/data.yaml") as file:
                new_yaml = yaml.safe_load(file)
            new_yaml["train"] = location + new_yaml["train"].lstrip("..")
            new_yaml["val"] = location + new_yaml["val"].lstrip("..")

            os.remove(location + "/data.yaml")

            with open(location + "/data.yaml", "w") as outfile:
                yaml.dump(new_yaml, outfile)

        if model_format == "mt-yolov6":
            with open(location + "/data.yaml") as file:
                new_yaml = yaml.safe_load(file)
            new_yaml["train"] = location + new_yaml["train"].lstrip(".")
            new_yaml["val"] = location + new_yaml["val"].lstrip(".")
            new_yaml["test"] = location + new_yaml["test"].lstrip(".")

            os.remove(location + "/data.yaml")

            with open(location + "/data.yaml", "w") as outfile:
                yaml.dump(new_yaml, outfile)

        os.remove(location + "/roboflow.zip")

        return Dataset(
            self.name, self.version, self.model_format, os.path.abspath(location)
        )

    def __get_download_url(self, download_type):
        temporary = self.id.rsplit("/")
        workspace, project = temporary[0], temporary[1]
        url = "".join(
            [
                API_URL + "/" + workspace + "/" + project,
                "/" + self.version,
                "/" + download_type,
                "?api_key=" + self.__api_key,
            ]
        )
        return url

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
