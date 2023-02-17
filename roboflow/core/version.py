import json
import os
import sys
import time
import zipfile
from importlib import import_module

import requests
import wget
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from roboflow.config import (
    API_URL,
    APP_URL,
    DEMO_KEYS,
    TYPE_CLASSICATION,
    TYPE_INSTANCE_SEGMENTATION,
    TYPE_OBJECT_DETECTION,
    TYPE_SEMANTIC_SEGMENTATION,
    UNIVERSE_URL,
)
from roboflow.core.dataset import Dataset
from roboflow.models.classification import ClassificationModel
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.models.semantic_segmentation import SemanticSegmentationModel
from roboflow.util.annotations import amend_data_yaml
from roboflow.util.versions import (
    get_wrong_dependencies_versions,
    print_warn_for_wrong_dependencies_versions,
)

load_dotenv()


class Version:
    def __init__(
        self,
        version_dict,
        type,
        api_key,
        name,
        version,
        model_format,
        local,
        workspace,
        project,
        public,
    ):
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
            self.workspace = workspace
            self.project = project
            self.public = public
            if "exports" in version_dict.keys():
                self.exports = version_dict["exports"]
            else:
                self.exports = []

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
                self.model = InstanceSegmentationModel(self.__api_key, self.id)
            elif self.type == TYPE_SEMANTIC_SEGMENTATION:
                self.model = SemanticSegmentationModel(self.__api_key, self.id)
            else:
                self.model = None

    def __check_if_generating(self):
        # check Roboflow API to see if this version is still generating

        url = f"{API_URL}/{self.workspace}/{self.project}/{self.version}?nocache=true"
        response = requests.get(url, params={"api_key": self.__api_key})
        response.raise_for_status()

        if response.json()["version"]["progress"] == None:
            progress = 0.0
        else:
            progress = float(response.json()["version"]["progress"])

        return response.json()["version"]["generating"], progress

    def __wait_if_generating(self, recurse=False):
        # checks if a given version is still in the progress of generating

        still_generating, progress = self.__check_if_generating()

        if still_generating:
            progress_message = (
                "Generating version still in progress. Progress: "
                + str(round(progress * 100, 2))
                + "%"
            )
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()
            time.sleep(5)
            return self.__wait_if_generating(recurse=True)

        else:
            if recurse:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

    def download(self, model_format=None, location=None, overwrite: bool = True):
        """
        Download and extract a ZIP of a version's dataset in a given format

        :param model_format: A format to use for downloading
        :param location: An optional path for saving the file
        :param overwrite: An optional flag to prevent dataset overwrite when dataset is already downloaded

        :return: Dataset
        """

        self.__wait_if_generating()

        if model_format == "yolov8":
            # if ultralytics is installed, we will assume users will want to use yolov8 and we check for the supported version
            try:
                import_module("ultralytics")
                print_warn_for_wrong_dependencies_versions(
                    [("ultralytics", "<=", "8.0.20")]
                )
            except ImportError as e:
                print(
                    "[WARNING] we noticed you are downloading a `yolov8` datasets but you don't have `ultralytics` installed. Roboflow `.deploy` supports only models trained with `ultralytics<=8.0.20`, to intall it `pip install ultralytics<=8.0.20`."
                )
                # silently fail
                pass

        model_format = self.__get_format_identifier(model_format)

        if model_format not in self.exports:
            self.export(model_format)

        # if model_format is not in

        if location is None:
            location = self.__get_download_location()
        if os.path.exists(location) and not overwrite:
            return Dataset(
                self.name, self.version, model_format, os.path.abspath(location)
            )

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

    def export(self, model_format=None):
        """
        Ask the Roboflow API to generate a version's dataset in a given format so that it can be downloaded via the `download()` method.
        The export will be asynchronously generated and available for download after some amount of seconds - depending on dataset size.

        :param model_format: A format to use for downloading

        :return: True
        :raises RuntimeError / HTTPError:
        """

        model_format = self.__get_format_identifier(model_format)

        self.__wait_if_generating()

        url = self.__get_download_url(model_format)
        response = requests.get(url, params={"api_key": self.__api_key})
        if not response.ok:
            try:
                raise RuntimeError(response.json())
            except requests.exceptions.JSONDecodeError:
                response.raise_for_status()

        # the rest api returns 202 if the export is still in progress
        if response.status_code == 202:
            status_code_check = 202
            while status_code_check == 202:
                time.sleep(1)
                response = requests.get(url, params={"api_key": self.__api_key})
                status_code_check = response.status_code
                if status_code_check == 202:
                    progress = response.json()["progress"]
                    progress_message = (
                        "Exporting format "
                        + model_format
                        + " in progress : "
                        + str(round(progress * 100, 2))
                        + "%"
                    )
                    sys.stdout.write("\r" + progress_message)
                    sys.stdout.flush()

        if response.status_code == 200:
            sys.stdout.write("\n")
            print("\r" + "Version export complete for " + model_format + " format")
            sys.stdout.flush()
            return True
        else:
            try:
                raise RuntimeError(response.json())
            except requests.exceptions.JSONDecodeError:
                response.raise_for_status()

    def train(self, speed=None, checkpoint=None) -> bool:
        """
        Ask the Roboflow API to train a previously exported version's dataset.
        Args:
            speed: Whether to train quickly or accurately. Note: accurate training is a paid feature. Default speed is `fast`.
            checkpoint: A string representing the checkpoint to use while training
        Returns:
            True
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """

        self.__wait_if_generating()

        train_model_format = "yolov5pytorch"

        if train_model_format not in self.exports:
            self.export(train_model_format)

        workspace, project, *_ = self.id.rsplit("/")
        url = f"{API_URL}/{workspace}/{project}/{self.version}/train"

        data = {}
        if speed:
            data["speed"] = speed

        if checkpoint:
            data["checkpoint"] = checkpoint

        sys.stdout.write("\r" + "Reaching out to Roboflow to start training...")
        sys.stdout.write("\n")
        sys.stdout.flush()

        response = requests.post(url, json=data, params={"api_key": self.__api_key})
        if not response.ok:
            try:
                raise RuntimeError(response.json())
            except requests.exceptions.JSONDecodeError:
                response.raise_for_status()

        sys.stdout.write("\r" + "Training model in progress...")
        sys.stdout.flush()

        return True

    # @warn_for_wrong_dependencies_versions([("ultralytics", "<=", "8.0.20")])
    def deploy(self, model_type: str, model_path: str) -> None:
        """Uploads provided weights file to Roboflow

        Args:
            model_path (str): File path to model weights to be uploaded
        """

        supported_models = ["yolov8", "yolov5", "yolov7-seg"]

        if model_type not in supported_models:
            raise (
                ValueError(
                    f"Model type {model_type} not supported. Supported models are {supported_models}"
                )
            )

        if model_type == "yolov8":
            try:
                import torch
                import ultralytics

            except ImportError as e:
                raise (
                    "The ultralytics python package is required to deploy yolov8 models. Please install it with `pip install ultralytics`"
                )

            print_warn_for_wrong_dependencies_versions(
                [("ultralytics", "<=", "8.0.20")]
            )

        elif model_type in ["yolov5", "yolov7-seg"]:
            try:
                import torch
            except ImportError as e:
                raise (
                    "The torch python package is required to deploy yolov5 models. Please install it with `pip install torch`"
                )

        model = torch.load(os.path.join(model_path, "weights/best.pt"))

        class_names = []
        for i, val in enumerate(model["model"].names):
            class_names.append((val, model["model"].names[val]))
        class_names.sort(key=lambda x: x[0])
        class_names = [x[1] for x in class_names]

        if model_type == "yolov8":
            # try except for backwards compatibility with older versions of ultralytics
            try:
                model_artifacts = {
                    "names": class_names,
                    "yaml": model["model"].yaml,
                    "nc": model["model"].nc,
                    "args": {
                        k: val
                        for k, val in model["model"].args.items()
                        if ((k == "model") or (k == "imgsz") or (k == "batch"))
                    },
                    "ultralytics_version": ultralytics.__version__,
                    "model_type": model_type,
                }
            except:
                model_artifacts = {
                    "names": class_names,
                    "yaml": model["model"].yaml,
                    "nc": model["model"].nc,
                    "args": {
                        k: val
                        for k, val in model["model"].args.__dict__.items()
                        if ((k == "model") or (k == "imgsz") or (k == "batch"))
                    },
                    "ultralytics_version": ultralytics.__version__,
                    "model_type": model_type,
                }
        elif model_type in ["yolov5", "yolov7-seg"]:
            # parse from yaml for yolov5

            with open(os.path.join(model_path, "opt.yaml"), "r") as stream:
                opts = yaml.safe_load(stream)

            model_artifacts = {
                "names": class_names,
                "yaml": model["model"].yaml,
                "nc": model["model"].nc,
                "args": {"imgsz": opts["imgsz"], "batch": opts["batch_size"]},
                "model_type": model_type,
            }

        with open(model_path + "model_artifacts.json", "w") as fp:
            json.dump(model_artifacts, fp)

        torch.save(model["model"].state_dict(), model_path + "state_dict.pt")

        lista_files = [
            "results.csv",
            "results.png",
            "model_artifacts.json",
            "state_dict.pt",
        ]

        with zipfile.ZipFile(model_path + "roboflow_deploy.zip", "w") as zipMe:
            for file in lista_files:
                if os.path.exists(model_path + file):
                    zipMe.write(
                        model_path + file,
                        arcname=file,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                else:
                    if file in ["model_artifacts.json", "state_dict.pt"]:
                        raise (
                            ValueError(
                                f"File {file} not found. Please make sure to provide a valid model path."
                            )
                        )

        res = requests.get(
            f"{API_URL}/{self.workspace}/{self.project}/{self.version}/uploadModel?api_key={self.__api_key}"
        )
        try:
            if res.status_code == 429:
                raise RuntimeError(
                    f"This version already has a trained model. Please generate and train a new version in order to upload model to Roboflow."
                )
            else:
                res.raise_for_status()
        except Exception as e:
            print(f"An error occured when getting the model upload URL: {e}")
            return

        res = requests.put(
            res.json()["url"],
            data=open(os.path.join(model_path + "roboflow_deploy.zip"), "rb"),
        )
        try:
            res.raise_for_status()

            if self.public:
                print(
                    f"View the status of your deployment at: {APP_URL}/{self.workspace}/{self.project}/deploy/{self.version}"
                )
                print(
                    f"Share your model with the world at: {UNIVERSE_URL}/{self.workspace}/{self.project}/model/{self.version}"
                )
            else:
                print(
                    f"View the status of your deployment at: {APP_URL}/{self.workspace}/{self.project}/deploy/{self.version}"
                )

        except Exception as e:
            print(f"An error occured when uploading the model: {e}")

    def __download_zip(self, link, location, format):
        """
        Download a dataset's zip file from the given URL and save it in the desired location

        :param location: link the URL of the remote zip file
        :param location: filepath of the data directory to save the zip file to
        :param format: the format identifier string

        :return None:
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

        try:
            wget.download(link, out=location + "/roboflow.zip", bar=bar_progress)
        except Exception as e:
            print(f"Error when trying to download dataset @ {link}")
            raise e
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __extract_zip(self, location, format):
        """
        This simply extracts the contents of a downloaded zip file and then deletes the zip

        :param location: filepath of the data directory that contains the zip file
        :param format: the format identifier string

        :return None:
        :raises RuntimeError:
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

        :return local path string:
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

        :param format: the format identifier string

        :return Roboflow API URL string:
        """
        workspace, project, *_ = self.id.rsplit("/")
        return f"{API_URL}/{workspace}/{project}/{self.version}/{format}"

    def __get_format_identifier(self, format):
        """
        If `format` is none, fall back to the instance's `model_format` value.
        If a human readable format name was passed, return the identifier that should be used for Roboflow API calls
        Otherwise, assume that the passed in format is also the identifier

        :param format: a human readable format string

        :return: format identifier string
        """
        if not format:
            format = self.model_format

        if not format:
            raise RuntimeError(
                "You must pass a format argument to version.download() or define a model in your Roboflow object"
            )

        friendly_formats = {"yolov5": "yolov5pytorch", "yolov7": "yolov7pytorch"}
        return friendly_formats.get(format, format)

    def __reformat_yaml(self, location: str, format: str):
        """
        Certain formats seem to require reformatting the downloaded YAML.
        It'd be nice if the API did this, but we're doing it in python for now.

        :param location: filepath of the data directory that contains the yaml file
        :param format: the format identifier string

        :return None:
        """
        data_path = os.path.join(location, "data.yaml")

        def callback(content: dict) -> dict:
            if format == "mt-yolov6":
                content["train"] = location + content["train"].lstrip(".")
                content["val"] = location + content["val"].lstrip(".")
                content["test"] = location + content["test"].lstrip(".")
            if format in ["yolov5pytorch", "yolov7pytorch", "yolov8"]:
                content["train"] = location + content["train"].lstrip("..")
                content["val"] = location + content["val"].lstrip("..")
            try:
                # get_wrong_dependencies_versions raises exception if ultralytics is not installed at all
                if not get_wrong_dependencies_versions(
                    dependencies_versions=[("ultralytics", ">=", "8.0.30")]
                ):
                    content["train"] = "train/images"
                    content["val"] = "valid/images"
                    content["test"] = "test/images"
            except ModuleNotFoundError:
                pass
            return content

        amend_data_yaml(path=data_path, callback=callback)

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
            "workspace": self.workspace,
        }
        return json.dumps(json_value, indent=2)
