from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
import zipfile
from typing import TYPE_CHECKING, Optional, Union

import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from roboflow.config import (
    API_URL,
    APP_URL,
    DEMO_KEYS,
    TQDM_DISABLE,
    TYPE_CLASSICATION,
    TYPE_INSTANCE_SEGMENTATION,
    TYPE_KEYPOINT_DETECTION,
    TYPE_OBJECT_DETECTION,
    TYPE_SEMANTIC_SEGMENTATION,
    UNIVERSE_URL,
)
from roboflow.core.dataset import Dataset
from roboflow.models.classification import ClassificationModel
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.models.keypoint_detection import KeypointDetectionModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.models.semantic_segmentation import SemanticSegmentationModel
from roboflow.util.annotations import amend_data_yaml
from roboflow.util.general import write_line
from roboflow.util.versions import get_wrong_dependencies_versions, print_warn_for_wrong_dependencies_versions

if TYPE_CHECKING:
    import numpy as np

    from roboflow.models.inference import InferenceModel

load_dotenv()


class Version:
    """
    Class representing a Roboflow dataset version.
    """

    model: Optional[InferenceModel]

    def __init__(
        self,
        version_dict,
        type,
        api_key,
        name,
        version,
        model_format,
        local: Optional[str],
        workspace,
        project,
        public,
        colors=None,
    ):
        """
        Initialize a Version object.
        """
        if api_key:
            self.__api_key = api_key
            self.name = name
            self.version = unwrap_version_id(version_id=version)
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
            self.colors = {} if colors is None else colors

            self.colors = colors
            if "exports" in version_dict.keys():
                self.exports = version_dict["exports"]
            else:
                self.exports = []

            version_without_workspace = os.path.basename(str(version))

            response = requests.get(f"{API_URL}/{workspace}/{project}/{self.version}?api_key={self.__api_key}")
            if response.ok:
                version_info = response.json()["version"]
                has_model = bool(version_info.get("train", {}).get("model"))
            else:
                has_model = False

            if not has_model:
                self.model = None
            elif self.type == TYPE_OBJECT_DETECTION:
                self.model = ObjectDetectionModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    version_without_workspace,
                    local=local,
                    colors=self.colors,
                    preprocessing=self.preprocessing,
                )
            elif self.type == TYPE_CLASSICATION:
                self.model = ClassificationModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    version_without_workspace,
                    local=local,
                    colors=self.colors,
                    preprocessing=self.preprocessing,
                )
            elif self.type == TYPE_INSTANCE_SEGMENTATION:
                self.model = InstanceSegmentationModel(
                    self.__api_key,
                    self.id,
                    colors=self.colors,
                    preprocessing=self.preprocessing,
                    local=local,
                )
            elif self.type == TYPE_SEMANTIC_SEGMENTATION:
                self.model = SemanticSegmentationModel(self.__api_key, self.id)
            elif self.type == TYPE_KEYPOINT_DETECTION:
                self.model = KeypointDetectionModel(self.__api_key, self.id, version=version_without_workspace)
            else:
                self.model = None

        elif DEMO_KEYS:
            api_key = DEMO_KEYS[0]
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

    def __check_if_generating(self):
        # check Roboflow API to see if this version is still generating

        url = f"{API_URL}/{self.workspace}/{self.project}/{self.version}?nocache=true"
        response = requests.get(url, params={"api_key": self.__api_key})
        response.raise_for_status()
        if response.json()["version"]["progress"] is None:
            progress = 0.0
        else:
            progress = float(response.json()["version"]["progress"])

        return response.json()["version"]["generating"], progress

    def __wait_if_generating(self, recurse=False):
        # checks if a given version is still in the progress of generating

        still_generating, progress = self.__check_if_generating()

        if still_generating:
            progress_message = "Generating version still in progress. Progress: " + str(round(progress * 100, 2)) + "%"
            sys.stdout.write("\r" + progress_message)
            sys.stdout.flush()
            time.sleep(5)
            return self.__wait_if_generating(recurse=True)

        else:
            if recurse:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

    def download(self, model_format=None, location=None, overwrite: bool = False):
        """
        Download and extract a ZIP of a version's dataset in a given format

        :param model_format: A format to use for downloading
        :param location: An optional path for saving the file
        :param overwrite: An optional flag to prevent dataset overwrite when dataset is already downloaded

        Args:
            model_format (str): A format to use for downloading
            location (str): An optional path for saving the file
            overwrite (bool): An optional flag to overwrite an existing dataset if the dataset has already downloaded

        Returns:
            Dataset Object

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """  # noqa: E501 // docs

        self.__wait_if_generating()

        model_format = self.__get_format_identifier(model_format)

        if model_format not in self.exports:
            self.export(model_format)

        # if model_format is not in

        if location is None:
            location = self.__get_download_location()
        if os.path.exists(location) and not overwrite:
            return Dataset(self.name, self.version, model_format, os.path.abspath(location))

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
                except json.JSONDecodeError:
                    response.raise_for_status()

        self.__download_zip(link, location, model_format)
        self.__extract_zip(location, model_format)
        self.__reformat_yaml(location, model_format)  # TODO: is roboflow-python a place to be munging yaml files?

        return Dataset(self.name, self.version, model_format, os.path.abspath(location))

    def export(self, model_format=None):
        """
        Ask the Roboflow API to generate a version's dataset in a given format so that it can be downloaded via the `download()` method.

        The export will be asynchronously generated and available for download after some amount of seconds - depending on dataset size.

        Args:
            model_format (str): A format to use for downloading

        Returns:
            True

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """  # noqa: E501 // docs

        model_format = self.__get_format_identifier(model_format)

        self.__wait_if_generating()

        url = self.__get_download_url(model_format)
        response = requests.get(url, params={"api_key": self.__api_key})
        if not response.ok:
            try:
                raise RuntimeError(response.json())
            except json.JSONDecodeError:
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
                        "Exporting format " + model_format + " in progress : " + str(round(progress * 100, 2)) + "%"
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
            except json.JSONDecodeError:
                response.raise_for_status()

    def train(self, speed=None, checkpoint=None, plot_in_notebook=False) -> InferenceModel:
        """
        Ask the Roboflow API to train a previously exported version's dataset.

        Args:
            speed: Whether to train quickly or accurately. Note: accurate training is a paid feature. Default speed is `fast`.
            checkpoint: A string representing the checkpoint to use while training
            plot: Whether to plot the training results. Default is `False`.

        Returns:
            An instance of the trained model class

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON
        """  # noqa: E501 // docs

        self.__wait_if_generating()

        train_model_format = "yolov5pytorch"

        if self.type == TYPE_CLASSICATION:
            train_model_format = "folder"

        if self.type == TYPE_INSTANCE_SEGMENTATION:
            train_model_format = "yolov5pytorch"

        if self.type == TYPE_SEMANTIC_SEGMENTATION:
            train_model_format = "png-mask-semantic"

        # if classification
        if train_model_format not in self.exports:
            self.export(train_model_format)

        workspace, project, *_ = self.id.rsplit("/")
        url = f"{API_URL}/{workspace}/{project}/{self.version}/train"

        data = {}
        if speed:
            data["speed"] = speed

        if checkpoint:
            data["checkpoint"] = checkpoint

        write_line("Reaching out to Roboflow to start training...")

        response = requests.post(url, json=data, params={"api_key": self.__api_key})
        if not response.ok:
            try:
                raise RuntimeError(response.json())
            except json.JSONDecodeError:
                response.raise_for_status()

        status = "training"

        if plot_in_notebook:
            from IPython.display import clear_output
            from matplotlib import pyplot as plt

            def live_plot(epochs, mAP, loss, title=""):
                clear_output(wait=True)

                plt.subplot(2, 1, 1)
                plt.plot(epochs, mAP, "#00FFCE")
                plt.title(title)
                plt.ylabel("mAP")

                plt.subplot(2, 1, 2)
                plt.plot(epochs, loss, "#A351FB")
                plt.xlabel("epochs")
                plt.ylabel("loss")
                plt.show()

        first_graph_write = False
        previous_epochs: Union[np.ndarray, list] = []
        num_machine_spin_dots = []

        while status == "training" or status == "running":
            url = f"{API_URL}/{self.workspace}/{self.project}/{self.version}?nocache=true"
            response = requests.get(url, params={"api_key": self.__api_key})
            response.raise_for_status()
            version = response.json()["version"]
            if "models" in version.keys():
                models = version["models"]
            else:
                models = {}

            if "train" in version.keys():
                if "results" in version["train"].keys():
                    status = "finished"
                    break
                if "status" in version["train"].keys():
                    if version["train"]["status"] == "failed":
                        write_line(line="Training failed")
                        break

            epochs: Union[np.ndarray, list]
            mAP: Union[np.ndarray, list]
            loss: Union[np.ndarray, list]

            if "roboflow-train" in models.keys():
                import numpy as np

                # training has started
                epochs = np.array([int(epoch["epoch"]) for epoch in models["roboflow-train"]["epochs"]])
                mAP = np.array([float(epoch["mAP"]) for epoch in models["roboflow-train"]["epochs"]])
                loss = np.array(
                    [
                        sum(float(epoch[key]) for key in ["box_loss", "class_loss", "obj_loss"] if key in epoch)
                        for epoch in models["roboflow-train"]["epochs"]
                    ]
                )

                title = "Training in Progress"
                # plottling logic
            else:
                num_machine_spin_dots.append(".")
                if len(num_machine_spin_dots) > 5:
                    num_machine_spin_dots = ["."]
                title = "Training Machine Spinning Up" + "".join(num_machine_spin_dots)

                epochs = []
                mAP = []
                loss = []

            if (len(epochs) > len(previous_epochs)) or (len(epochs) == 0):
                if plot_in_notebook:
                    live_plot(epochs, mAP, loss, title)
                else:
                    if len(epochs) > 0:
                        title = (
                            title + ": Epoch: " + str(epochs[-1]) + " mAP: " + str(mAP[-1]) + " loss: " + str(loss[-1])
                        )
                    if not first_graph_write:
                        write_line(title)
                        first_graph_write = True

            previous_epochs = copy.deepcopy(epochs)

            time.sleep(5)

        if not self.model:
            if self.type == TYPE_OBJECT_DETECTION:
                self.model = ObjectDetectionModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    self.version,
                    colors=self.colors,
                    preprocessing=self.preprocessing,
                )
            elif self.type == TYPE_CLASSICATION:
                self.model = ClassificationModel(
                    self.__api_key,
                    self.id,
                    self.name,
                    self.version,
                    colors=self.colors,
                    preprocessing=self.preprocessing,
                )
            elif self.type == TYPE_INSTANCE_SEGMENTATION:
                self.model = InstanceSegmentationModel(
                    self.__api_key,
                    self.id,
                    colors=self.colors,
                    preprocessing=self.preprocessing,
                )
            elif self.type == TYPE_SEMANTIC_SEGMENTATION:
                self.model = SemanticSegmentationModel(self.__api_key, self.id)
            elif self.type == TYPE_KEYPOINT_DETECTION:
                self.model = KeypointDetectionModel(self.__api_key, self.id, version=self.version)
            else:
                raise ValueError(f"Unsupported model type: {self.type}")

        # return the model object
        assert self.model
        return self.model

    # @warn_for_wrong_dependencies_versions([("ultralytics", "==", "8.0.196")])
    def deploy(self, model_type: str, model_path: str, filename: str = "weights/best.pt") -> None:
        """Uploads provided weights file to Roboflow.

        Args:
            model_type (str): The type of the model to be deployed.
            model_path (str): File path to the model weights to be uploaded.
            filename (str, optional): The name of the weights file. Defaults to "weights/best.pt".
        """
        if model_type.startswith("yolo11"):
            model_type = model_type.replace("yolo11", "yolov11")

        supported_models = [
            "yolov5",
            "yolov7-seg",
            "yolov8",
            "yolov9",
            "yolonas",
            "paligemma",
            "yolov10",
            "florence-2",
            "yolov11",
        ]

        if not any(supported_model in model_type for supported_model in supported_models):
            raise (ValueError(f"Model type {model_type} not supported. Supported models are" f" {supported_models}"))

        if model_type.startswith(("paligemma", "florence-2")):
            if "paligemma" in model_type or "florence-2" in model_type:
                supported_hf_types = [
                    "florence-2-base",
                    "florence-2-large",
                    "paligemma-3b-pt-224",
                    "paligemma-3b-pt-448",
                    "paligemma-3b-pt-896",
                ]
                if model_type not in supported_hf_types:
                    raise RuntimeError(
                        f"{model_type} not supported for this type of upload."
                        f"Supported upload types are {supported_hf_types}"
                    )
            self.deploy_huggingface(model_type, model_path, filename)
            return

        if "yolonas" in model_type:
            self.deploy_yolonas(model_type, model_path, filename)
            return

        if "yolov8" in model_type:
            try:
                import torch
                import ultralytics

            except ImportError:
                raise RuntimeError(
                    "The ultralytics python package is required to deploy yolov8"
                    " models. Please install it with `pip install ultralytics`"
                )

            print_warn_for_wrong_dependencies_versions([("ultralytics", "==", "8.0.196")], ask_to_continue=True)

        elif "yolov10" in model_type:
            try:
                import torch
                import ultralytics

            except ImportError:
                raise RuntimeError(
                    "The ultralytics python package is required to deploy yolov10"
                    " models. Please install it with `pip install ultralytics`"
                )

        elif "yolov5" in model_type or "yolov7" in model_type or "yolov9" in model_type:
            try:
                import torch
            except ImportError:
                raise RuntimeError(
                    "The torch python package is required to deploy yolov5 models."
                    " Please install it with `pip install torch`"
                )

        elif "yolov11" in model_type:
            try:
                import torch
                import ultralytics

            except ImportError:
                raise RuntimeError(
                    "The ultralytics python package is required to deploy yolov10"
                    " models. Please install it with `pip install ultralytics`"
                )

            print_warn_for_wrong_dependencies_versions([("ultralytics", ">=", "8.3.0")], ask_to_continue=True)

        model = torch.load(os.path.join(model_path, filename))

        if isinstance(model["model"].names, list):
            class_names = model["model"].names
        else:
            class_names = []
            for i, val in enumerate(model["model"].names):
                class_names.append((val, model["model"].names[val]))
            class_names.sort(key=lambda x: x[0])
            class_names = [x[1] for x in class_names]

        if "yolov8" in model_type or "yolov10" in model_type or "yolov11" in model_type:
            # try except for backwards compatibility with older versions of ultralytics
            if "-cls" in model_type or model_type.startswith("yolov10") or model_type.startswith("yolov11"):
                nc = model["model"].yaml["nc"]
                args = model["train_args"]
            else:
                nc = model["model"].nc
                args = model["model"].args
            try:
                model_artifacts = {
                    "names": class_names,
                    "yaml": model["model"].yaml,
                    "nc": nc,
                    "args": {k: val for k, val in args.items() if ((k == "model") or (k == "imgsz") or (k == "batch"))},
                    "ultralytics_version": ultralytics.__version__,
                    "model_type": model_type,
                }
            except Exception:
                model_artifacts = {
                    "names": class_names,
                    "yaml": model["model"].yaml,
                    "nc": nc,
                    "args": {
                        k: val
                        for k, val in args.__dict__.items()
                        if ((k == "model") or (k == "imgsz") or (k == "batch"))
                    },
                    "ultralytics_version": ultralytics.__version__,
                    "model_type": model_type,
                }
        elif "yolov5" in model_type or "yolov7" in model_type or "yolov9" in model_type:
            # parse from yaml for yolov5

            with open(os.path.join(model_path, "opt.yaml")) as stream:
                opts = yaml.safe_load(stream)

            model_artifacts = {
                "names": class_names,
                "nc": model["model"].nc,
                "args": {
                    "imgsz": opts["imgsz"] if "imgsz" in opts else opts["img_size"],
                    "batch": opts["batch_size"],
                },
                "model_type": model_type,
            }
            if hasattr(model["model"], "yaml"):
                model_artifacts["yaml"] = model["model"].yaml

        with open(os.path.join(model_path, "model_artifacts.json"), "w") as fp:
            json.dump(model_artifacts, fp)

        torch.save(model["model"].state_dict(), os.path.join(model_path, "state_dict.pt"))

        list_files = [
            "results.csv",
            "results.png",
            "model_artifacts.json",
            "state_dict.pt",
        ]

        with zipfile.ZipFile(os.path.join(model_path, "roboflow_deploy.zip"), "w") as zipMe:
            for file in list_files:
                if os.path.exists(os.path.join(model_path, file)):
                    zipMe.write(
                        os.path.join(model_path, file),
                        arcname=file,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                else:
                    if file in ["model_artifacts.json", "state_dict.pt"]:
                        raise (ValueError(f"File {file} not found. Please make sure to provide a" " valid model path."))

        self.upload_zip(model_type, model_path)

    def deploy_huggingface(
        self, model_type: str, model_path: str, filename: str = "fine-tuned-paligemma-3b-pt-224.f16.npz"
    ) -> None:
        # Check if model_path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        model_files = os.listdir(model_path)
        print(f"Model files found in {model_path}: {model_files}")

        files_to_deploy = []

        # Find first .npz file in model_path
        npz_filename = next((file for file in model_files if file.endswith(".npz")), None)
        if any([file.endswith(".safetensors") for file in model_files]):
            print(f"Found .safetensors file in model path. Deploying PyTorch {model_type} model.")
            necessary_files = [
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "tokenizer.json",
            ]
            for file in necessary_files:
                if file not in model_files:
                    print("Missing necessary file", file)
                    res = input("Do you want to continue? (y/n)")
                    if res.lower() != "y":
                        exit(1)
            for file in model_files:
                files_to_deploy.append(file)
        elif npz_filename is not None:
            print(f"Found .npz file {npz_filename} in model path. Deploying JAX PaliGemma model.")
            files_to_deploy.append(npz_filename)
        else:
            raise FileNotFoundError(f"No .npz or .safetensors file found in model path {model_path}.")

        if len(files_to_deploy) == 0:
            raise FileNotFoundError(f"No valid files found in model path {model_path}.")
        print(f"Zipping files for deploy: {files_to_deploy}")

        import tarfile

        with tarfile.open(os.path.join(model_path, "roboflow_deploy.tar"), "w") as tar:
            for file in files_to_deploy:
                tar.add(os.path.join(model_path, file), arcname=file)

        print("Uploading to Roboflow... May take several minutes.")
        self.upload_zip(model_type, model_path, "roboflow_deploy.tar")

    def deploy_yolonas(self, model_type: str, model_path: str, filename: str = "weights/best.pt") -> None:
        try:
            import torch
        except ImportError:
            raise RuntimeError(
                "The torch python package is required to deploy yolonas models."
                " Please install it with `pip install torch`"
            )

        model = torch.load(os.path.join(model_path, filename), map_location="cpu")
        class_names = model["processing_params"]["class_names"]

        opt_path = os.path.join(model_path, "opt.yaml")
        if not os.path.exists(opt_path):
            raise RuntimeError(
                f"You must create an opt.yaml file at {os.path.join(model_path, '')} of the format:\n"
                f"imgsz: <resolution of model>\n"
                f"batch_size: <batch size of inference model>\n"
                f"architecture: <one of [yolo_nas_s, yolo_nas_m, yolo_nas_l]."
                f"s, m, l refer to small, medium, large architecture sizes, respectively>\n"
            )
        with open(os.path.join(model_path, "opt.yaml")) as stream:
            opts = yaml.safe_load(stream)
        required_keys = ["imgsz", "batch_size", "architecture"]
        for key in required_keys:
            if key not in opts:
                raise RuntimeError(f"{opt_path} lacks required key {key}. Required keys: {required_keys}")

        model_artifacts = {
            "names": class_names,
            "nc": len(class_names),
            "args": {
                "imgsz": opts["imgsz"] if "imgsz" in opts else opts["img_size"],
                "batch": opts["batch_size"],
                "architecture": opts["architecture"],
            },
            "model_type": model_type,
        }

        with open(os.path.join(model_path, "model_artifacts.json"), "w") as fp:
            json.dump(model_artifacts, fp)

        shutil.copy(os.path.join(model_path, filename), os.path.join(model_path, "state_dict.pt"))

        list_files = [
            "results.json",
            "results.png",
            "model_artifacts.json",
            "state_dict.pt",
        ]

        with zipfile.ZipFile(os.path.join(model_path, "roboflow_deploy.zip"), "w") as zipMe:
            for file in list_files:
                if os.path.exists(os.path.join(model_path, file)):
                    zipMe.write(
                        os.path.join(model_path, file),
                        arcname=file,
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                else:
                    if file in ["model_artifacts.json", filename]:
                        raise (ValueError(f"File {file} not found. Please make sure to provide a" " valid model path."))

        self.upload_zip(model_type, model_path)

    def upload_zip(self, model_type: str, model_path: str, model_file_name: str = "roboflow_deploy.zip"):
        res = requests.get(
            f"{API_URL}/{self.workspace}/{self.project}/{self.version}"
            f"/uploadModel?api_key={self.__api_key}&modelType={model_type}&nocache=true"
        )
        try:
            if res.status_code == 429:
                raise RuntimeError(
                    "This version already has a trained model. Please generate and"
                    " train a new version in order to upload model to Roboflow."
                )
            else:
                res.raise_for_status()
        except Exception as e:
            print(f"An error occured when getting the model upload URL: {e}")
            return

        res = requests.put(
            res.json()["url"],
            data=open(os.path.join(model_path, model_file_name), "rb"),
        )
        try:
            res.raise_for_status()

            if self.public:
                print(
                    "View the status of your deployment at:"
                    f" {APP_URL}/{self.workspace}/{self.project}/{self.version}"
                )
                print(
                    "Share your model with the world at:"
                    f" {UNIVERSE_URL}/{self.workspace}/{self.project}/"
                    f"model/{self.version}"
                )
            else:
                print(
                    "View the status of your deployment at:"
                    f" {APP_URL}/{self.workspace}/{self.project}/{self.version}"
                )

        except Exception as e:
            print(f"An error occured when uploading the model: {e}")

    def __download_zip(self, link, location, format):
        """
        Download a dataset's zip file from the given URL and save it in the desired location

        Args:
            link (str): link the URL of the remote zip file
            location (str): filepath of the data directory to save the zip file to
            format (str): the format identifier string
        """  # noqa: E501 // docs
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
            response = requests.get(link, stream=True)

            # write the zip file to the desired location
            with open(location + "/roboflow.zip", "wb") as f:
                total_length = int(response.headers.get("content-length"))  # type: ignore[arg-type]
                desc = None if TQDM_DISABLE else f"Downloading Dataset Version Zip in {location} to {format}:"
                for chunk in tqdm(
                    response.iter_content(chunk_size=1024),
                    desc=desc,
                    total=int(total_length / 1024) + 1,
                ):
                    if chunk:
                        f.write(chunk)
                        f.flush()

        except Exception as e:
            print(f"Error when trying to download dataset @ {link}")
            raise e
        sys.stdout.write("\n")
        sys.stdout.flush()

    def __extract_zip(self, location, format):
        """
        Extracts the contents of a downloaded ZIP file and then deletes the zipped file.

        Args:
            location (str): filepath of the data directory that contains the ZIP file
            format (str): the format identifier string

        Raises:
            RuntimeError: If there is an error unzipping the file
        """  # noqa: E501 // docs
        desc = None if TQDM_DISABLE else f"Extracting Dataset Version Zip to {location} in {format}:"
        with zipfile.ZipFile(location + "/roboflow.zip", "r") as zip_ref:
            for member in tqdm(
                zip_ref.infolist(),
                desc=desc,
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
            str: the local path
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
            format (str): the format identifier string

        Returns:
            str: the Roboflow API URL
        """
        workspace, project, *_ = self.id.rsplit("/")
        return f"{API_URL}/{workspace}/{project}/{self.version}/{format}"

    def __get_format_identifier(self, format):
        """
        If `format` is none, fall back to the instance's `model_format` value.

        If a human readable format name was passed, return the identifier that should be used for Roboflow API calls

        Otherwise, assume that the passed in format is also the identifier

        Args:
            format (str): a human readable format string

        Returns:
            str: format identifier string
        """  # noqa: E501 // docs
        if not format:
            format = self.model_format

        if not format:
            raise RuntimeError(
                "You must pass a format argument to version.download() or define a" " model in your Roboflow object"
            )

        friendly_formats = {"yolov5": "yolov5pytorch", "yolov7": "yolov7pytorch"}

        return friendly_formats.get(format, format)

    def __reformat_yaml(self, location: str, format: str):
        """
        Certain formats seem to require reformatting the downloaded YAML.

        Args:
            location (str): filepath of the data directory that contains the yaml file
            format (str): the format identifier string
        """  # noqa: E501 // docs
        data_path = os.path.join(location, "data.yaml")

        def data_yaml_callback(content: dict) -> dict:
            if format == "mt-yolov6":
                content["train"] = location + content["train"].lstrip(".")
                content["val"] = location + content["val"].lstrip(".")
                content["test"] = location + content["test"].lstrip(".")
            if format in ["yolov5pytorch", "yolov7pytorch"]:
                content["train"] = location + content["train"].lstrip("..")
                content["val"] = location + content["val"].lstrip("..")
            try:
                # get_wrong_dependencies_versions raises exception if ultralytics is not installed at all  # noqa: E501 // docs
                if format == "yolov8" and not get_wrong_dependencies_versions(
                    dependencies_versions=[("ultralytics", "==", "8.0.196")]
                ):
                    content["train"] = "train/images"
                    content["val"] = "valid/images"
                    content["test"] = "test/images"
            except ModuleNotFoundError:
                pass
            return content

        if format in ["yolov5pytorch", "mt-yolov6", "yolov7pytorch", "yolov8", "yolov9"]:
            amend_data_yaml(path=data_path, callback=data_yaml_callback)

    def __str__(self):
        """
        String representation of version object.
        """
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


def unwrap_version_id(version_id: str) -> str:
    return version_id if "/" not in str(version_id) else version_id.split("/")[-1]
