import datetime
import json
import mimetypes
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Union

import filetype
import requests

from roboflow.adapters import rfapi
from roboflow.adapters.rfapi import ImageUploadError
from roboflow.config import API_URL, DEMO_KEYS
from roboflow.core.version import Version
from roboflow.util.general import Retry
from roboflow.util.image_utils import load_labelmap

ACCEPTED_IMAGE_FORMATS = {
    "image/bmp",
    "image/jpeg",
    "image/png",
    "image/webp",
}


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning  # type: ignore[assignment]


class Project:
    """
    A Roboflow Project.
    """

    def __init__(self, api_key: str, a_project: dict, model_format: Optional[str] = None):
        """
        Create a Project object that represents a Project associated with a Workspace.

        Args:
            api_key (str): private roboflow api key
            a_project (dict): the project information dictionary
            model_format (str): the model format of the project

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")
        """

        if api_key:
            self.__api_key = api_key
            self.annotation = a_project["annotation"]
            self.classes = a_project["classes"]
            self.colors = a_project["colors"]
            self.created = datetime.datetime.fromtimestamp(a_project["created"]) if a_project["created"] else None
            self.id = a_project["id"]
            self.images = a_project["images"]
            self.name = a_project["name"]
            self.public = a_project["public"]
            self.splits = a_project["splits"]
            self.type = a_project["type"]
            self.multilabel = a_project.get("multilabel", False)
            self.unannotated = a_project["unannotated"]
            self.updated = datetime.datetime.fromtimestamp(a_project["updated"]) if a_project["updated"] else None
            self.model_format = model_format

            temp = self.id.rsplit("/")
            self.__workspace = temp[0]
            self.__project_name = temp[1]

        elif DEMO_KEYS:
            self.__api_key = DEMO_KEYS[0]
            self.model_format = model_format

        else:
            raise ValueError("A valid API key must be provided.")

    def get_version_information(self):
        """
        Retrieve all versions of a project.

        Returns:
            A list of all versions of the project.

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> version_info = project.get_version_information()
        """
        dataset_info = requests.get(
            API_URL + "/" + self.__workspace + "/" + self.__project_name + "?api_key=" + self.__api_key
        )

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset # noqa: E501 // docs
        if dataset_info.status_code != 200:
            raise RuntimeError(dataset_info.text)

        dataset_info = dataset_info.json()
        return dataset_info["versions"]

    def list_versions(self):
        """
        Print out versions for that specific project.

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> project.list_versions()
        """
        version_info = self.get_version_information()
        print(version_info)

    def versions(self):
        """
        Return all versions in the project as Version objects.

        Returns:
            A list of Version objects.

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> versions = project.versions()
        """
        version_info = self.get_version_information()
        version_array = []
        for a_version in version_info:
            version_object = Version(
                a_version,
                (self.type if "model" in a_version else None),
                self.__api_key,
                self.name,
                a_version["id"],
                self.model_format,
                local=None,
                workspace=self.__workspace,
                project=self.__project_name,
                public=self.public,
                colors=self.colors,
            )
            version_array.append(version_object)
        return version_array

    def generate_version(self, settings):
        """
        Generate a version of a dataset hosted on Roboflow.

        Args:
            settings: A Python dict with augmentation and preprocessing keys and specifications for generation. These settings mirror capabilities available via the Roboflow UI.
                    For example:
                        {
                            "augmentation": {
                                "bbblur": { "pixels": 1.5 },
                                "bbbrightness": { "brighten": true, "darken": false, "percent": 91 },
                                "bbcrop": { "min": 12, "max": 71 },
                                "bbexposure": { "percent": 30 },
                                "bbflip": { "horizontal": true, "vertical": false },
                                "bbnoise": { "percent": 50 },
                                "bbninety": { "clockwise": true, "counter-clockwise": false, "upside-down": false },
                                "bbrotate": { "degrees": 45 },
                                "bbshear": { "horizontal": 45, "vertical": 45 },
                                "blur": { "pixels": 1.5 },
                                "brightness": { "brighten": true, "darken": false, "percent": 91 },
                                "crop": { "min": 12, "max": 71 },
                                "cutout": { "count": 26, "percent": 71 },
                                "exposure": { "percent": 30 },
                                "flip": { "horizontal": true, "vertical": false },
                                "hue": { "degrees": 180 },
                                "image": { "versions": 32 },
                                "mosaic": true,
                                "ninety": { "clockwise": true, "counter-clockwise": false, "upside-down": false },
                                "noise": { "percent": 50 },
                                "rgrayscale": { "percent": 50 },
                                "rotate": { "degrees": 45 },
                                "saturation": { "percent": 50 },
                                "shear": { "horizontal": 45, "vertical": 45 }
                            },
                            "preprocessing": {
                                "auto-orient": true,
                                "contrast": { "type": "Contrast Stretching" },
                                "filter-null": { "percent": 50 },
                                "grayscale": true,
                                "isolate": true,
                                "remap": { "original_class_name": "new_class_name" },
                                "resize": { "width": 200, "height": 200, "format": "Stretch to" },
                                "static-crop": { "x_min": 10, "x_max": 90, "y_min": 10, "y_max": 90 },
                                "tile": { "rows": 2, "columns": 2 }
                            }
                        }

        Returns:
            int: The version number that is being generated.

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> versions = project.generate_version(settings={...})
        """  # noqa: E501 // docs

        if not {"augmentation", "preprocessing"} <= settings.keys():
            raise (
                RuntimeError(
                    "augmentation and preprocessing keys are required to generate. If"
                    " none are desired specify empty dict associated with that key."
                )
            )

        r = requests.post(
            f"{API_URL}/{self.__workspace}/{self.__project_name}/" f"generate?api_key={self.__api_key}",
            json=settings,
        )

        try:
            r_json = r.json()
        except Exception:
            raise RuntimeError("Error when requesting to generate a new version for project.")

        # if the generation succeeds, return the version that is being generated
        if r.status_code == 200:
            sys.stdout.write("\r" + r_json["message"] + " for new version " + str(r_json["version"]) + ".")
            sys.stdout.write("\n")
            sys.stdout.flush()
            return int(r_json["version"])
        else:
            if "error" in r_json.keys():
                raise RuntimeError(r_json["error"])
            else:
                raise RuntimeError(json.dumps(r_json))

    def train(
        self,
        new_version_settings={
            "preprocessing": {
                "auto-orient": True,
                "resize": {"width": 640, "height": 640, "format": "Stretch to"},
            },
            "augmentation": {},
        },
        speed=None,
        checkpoint=None,
        plot_in_notebook=False,
    ):
        """
        Ask the Roboflow API to train a previously exported version's dataset.

        Args:
            speed: Whether to train quickly or accurately. Note: accurate training is a paid feature. Default speed is `fast`.
            checkpoint: A string representing the checkpoint to use while training
            plot: Whether to plot the training loss curve. Default is False.

        Returns:
            True

        Raises:
            RuntimeError: If the Roboflow API returns an error with a helpful JSON body
            HTTPError: If the Network/Roboflow API fails and does not return JSON

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> version = project.version(1)

            >>> version.train()
        """  # noqa: E501 // docs

        new_version = self.generate_version(settings=new_version_settings)
        new_version = self.version(new_version)
        new_model = new_version.train(speed=speed, checkpoint=checkpoint, plot_in_notebook=plot_in_notebook)

        return new_model

    def version(self, version_number: int, local: Optional[str] = None):
        """
        Retrieves information about a specific version and returns a Version() object.

        Args:
            version_number (int): the version number that you want to retrieve
            local (str): specifies the localhost address and port if pointing towards local inference engine

        Returns:
            Version() object
        """  # noqa: E501 // docs

        if self.__api_key in DEMO_KEYS:
            name = ""
            if self.__api_key == "coco-128-sample":
                name = "coco-128"
            else:
                name = "chess-pieces-new"
            return Version(
                {},
                "type",
                self.__api_key,
                name,
                version_number,
                self.model_format,
                local=None,
                workspace="",
                project="",
                public=True,
            )

        version_info = self.get_version_information()

        for version_object in version_info:
            current_version_num = os.path.basename(version_object["id"])
            if current_version_num == str(version_number):
                vers = Version(
                    version_object,
                    self.type,
                    self.__api_key,
                    self.name,
                    current_version_num,
                    self.model_format,
                    local=local,
                    workspace=self.__workspace,
                    project=self.__project_name,
                    public=self.public,
                    colors=self.colors,
                )
                return vers

        raise RuntimeError(f"Version number {version_number} is not found.")

    def check_valid_image(self, image_path: str) -> bool:
        """
        Check if an image is valid. Useful before attempting to upload an image to Roboflow.

        Args:
            image_path (str): path to image you'd like to check

        Returns:
            bool: whether the image is valid or not
        """
        kind = filetype.guess(image_path)

        if kind is None:
            return False

        extension_mimetype, _ = mimetypes.guess_type(image_path)

        if extension_mimetype and extension_mimetype != kind.mime:
            print(f"[{image_path}] file type ({kind.mime}) does not match filename extension.")

        return kind.mime in ACCEPTED_IMAGE_FORMATS

    def upload(
        self,
        image_path: str,
        annotation_path: Optional[str] = None,
        hosted_image: bool = False,
        image_id: Optional[str] = None,
        split: str = "train",
        num_retry_uploads: int = 0,
        batch_name: Optional[str] = None,
        tag_names: list = [],
        is_prediction: bool = False,
        **kwargs,
    ):
        """
        Upload an image or annotation to the Roboflow API.

        Args:
            image_path (str): path to image you'd like to upload
            annotation_path (str): path to the annotation file. If not provided, the image will be uploaded without annotation.
                Special case: in classification projects, this can instead be a class name. e.g. "dog".
            hosted_image (bool): whether the image is hosted
            image_id (str): id of the image
            split (str): which split to upload to - "train", "valid" or "test"
            num_retry_uploads (int): how many times to retry upload on failure
            batch_name (str): name of batch to upload to within project
            tag_names (list[str]): tags to be applied to an image
            is_prediction (bool): whether the annotation data is a prediction rather than ground truth

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> project.upload(image_path="YOUR_IMAGE.jpg")
        """  # noqa: E501 // docs

        is_hosted = image_path.startswith("http://") or image_path.startswith("https://")

        is_file = os.path.isfile(image_path) or is_hosted
        is_dir = os.path.isdir(image_path)

        if not is_file and not is_dir:
            raise RuntimeError(
                f"The provided image path [ {image_path} ] is not a valid path. Please provide a"
                " path to an image or a directory."
            )

        if is_file:
            is_image = is_hosted or self.check_valid_image(image_path)

            if not is_image:
                raise RuntimeError(
                    "The image you provided {} is not a supported file format. We" " currently support: {}.".format(
                        image_path, ", ".join(ACCEPTED_IMAGE_FORMATS)
                    )
                )

            self.single_upload(
                image_path=image_path,
                annotation_path=annotation_path,
                hosted_image=hosted_image,
                image_id=image_id,
                split=split,
                num_retry_uploads=num_retry_uploads,
                batch_name=batch_name,
                tag_names=tag_names,
                is_prediction=is_prediction,
                **kwargs,
            )

        else:
            images = os.listdir(image_path)
            for image in images:
                path = image_path + "/" + image
                if self.check_valid_image(path):
                    self.single_upload(
                        image_path=path,
                        annotation_path=annotation_path,
                        hosted_image=hosted_image,
                        image_id=image_id,
                        split=split,
                        num_retry_uploads=num_retry_uploads,
                        batch_name=batch_name,
                        tag_names=tag_names,
                        is_prediction=is_prediction,
                        **kwargs,
                    )
                    print("[ " + path + " ] was uploaded succesfully.")
                else:
                    print("[ " + path + " ] was skipped.")
                    continue

    def upload_image(
        self,
        image_path=None,
        hosted_image=False,
        split="train",
        num_retry_uploads=0,
        batch_name=None,
        tag_names=[],
        sequence_number=None,
        sequence_size=None,
        **kwargs,
    ):
        project_url = self.id.rsplit("/")[1]

        t0 = time.time()
        upload_retry_attempts = 0
        retry = Retry(num_retry_uploads, ImageUploadError)

        try:
            image = retry(
                rfapi.upload_image,
                self.__api_key,
                project_url,
                image_path,
                hosted_image=hosted_image,
                split=split,
                batch_name=batch_name,
                tag_names=tag_names,
                sequence_number=sequence_number,
                sequence_size=sequence_size,
                **kwargs,
            )
            upload_retry_attempts = retry.retries
        except ImageUploadError as e:
            e.retries = upload_retry_attempts
            raise e

        upload_time = time.time() - t0

        return image, upload_time, upload_retry_attempts

    def save_annotation(
        self,
        annotation_path=None,
        annotation_labelmap=None,
        image_id=None,
        job_name=None,
        is_prediction: bool = False,
        annotation_overwrite=False,
    ):
        project_url = self.id.rsplit("/")[1]
        annotation_name, annotation_str = self._annotation_params(annotation_path)
        t0 = time.time()

        annotation = rfapi.save_annotation(
            self.__api_key,
            project_url,
            annotation_name,  # type: ignore[type-var]
            annotation_str,  # type: ignore[type-var]
            image_id,
            job_name=job_name,  # type: ignore[type-var]
            is_prediction=is_prediction,
            annotation_labelmap=annotation_labelmap,
            overwrite=annotation_overwrite,
        )

        upload_time = time.time() - t0

        return annotation, upload_time

    def single_upload(
        self,
        image_path=None,
        annotation_path=None,
        annotation_labelmap=None,
        hosted_image=False,
        image_id=None,
        split="train",
        num_retry_uploads=0,
        batch_name=None,
        tag_names=[],
        is_prediction: bool = False,
        annotation_overwrite=False,
        sequence_number=None,
        sequence_size=None,
        **kwargs,
    ):
        if image_path and image_id:
            raise Exception("You can't pass both image_id and image_path")
        if not (image_path or image_id):
            raise Exception("You need to pass image_path or image_id")
        if isinstance(annotation_labelmap, str):
            annotation_labelmap = load_labelmap(annotation_labelmap)

        uploaded_image, uploaded_annotation = None, None
        upload_time, annotation_time = None, None
        upload_retry_attempts = 0

        if image_path:
            uploaded_image, upload_time, upload_retry_attempts = self.upload_image(
                image_path,
                hosted_image,
                split,
                num_retry_uploads,
                batch_name,
                tag_names,
                sequence_number,
                sequence_size,
                **kwargs,
            )
            image_id = uploaded_image["id"]  # type: ignore[index]

        if annotation_path and image_id:
            uploaded_annotation, annotation_time = self.save_annotation(
                annotation_path,
                annotation_labelmap,
                image_id,
                batch_name,
                is_prediction,
                annotation_overwrite,
            )

        return {
            "image": uploaded_image,
            "annotation": uploaded_annotation,
            "upload_time": upload_time,
            "annotation_time": annotation_time,
            "upload_retry_attempts": upload_retry_attempts,
        }

    def _annotation_params(self, annotation_path):
        annotation_name, annotation_string = None, None
        if isinstance(annotation_path, dict) and annotation_path.get("rawText"):
            annotation_name = annotation_path["name"]
            annotation_string = annotation_path["rawText"]
        elif os.path.exists(annotation_path):  # type: ignore[arg-type]
            with open(annotation_path):  # type: ignore[arg-type]
                annotation_string = open(annotation_path).read()  # type: ignore[arg-type]
            annotation_name = os.path.basename(annotation_path)  # type: ignore[arg-type]
        elif self.type == "classification":
            print(f"-> using {annotation_path} as classname for classification project")
            annotation_string = annotation_path
            annotation_name = annotation_path
        else:
            raise Exception(
                f"File not found or uploading to non-classification "
                f"type project with invalid string. - {annotation_path}"
            )
        return annotation_name, annotation_string

    def search(
        self,
        like_image: Optional[str] = None,
        prompt: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tag: Optional[str] = None,
        class_name: Optional[str] = None,
        in_dataset: Optional[str] = None,
        batch: bool = False,
        batch_id: Optional[str] = None,
        fields: list = ["id", "created", "name", "labels"],
    ):
        """
        Search for images in a project.

        Args:
            like_image (str): name of an image in your dataset to use if you want to find images similar to that one
            prompt (str): search prompt
            offset (int): offset of results
            limit (int): limit of results
            tag (str): tag that an image must have
            class_name (str): class name that an image must have
            in_dataset (str): dataset that an image must be in
            batch (bool): whether the image must be in a batch
            batch_id (str): batch id that an image must be in
            fields (list): fields to return in results (default: ["id", "created", "name", "labels"])

        Returns:
            A list of images that match the search criteria.

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> results = project.search(query="cat", limit=10)
        """  # noqa: E501 // docs
        payload: Dict[str, Union[str, int, List[str]]] = {}

        if like_image is not None:
            payload["like_image"] = like_image

        if prompt is not None:
            payload["prompt"] = prompt

        if offset is not None:
            payload["offset"] = offset

        if limit is not None:
            payload["limit"] = limit

        if tag is not None:
            payload["tag"] = tag

        if class_name is not None:
            payload["class_name"] = class_name

        if in_dataset is not None:
            payload["in_dataset"] = in_dataset

        if batch is not None:
            payload["batch"] = batch

        if batch_id is not None:
            payload["batch_id"] = batch_id

        payload["fields"] = fields

        data = requests.post(
            API_URL + "/" + self.__workspace + "/" + self.__project_name + "/search?api_key=" + self.__api_key,
            json=payload,
        )

        return data.json()["results"]

    def search_all(
        self,
        like_image: Optional[str] = None,
        prompt: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        tag: Optional[str] = None,
        class_name: Optional[str] = None,
        in_dataset: Optional[str] = None,
        batch: bool = False,
        batch_id: Optional[str] = None,
        fields: list = ["id", "created"],
    ):
        """
        Create a paginated list of search results for use in searching the images in a project.

        Args:
            like_image (str): name of an image in your dataset to use if you want to find images similar to that one
            prompt (str): search prompt
            offset (int): offset of results
            limit (int): limit of results
            tag (str): tag that an image must have
            class_name (str): class name that an image must have
            in_dataset (str): dataset that an image must be in
            batch (bool): whether the image must be in a batch
            batch_id (str): batch id that an image must be in
            fields (list): fields to return in results (default: ["id", "created", "name", "labels"])

        Returns:
            A list of images that match the search criteria.

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> results = project.search_all(query="cat", limit=10)

            >>> for result in results:

            >>>     print(result)
        """  # noqa: E501 // docs
        while True:
            data = self.search(
                like_image=like_image,
                prompt=prompt,
                offset=offset,
                limit=limit,
                tag=tag,
                class_name=class_name,
                in_dataset=in_dataset,
                batch=batch,
                batch_id=batch_id,
                fields=fields,
            )

            yield data

            if len(data) < limit:
                break

            offset += limit

    def __str__(self):
        """
        Show a string representation of a Project object.
        """
        # String representation of project
        json_str = {"name": self.name, "type": self.type, "workspace": self.__workspace}

        return json.dumps(json_str, indent=2)
