import datetime
import io
import json
import os
import sys
import urllib
import warnings

import cv2
import requests
from PIL import Image, UnidentifiedImageError
from requests_toolbelt.multipart.encoder import MultipartEncoder

from roboflow.config import API_URL, DEFAULT_BATCH_NAME, DEMO_KEYS
from roboflow.core.version import Version
from roboflow.util.general import retry

ACCEPTED_IMAGE_FORMATS = ["PNG", "JPEG"]


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning


class UploadError(Exception):
    pass


class Project:
    """
    A Roboflow Project.
    """

    def __init__(self, api_key: str, a_project: str, model_format: str = None):
        """
        Create a Project object that represents a Project associated with a Workspace.

        Args:
            api_key (str): private roboflow api key
            a_project (str): the project id
            model_format (str): the model format of the project

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")
        """
        if api_key in DEMO_KEYS:
            self.__api_key = api_key
            self.model_format = model_format
        else:
            self.__api_key = api_key
            self.annotation = a_project["annotation"]
            self.classes = a_project["classes"]
            self.colors = a_project["colors"]
            self.created = datetime.datetime.fromtimestamp(a_project["created"])
            self.id = a_project["id"]
            self.images = a_project["images"]
            self.name = a_project["name"]
            self.public = a_project["public"]
            self.splits = a_project["splits"]
            self.type = a_project["type"]
            self.unannotated = a_project["unannotated"]
            self.updated = datetime.datetime.fromtimestamp(a_project["updated"])
            self.model_format = model_format

            temp = self.id.rsplit("/")
            self.__workspace = temp[0]
            self.__project_name = temp[1]

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
            API_URL
            + "/"
            + self.__workspace
            + "/"
            + self.__project_name
            + "?api_key="
            + self.__api_key
        )

        # Throw error if dataset isn't valid/user doesn't have permissions to access the dataset
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
        """

        if not {"augmentation", "preprocessing"} <= settings.keys():
            raise (
                RuntimeError(
                    "augmentation and preprocessing keys are required to generate. If none are desired specify empty dict associated with that key."
                )
            )

        r = requests.post(
            f"{API_URL}/{self.__workspace}/{self.__project_name}/generate?api_key={self.__api_key}",
            json=settings,
        )

        try:
            r_json = r.json()
        except:
            raise ("Error when requesting to generate a new version for project.")

        # if the generation succeeds, return the version that is being generated
        if r.status_code == 200:
            sys.stdout.write(
                "\r"
                + r_json["message"]
                + " for new version "
                + str(r_json["version"])
                + "."
            )
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
    ) -> bool:
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
        """

        new_version = self.generate_version(settings=new_version_settings)
        new_version = self.version(new_version)
        new_model = new_version.train(
            speed=speed, checkpoint=checkpoint, plot_in_notebook=plot_in_notebook
        )

        return new_model

    def version(self, version_number: int, local: str = None):
        """
        Retrieves information about a specific version and returns a Version() object.

        Args:
            version_number (int): the version number that you want to retrieve
            local (str): specifies the localhost address and port if pointing towards local inference engine

        Returns:
            Version() object
        """

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

        raise RuntimeError("Version number {} is not found.".format(version_number))

    def __image_upload(
        self,
        image_path: str,
        hosted_image: bool = False,
        split: str = "train",
        batch_name: str = DEFAULT_BATCH_NAME,
        tag_names: list = [],
        **kwargs,
    ):
        """
        Upload an image to a specific project.

        Args:
            image_path (str): path to image you'd like to upload
            hosted_image (bool): whether the image is hosted on Roboflow
            split (str): the dataset split the image to
        """

        # If image is not a hosted image
        if not hosted_image:
            batch_name = (
                batch_name
                if batch_name and isinstance(batch_name, str)
                else DEFAULT_BATCH_NAME
            )

            project_name = self.id.rsplit("/")[1]
            image_name = os.path.basename(image_path)

            # Construct URL for local image upload
            self.image_upload_url = "".join(
                [
                    API_URL + "/dataset/",
                    project_name,
                    "/upload",
                    "?api_key=",
                    self.__api_key,
                    "&batch=",
                    batch_name,
                ]
            )
            for key, value in kwargs.items():
                self.image_upload_url += "&" + str(key) + "=" + str(value)

            for tag in tag_names:
                self.image_upload_url = self.image_upload_url + f"&tag={tag}"

            # Convert to PIL Image
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(image)

            # Convert to JPEG Buffer
            buffered = io.BytesIO()
            pilImage.save(buffered, quality=100, format="JPEG")

            # Build multipart form and post request
            m = MultipartEncoder(
                fields={
                    "name": image_name,
                    "split": split,
                    "file": ("imageToUpload", buffered.getvalue(), "image/jpeg"),
                }
            )
            response = requests.post(
                self.image_upload_url, data=m, headers={"Content-Type": m.content_type}
            )

        else:
            # Hosted image upload url
            project_name = self.id.rsplit("/")[1]

            upload_url = "".join(
                [
                    API_URL + "/dataset/" + self.project_name + "/upload",
                    "?api_key=" + self.__api_key,
                    "&name=" + os.path.basename(image_path),
                    "&split=" + split,
                    "&image=" + urllib.parse.quote_plus(image_path),
                ]
            )
            # Get response
            response = requests.post(upload_url)
        responsejson = None
        try:
            responsejson = response.json()
        except:
            pass
        if response.status_code == 200:
            if responsejson:
                if "duplicate" in responsejson.keys():
                    print(f"Duplicate image not uploaded: {image_path}")
                elif not responsejson.get("success"):
                    raise UploadError(f"Server rejected image: {responsejson}")
                return responsejson.get("id")
            else:
                warnings.warn(
                    f"upload image {image_path} 200 OK, weird response: {response}"
                )
                return None
        else:
            if responsejson:
                raise UploadError(
                    f"Bad response: {response.status_code}: {responsejson}"
                )
            else:
                raise UploadError(f"Bad response: {response}")

    def __annotation_upload(
        self, annotation_path: str, image_id: str, is_prediction: bool = False
    ):
        """
        Upload an annotation to a specific project.

        Args:
            annotation_path (str): path to annotation you'd like to upload
            image_id (str): image id you'd like to upload that has annotations for it.
        """

        # stop on empty string
        if len(annotation_path) == 0:
            print("Please provide a non-empty string for annotation_path.")
            return {"result": "Please provide a non-empty string for annotation_path."}

        # check if annotation file exists
        elif os.path.exists(annotation_path):
            # print("-> found given annotation file")
            annotation_string = open(annotation_path, "r").read()

        # if not annotation file, check if user wants to upload regular as classification annotation
        elif self.type == "classification":
            print(f"-> using {annotation_path} as classname for classification project")
            annotation_string = annotation_path

        # don't attempt upload otherwise
        else:
            print(
                "File not found or uploading to non-classification type project with invalid string"
            )
            return {
                "result": "File not found or uploading to non-classification type project with invalid string"
            }

        self.annotation_upload_url = "".join(
            [
                API_URL + "/dataset/",
                self.__project_name,
                "/annotate/",
                image_id,
                "?api_key=",
                self.__api_key,
                "&name=" + os.path.basename(annotation_path),
                "&prediction=true" if is_prediction else "",
            ]
        )

        response = requests.post(
            self.annotation_upload_url,
            data=annotation_string,
            headers={"Content-Type": "text/plain"},
        )
        responsejson = None
        try:
            responsejson = response.json()
        except:
            pass
        if response.status_code == 200:
            if responsejson:
                if responsejson.get("error"):
                    raise UploadError(
                        f"Failed to save annotation for {image_id}: {responsejson['error']}"
                    )
                elif not responsejson.get("success"):
                    raise UploadError(
                        f"Failed to save annotation for {image_id}: {responsejson}"
                    )
            else:
                warnings.warn(
                    f"save annotation {annotation_path} 200 OK, weird response: {response}"
                )
        elif response.status_code == 409 and "already annotated" in (
            responsejson or {}
        ).get("error", {}).get("message"):
            print(f"image already annotated: {annotation_path}")
        else:
            if responsejson:
                if responsejson.get("error"):
                    raise UploadError(
                        f"save annotation for {image_id} / bad response: {response.status_code}: {responsejson['error']}"
                    )
                else:
                    raise UploadError(
                        f"save annotation for {image_id} / bad response: {response.status_code}: {responsejson}"
                    )
            else:
                raise UploadError(
                    f"save annotation for {image_id} bad response: {response}"
                )

    def check_valid_image(self, image_path: str):
        """
        Check if an image is valid. Useful before attempting to upload an image to Roboflow.

        Args:
            image_path (str): path to image you'd like to check

        Returns:
            bool: whether the image is valid or not
        """
        try:
            img = Image.open(image_path)
            valid = img.format in ACCEPTED_IMAGE_FORMATS
            img.close()
        except UnidentifiedImageError:
            return False

        return valid

    def upload(
        self,
        image_path: str = None,
        annotation_path: str = None,
        hosted_image: bool = False,
        image_id: str = None,
        split: str = "train",
        num_retry_uploads: int = 0,
        batch_name: str = DEFAULT_BATCH_NAME,
        tag_names: list = [],
        is_prediction: bool = False,
        **kwargs,
    ):
        """
        Upload an image or annotation to the Roboflow API.

        Args:
            image_path (str): path to image you'd like to upload
            annotation_path (str): if you're upload annotation, path to it
            hosted_image (bool): whether the image is hosted
            image_id (str): id of the image
            split (str): to upload the image to
            num_retry_uploads (int): how many times to retry upload on failure
            batch_name (str): name of batch to upload to within project
            tag_names (list[str]): tags to be applied to an image
            is_prediction (bool): whether the annotation data is a prediction rather than ground truth

        Example:
            >>> import roboflow

            >>> rf = roboflow.Roboflow(api_key="")

            >>> project = rf.workspace().project("PROJECT_ID")

            >>> project.upload(image_path="YOUR_IMAGE.jpg")
        """

        is_hosted = image_path.startswith("http://") or image_path.startswith(
            "https://"
        )

        is_file = os.path.isfile(image_path) or is_hosted
        is_dir = os.path.isdir(image_path)

        if not is_file and not is_dir:
            raise RuntimeError(
                "The provided image path [ {} ] is not a valid path. Please provide a path to an image or a directory.".format(
                    image_path
                )
            )

        if is_file:
            is_image = self.check_valid_image(image_path) or is_hosted

            if not is_image:
                raise RuntimeError(
                    "The image you provided {} is not a supported file format. We currently support: {}.".format(
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
                if self.check_valid_image(image):
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

    def single_upload(
        self,
        image_path=None,
        annotation_path=None,
        hosted_image=False,
        image_id=None,
        split="train",
        num_retry_uploads=0,
        batch_name=DEFAULT_BATCH_NAME,
        tag_names=[],
        is_prediction: bool = False,
        **kwargs,
    ):
        success = False
        annotation_success = False
        if image_path is not None:
            try:
                image_id = retry(
                    num_retry_uploads,
                    Exception,
                    self.__image_upload,
                    image_path,
                    hosted_image=hosted_image,
                    split=split,
                    batch_name=batch_name,
                    tag_names=tag_names,
                    **kwargs,
                )
                success = True
            except BaseException as e:
                print(
                    f"{image_path} ERROR uploading image after {num_retry_uploads} retries: {e}",
                    file=sys.stderr,
                )
                return

        # Upload only annotations to image based on image Id (no image)
        if annotation_path is not None and image_id is not None and success:
            # Get annotation upload response
            try:
                self.__annotation_upload(
                    annotation_path, image_id, is_prediction=is_prediction
                )
                annotation_success = True
            except BaseException as e:
                print(
                    f"{annotation_path} ERROR saving annotation: {e}", file=sys.stderr
                )
                return False
            # Give user warning that annotation failed to upload
            if not annotation_success:
                warnings.warn(
                    "Annotation, "
                    + annotation_path
                    + "failed to upload!\n Upload correct annotation file to image_id: "
                    + image_id
                )
        else:
            annotation_success = True

        overall_success = success and annotation_success
        return overall_success

    def search(
        self,
        like_image: str = None,
        prompt: str = None,
        offset: int = 0,
        limit: int = 100,
        tag: str = None,
        class_name: str = None,
        in_dataset: str = None,
        batch: bool = False,
        batch_id: str = None,
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
        """
        payload = {}

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
            API_URL
            + "/"
            + self.__workspace
            + "/"
            + self.__project_name
            + "/search?api_key="
            + self.__api_key,
            json=payload,
        )

        return data.json()["results"]

    def search_all(
        self,
        like_image: str = None,
        prompt: str = None,
        offset: int = 0,
        limit: int = 100,
        tag: str = None,
        class_name: str = None,
        in_dataset: str = None,
        batch: bool = False,
        batch_id: str = None,
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
        """
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
