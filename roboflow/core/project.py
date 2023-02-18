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

ACCEPTED_IMAGE_FORMATS = ["PNG", "JPEG"]


def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return str(msg) + "\n"


warnings.formatwarning = custom_formatwarning


class Project:
    def __init__(self, api_key, a_project, model_format=None):
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
        """Helper function to get version information from the REST API.
        :returns dictionary with information about all of the versions directly from the API.
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
        """Prints out versions for that specific project"""
        version_info = self.get_version_information()
        print(version_info)

    def versions(self):
        """function to return all versions in the project as Version objects.
        :returns an array of Version() objects.
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
            )
            version_array.append(version_object)
        return version_array

    def generate_version(self, settings):
        """
        Settings, a python dict with augmentation and preprocessing keys and specifications for generation.
        These settings mirror capabilities available via the Roboflow UI.
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

        Returns: The version number that is being generated
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
    ) -> bool:
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

        new_version = self.generate_version(settings=new_version_settings)
        new_version = self.version(new_version)
        new_version.train(speed=speed, checkpoint=checkpoint)

        return True

    def version(self, version_number, local=None):
        """Retrieves information about a specific version, and throws it into an object.
        :param version_number: the version number that you want to retrieve
        :local: specifies the localhost address and port if pointing towards local inference engine
        :return Version() object
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
                )
                return vers

        raise RuntimeError("Version number {} is not found.".format(version_number))

    def __image_upload(
        self,
        image_path,
        hosted_image=False,
        split="train",
        batch_name=DEFAULT_BATCH_NAME,
        tag_names=[],
        **kwargs,
    ):
        """function to upload image to the specific project
        :param image_path: path to image you'd like to upload.
        :param hosted_image: if the image is hosted online, then this should be modified
        :param split: the dataset split to upload the project to.
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
                    "https://api.roboflow.com/dataset/",
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
        # Return response

        return response

    def __annotation_upload(self, annotation_path: str, image_id: str):
        """function to upload annotation to the specific project
        :param annotation_path: path to annotation you'd like to upload
        :param image_id: image id you'd like to upload that has annotations for it.
        """

        # stop on empty string
        if len(annotation_path) == 0:
            print("Please provide a non-empty string for annotation_path.")
            return {"result": "Please provide a non-empty string for annotation_path."}

        # check if annotation file exists
        elif os.path.exists(annotation_path):
            print("-> found given annotation file")
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

        # Set annotation upload url
        self.annotation_upload_url = "".join(
            [
                API_URL + "/dataset/",
                self.name,
                "/annotate/",
                image_id,
                "?api_key=",
                self.__api_key,
                "&name=" + os.path.basename(annotation_path),
            ]
        )
        # Get annotation response
        annotation_response = requests.post(
            self.annotation_upload_url,
            data=annotation_string,
            headers={"Content-Type": "text/plain"},
        )
        # Return annotation response
        return annotation_response

    def check_valid_image(self, image_path):
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
        image_id: int = None,
        split: str = "train",
        num_retry_uploads: int = 0,
        batch_name: str = DEFAULT_BATCH_NAME,
        tag_names: list = [],
        **kwargs,
    ):
        """Upload image function based on the RESTful API

        Args:
            image_path (str) - path to image you'd like to upload
            annotation_path (str) - if you're upload annotation, path to it
            hosted_image (bool) - whether the image is hosted
            image_id (int) - id of the image
            split (str) - to upload the image to
            num_retry_uploads (int) - how many times to retry upload on failure
            batch_name (str) - name of batch to upload to within project
            tag_names (list[str]) - tags to be applied to an image

        Returns:
            None - returns nothing
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
        **kwargs,
    ):
        success = False
        annotation_success = False
        # User gives image path
        if image_path is not None:
            # Upload Image Response
            response = self.__image_upload(
                image_path,
                hosted_image=hosted_image,
                split=split,
                batch_name=batch_name,
                tag_names=tag_names,
                **kwargs,
            )
            # Get JSON response values
            try:
                if "duplicate" in response.json().keys():
                    if response.json()["duplicate"]:
                        success = True
                        warnings.warn("Duplicate image not uploaded:  " + image_path)
                else:
                    success, image_id = (
                        response.json()["success"],
                        response.json()["id"],
                    )

                if not success:
                    warnings.warn(f"Server rejected image: {response.json()}")

            except Exception:
                # Image fails to upload
                warnings.warn(f"Bad response: {response}")
                success = False
            # Give user warning that image failed to upload
            if not success:
                warnings.warn(
                    "Upload api failed with response: " + str(response.json())
                )
                if num_retry_uploads > 0:
                    warnings.warn(
                        "Image, "
                        + image_path
                        + ", failed to upload! Retrying for this many times: "
                        + str(num_retry_uploads)
                    )
                    self.single_upload(
                        image_path=image_path,
                        annotation_path=annotation_path,
                        hosted_image=hosted_image,
                        image_id=image_id,
                        split=split,
                        num_retry_uploads=num_retry_uploads - 1,
                        **kwargs,
                    )
                    return
                else:
                    warnings.warn(
                        "Image, "
                        + image_path
                        + ", failed to upload! You can specify num_retry_uploads to retry a number of times."
                    )

        # Upload only annotations to image based on image Id (no image)
        if annotation_path is not None and image_id is not None and success:
            # Get annotation upload response
            annotation_response = self.__annotation_upload(annotation_path, image_id)
            # Check if upload was a success
            try:
                annotation_success = annotation_response.json()["success"]
            except Exception:
                warnings.warn(f"Bad response: {response}")
                annotation_success = False
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

    def __str__(self):
        # String representation of project
        json_str = {"name": self.name, "type": self.type, "workspace": self.__workspace}

        return json.dumps(json_str, indent=2)
