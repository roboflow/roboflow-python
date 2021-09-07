import base64
import io
import json
import os
import pathlib
import urllib
import warnings

import cv2
import requests
from PIL import Image

from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.core.version import Version

from roboflow.config import *

#version class that should return

class Project():
    def __init__(self, api_key, dataset_slug, type, versions):
        self.api_key = api_key
        self.dataset_slug = dataset_slug
        self.type = type
        # Dictionary of versions + names
        self.versions_and_names = versions
        # List of all versions to choose from

        version_array = []

        for a_version in versions:
            version_object = Version(a_version)
            version_array.append(version_object)

        self.all_versions = version_array


    def version(self, version_number):
        print(self.versions)
        for version_object in self.all_versions:
            id = version_object.version_id
            if id or os.path.basename(id) == version_number:
                return version_object

        raise RuntimeError("Version number {} is not found.".format(version_number))

    def versions(self):
        return self.versions

    def model(self, version, local=False):

        # Check if version number is an available version to choose from
        if version not in self.versions:
            raise RuntimeError(
                str(version) + " is an invalid version; please select a different version from " + str(self.versions))

        dataset_name = os.path.basename(self.dataset_slug)

        # Check whether model exists before initializing model
        model_info_response = requests.get(
            API_URL + "/model/" + dataset_name + "/" + str(version) + "?api_key=" + self.api_key)

        if model_info_response.status_code != 200:
            raise RuntimeError(model_info_response.text)

        model_info_response = model_info_response.json()

        # Return appropriate model if model does exist
        if model_info_response['exists']:
            if self.type == "object-detection":
                return ObjectDetectionModel(self.api_key, self.dataset_slug, version, local=local)
            elif self.type == "classification":
                return ClassificationModel(self.api_key, self.dataset_slug, version, local=local)

    def __image_upload(self, image_path, hosted_image=False, split="train"):



        # If image is not a hosted image
        if not hosted_image:
            project_name = os.path.basename(self.dataset_slug)
            image_name = os.path.basename(image_path)
            # Construct URL for local image upload
            self.image_upload_url = "".join([
                "https://api.roboflow.com/dataset/", project_name, "/upload",
                "?api_key=", self.api_key,
                "&name=" + image_name,
                "&split=" + split
            ])

            # Convert to PIL Image
            img = cv2.imread(image_path)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            pilImage = Image.fromarray(image)
            # Convert to JPEG Buffer
            buffered = io.BytesIO()
            pilImage.save(buffered, quality=100, format="JPEG")
            # Base 64 Encode
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode("ascii")
            # Post Base 64 Data to Upload API

            response = requests.post(self.image_upload_url, data=img_str, headers={
                "Content-Type": "application/x-www-form-urlencoded"
            })


        else:
            # Hosted image upload url
            upload_url = "".join([
                "https://api.roboflow.com/dataset/" + self.dataset_slug + "/upload",
                "?api_key=" + self.api_key,
                "&name=" + os.path.basename(image_path),
                "&split=" + split,
                "&image=" + urllib.parse.quote_plus(image_path)
            ])
            # Get response
            response = requests.post(upload_url)
        # Return response

        return response

    def __annotation_upload(self, annotation_path, image_id):
        # Get annotation string
        annotation_string = open(annotation_path, "r").read()
        # Set annotation upload url
        self.annotation_upload_url = "".join([
            "https://api.roboflow.com/dataset/", self.dataset_slug, "/annotate/", image_id,
            "?api_key=", self.api_key,
            "&name=" + os.path.basename(annotation_path)
        ])
        # Get annotation response
        annotation_response = requests.post(self.annotation_upload_url, data=annotation_string, headers={
            "Content-Type": "text/plain"
        })
        # Return annotation response
        return annotation_response

    def upload(self, image_path=None, annotation_path=None, hosted_image=False, image_id=None, split='train'):
        success = False
        # User gives image path
        if image_path is not None:
            # Upload Image Response
            response = self.__image_upload(image_path, hosted_image=hosted_image, split=split)

            # Get JSON response values
            try:
                success, image_id = response.json()['success'], response.json()['id']
            except Exception:
                # Image fails to upload
                success = False
            # Give user warning that image failed to upload
            if not success:
                warnings.warn("Image, " + image_path + ", failed to upload!")

            # Check if image uploaded successfully + check if there are annotations to upload
            if annotation_path is not None and image_id is not None and success:
                # Upload annotation to API
                annotation_response = self.__annotation_upload(annotation_path, image_id)
                try:
                    success = annotation_response.json()['success']
                except Exception:
                    success = False
            # Give user warning that annotation failed to upload
            if not success:
                warnings.warn("Annotation, " + annotation_path + ", failed to upload!")
        # Upload only annotations to image based on image Id (no image)
        elif annotation_path is not None and image_id is not None:
            # Get annotation upload response
            annotation_response = self.__annotation_upload(annotation_path, image_id)
            # Check if upload was a success
            try:
                success = annotation_response.json()['success']
            except Exception:
                success = False
        # Give user warning that annotation failed to upload
        if not success:
            warnings.warn("Annotation, " + annotation_path + ", failed to upload!")

    def __str__(self):
        # String representation of project
        json_str = {
            "dataset_slug": self.dataset_slug,
            "dataset_type": self.type,
            "dataset_versions": self.versions_and_names
        }

        return json.dumps(json_str, indent=2)
