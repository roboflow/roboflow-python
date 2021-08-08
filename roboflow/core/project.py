import base64
import io
import json
import os
import pathlib
import urllib

import cv2
import requests
from PIL import Image

from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.config import *


class Project():
    def __init__(self, api_key, dataset_slug, type, versions, access_token, publishable_key):
        self.api_key = api_key
        self.dataset_slug = dataset_slug
        self.type = type
        self.access_token = access_token
        self.publishable_key = publishable_key
        # Dictionary of versions + names
        self.versions_and_names = versions
        # List of all versions to choose from
        self.versions = versions.keys()

    def model(self, version):
        # Check if version number is an available version to choose from
        if str(version) not in self.versions:
            raise RuntimeError(
                version + " is an invalid version; please select a different version from " + str(self.versions))

        # Currently uses TFJS endpoint to figure out whether model exists
        # TODO: (Optional) Consider writing a separate endpoint for this to keep this organized
        # Check whether model exists before initializing model
        model_info_response = requests.get(
            API_URL + "/tfjs/" + self.dataset_slug + "/" + str(version) + "?publishable_key=" + self.publishable_key)
        if model_info_response.status_code != 200:
            raise RuntimeError(model_info_response.text)

        # Return appropriate model if model does exist
        if self.type == "object-detection":
            return ObjectDetectionModel(self.api_key, self.dataset_slug, version)
        elif self.type == "classification":
            return ClassificationModel(self.api_key, self.dataset_slug, version)

    def upload(self, image_path, annotation_path=None, hosted_image=False, split='train'):
        success = False
        image_id = None
        image_type = pathlib.Path(image_path).suffix
        if not hosted_image:
            self.image_upload_url = "".join([
                "https://api.roboflow.com/dataset/", self.dataset_slug, "/upload",
                "?api_key=", self.api_key,
                "&name=" + os.path.basename(image_path),
                "&split=" + split
            ])
            image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
            pilImage = Image.fromarray(image)
            # Convert to JPEG Buffer
            buffered = io.BytesIO()
            pilImage.save(buffered, quality=100, format="JPEG")

            # Base 64 Encode
            img_str = base64.b64encode(buffered.getvalue())
            img_str = img_str.decode("ascii")

            response = requests.post(self.image_upload_url, data=img_str, headers={
                "Content-Type": "application/x-www-form-urlencoded"
            })
            success, image_id = response.json()['success'], response.json()['id']

        else:
            upload_url = "".join([
                "https://api.roboflow.com/dataset/" + self.dataset_slug + "/upload",
                "?api_key=" + self.api_key,
                "&name=" + os.path.basename(image_path),
                "&split=" + split,
                "&image=" + urllib.parse.quote_plus(image_path)
            ])

            response = requests.post(upload_url)
            success, image_id = response.json()['success'], response.json()['id']

        # To upload annotations
        if annotation_path is not None and image_id is not None and success:
            annotation_string = open(annotation_path, "r").read()

            self.annotation_upload_url = "".join([
                "https://api.roboflow.com/dataset/", self.dataset_slug, "/annotate/", image_id,
                "?api_key=", self.api_key,
                "&name=" + os.path.basename(annotation_path)
            ])

            annotation_response = requests.post(self.annotation_upload_url, data=annotation_string, headers={
                "Content-Type": "text/plain"
            }).json()

            success = annotation_response['success']

    def __str__(self):
        json_str = {
            "dataset_slug": self.dataset_slug,
            "dataset_type": self.type,
            "dataset_versions": self.versions_and_names
        }

        return json.dumps(json_str, indent=2)
