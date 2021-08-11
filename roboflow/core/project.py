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
        self.versions = list(int(vers) for vers in versions.keys())

    def model(self, version):
        # Check if version number is an available version to choose from
        if version not in self.versions:
            raise RuntimeError(
                str(version) + " is an invalid version; please select a different version from " + str(self.versions))

        # Currently uses TFJS endpoint to figure out whether model exists
        # TODO: Write a endpoint to check if model exists
        # Check whether model exists before initializing model
        model_info_response = requests.get(
            API_URL + "/model/" + self.dataset_slug + "/" + str(version) + "?access_token=" + self.access_token)
        if model_info_response.status_code != 200:
            raise RuntimeError(model_info_response.text)

        model_info_response = model_info_response.json()
        # Return appropriate model if model does exist
        if model_info_response['exists']:
            if self.type == "object-detection":
                return ObjectDetectionModel(self.api_key, self.dataset_slug, version)
            elif self.type == "classification":
                return ClassificationModel(self.api_key, self.dataset_slug, version)

    def __image_upload(self, image_path, hosted_image=False, split="train"):
        # If image is not a hosted image
        if not hosted_image:
            # Construct URL for local image upload
            self.image_upload_url = "".join([
                "https://api.roboflow.com/dataset/", self.dataset_slug, "/upload",
                "?api_key=", self.api_key,
                "&name=" + os.path.basename(image_path),
                "&split=" + split
            ])
            # Convert to PIL Image
            image = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)
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
            upload_url = "".join([
                "https://api.roboflow.com/dataset/" + self.dataset_slug + "/upload",
                "?api_key=" + self.api_key,
                "&name=" + os.path.basename(image_path),
                "&split=" + split,
                "&image=" + urllib.parse.quote_plus(image_path)
            ])

            response = requests.post(upload_url)

        return response

    def __annotation_upload(self, annotation_path, image_id):
        annotation_string = open(annotation_path, "r").read()

        self.annotation_upload_url = "".join([
            "https://api.roboflow.com/dataset/", self.dataset_slug, "/annotate/", image_id,
            "?api_key=", self.api_key,
            "&name=" + os.path.basename(annotation_path)
        ])

        annotation_response = requests.post(self.annotation_upload_url, data=annotation_string, headers={
            "Content-Type": "text/plain"
        })

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
        json_str = {
            "dataset_slug": self.dataset_slug,
            "dataset_type": self.type,
            "dataset_versions": self.versions_and_names
        }

        return json.dumps(json_str, indent=2)
