import base64
import io
import os
import pathlib
import urllib

import cv2
import requests
from PIL import Image

from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel


class Project():
    def __init__(self, api_key, dataset_slug, type, exports):
        # TODO: Write JS endpoint to get all this Project info
        self.api_key = api_key
        self.dataset_slug = dataset_slug
        self.type = type
        # List of all versions
        self.exports = exports

    def model(self, version):
        if version not in self.exports:
            raise RuntimeError(
                version + " is an invalid version; please export a different version from " + str(self.exports))
        # TODO: Write JS endpoint to get model info
        # Check whether model exists before initializing model
        MODEL_INFO_ENDPOINT = "" + version
        model_info = requests.get(MODEL_INFO_ENDPOINT).json()
        if not model_info['exists']:
            raise RuntimeError("Model does not exist for this version (" + version + ")")

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
