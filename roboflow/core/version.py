import pathlib
import urllib
import warnings

import cv2
import requests
from PIL import Image

from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
from roboflow.config import *

class Version():
    def __init__(self, a_version):
        self.version_id = a_version['id']
        self.model = a_version['model']