from roboflow.core.model import Model
from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
import os


class Version():
    def __init__(self, type, api_key, dataset_slug, version, local):
        self.api_key = api_key
        self.dataset_slug = dataset_slug
        self.version_id = os.path.basename(version)

        if type == "object-detection":
            self.model = ObjectDetectionModel(self.api_key, self.dataset_slug, self.version_id, local=local)
        elif type == "classification":
            self.model = ClassificationModel(self.api_key, self.dataset_slug, self.version_id, local=local)



