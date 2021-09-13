from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
import os
import json


class Version():
    def __init__(self, type, api_key, dataset_slug, version, local):
        self.api_key = api_key
        self.name = dataset_slug
        self.version = version
        self.category = type

        version_without_workspace = os.path.basename(version)

        if self.category == "object-detection":
            self.model = ObjectDetectionModel(self.api_key, self.name, version_without_workspace, local=local)
        elif self.category == "classification":
            self.model = ClassificationModel(self.api_key, self.name, version_without_workspace, local=local)
        else:
            self.model = None

    def __str__(self):
        json_value = {'api_key': self.api_key,
                      'name': self.name,
                      'model_type': str(self.model),
                      'version': self.version}
        return json.dumps(json_value, indent=2)



