from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
import os
import json


class Version():
    def __init__(self, type, api_key, dataset_slug, version, local):
        self.api_key = api_key
        self.dataset_slug = dataset_slug
        self.version_id = version


        version_without_workspace = os.path.basename(version)

        if type == "object-detection":
            self.model = ObjectDetectionModel(self.api_key, self.dataset_slug, version_without_workspace, local=local)
        elif type == "classification":
            self.model = ClassificationModel(self.api_key, self.dataset_slug, version_without_workspace, local=local)
        else:
            self.model = None

    def __str__(self):
        json_value = {'api_key': self.api_key,
                      'model_type': str(self.model),
                      'version': self.version_id}
        return json.dumps(json_value, indent=2)



