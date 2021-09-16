from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
import os
import json


class Version():

    def __init__(self, version_dict, type, api_key, name, version, local):
        """Version object that stores information about a specific version
        :param version_dict: dictionary returned from the API containing most of the information
        :param type: type of task (object detection vs classification)
        :param api_key: private roboflow key
        :param name: is the name of the version
        :param version: the version number
        :param local: whether the version is stored locally
        :return a Version object
        """
        self.__api_key = api_key
        self.name = name
        self.version = version
        self.type = type
        self.augmentation = version_dict['augmentation']
        self.created = version_dict['created']
        self.id = version_dict['id']
        self.images = version_dict['images']
        self.preprocessing = version_dict['preprocessing']
        self.splits = version_dict['splits']

        version_without_workspace = os.path.basename(version)

        if self.type == "object-detection":
            self.model = ObjectDetectionModel(self.__api_key, self.id, self.name, version_without_workspace, local=local)
        elif self.type == "classification":
            self.model = ClassificationModel(self.__api_key, self.id, self.name, version_without_workspace, self.id, local=local)
        else:
            self.model = None

    def __str__(self):
        """string representation of version object.
        """
        json_value = {
            'name': self.name,
            'type': self.type,
            'version': self.version,
            'augmentation': self.augmentation,
            'created': self.created,
            'preprocessing': self.preprocessing,
            'splits': self.splits}
        return json.dumps(json_value, indent=2)
