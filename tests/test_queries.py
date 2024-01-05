from _datetime import datetime

from roboflow.core.project import Project
from roboflow.core.version import Version
from roboflow.models.classification import ClassificationModel
from roboflow.models.object_detection import ObjectDetectionModel
from tests import PROJECT_NAME, RoboflowTest, ordered


class TestQueries(RoboflowTest):
    @ordered
    def test_workspace_fields(self):
        self.assertTrue(isinstance(self.workspace.name, str))
        self.assertTrue(isinstance(self.workspace.name, str))
        self.assertTrue(isinstance(self.workspace.project_list, list))
        self.assertTrue(isinstance(self.workspace.members, int))

    @ordered
    def test_workspace_methods(self):
        print_projects = self.workspace.list_projects()
        project_array = self.workspace.projects()
        project_obj = self.workspace.project(PROJECT_NAME)

        self.assertIsNone(print_projects)
        self.assertTrue(isinstance(project_array, list))
        self.assertTrue(isinstance(project_obj, Project))

    @ordered
    def test_project_fields(self):
        self.assertTrue(isinstance(self.project.annotation, str))
        self.assertTrue(isinstance(self.project.classes, dict))
        self.assertTrue(isinstance(self.project.colors, dict))
        self.assertTrue(isinstance(self.project.created, datetime))
        self.assertTrue(isinstance(self.project.id, str))
        self.assertTrue(isinstance(self.project.images, int))
        self.assertTrue(isinstance(self.project.public, bool))
        self.assertTrue(isinstance(self.project.splits, dict))
        self.assertTrue(isinstance(self.project.type, str))
        self.assertTrue(isinstance(self.project.updated, datetime))

    @ordered
    def test_project_methods(self):
        version_information = self.project.get_version_information()
        print_versions = self.project.list_versions()
        list_versions = self.project.versions()
        upload = self.project.upload("tests/images/rabbit2.jpg")

        self.assertEqual(len(version_information), 2)
        self.assertIsNone(print_versions)
        self.assertTrue(all(map(lambda x: isinstance(x, Version), list_versions)))
        self.assertIsNone(upload)

    @ordered
    def test_version_fields(self):
        self.assertTrue(isinstance(self.version.name, str))
        self.assertTrue(isinstance(self.version.version, str))
        self.assertTrue(isinstance(self.version.type, str))
        self.assertTrue(isinstance(self.version.augmentation, dict))
        self.assertTrue(isinstance(self.version.created, float))
        self.assertTrue(isinstance(self.version.id, str))
        self.assertTrue(isinstance(self.version.images, int))
        self.assertTrue(isinstance(self.version.preprocessing, dict))
        self.assertTrue(isinstance(self.version.splits, dict))

    @ordered
    def test_version_methods(self):
        self.assertTrue(
            (
                isinstance(self.version.model, ClassificationModel)
                or (isinstance(self.version.model, ObjectDetectionModel))
            )
        )
