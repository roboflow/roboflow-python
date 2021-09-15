import os
import roboflow
from roboflow.core.project import Project
from roboflow.core.version import Version
import unittest
from _datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def make_orderer():
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        return [1, -1][order[a] < order[b]]

    return ordered, compare


ordered, compare = make_orderer()
unittest.defaultTestLoader.sortTestMethodsUsing = compare


class TestQueries(unittest.TestCase):
    rf = roboflow.Roboflow(api_key="")
    workspace = rf.workspace("")
    project = workspace.project("")
    """
    TEST QUERIES

    Tests some queries in queries.py
    """
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
        project_obj = self.workspace.project("")

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
        upload = self.project.upload("")

        self.assertTrue(len(version_information) == 2)
        self.assertIsNone(print_versions)
        self.assertTrue(all(map(lambda x: isinstance(x, Version), list_versions)))
        self.assertIsNone(upload)

if __name__ == '__main__':
    unittest.main()

