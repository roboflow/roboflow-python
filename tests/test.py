import os
import roboflow
from roboflow.core.project import Project
import unittest

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
    rf = roboflow.Roboflow(api_key="MY_KEY")
    workspace = rf.workspace("A_WORKSPACE")
    """
    TEST QUERIES

    Tests some queries in queries.py
    """

    # Tests whether regression_ann_query works without errors, and creates a key in models dictionary
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
        project_obj = self.workspace.project("PROJECT_NAME")

        self.assertIsNone(print_projects)
        self.assertTrue(isinstance(project_array, list))
        self.assertTrue(isinstance(project_obj, Project))



if __name__ == '__main__':
    unittest.main()

