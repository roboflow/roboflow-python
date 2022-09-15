import os
import unittest

import roboflow

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY")
WORKSPACE_NAME = os.environ.get("WORKSPACE_NAME", "")
PROJECT_NAME = os.environ.get("PROJECT_NAME")
PROJECT_VERSION = os.environ.get("PROJECT_VERSION", "1")


def make_orderer():
    order = {}

    def ordered(f):
        order[f.__name__] = len(order)
        return f

    def compare(a, b):
        if a not in order or b not in order:
            return 1

        return [1, -1][order[a] < order[b]]

    return ordered, compare


ordered, compare = make_orderer()

unittest.defaultTestLoader.sortTestMethodsUsing = compare


class RoboflowTest(unittest.TestCase):
    rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace()
    project = workspace.project(PROJECT_NAME)
    version = project.version(PROJECT_VERSION)
