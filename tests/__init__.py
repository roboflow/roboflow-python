import os
import unittest

import responses

import roboflow
from roboflow.config import API_URL

ROBOFLOW_API_KEY = "my-test-app"
WORKSPACE_NAME = "my-workspace"
PROJECT_NAME = "test-project"
PROJECT_VERSION = "1"


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

    def setUp(self):
        super(RoboflowTest, self).setUp()
        responses.start()

        # Check key
        responses.add(
            responses.POST,
            f"{API_URL}/?api_key={ROBOFLOW_API_KEY}",
            json={
                "welcome": "Welcome to the Roboflow API.",
                "instructions": "You are successfully authenticated.",
                "docs": "https://docs.roboflow.com",
                "workspace": WORKSPACE_NAME
            },
            status=200
        )

        # Get workspace
        responses.add(
            responses.GET,
            f"{API_URL}/{WORKSPACE_NAME}?api_key={ROBOFLOW_API_KEY}",
            json={
                'workspace': {
                    'name': WORKSPACE_NAME,
                    'url': WORKSPACE_NAME,
                    'members': 1,
                    'projects': [
                        {'id': f'{WORKSPACE_NAME}/{PROJECT_NAME}', 'type': 'object-detection', 'name': 'Hard Hat Sample', 'created': 1593802673.521, 'updated': 1663269501.654, 'images': 100, 'unannotated': 3, 'annotation': 'Workers', 'versions': 2, 'public': False, 'splits': {'train': 70, 'test': 10, 'valid': 20}, 'colors': {'head': '#8622FF', 'person': '#FF00FF', 'helmet': '#C7FC00'}, 'classes': {'person': 9, 'helmet': 287, 'head': 90}}
                    ]
                }
            },
            status=200
        )

        # Get project
        responses.add(
            responses.GET,
            f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}?api_key={ROBOFLOW_API_KEY}",
            json={
                'workspace': {
                    'name': WORKSPACE_NAME,
                    'url': WORKSPACE_NAME,
                    'members': 1
                },
                'project': {
                    'id': f'{WORKSPACE_NAME}/{PROJECT_NAME}', 'type': 'object-detection', 'name': 'Hard Hat Sample', 'created': 1593802673.521, 'updated': 1663269501.654, 'images': 100, 'unannotated': 3, 'annotation': 'Workers', 'versions': 2, 'public': False, 'splits': {'test': 10, 'train': 70, 'valid': 20}, 'colors': {'person': '#FF00FF', 'helmet': '#C7FC00', 'head': '#8622FF'}, 'classes': {'person': 9, 'helmet': 287, 'head': 90}
                },
                'versions': [
                    {'id': f'{WORKSPACE_NAME}/{PROJECT_NAME}/2', 'name': 'augmented-416x416', 'created': 1663104679.539, 'images': 240, 'splits': {'train': 210, 'test': 10, 'valid': 20}, 'preprocessing': {'resize': {'height': '416', 'enabled': True, 'width': '416', 'format': 'Stretch to'}, 'auto-orient': {'enabled': True}}, 'augmentation': {'blur': {'enabled': True, 'pixels': 1.5}, 'image': {'enabled': True, 'versions': 3}, 'rotate': {'degrees': 15, 'enabled': True}, 'crop': {'enabled': True, 'percent': 40, 'min': 0}, 'flip': {'horizontal': True, 'enabled': True, 'vertical': False}}, 'exports': []},
                    {'id': f'{WORKSPACE_NAME}/{PROJECT_NAME}/1', 'name': 'raw', 'created': 1663104679.538, 'images': 100, 'splits': {'train': 70, 'test': 10, 'valid': 20}, 'preprocessing': {}, 'augmentation': {}, 'exports': []}
                ]
            },
            status=200
        )

        # Upload image
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}",
            json={'duplicate': True, 'id': 'hbALkCFdNr9rssgOUXug'},
            status=200
        )

        self.connect_to_roboflow()

    def tearDown(self):
        super(RoboflowTest, self).tearDown()
        responses.stop()
        responses.reset()

    def connect_to_roboflow(self):
        self.rf = roboflow.Roboflow(api_key=ROBOFLOW_API_KEY)
        self.workspace = self.rf.workspace()
        self.project = self.workspace.project(PROJECT_NAME)
        self.version = self.project.version(PROJECT_VERSION)
