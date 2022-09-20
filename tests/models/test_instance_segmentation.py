import unittest

import responses

from roboflow.config import INSTANCE_SEGMENTATION_URL
from roboflow.models.instance_segmentation import InstanceSegmentationModel
from roboflow.util.prediction import PredictionGroup


class TestInstanceSegmentation(unittest.TestCase):

    api_key = 'my-api-key'
    workspace = 'roboflow'
    dataset_id = 'test-123'
    version = '23'

    def setUp(self):
        super(TestInstanceSegmentation, self).setUp()
        self.version_id = f'{self.workspace}/{self.dataset_id}/{self.version}'

    def test_init_sets_attributes(self):
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        self.assertEqual(instance.id, self.version_id)
        self.assertEqual(instance.api_url, f'{INSTANCE_SEGMENTATION_URL}/{self.dataset_id}/{self.version}')

    @responses.activate
    def test_predict_with_local_image(self):
        image_path = 'tests/images/rabbit.JPG'
        instance = InstanceSegmentationModel(self.api_key, self.version_id)

        responses.add(responses.POST, instance.api_url, json={
            "predictions": [
                {
                    "x": 812.0,
                    "y": 362.9,
                    "width": 277,
                    "height": 206,
                    "class": "J",
                    "confidence": 0.598,
                    "points": [
                        {
                            "x": 831.0,
                            "y": 527.0
                        },
                        {
                            "x": 931.0,
                            "y": 389.0
                        },
                        {
                            "x": 831.0,
                            "y": 527.0
                        }
                    ]
                },
                {
                    "x": 363.8,
                    "y": 665.5,
                    "width": 707,
                    "height": 669,
                    "class": "K",
                    "confidence": 0.52,
                    "points": [
                        {
                            "x": 131.0,
                            "y": 999.0
                        },
                        {
                            "x": 269.0,
                            "y": 666.0
                        },

                        {
                            "x": 131.0,
                            "y": 999.0
                        }
                    ]
                }
            ],
            "image": {
                "width": 1333,
                "height": 1000
            }
        })

        group = instance.predict(image_path)
        self.assertIsInstance(group, PredictionGroup)
