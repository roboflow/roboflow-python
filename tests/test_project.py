import responses

from roboflow import API_URL
from roboflow.config import DEFAULT_BATCH_NAME
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, RoboflowTest


class TestProject(RoboflowTest):
    def test_check_valid_image_with_accepted_formats(self):
        # Mock dataset upload
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={"duplicate": True, "id": "hbALkCFdNr9rssgOUXug"},
            status=200,
        )

        images_to_test = [
            "rabbit.JPG",
            "rabbit2.jpg",
            "hand-rabbit.PNG",
            "woodland-rabbit.png",
        ]

        for image in images_to_test:
            self.assertTrue(self.project.check_valid_image(f"tests/images/{image}"))

    def test_check_valid_image_with_unaccepted_formats(self):
        # Mock dataset upload
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={"duplicate": True, "id": "hbALkCFdNr9rssgOUXug"},
            status=200,
        )

        images_to_test = [
            "sky-rabbit.gif",
            "sky-rabbit.heic",
        ]

        for image in images_to_test:
            self.assertFalse(self.project.check_valid_image(f"tests/images/{image}"))
