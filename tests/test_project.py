import responses

from roboflow import API_URL
from roboflow.config import DEFAULT_BATCH_NAME
from roboflow.core.exceptions import UploadImageError
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

    def test_upload_image_that_already_is_annotated_raises_upload_image_error(self):
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={
                "message": "Image was already annotated.",
                "type": "InvalidImageException",
                "hint": "This image was already annotated; to overwrite the annotation, pass overwrite=true...",
            },
            status=200,
        )

        with self.assertRaises(UploadImageError) as error:
            self.project.upload(
                "tests/images/rabbit.JPG",
                annotation_path="tests/annotations/valid_annotation.json",
            )

        self.assertEqual(str(error.exception), "Error uploading image: Image was already annotated.")
