import responses

from roboflow import API_URL
from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError
from roboflow.config import DEFAULT_BATCH_NAME
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, RoboflowTest


class TestProject(RoboflowTest):
    def test_check_valid_image_with_accepted_formats(self):
        images_to_test = [
            "rabbit.JPG",
            "rabbit2.jpg",
            "hand-rabbit.PNG",
            "woodland-rabbit.png",
        ]

        for image in images_to_test:
            self.assertTrue(self.project.check_valid_image(f"tests/images/{image}"))

    def test_check_valid_image_with_unaccepted_formats(self):
        images_to_test = [
            "sky-rabbit.gif",
            "sky-rabbit.heic",
        ]

        for image in images_to_test:
            self.assertFalse(self.project.check_valid_image(f"tests/images/{image}"))

    def test_upload_raises_upload_image_error(self):
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={
                "error": {
                    "message": "Invalid image.",
                    "type": "InvalidImageException",
                    "hint": "This image was already annotated; to overwrite the annotation, pass overwrite=true...",
                }
            },
            status=400,
        )

        with self.assertRaises(ImageUploadError) as error:
            self.project.upload(
                "tests/images/rabbit.JPG",
                annotation_path="tests/annotations/valid_annotation.json",
            )

        self.assertEqual(str(error.exception), "Invalid image.")

    def test_upload_raises_upload_annotation_error(self):
        image_id = "hbALkCFdNr9rssgOUXug"
        image_name = "invalid_annotation.json"

        # Image upload
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={"success": True, "id": image_id},
            status=200,
        )

        # Annotation
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/annotate/{image_id}?api_key={ROBOFLOW_API_KEY}" f"&name={image_name}",
            json={
                "error": {
                    "message": "Image was already annotated.",
                    "type": "InvalidImageException",
                    "hint": "This image was already annotated; to overwrite the annotation, pass overwrite=true...",
                }
            },
            status=400,
        )

        with self.assertRaises(AnnotationSaveError) as error:
            self.project.upload(
                "tests/images/rabbit.JPG",
                annotation_path=f"tests/annotations/{image_name}",
            )

        self.assertEqual(str(error.exception), "Image was already annotated.")
