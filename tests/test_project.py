import responses

from roboflow import API_URL
from roboflow.config import DEFAULT_BATCH_NAME
from roboflow.core.exceptions import UploadAnnotationError, UploadImageError
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

    def test_upload_raises_upload_image_error_response_200(self):
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={
                "message": "Invalid Image",
                "type": "InvalidImageException",
            },
            status=200,
        )

        with self.assertRaises(UploadImageError) as error:
            self.project.upload(
                "tests/images/rabbit.JPG",
                annotation_path="tests/annotations/valid_annotation.json",
            )

        self.assertEqual(str(error.exception), "Error uploading image: Invalid Image")

    def test_upload_raises_upload_image_error_response_400(self):
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}" f"&batch={DEFAULT_BATCH_NAME}",
            json={
                "message": "Invalid Image",
                "type": "InvalidImageException",
            },
            status=400,
        )

        with self.assertRaises(UploadImageError) as error:
            self.project.upload(
                "tests/images/rabbit.JPG",
                annotation_path="tests/annotations/valid_annotation.json",
            )

        self.assertEqual(str(error.exception), "Error uploading image: Invalid Image")

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
                "message": "Image was already annotated.",
                "type": "InvalidImageException",
                "hint": "This image was already annotated; to overwrite the annotation, pass overwrite=true...",
            },
            status=400,
        )

        with self.assertRaises(UploadAnnotationError) as error:
            self.project.upload(
                "tests/images/rabbit.JPG",
                annotation_path=f"tests/annotations/{image_name}",
            )

        self.assertEqual(str(error.exception), "Error uploading annotation: Image was already annotated.")
