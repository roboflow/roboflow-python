import requests
import responses
from responses.matchers import json_params_matcher

from roboflow import API_URL
from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError
from roboflow.config import DEFAULT_BATCH_NAME
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest


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

    def test_image_success(self):
        image_id = "test-image-id"
        expected_url = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/images/{image_id}?api_key={ROBOFLOW_API_KEY}"
        mock_response = {
            "image": {
                "id": image_id,
                "name": "test_image.jpg",
                "annotation": {
                    "key": "some-key",
                    "width": 640,
                    "height": 480,
                    "boxes": [{"label": "person", "x": 100, "y": 150, "width": 50, "height": 80}],
                },
                "labels": ["person"],
                "split": "train",
                "tags": ["tag1", "tag2"],
                "created": 1616161616,
                "urls": {
                    "original": "https://example.com/image.jpg",
                    "thumb": "https://example.com/thumb.jpg",
                    "annotation": "https://example.com/annotation.json",
                },
                "embedding": [0.1, 0.2, 0.3],
            }
        }

        responses.add(responses.GET, expected_url, json=mock_response, status=200)

        image_details = self.project.image(image_id)

        self.assertIsInstance(image_details, dict)
        self.assertEqual(image_details["id"], image_id)
        self.assertEqual(image_details["name"], "test_image.jpg")
        self.assertIn("annotation", image_details)
        self.assertIn("labels", image_details)
        self.assertEqual(image_details["split"], "train")

    def test_image_not_found(self):
        image_id = "nonexistent-image-id"
        expected_url = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/images/{image_id}?api_key={ROBOFLOW_API_KEY}"
        mock_response = {"error": "Image not found."}

        responses.add(responses.GET, expected_url, json=mock_response, status=404)

        with self.assertRaises(RuntimeError) as context:
            self.project.image(image_id)

            self.assertIn("HTTP error occurred while fetching image details", str(context.exception))

    def test_image_invalid_json_response(self):
        image_id = "invalid-json-image-id"
        expected_url = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/images/{image_id}?api_key={ROBOFLOW_API_KEY}"
        invalid_json = "Invalid JSON response"

        responses.add(responses.GET, expected_url, body=invalid_json, status=200)

        with self.assertRaises(requests.exceptions.JSONDecodeError) as context:
            self.project.image(image_id)

        self.assertIn("Expecting value", str(context.exception))

    def test_create_annotation_job_success(self):
        job_name = "Test Job"
        batch_id = "test-batch-123"
        num_images = 10
        labeler_email = "labeler@example.com"
        reviewer_email = "reviewer@example.com"

        expected_response = {
            "success": True,
            "job": {
                "id": "job-123",
                "name": job_name,
                "batch": batch_id,
                "num_images": num_images,
                "labeler": labeler_email,
                "reviewer": reviewer_email,
                "status": "created",
                "created": 1616161616,
            },
        }

        expected_url = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/jobs?api_key={ROBOFLOW_API_KEY}"

        responses.add(
            responses.POST,
            expected_url,
            json=expected_response,
            status=200,
            match=[
                json_params_matcher(
                    {
                        "name": job_name,
                        "batch": batch_id,
                        "num_images": num_images,
                        "labelerEmail": labeler_email,
                        "reviewerEmail": reviewer_email,
                    }
                )
            ],
        )

        result = self.project.create_annotation_job(
            name=job_name,
            batch_id=batch_id,
            num_images=num_images,
            labeler_email=labeler_email,
            reviewer_email=reviewer_email,
        )

        self.assertEqual(result, expected_response)
        self.assertTrue(result["success"])
        self.assertEqual(result["job"]["id"], "job-123")
        self.assertEqual(result["job"]["name"], job_name)

    def test_create_annotation_job_error(self):
        job_name = "Test Job"
        batch_id = "invalid-batch"
        num_images = 10
        labeler_email = "labeler@example.com"
        reviewer_email = "reviewer@example.com"

        error_response = {"error": "Batch not found"}

        expected_url = f"{API_URL}/{WORKSPACE_NAME}/{PROJECT_NAME}/jobs?api_key={ROBOFLOW_API_KEY}"

        responses.add(responses.POST, expected_url, json=error_response, status=404)

        with self.assertRaises(RuntimeError) as context:
            self.project.create_annotation_job(
                name=job_name,
                batch_id=batch_id,
                num_images=num_images,
                labeler_email=labeler_email,
                reviewer_email=reviewer_email,
            )

        self.assertEqual(str(context.exception), "Batch not found")
