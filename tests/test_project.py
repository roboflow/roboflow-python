from unittest.mock import patch

import requests
import responses
from responses.matchers import json_params_matcher

from roboflow import API_URL
from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError
from roboflow.config import DEFAULT_BATCH_NAME
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest, ordered


class TestProject(RoboflowTest):
    def _create_test_dataset(self, images=None):
        """
        Create a test dataset with specified images or a default image

        Args:
            images: List of image dictionaries. If None, a default image will be used.

        Returns:
            Dictionary representing a parsed dataset
        """
        if images is None:
            images = [{"file": "image1.jpg", "split": "train", "annotationfile": {"file": "image1.xml"}}]

        return {"location": "/test/location/", "images": images}

    def _setup_upload_dataset_mocks(
        self,
        test_dataset=None,
        image_return=None,
        annotation_return=None,
        project_created=False,
        save_annotation_side_effect=None,
        upload_image_side_effect=None,
    ):
        """
        Set up common mocks for upload_dataset tests

        Args:
            test_dataset: The dataset to return from parsefolder. If None, creates a default dataset
            image_return: Return value for upload_image. Default is successful upload
            annotation_return: Return value for save_annotation. Default is successful annotation
            project_created: Whether to simulate a newly created project
            save_annotation_side_effect: Side effect function for save_annotation
            upload_image_side_effect: Side effect function for upload_image

        Returns:
            Dictionary of mock objects with start and stop methods
        """
        if test_dataset is None:
            test_dataset = self._create_test_dataset()

        if image_return is None:
            image_return = ({"id": "test-id", "success": True}, 0.1, 0)

        if annotation_return is None:
            annotation_return = ({"success": True}, 0.1, 0)

        # Create the mock objects
        mocks = {
            "parser": patch("roboflow.core.workspace.folderparser.parsefolder", return_value=test_dataset),
            "upload": patch("roboflow.core.workspace.Project.upload_image", side_effect=upload_image_side_effect)
            if upload_image_side_effect
            else patch("roboflow.core.workspace.Project.upload_image", return_value=image_return),
            "save_annotation": patch(
                "roboflow.core.workspace.Project.save_annotation", side_effect=save_annotation_side_effect
            )
            if save_annotation_side_effect
            else patch("roboflow.core.workspace.Project.save_annotation", return_value=annotation_return),
            "get_project": patch(
                "roboflow.core.workspace.Workspace._get_or_create_project", return_value=(self.project, project_created)
            ),
        }

        return mocks

    def test_check_valid_image_with_accepted_formats(self):
        images_to_test = [
            "rabbit.JPG",
            "rabbit2.jpg",
            "hand-rabbit.PNG",
            "woodland-rabbit.png",
            "file_example_TIFF_1MB.tiff",
            "sky-rabbit.heic",
        ]

        for image in images_to_test:
            self.assertTrue(self.project.check_valid_image(f"tests/images/{image}"))

    def test_check_valid_image_with_unaccepted_formats(self):
        images_to_test = [
            "sky-rabbit.gif",
        ]

        for image in images_to_test:
            self.assertFalse(self.project.check_valid_image(f"tests/images/{image}"))

    def test_upload_raises_upload_image_error(self):
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}&batch={DEFAULT_BATCH_NAME}",
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
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}&batch={DEFAULT_BATCH_NAME}",
            json={"success": True, "id": image_id},
            status=200,
        )

        # Annotation
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/annotate/{image_id}?api_key={ROBOFLOW_API_KEY}&name={image_name}",
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

    @ordered
    @responses.activate
    def test_project_upload_dataset(self):
        """Test upload_dataset functionality with various scenarios"""
        test_scenarios = [
            {
                "name": "string_annotationdesc",
                "dataset": [{"file": "test_image.jpg", "split": "train", "annotationfile": "string_annotation.txt"}],
                "params": {"num_workers": 1},
                "assertions": {},
            },
            {
                "name": "success_basic",
                "dataset": [
                    {"file": "image1.jpg", "split": "train", "annotationfile": {"file": "image1.xml"}},
                    {"file": "image2.jpg", "split": "valid", "annotationfile": {"file": "image2.xml"}},
                ],
                "params": {},
                "assertions": {"parser": [("/test/dataset",)], "upload": {"count": 2}, "save_annotation": {"count": 2}},
                "image_return": ({"id": "test-id-1", "success": True}, 0.1, 0),
            },
            {
                "name": "custom_parameters",
                "dataset": None,
                "params": {
                    "num_workers": 2,
                    "project_license": "CC BY 4.0",
                    "project_type": "classification",
                    "batch_name": "test-batch",
                    "num_retries": 3,
                },
                "assertions": {"upload": {"count": 1, "kwargs": {"batch_name": "test-batch", "num_retry_uploads": 3}}},
            },
            {
                "name": "project_creation",
                "dataset": None,
                "params": {"project_name": "new-project"},
                "assertions": {},
                "project_created": True,
            },
            {
                "name": "with_labelmap",
                "dataset": [
                    {
                        "file": "image1.jpg",
                        "split": "train",
                        "annotationfile": {"file": "image1.xml", "labelmap": "path/to/labelmap.json"},
                    }
                ],
                "params": {},
                "assertions": {"save_annotation": {"count": 1}, "load_labelmap": {"count": 1}},
                "extra_mocks": [
                    (
                        "load_labelmap",
                        "roboflow.core.workspace.load_labelmap",
                        {"return_value": {"old_label": "new_label"}},
                    )
                ],
            },
            {
                "name": "concurrent_uploads",
                "dataset": [{"file": f"image{i}.jpg", "split": "train"} for i in range(10)],
                "params": {"num_workers": 5},
                "assertions": {"thread_pool": {"count": 1, "kwargs": {"max_workers": 5}}},
                "extra_mocks": [("thread_pool", "concurrent.futures.ThreadPoolExecutor", {})],
            },
            {"name": "empty_dataset", "dataset": [], "params": {}, "assertions": {"upload": {"count": 0}}},
            {
                "name": "raw_text_annotation",
                "dataset": [
                    {
                        "file": "image1.jpg",
                        "split": "train",
                        "annotationfile": {"rawText": "annotation content here", "format": "json"},
                    }
                ],
                "params": {},
                "assertions": {"save_annotation": {"count": 1}},
            },
        ]

        error_cases = [
            {
                "name": "image_upload_error",
                "side_effect": {
                    "upload_image_side_effect": lambda *args, **kwargs: (_ for _ in ()).throw(
                        ImageUploadError("Failed to upload image")
                    )
                },
                "params": {"num_workers": 1},
            },
            {
                "name": "annotation_upload_error",
                "side_effect": {
                    "save_annotation_side_effect": lambda *args, **kwargs: (_ for _ in ()).throw(
                        AnnotationSaveError("Failed to save annotation")
                    )
                },
                "params": {"num_workers": 1},
            },
        ]

        for scenario in test_scenarios:
            test_dataset = (
                self._create_test_dataset(scenario.get("dataset")) if scenario.get("dataset") is not None else None
            )

            extra_mocks = {}
            if "extra_mocks" in scenario:
                for mock_name, target, config in scenario.get("extra_mocks", []):
                    extra_mocks[mock_name] = patch(target, **config)

            mocks = self._setup_upload_dataset_mocks(
                test_dataset=test_dataset,
                image_return=scenario.get("image_return"),
                project_created=scenario.get("project_created", False),
            )

            mock_objects = {}
            for name, mock in mocks.items():
                mock_objects[name] = mock.start()

            for name, mock in extra_mocks.items():
                mock_objects[name] = mock.start()

            try:
                params = {"dataset_path": "/test/dataset", "project_name": PROJECT_NAME}
                params.update(scenario.get("params", {}))

                self.workspace.upload_dataset(**params)

                for mock_name, assertion in scenario.get("assertions", {}).items():
                    if isinstance(assertion, list):
                        mock_obj = mock_objects.get(mock_name)
                        call_args_list = [args for args, _ in mock_obj.call_args_list]
                        for expected_args in assertion:
                            self.assertIn(expected_args, call_args_list)
                    elif isinstance(assertion, dict):
                        mock_obj = mock_objects.get(mock_name)
                        if "count" in assertion:
                            self.assertEqual(mock_obj.call_count, assertion["count"])
                        if "kwargs" in assertion and mock_obj.call_count > 0:
                            _, kwargs = mock_obj.call_args
                            for key, value in assertion["kwargs"].items():
                                self.assertEqual(kwargs.get(key), value)
            finally:
                for mock in list(mocks.values()) + list(extra_mocks.values()):
                    mock.stop()

        for case in error_cases:
            mocks = self._setup_upload_dataset_mocks(**case.get("side_effect", {}))

            for mock in mocks.values():
                mock.start()

            try:
                params = {"dataset_path": "/test/dataset", "project_name": PROJECT_NAME}
                params.update(case.get("params", {}))
                self.workspace.upload_dataset(**params)
            finally:
                for mock in mocks.values():
                    mock.stop()
