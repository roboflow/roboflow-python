"""Regression tests for the Project.save_annotation / Project.upload_image retry contract.

`Project.save_annotation` built a `Retry` helper but called the HTTP adapter
directly, so a caller-requested retry budget was silently ignored. Both
`save_annotation` and `upload_image` also reported `error.retries == 0` on
exhaustion because the retry count was copied into a local variable only on
the success path. See roboflow/util/general.py::Retry and PR #362, which
originally intended annotation retries.

HTTP is intercepted with `responses`; nothing here touches the network or
needs credentials. A simulated HTTP 503 stands in for a transient pre-commit
failure. These tests say nothing about exactly-once behavior when a write
succeeds server-side but the response is lost (an ambiguous remote commit).
"""

import json
import re
from unittest.mock import patch

import responses

from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError
from roboflow.config import API_URL, DEFAULT_BATCH_NAME
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, RoboflowTest

ANNOTATION_IMAGE_ID = "test-image"
ANNOTATION_BASE_URL = f"{API_URL}/dataset/{PROJECT_NAME}/annotate/{ANNOTATION_IMAGE_ID}"
ANNOTATION_URL = f"{ANNOTATION_BASE_URL}?api_key={ROBOFLOW_API_KEY}&name=test.txt"

UPLOAD_BASE_URL = f"{API_URL}/dataset/{PROJECT_NAME}/upload"
UPLOAD_URL_PATTERN = re.compile(re.escape(f"{UPLOAD_BASE_URL}?api_key={ROBOFLOW_API_KEY}"))


def _error_body(message="simulated pre-commit failure"):
    # Nested {"error": {"message": ...}} shape: rfapi's error handling assumes
    # a dict here and mishandles a bare string, which is an unrelated bug.
    return {"error": {"message": message}}


class TestSaveAnnotationRetries(RoboflowTest):
    """save_annotation must route through Retry and report the real retry count."""

    def _save(self, num_retry_uploads):
        return self.project.save_annotation(
            annotation_path={"name": "test.txt", "rawText": "0 0.5 0.5 0.2 0.2"},
            image_id=ANNOTATION_IMAGE_ID,
            num_retry_uploads=num_retry_uploads,
        )

    def _annotate_calls(self):
        return [c.request for c in responses.calls if c.request.url.startswith(ANNOTATION_BASE_URL)]

    def test_immediate_success_makes_one_request_and_reports_zero_retries(self):
        responses.add(responses.POST, ANNOTATION_URL, json={"success": True, "id": ANNOTATION_IMAGE_ID}, status=200)

        annotation, _, retries = self._save(num_retry_uploads=2)

        self.assertTrue(annotation["success"])
        self.assertEqual(retries, 0)
        self.assertEqual(len(self._annotate_calls()), 1)

    def test_zero_retry_budget_fails_after_one_request(self):
        responses.add(responses.POST, ANNOTATION_URL, json=_error_body(), status=503)

        with self.assertRaises(AnnotationSaveError) as caught:
            self._save(num_retry_uploads=0)

        self.assertEqual(caught.exception.retries, 0)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(len(self._annotate_calls()), 1)

    @patch("roboflow.util.general.time.sleep")
    def test_transient_failure_is_retried_and_succeeds(self, mock_sleep):
        responses.add(responses.POST, ANNOTATION_URL, json=_error_body(), status=503)
        responses.add(responses.POST, ANNOTATION_URL, json={"success": True, "id": ANNOTATION_IMAGE_ID}, status=200)

        annotation, _, retries = self._save(num_retry_uploads=1)

        self.assertTrue(annotation["success"])
        self.assertEqual(retries, 1)
        self.assertEqual(len(self._annotate_calls()), 2)
        mock_sleep.assert_called_once()

    @patch("roboflow.util.general.time.sleep")
    def test_exhaustion_reports_actual_retry_count_and_preserves_error(self, mock_sleep):
        for _ in range(3):
            responses.add(responses.POST, ANNOTATION_URL, json=_error_body(), status=503)

        with self.assertRaises(AnnotationSaveError) as caught:
            self._save(num_retry_uploads=2)

        self.assertEqual(caught.exception.retries, 2)
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(str(caught.exception), "simulated pre-commit failure")
        self.assertEqual(len(self._annotate_calls()), 3)
        self.assertEqual(mock_sleep.call_count, 2)

    def test_unexpected_exception_type_is_not_retried(self):
        """Only AnnotationSaveError is retryable; anything else must propagate on the first attempt."""
        with patch("roboflow.adapters.rfapi.save_annotation", side_effect=ValueError("boom")) as mock_save:
            with self.assertRaises(ValueError):
                self._save(num_retry_uploads=3)
            self.assertEqual(mock_save.call_count, 1)

    @patch("roboflow.util.general.time.sleep")
    def test_retry_does_not_change_the_request_sent(self, mock_sleep):
        """Image identity, annotation name/content, labelmap, job name, prediction flag, and
        overwrite option are all encoded in the request URL/body, so an identical repeat
        proves none of them were mutated by the retry wiring."""
        responses.add(responses.POST, ANNOTATION_URL, json=_error_body(), status=503)
        responses.add(responses.POST, ANNOTATION_URL, json={"success": True, "id": ANNOTATION_IMAGE_ID}, status=200)

        self._save(num_retry_uploads=1)

        calls = self._annotate_calls()
        self.assertEqual(len(calls), 2)
        first, second = calls
        self.assertEqual(second.url, first.url)
        self.assertEqual(second.body, first.body)

    @patch("roboflow.util.general.time.sleep")
    def test_non_default_annotation_parameters_survive_multiple_retries(self, mock_sleep):
        """The above test only proves two default-valued requests match each other,
        which a bug that resets non-default arguments to their defaults on every
        attempt could not fail. Use every non-default parameter save_annotation
        accepts, across two failures then a success, and assert their actual
        encoded values -- not just that repeats match -- survive every attempt."""
        job_name = "custom-job-42"
        labelmap = {"0": "person", "1": "car"}
        non_default_url = (
            f"{ANNOTATION_BASE_URL}?api_key={ROBOFLOW_API_KEY}&name=test.txt"
            f"&jobName={job_name}&prediction=true&overwrite=true"
        )
        for _ in range(2):
            responses.add(responses.POST, non_default_url, json=_error_body(), status=503)
        responses.add(responses.POST, non_default_url, json={"success": True, "id": ANNOTATION_IMAGE_ID}, status=200)

        annotation, _, retries = self.project.save_annotation(
            annotation_path={"name": "test.txt", "rawText": "0 0.5 0.5 0.2 0.2"},
            annotation_labelmap=labelmap,
            image_id=ANNOTATION_IMAGE_ID,
            job_name=job_name,
            is_prediction=True,
            annotation_overwrite=True,
            num_retry_uploads=2,
        )

        self.assertTrue(annotation["success"])
        self.assertEqual(retries, 2)

        calls = self._annotate_calls()
        self.assertEqual(len(calls), 3)
        for call in calls:
            self.assertEqual(call.url, non_default_url)
            self.assertEqual(json.loads(call.body)["labelmap"], labelmap)


class TestUploadImageRetries(RoboflowTest):
    """upload_image already used the retry helper; lock in its behavior and fix retry-count reporting."""

    def _upload(self, num_retry_uploads):
        return self.project.upload_image(
            image_path="https://example.invalid/test.jpg",
            hosted_image=True,
            num_retry_uploads=num_retry_uploads,
        )

    def _upload_calls(self):
        return [c.request for c in responses.calls if c.request.url.startswith(UPLOAD_BASE_URL)]

    def test_immediate_success_makes_one_request_and_reports_zero_retries(self):
        responses.add(responses.POST, UPLOAD_URL_PATTERN, json={"success": True, "id": "test-id"}, status=200)

        image, _, retries = self._upload(num_retry_uploads=2)

        self.assertTrue(image["success"])
        self.assertEqual(retries, 0)
        self.assertEqual(len(self._upload_calls()), 1)

    def test_zero_retry_budget_fails_after_one_request(self):
        responses.add(responses.POST, UPLOAD_URL_PATTERN, json=_error_body(), status=503)

        with self.assertRaises(ImageUploadError) as caught:
            self._upload(num_retry_uploads=0)

        self.assertEqual(caught.exception.retries, 0)
        self.assertEqual(len(self._upload_calls()), 1)

    @patch("roboflow.util.general.time.sleep")
    def test_transient_failure_is_retried_and_succeeds(self, mock_sleep):
        responses.add(responses.POST, UPLOAD_URL_PATTERN, json=_error_body(), status=503)
        responses.add(responses.POST, UPLOAD_URL_PATTERN, json={"success": True, "id": "test-id"}, status=200)

        image, _, retries = self._upload(num_retry_uploads=1)

        self.assertTrue(image["success"])
        self.assertEqual(retries, 1)
        self.assertEqual(len(self._upload_calls()), 2)
        mock_sleep.assert_called_once()

    @patch("roboflow.util.general.time.sleep")
    def test_exhaustion_reports_actual_retry_count(self, mock_sleep):
        for _ in range(3):
            responses.add(responses.POST, UPLOAD_URL_PATTERN, json=_error_body(), status=503)

        with self.assertRaises(ImageUploadError) as caught:
            self._upload(num_retry_uploads=2)

        self.assertEqual(caught.exception.retries, 2)
        self.assertEqual(len(self._upload_calls()), 3)
        self.assertEqual(mock_sleep.call_count, 2)


class TestUploadPathForwardsRetryBudget(RoboflowTest):
    """The public upload() -> single_upload() path must forward num_retry_uploads to
    annotation persistence. Project.save_annotation is NOT mocked here, unlike the
    upload_dataset tests above, so this exercises the real wiring end to end."""

    @patch("roboflow.util.general.time.sleep")
    def test_upload_retries_annotation_persistence_through_the_real_path(self, mock_sleep):
        image_id = "real-path-image-id"

        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}&batch={DEFAULT_BATCH_NAME}",
            json={"success": True, "id": image_id},
            status=200,
        )

        annotation_url = (
            f"{API_URL}/dataset/{PROJECT_NAME}/annotate/{image_id}"
            f"?api_key={ROBOFLOW_API_KEY}&name=valid_annotation.json"
        )
        responses.add(responses.POST, annotation_url, json=_error_body(), status=503)
        responses.add(responses.POST, annotation_url, json={"success": True}, status=200)

        results = self.project.upload(
            "tests/images/rabbit.JPG",
            annotation_path="tests/annotations/valid_annotation.json",
            num_retry_uploads=1,
        )

        result = results[0]
        self.assertTrue(result["annotation"]["success"])
        self.assertEqual(result["annotation_upload_retry_attempts"], 1)

        annotate_calls = [
            c
            for c in responses.calls
            if c.request.url.startswith(f"{API_URL}/dataset/{PROJECT_NAME}/annotate/{image_id}")
        ]
        self.assertEqual(len(annotate_calls), 2)
