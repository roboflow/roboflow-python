"""User-facing SDK calls against isolated public annotate request fixtures."""

import json
import unittest
from urllib.parse import parse_qs, urlparse

import requests
import responses

from roboflow.adapters.rfapi import AnnotationSaveError
from roboflow.config import API_URL
from roboflow.core.project import Project


class TestVideoSegmentAnnotation(unittest.TestCase):
    def setUp(self):
        self.project = Project(
            "test-key",
            {
                "annotation": "actions",
                "classes": {},
                "colors": {},
                "created": 0,
                "id": "my-workspace/actions",
                "images": 0,
                "name": "Actions",
                "public": False,
                "splits": {},
                "type": "action-recognition",
                "unannotated": 0,
                "updated": 0,
            },
        )
        self.url = f"{API_URL}/my-workspace/actions/annotate/final-source-id"
        self.document = {
            "info": {"format": "roboflow-video-coco"},
            "videos": [
                {
                    "id": 1,
                    "file_name": "clip.mp4",
                    "width": 640,
                    "height": 360,
                    "duration": 10,
                    "fps": 24,
                    "time_base": {"numerator": 1, "denominator": 12288},
                }
            ],
            "categories": [{"id": 1, "name": "jumping"}],
            "segments": [
                {
                    "id": 1,
                    "video_id": 1,
                    "category_id": 1,
                    "start_frame": 48,
                    "end_frame": 95,
                    "start_pts": 24576,
                    "end_pts": 49152,
                }
            ],
        }

    def test_acceptance_and_identical_retry_forward_the_canonical_document(self):
        first = {"success": True, "inDataset": True, "createdClasses": ["jumping"]}
        retry = {"success": True, "inDataset": True, "createdClasses": []}
        with responses.RequestsMock() as http:
            http.add(responses.POST, self.url, json=first)
            http.add(responses.POST, self.url, json=retry)
            self.assertEqual(self.project.annotate_video_segments("final-source-id", self.document), first)
            self.assertEqual(self.project.annotate_video_segments("final-source-id", self.document), retry)

            self.assertEqual(len(http.calls), 2)
            for call in http.calls:
                query = parse_qs(urlparse(call.request.url).query)
                self.assertEqual(query, {"api_key": ["test-key"], "name": ["annotations.json"]})
                self.assertEqual(json.loads(call.request.body), {"annotationFile": json.dumps(self.document)})

    def test_conflict_is_raised_and_explicit_overwrite_and_split_are_forwarded(self):
        def annotate(request):
            query = parse_qs(urlparse(request.url).query)
            if "overwrite" not in query:
                return (
                    409,
                    {"Content-Type": "application/json"},
                    json.dumps(
                        {
                            "error": {
                                "message": "This video already has annotations. Send overwrite=true to replace them."
                            }
                        }
                    ),
                )
            self.assertEqual(query["overwrite"], ["true"])
            self.assertEqual(query["split"], ["valid"])
            self.assertEqual(query["addToDataset"], ["false"])
            return (
                200,
                {"Content-Type": "application/json"},
                json.dumps({"success": True, "inDataset": False, "createdClasses": []}),
            )

        with responses.RequestsMock() as http:
            http.add_callback(responses.POST, self.url, callback=annotate)
            with self.assertRaises(AnnotationSaveError) as error:
                self.project.annotate_video_segments("final-source-id", self.document)
            self.assertEqual(error.exception.status_code, 409)
            self.assertIn("already has annotations", str(error.exception))

            result = self.project.annotate_video_segments(
                "final-source-id", self.document, overwrite=True, split="valid", add_to_dataset=False
            )
            self.assertEqual(result, {"success": True, "inDataset": False, "createdClasses": []})

    def test_server_validation_and_transport_errors_are_not_recast_as_success(self):
        with responses.RequestsMock() as http:
            http.add(
                responses.POST,
                self.url,
                json={"error": {"message": "Invalid roboflow-video-coco document"}},
                status=400,
            )
            with self.assertRaises(AnnotationSaveError) as error:
                self.project.annotate_video_segments("final-source-id", self.document)
            self.assertEqual(error.exception.status_code, 400)
            self.assertEqual(str(error.exception), "Invalid roboflow-video-coco document")

        with responses.RequestsMock() as http:
            http.add(responses.POST, self.url, body=requests.ConnectionError("connection lost"))
            with self.assertRaises(AnnotationSaveError) as error:
                self.project.annotate_video_segments("final-source-id", self.document)
            self.assertIn("connection lost", str(error.exception))


if __name__ == "__main__":
    unittest.main()
