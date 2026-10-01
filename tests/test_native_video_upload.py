"""Public Project calls through the native video preparation, PUT and status API."""

import json
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

import requests
import responses
from responses.matchers import json_params_matcher

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL
from roboflow.core.project import Project
from tests import ROBOFLOW_API_KEY, WORKSPACE_NAME, RoboflowTest


class TestNativeVideoUpload(RoboflowTest):
    def setUp(self):
        super().setUp()
        self.project.type = "action-recognition"
        self.temp_dir = tempfile.TemporaryDirectory(dir=".")
        self.video_path = os.path.join(self.temp_dir.name, "clip.mp4")
        with open(self.video_path, "wb") as video:
            video.write(b"original video bytes")
        self.addCleanup(self.temp_dir.cleanup)
        self.prepare_url = f"{API_URL}/{WORKSPACE_NAME}/upload/video?api_key={ROBOFLOW_API_KEY}"
        self.status_url = f"{API_URL}/{WORKSPACE_NAME}/upload/video/upload-1?api_key={ROBOFLOW_API_KEY}"

    def test_upload_streams_original_bytes_and_returns_canonical_source(self):
        headers = {"x-goog-content-length-range": "1,100", "x-goog-if-generation-match": "0"}
        responses.add(
            responses.POST,
            self.prepare_url,
            json={"videoId": "upload-1", "signedUrl": "https://signed.example/video", "requiredHeaders": headers},
            status=200,
            match=[
                json_params_matcher(
                    {
                        "project": "test-project",
                        "name": "clip.mp4",
                        "contentType": "video/mp4",
                        "batch": "clips",
                        "tag": ["indoor"],
                        "metadata": {"camera": "one"},
                        "split": "valid",
                    }
                )
            ],
        )
        uploaded = {}

        def receive_video(request):
            uploaded["body"] = request.body
            uploaded["headers"] = request.headers
            uploaded["url"] = request.url
            return 200, {}, ""

        responses.add_callback(responses.PUT, "https://signed.example/video", callback=receive_video)
        responses.add(
            responses.GET,
            self.status_url,
            json={"videoId": "source-2", "status": "uploaded", "duplicate": True, "resolvedBatch": None},
            status=200,
        )

        result = self.project.upload_video(
            self.video_path,
            batch_name="clips",
            tag_names=["indoor"],
            metadata={"camera": "one"},
            split="valid",
        )

        self.assertEqual(result["videoId"], "source-2")
        self.assertEqual(result["status"], "uploaded")
        self.assertEqual(uploaded["body"], b"original video bytes")
        self.assertEqual(uploaded["headers"]["Content-Type"], "video/mp4")
        for key, value in headers.items():
            self.assertEqual(uploaded["headers"][key], value)
        self.assertNotIn("api_key", uploaded["url"])

    def test_pending_then_wait_returns_uploaded_with_batch(self):
        responses.add(
            responses.POST,
            self.prepare_url,
            json={"videoId": "upload-1", "signedUrl": "https://signed.example/video", "requiredHeaders": {}},
            status=200,
        )
        responses.add(responses.PUT, "https://signed.example/video", status=200)
        responses.add(responses.GET, self.status_url, json={"videoId": "upload-1", "status": "pending"}, status=200)
        responses.add(responses.GET, self.status_url, json={"videoId": "upload-1", "status": "pending"}, status=200)
        responses.add(
            responses.GET,
            self.status_url,
            json={"videoId": "upload-1", "status": "uploaded", "resolvedBatch": {"id": "b1", "name": "clips"}},
            status=200,
        )

        first = self.project.upload_video(self.video_path)
        with patch("roboflow.core.project.time.sleep"):
            final = self.project.wait_for_video_upload(first["videoId"], poll_interval=0.1, poll_timeout=10)

        self.assertEqual(first["status"], "pending")
        self.assertEqual(final["resolvedBatch"], {"id": "b1", "name": "clips"})

    def test_server_errors_and_bounded_wait(self):
        responses.add(responses.POST, self.prepare_url, json={"error": "quota exceeded"}, status=403)
        with self.assertRaises(RoboflowError) as error:
            self.project.upload_video(self.video_path)
        self.assertIn("quota exceeded", str(error.exception))
        self.assertEqual(error.exception.status_code, 403)

        responses.add(responses.GET, self.status_url, json={"videoId": "upload-1", "status": "pending"}, status=200)
        with self.assertRaises(RoboflowError) as timeout:
            self.project.wait_for_video_upload("upload-1", poll_timeout=0)
        self.assertIn("upload-1", str(timeout.exception))

    def test_signed_put_error_keeps_server_response(self):
        responses.add(
            responses.POST,
            self.prepare_url,
            json={"videoId": "upload-1", "signedUrl": "https://signed.example/video", "requiredHeaders": {}},
            status=200,
        )
        responses.add(responses.PUT, "https://signed.example/video", body="object already exists", status=412)

        with self.assertRaises(RoboflowError) as error:
            self.project.upload_video(self.video_path)

        self.assertIn("object already exists", str(error.exception))
        self.assertEqual(error.exception.status_code, 412)
        self.assertFalse(
            any(call.request.method == "GET" and "/upload/video/" in call.request.url for call in responses.calls)
        )

    def test_failed_processing_returns_api_message(self):
        responses.add(
            responses.GET,
            self.status_url,
            json={"videoId": "upload-1", "status": "failed", "message": "Invalid video media"},
            status=200,
        )
        self.assertEqual(
            self.project.wait_for_video_upload("upload-1", poll_timeout=0),
            {"videoId": "upload-1", "status": "failed", "message": "Invalid video media"},
        )

    def test_invalid_file_does_not_prepare_upload(self):
        before = len(responses.calls)
        with self.assertRaises(ValueError):
            self.project.upload_video(os.path.join(self.temp_dir.name, "missing.mp4"))
        self.assertEqual(len(responses.calls), before)

    def test_preparation_and_status_transport_errors_are_sanitized(self):
        with patch("roboflow.adapters.rfapi.requests.post", side_effect=requests.exceptions.ConnectTimeout) as post:
            with self.assertRaises(RoboflowError) as preparation:
                self.project.upload_video(self.video_path)
        self.assertIn("ConnectTimeout", str(preparation.exception))
        self.assertNotIn(ROBOFLOW_API_KEY, str(preparation.exception))
        self.assertEqual(post.call_args.kwargs["timeout"], (5, 30))

        with patch("roboflow.adapters.rfapi.requests.get", side_effect=requests.exceptions.ConnectionError) as get:
            with self.assertRaises(RoboflowError) as status:
                self.project.get_video_upload_status("upload-1")
        self.assertIn("ConnectionError", str(status.exception))
        self.assertNotIn(ROBOFLOW_API_KEY, str(status.exception))
        self.assertEqual(get.call_args.kwargs["timeout"], 30)


class TestNativeVideoUploadOverHttp(unittest.TestCase):
    @staticmethod
    def make_project():
        return Project(
            "test-key",
            {
                "annotation": "",
                "classes": {},
                "colors": {},
                "created": 0,
                "id": "region-workspace/actions",
                "images": 0,
                "name": "Actions",
                "public": False,
                "splits": {},
                "type": "action-recognition",
                "unannotated": 0,
                "updated": 0,
            },
        )

    def test_public_call_uses_configured_host_and_streams_original_bytes(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def respond(self, body):
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_POST(self):
                received["prepare_path"] = self.path
                received["prepare_body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.respond(
                    {
                        "videoId": "upload-1",
                        "signedUrl": f"http://127.0.0.1:{self.server.server_port}/signed",
                        "requiredHeaders": {"x-goog-if-generation-match": "0"},
                    }
                )

            def do_PUT(self):
                received["put_path"] = self.path
                received["put_body"] = self.rfile.read(int(self.headers["Content-Length"]))
                received["put_header"] = self.headers["x-goog-if-generation-match"]
                self.respond({})

            def do_GET(self):
                received["status_path"] = self.path
                self.respond({"videoId": "source-2", "status": "uploaded", "resolvedBatch": None})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        with tempfile.TemporaryDirectory(dir=".") as temp_dir:
            video_path = os.path.join(temp_dir, "native.mp4")
            with open(video_path, "wb") as video:
                video.write(b"original video bytes")
            project = self.make_project()
            with patch("roboflow.adapters.rfapi.API_URL", f"http://127.0.0.1:{server.server_port}"):
                status = project.upload_video(video_path)

        self.assertEqual(status["videoId"], "source-2")
        self.assertEqual(
            received["prepare_body"], {"project": "actions", "name": "native.mp4", "contentType": "video/mp4"}
        )
        self.assertEqual(urlparse(received["prepare_path"]).path, "/region-workspace/upload/video")
        self.assertEqual(parse_qs(urlparse(received["prepare_path"]).query), {"api_key": ["test-key"]})
        self.assertEqual(received["put_path"], "/signed")
        self.assertEqual(received["put_body"], b"original video bytes")
        self.assertEqual(received["put_header"], "0")
        self.assertEqual(urlparse(received["status_path"]).path, "/region-workspace/upload/video/upload-1")

    def test_wait_uses_remaining_budget_for_silent_status_server(self):
        entered = threading.Event()
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                entered.set()
                release.wait(2)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(thread.join)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        self.addCleanup(release.set)

        with patch("roboflow.adapters.rfapi.API_URL", f"http://127.0.0.1:{server.server_port}"):
            started = time.monotonic()
            with self.assertRaises(RoboflowError) as error:
                self.make_project().wait_for_video_upload("upload-1", poll_timeout=0.15)
            elapsed = time.monotonic() - started

        self.assertTrue(entered.is_set())
        self.assertIn("ReadTimeout", str(error.exception))
        self.assertLess(elapsed, 1.5)
