import gzip
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests
import responses
from urllib3.response import HTTPResponse

from roboflow.config import API_URL
from roboflow.models.inference import InferenceModel
from roboflow.models.object_detection import ObjectDetectionModel


class TestVideoUpload(unittest.TestCase):
    @responses.activate
    def test_stream_is_open_during_upload_and_closed_afterwards(self):
        model = ObjectDetectionModel("test-key", "workspace/project/1")
        signed_url = "https://upload.test/video?signature=test&X-Goog-Expires=3600"
        responses.post(f"{API_URL}/video_upload_signed_url", json={"signed_url": signed_url})
        responses.post(f"{API_URL}/videoinfer/", json={"job_id": "job-1"})
        with tempfile.TemporaryDirectory() as directory:
            video_path = os.path.join(directory, "video.mp4")
            with open(video_path, "wb") as video:
                video.write(b"video bytes")
            for outcome in (200, 403, requests.ConnectionError("Connection lost")):
                with self.subTest(outcome=outcome):

                    def upload(url, data, headers):
                        self.assertEqual(url, signed_url)
                        self.assertFalse(data.closed)
                        self.assertEqual(data.read(5), b"video")
                        self.assertEqual(data.read(6), b" bytes")
                        if isinstance(outcome, Exception):
                            raise outcome
                        response = requests.Response()
                        response.status_code = outcome
                        response._content = b"Upload rejected" if outcome == 403 else b""
                        return response

                    before = len(responses.calls)
                    with patch("roboflow.models.inference.http.put", side_effect=upload) as put:
                        if outcome == 200:
                            self.assertEqual(model.predict_video(video_path), ("job-1", signed_url, "3600"))
                            self.assertEqual(len(responses.calls) - before, 2)
                        else:
                            with self.assertRaisesRegex(Exception, "error uploading the video"):
                                model.predict_video(video_path)
                            self.assertEqual(len(responses.calls) - before, 1)
                    self.assertTrue(put.call_args.kwargs["data"].closed)

    @responses.activate
    def test_hosted_video_does_not_upload_a_file(self):
        model = ObjectDetectionModel("test-key", "workspace/project/1")
        responses.post(f"{API_URL}/videoinfer/", json={"job_id": "job-1"})
        with patch("roboflow.models.inference.http.put") as put:
            result = model.predict_video("https://video.test/video.mp4")
        self.assertEqual(result, ("job-1", "https://video.test/video.mp4", None))
        put.assert_not_called()


class TestDownloadWeights(unittest.TestCase):
    def setUp(self):
        self.model = InferenceModel("test-key", "workspace/project/1")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.weights_path = os.path.join(self.directory.name, "weights.pt")
        with open(self.weights_path, "wb") as f:
            f.write(b"existing weights")
        self.metadata_url = f"{API_URL}/workspace/project/1/ptFile"
        self.weights_url = "https://download.test/weights.pt"

    @responses.activate
    def test_http_error_does_not_replace_existing_weights(self):
        responses.get(self.metadata_url, json={"weightsUrl": self.weights_url})
        for status in (403, 404, 500):
            with self.subTest(status=status):
                responses.get(self.weights_url, status=status, body="error")
                with self.assertRaises(requests.HTTPError):
                    self.model.download(location=self.directory.name)
                with open(self.weights_path, "rb") as f:
                    self.assertEqual(f.read(), b"existing weights")
        self.assertEqual(os.listdir(self.directory.name), ["weights.pt"])

    def test_interrupted_stream_preserves_weights_and_cleans_temporary_files(self):
        def broken_stream(**kwargs):
            yield b"partial weights"
            raise requests.exceptions.ChunkedEncodingError("Download interrupted")

        metadata = MagicMock()
        metadata.json.return_value = {"weightsUrl": self.weights_url}
        response = MagicMock(headers={"content-length": "1024"})
        response.__enter__.return_value = response
        response.iter_content.side_effect = broken_stream
        with patch("roboflow.models.inference.http.get", side_effect=[metadata, response]):
            with self.assertRaisesRegex(requests.exceptions.ChunkedEncodingError, "interrupted"):
                self.model.download(location=self.directory.name)
        with open(self.weights_path, "rb") as f:
            self.assertEqual(f.read(), b"existing weights")
        self.assertEqual(os.listdir(self.directory.name), ["weights.pt"])
        response.__exit__.assert_called_once()

    def test_short_stream_without_transport_error_does_not_replace_weights(self):
        metadata = MagicMock()
        metadata.json.return_value = {"weightsUrl": self.weights_url}
        for encoding, body in (("identity", b"truncated"), ("gzip", gzip.compress(b"partial weights"))):
            with self.subTest(encoding=encoding):
                response = requests.Response()
                response.status_code = 200
                response.headers.update({"content-length": str(len(body) + 10), "content-encoding": encoding})
                response.raw = HTTPResponse(
                    body=io.BytesIO(body), headers=response.headers, preload_content=False, enforce_content_length=False
                )
                with patch("roboflow.models.inference.http.get", side_effect=[metadata, response]):
                    with self.assertRaises(requests.exceptions.ChunkedEncodingError):
                        self.model.download(location=self.directory.name)
                with open(self.weights_path, "rb") as f:
                    self.assertEqual(f.read(), b"existing weights")
        self.assertEqual(os.listdir(self.directory.name), ["weights.pt"])

    @responses.activate
    def test_success_replaces_weights_with_exact_response_bytes(self):
        weights = b"new weights" * 1000
        responses.get(self.metadata_url, json={"weightsUrl": self.weights_url})
        responses.get(self.weights_url, body=weights, headers={"Content-Length": str(len(weights))})
        self.model.download(location=self.directory.name)
        with open(self.weights_path, "rb") as f:
            self.assertEqual(f.read(), weights)
        self.assertEqual(os.listdir(self.directory.name), ["weights.pt"])
