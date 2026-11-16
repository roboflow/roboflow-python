"""Model upload failures must reach SDK callers and close the upload file."""

import os
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

import requests
import responses

from roboflow.config import API_URL
from roboflow.util import http
from tests.helpers import get_version
from tests.test_workspace import _make_workspace


class TestModelUpload(unittest.TestCase):
    @responses.activate
    def test_upload_status_and_file_lifetime(self):
        version = get_version(workspace_name="test-ws", project_name="project")
        workspace = _make_workspace()
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "model.zip"), "wb") as model_file:
                model_file.write(b"model archive")
            for scope in ("version", "workspace"):
                for prepare_status, upload_status in ((403, 200), (429, 200), (200, 500), (200, 200)):
                    with self.subTest(scope=scope, prepare=prepare_status, upload=upload_status):
                        responses.reset()
                        if scope == "version":
                            responses.get(
                                f"{API_URL}/test-ws/project/1/uploadModel",
                                status=prepare_status,
                                json={"url": "https://upload.test/model"},
                            )
                            upload = partial(version._upload_zip, "yolov8", directory, "model.zip")
                        else:
                            responses.post(
                                f"{API_URL}/test-ws/models/prepareUpload",
                                status=prepare_status,
                                json={"url": "https://upload.test/model"},
                            )
                            upload = partial(
                                workspace._upload_zip, "yolov8", directory, ["project"], "model", "model.zip"
                            )

                        if prepare_status == 200:
                            responses.put("https://upload.test/model", status=upload_status)
                        with patch("roboflow.util.http.put", wraps=http.put) as put:
                            if prepare_status != 200 or upload_status != 200:
                                with self.assertRaises(requests.HTTPError):
                                    upload()
                            else:
                                upload()
                        if prepare_status == 200:
                            self.assertTrue(put.call_args.kwargs["data"].closed)
                            self.assertEqual(responses.calls[-1].request.body, b"model archive")
                        else:
                            self.assertEqual(len(responses.calls), 1)
