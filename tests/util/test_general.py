import gzip
import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests
import responses

from roboflow.util import http
from roboflow.util.general import save_response_content


class TestSaveResponseContent(unittest.TestCase):
    def test_preserves_bytes_with_known_unknown_or_invalid_content_length(self):
        content = bytes(range(256)) * 1024 + b"tail"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "download")
            for length in (str(len(content)), None, "invalid", "-1"):
                with self.subTest(length=length):
                    response = requests.Response()
                    response.raw = io.BytesIO(content)
                    if length is not None:
                        response.headers["content-length"] = length
                    save_response_content(response, path)
                    with open(path, "rb") as f:
                        self.assertEqual(f.read(), content)

    @patch("roboflow.util.general.tqdm")
    def test_progress_counts_bytes_and_skips_empty_chunks(self, tqdm):
        response = MagicMock(headers={})
        response.iter_content.return_value = iter([b"ab", b"", b"cde"])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "download")
            save_response_content(response, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"abcde")
        self.assertIsNone(tqdm.call_args.kwargs["total"])
        self.assertEqual(
            [call.args[0] for call in tqdm.return_value.__enter__.return_value.update.call_args_list], [2, 3]
        )

    def test_empty_response_creates_empty_file(self):
        response = requests.Response()
        response.raw = io.BytesIO()
        response.headers["content-length"] = "0"
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "download")
            save_response_content(response, path)
            self.assertEqual(os.path.getsize(path), 0)

    @responses.activate
    @patch("roboflow.util.general.tqdm")
    def test_compressed_response_uses_decoded_bytes_without_a_wire_length_total(self, tqdm):
        content = b"model weights" * 10000
        compressed = gzip.compress(content)
        responses.get(
            "https://download.test/weights",
            body=compressed,
            headers={"Content-Encoding": "gzip", "Content-Length": str(len(compressed))},
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            http.get("https://download.test/weights", stream=True) as response,
        ):
            path = os.path.join(directory, "download")
            save_response_content(response, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), content)
        self.assertIsNone(tqdm.call_args.kwargs["total"])

    @patch("roboflow.util.general.tqdm")
    def test_transfer_encoding_takes_precedence_over_content_length(self, tqdm):
        response = requests.Response()
        response.raw = io.BytesIO(b"decoded chunks")
        response.headers.update({"Transfer-Encoding": "chunked", "Content-Length": "999"})
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "download")
            save_response_content(response, path)
            with open(path, "rb") as f:
                self.assertEqual(f.read(), b"decoded chunks")
        self.assertIsNone(tqdm.call_args.kwargs["total"])
