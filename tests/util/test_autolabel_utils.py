"""Unit tests for roboflow.util.autolabel_utils."""

import base64
import os
import tempfile
import unittest

from roboflow.util.autolabel_utils import image_payload, resolve_model


class TestImagePayload(unittest.TestCase):
    def test_url(self):
        self.assertEqual(
            image_payload("https://example.com/cat.jpg"),
            {"type": "url", "value": "https://example.com/cat.jpg"},
        )

    def test_local_file_is_base64_encoded(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
            handle.write(b"fake-image-bytes")
            path = handle.name
        try:
            payload = image_payload(path)
        finally:
            os.unlink(path)
        self.assertEqual(payload["type"], "base64")
        self.assertEqual(base64.b64decode(payload["value"]), b"fake-image-bytes")

    def test_other_strings_are_treated_as_base64(self):
        encoded = base64.b64encode(b"bytes").decode("ascii")
        self.assertEqual(image_payload(encoded), {"type": "base64", "value": encoded})


class TestResolveModel(unittest.TestCase):
    def test_foundational_is_pass_through(self):
        self.assertEqual(resolve_model("gpt-6-astra-boxes", "foundational"), ("gpt-6-astra-boxes", None))
        self.assertEqual(
            resolve_model("sam3-rle", "foundational", {"outputFormat": "rle"}),
            ("sam3-rle", {"outputFormat": "rle"}),
        )

    def test_roboflow_model_rides_in_model_options(self):
        self.assertEqual(
            resolve_model("proj/3", "roboflow", {"outputFormat": "polygon"}),
            ("custom_roboflow", {"outputFormat": "polygon", "modelId": "proj/3"}),
        )
        self.assertEqual(resolve_model("proj/3", "roboflow"), ("custom_roboflow", {"modelId": "proj/3"}))

    def test_unknown_model_type_raises(self):
        with self.assertRaises(ValueError):
            resolve_model("x", "hosted")
