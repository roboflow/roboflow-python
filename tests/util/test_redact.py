import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import responses

from roboflow.adapters.rfapi import AnnotationSaveError, ImageUploadError, RoboflowError
from roboflow.util.redact import redact_api_key
from tests.helpers import get_version

API_KEY = "abc-DEF_123"


class TestRedactApiKey(unittest.TestCase):
    def test_redacts_query_parameter(self) -> None:
        text = f"Max retries exceeded with url: /ws/proj?api_key={API_KEY}&nocache=true (Caused by ...)"
        self.assertEqual(
            redact_api_key(text),
            "Max retries exceeded with url: /ws/proj?api_key=***&nocache=true (Caused by ...)",
        )

    def test_redacts_every_occurrence(self) -> None:
        text = f"'https://a/b?api_key={API_KEY}' and \"https://c/d?x=1&api_key={API_KEY}\""
        self.assertNotIn(API_KEY, redact_api_key(text))
        self.assertEqual(redact_api_key(text).count("api_key=***"), 2)

    def test_leaves_other_values_unchanged(self) -> None:
        self.assertEqual(redact_api_key("no key here"), "no key here")
        self.assertEqual(redact_api_key({"api_key": API_KEY}), {"api_key": API_KEY})


class TestErrorMessages(unittest.TestCase):
    def test_exceptions_redact_their_message(self) -> None:
        message = f"500 Server Error for url: https://api.roboflow.com/ws/proj?api_key={API_KEY}"
        for error_class in (RoboflowError, ImageUploadError, AnnotationSaveError):
            error = error_class(message)
            self.assertNotIn(API_KEY, str(error))
            self.assertIn("api_key=***", str(error))
        self.assertNotIn(API_KEY, ImageUploadError(message).message)
        self.assertNotIn(API_KEY, AnnotationSaveError(message).message)

    def test_version_deploy_error_is_printed_without_key(self) -> None:
        bundle = SimpleNamespace(model_type="yolov8n", archive_path=SimpleNamespace(name="roboflow_deploy.zip"))
        stdout = io.StringIO()
        with (
            responses.RequestsMock() as rsps,
            patch("roboflow.core.version.package_custom_weights_interactive", return_value=bundle),
            contextlib.redirect_stdout(stdout),
        ):
            rsps.add(
                responses.GET,
                "https://api.roboflow.com/Test Workspace Name/Test Project Name/1/uploadModel",
                status=500,
            )
            get_version(api_key=API_KEY).deploy("yolov8n", "weights-dir", "weights/best.pt")

        self.assertIn("api_key=***", stdout.getvalue())
        self.assertNotIn(API_KEY, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
