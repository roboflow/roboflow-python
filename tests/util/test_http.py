import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

import responses

import roboflow
from roboflow.config import API_URL, DEFAULT_BATCH_NAME
from roboflow.util import http
from roboflow.util.prediction import Prediction, PredictionGroup
from tests import PROJECT_NAME, ROBOFLOW_API_KEY, RoboflowTest

URL = "https://api.roboflow.com/test"


class TestHttp(unittest.TestCase):
    def test_user_agent_names_the_sdk_and_its_version(self):
        self.assertEqual(http.USER_AGENT, f"roboflow-python/{roboflow.__version__}")

    @responses.activate
    def test_every_method_sends_the_user_agent(self):
        methods = [
            ("GET", http.get),
            ("POST", http.post),
            ("PUT", http.put),
            ("PATCH", http.patch),
            ("DELETE", http.delete),
            ("HEAD", http.head),
            ("POST", lambda url: http.request("POST", url)),
        ]
        for method, send in methods:
            with self.subTest(method=method):
                responses.add(method, URL)
                send(URL)
                self.assertEqual(responses.calls[-1].request.headers["User-Agent"], http.USER_AGENT)

    @responses.activate
    def test_keeps_the_callers_headers(self):
        responses.add(responses.POST, URL)
        http.post(URL, headers={"Content-Type": "application/json"})
        headers = responses.calls[0].request.headers
        self.assertEqual(headers["User-Agent"], http.USER_AGENT)
        self.assertEqual(headers["Content-Type"], "application/json")

    @responses.activate
    def test_callers_user_agent_wins(self):
        responses.add(responses.GET, URL)
        http.get(URL, headers={"user-agent": "my-app/1.0"})
        self.assertEqual(responses.calls[0].request.headers["User-Agent"], "my-app/1.0")

    @responses.activate
    def test_session_sends_the_user_agent(self):
        responses.add(responses.GET, URL)
        with http.Session() as session:
            session.get(URL)
        self.assertEqual(responses.calls[0].request.headers["User-Agent"], http.USER_AGENT)

    @mock.patch("urllib.request.urlopen")
    def test_hosted_prediction_image_is_fetched_with_the_user_agent(self, urlopen):
        urlopen.return_value.read.return_value = Path("tests/images/rabbit.JPG").read_bytes()
        image_url = "https://example.com/rabbit.jpg"
        prediction = Prediction(
            {"x": 50, "y": 50, "width": 20, "height": 20, "class": "rabbit", "confidence": 0.9}, image_url
        )
        group = PredictionGroup({"width": 100, "height": 100}, image_url, prediction)

        with tempfile.TemporaryDirectory() as tmp:
            group.save(os.path.join(tmp, "prediction.jpg"))

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, image_url)
        self.assertEqual(request.get_header("User-agent"), http.USER_AGENT)


class TestSdkRequestsSendUserAgent(RoboflowTest):
    def assert_every_request_sent_the_user_agent(self):
        self.assertGreater(len(responses.calls), 0)
        for call in responses.calls:
            with self.subTest(url=call.request.url):
                self.assertEqual(call.request.headers["User-Agent"], http.USER_AGENT)

    def test_connecting_to_a_version(self):
        responses.calls.reset()

        self.connect_to_roboflow()

        self.assert_every_request_sent_the_user_agent()

    def test_uploading_an_annotated_image(self):
        image_id = "hbALkCFdNr9rssgOUXug"
        responses.add(
            responses.POST,
            f"{API_URL}/dataset/{PROJECT_NAME}/upload?api_key={ROBOFLOW_API_KEY}&batch={DEFAULT_BATCH_NAME}",
            json={"success": True, "id": image_id},
        )
        responses.add(
            responses.POST,
            re.compile(rf"{API_URL}/dataset/{PROJECT_NAME}/annotate/{image_id}\?.*"),
            json={"success": True},
        )
        responses.calls.reset()

        self.project.upload("tests/images/rabbit.JPG", annotation_path="tests/annotations/valid_annotation.json")

        paths = [urlparse(call.request.url).path for call in responses.calls]
        self.assertEqual(paths, [f"/dataset/{PROJECT_NAME}/upload", f"/dataset/{PROJECT_NAME}/annotate/{image_id}"])
        self.assert_every_request_sent_the_user_agent()


if __name__ == "__main__":
    unittest.main()
