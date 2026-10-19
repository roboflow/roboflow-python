import unittest

import responses

import roboflow
from roboflow.util import http

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


if __name__ == "__main__":
    unittest.main()
