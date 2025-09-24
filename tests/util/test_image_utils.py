import unittest

import responses

from roboflow.util.image_utils import check_image_path, check_image_url, load_labelmap


class TestCheckImagePath(unittest.TestCase):
    def test_valid_path(self):
        self.assertTrue(check_image_path("tests/images/rabbit.JPG"))

    def test_invalid_paths(self):
        self.assertFalse(check_image_path("tests/images/notfound.jpg"))


class TestCheckImageURL(unittest.TestCase):
    @responses.activate
    def test_valid_url(self):
        url = "https://example.com/found.png"
        responses.add(responses.HEAD, url)
        self.assertTrue(check_image_url(url))

    def test_invalid_url(self):
        paths = [
            "ftp://example.com/found.png",
            "/found.png",
            None,
        ]
        for path in paths:
            self.assertFalse(check_image_url(path))

    def test_url_not_found(self):
        url = "https://example.com/notfound.png"
        responses.add(responses.HEAD, url, status=404)
        self.assertFalse(check_image_url(url))


class TestLoadLabelmap(unittest.TestCase):
    def test_yaml_dict_names(self):
        labelmap = load_labelmap("tests/annotations/dict_names.yaml")
        self.assertEqual(labelmap, {0: "cat", 1: "dog", 2: "fish"})
