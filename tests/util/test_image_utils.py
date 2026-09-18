import importlib
import sys
import unittest
from unittest import mock

import responses

from roboflow.util import image_utils
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

    @responses.activate
    def test_url_not_found(self):
        url = "https://roboflow.com/not-found.png"
        responses.add(responses.HEAD, url, status=404)
        self.assertFalse(check_image_url(url))


class TestHeifOpenerRegistration(unittest.TestCase):
    def tearDown(self):
        importlib.reload(image_utils)

    def _reload_with_modules(self, modules):
        with mock.patch.dict(sys.modules, modules):
            importlib.reload(image_utils)

    def test_registers_pillow_heif_when_installed(self):
        pillow_heif = mock.MagicMock()
        self._reload_with_modules({"pillow_heif": pillow_heif})
        pillow_heif.register_heif_opener.assert_called_once_with(thumbnails=False)

    def test_does_not_register_pi_heif(self):
        pi_heif = mock.MagicMock()
        # None in sys.modules makes `import pillow_heif` raise ImportError
        self._reload_with_modules({"pillow_heif": None, "pi_heif": pi_heif})
        pi_heif.register_heif_opener.assert_not_called()


class TestLoadLabelmap(unittest.TestCase):
    def test_yaml_dict_names(self):
        labelmap = load_labelmap("tests/annotations/dict_names.yaml")
        self.assertEqual(labelmap, {0: "cat", 1: "dog", 2: "fish"})
