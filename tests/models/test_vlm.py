"""Unit tests for roboflow.models.vlm.VLMModel."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from roboflow.models.vlm import VLMModel


class TestVLMModel(unittest.TestCase):
    def _make(self) -> VLMModel:
        return VLMModel(api_key="k", id="ws/proj/3", name="proj", version="3")

    @patch("roboflow.models.vlm.check_image_url", return_value=True)
    @patch("roboflow.models.vlm.requests.get")
    def test_predict_url_returns_raw_dict(self, mock_get: MagicMock, _chk: MagicMock) -> None:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {"response": {">": "box<loc_1><loc_2><loc_3><loc_4>"}},
        )
        model = self._make()
        result = model.predict("https://example.com/img.jpg")

        self.assertEqual(result, {"response": {">": "box<loc_1><loc_2><loc_3><loc_4>"}})
        called_url = mock_get.call_args[0][0]
        self.assertIn("https://serverless.roboflow.com/proj/3", called_url)
        self.assertIn("api_key=k", called_url)
        self.assertIn("image=", called_url)

    @patch("roboflow.models.vlm.check_image_url", return_value=True)
    @patch("roboflow.models.vlm.requests.get")
    def test_predict_forwards_extra_kwargs_as_query(self, mock_get: MagicMock, _chk: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})
        self._make().predict("https://example.com/img.jpg", prompt="caption")

        called_url = mock_get.call_args[0][0]
        self.assertIn("prompt=caption", called_url)

    @patch("roboflow.models.vlm.check_image_url", return_value=True)
    @patch("roboflow.models.vlm.requests.get")
    def test_predict_non_200_raises(self, mock_get: MagicMock, _chk: MagicMock) -> None:
        mock_get.return_value = MagicMock(status_code=401, text="unauthorized")
        with self.assertRaises(Exception) as ctx:
            self._make().predict("https://example.com/img.jpg")
        self.assertIn("unauthorized", str(ctx.exception))

    @patch("roboflow.models.vlm.os.path.exists", return_value=True)
    @patch("roboflow.models.vlm.Image.open")
    @patch("roboflow.models.vlm.requests.post")
    def test_predict_local_path_posts_base64(
        self, mock_post: MagicMock, mock_open: MagicMock, _exists: MagicMock
    ) -> None:
        mock_img = MagicMock()
        mock_img.convert.return_value = mock_img

        def _save(buf: object, **_kw: object) -> None:
            buf.write(b"fakejpeg")  # type: ignore[attr-defined]

        mock_img.save.side_effect = _save
        mock_open.return_value = mock_img
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"ok": True})

        result = self._make().predict("/tmp/x.jpg")
        self.assertEqual(result, {"ok": True})
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["headers"], {"Content-Type": "application/x-www-form-urlencoded"})
        self.assertIsInstance(kwargs["data"], str)

    def test_predict_missing_local_file_raises(self) -> None:
        with self.assertRaises(Exception) as ctx:
            self._make().predict("/definitely/not/a/real/path.jpg")
        self.assertIn("does not exist", str(ctx.exception))

    def test_endpoint_uses_id_parts_when_version_unset(self) -> None:
        model = VLMModel(api_key="k", id="ws/proj/7")
        model.version = None
        self.assertEqual(model._endpoint(), "https://serverless.roboflow.com/proj/7")


if __name__ == "__main__":
    unittest.main()
