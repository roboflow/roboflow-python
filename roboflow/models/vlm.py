"""Vision-language (text-image-pairs) hosted inference.

Wraps the serverless endpoint for VLM-style projects (e.g. PaliGemma).
Unlike detection/classification models, the response shape is free-form:
captions, VQA answers, OCR text, or tokenized detections depending on the
underlying model. `predict` returns the raw serverless JSON unchanged so
callers can interpret the payload for their specific model.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.parse
from typing import Any, Optional

import requests
from PIL import Image

from roboflow.models.inference import InferenceModel
from roboflow.util.image_utils import check_image_url


class VLMModel(InferenceModel):
    """Run inference on a hosted text-image-pairs (VLM) model."""

    def __init__(
        self,
        api_key: str,
        id: str,
        name: Optional[str] = None,
        version: Optional[str] = None,
        local: Optional[str] = None,
        colors: Optional[dict] = None,
        preprocessing: Optional[dict] = None,
    ) -> None:
        super().__init__(api_key, id, version=version)
        self.__api_key = api_key
        self.id = id
        self.name = name
        self.version = version
        self.base_url = local if local else "https://serverless.roboflow.com/"
        self.colors = {} if colors is None else colors
        self.preprocessing = {} if preprocessing is None else preprocessing

    def _endpoint(self) -> str:
        parts = self.id.rsplit("/")
        without_workspace = parts[1]
        version = self.version
        if not version and len(parts) > 2:
            version = parts[2]
        base = self.base_url if self.base_url.endswith("/") else self.base_url + "/"
        return f"{base}{without_workspace}/{version}"

    def predict(self, image_path: str, **kwargs: Any) -> dict:  # type: ignore[override]
        """Run inference and return the raw serverless response.

        Args:
            image_path: local path or http(s) URL to an image.
            **kwargs: extra query params forwarded to the endpoint.

        Returns:
            The raw JSON response as a dict. Shape depends on the underlying
            VLM (e.g. `{"response": {">": "..."}}` for PaliGemma).
        """
        is_url = urllib.parse.urlparse(image_path).scheme in ("http", "https")

        params: dict[str, Any] = {"api_key": self.__api_key}
        params.update(kwargs)

        if is_url:
            if not check_image_url(image_path):
                raise Exception(f"Image URL is not reachable: {image_path}")
            params["image"] = image_path
            url = f"{self._endpoint()}?{urllib.parse.urlencode(params)}"
            resp = requests.get(url)
        else:
            if not os.path.exists(image_path):
                raise Exception(f"Image does not exist at {image_path}!")
            image = Image.open(image_path).convert("RGB")
            buffered = io.BytesIO()
            image.save(buffered, quality=90, format="JPEG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
            url = f"{self._endpoint()}?{urllib.parse.urlencode(params)}"
            resp = requests.post(
                url,
                data=img_b64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )

        if resp.status_code != 200:
            raise Exception(resp.text)

        return resp.json()
