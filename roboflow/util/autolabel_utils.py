"""Helpers shared by the SDK and CLI for hosted auto-label requests."""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional, Tuple

MODEL_TYPES = ("foundational", "roboflow")


def image_payload(image: str) -> Dict[str, str]:
    """Build the ``{type, value}`` image payload for the auto-label preview endpoint.

    Accepts an HTTP(S) URL, a local file path (read and base64-encoded), or an
    already base64-encoded string.
    """
    if image.startswith(("http://", "https://")):
        return {"type": "url", "value": image}
    if os.path.isfile(image):
        with open(image, "rb") as handle:
            return {"type": "base64", "value": base64.b64encode(handle.read()).decode("ascii")}
    return {"type": "base64", "value": image}


def resolve_model(
    model: str, model_type: str, model_options: Optional[Dict[str, Any]] = None
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Translate the public ``model``/``model_type`` pair into wire values.

    Foundation models are sent as-is (the backend resolves catalog ids such as
    ``gpt-6-astra-boxes``). Roboflow-trained models are sent as
    ``custom_roboflow`` with the model id in ``modelOptions.modelId``.
    """
    if model_type == "roboflow":
        return "custom_roboflow", {**(model_options or {}), "modelId": model}
    if model_type == "foundational":
        return model, model_options
    raise ValueError("model_type must be 'foundational' or 'roboflow'")
