"""Helpers shared by the SDK and CLI for hosted auto-label requests."""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

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


def ontology_payload(
    ontology: Optional[Union[Dict[str, str], Iterable[str]]],
) -> Optional[List[Dict[str, str]]]:
    """Serialize an ontology into the ``[{"class", "prompt"}]`` wire form.

    The public SDK/CLI signature is the intuitive ``{"class name": "text
    prompt"}``. The API's object form means the opposite (``{prompt: class}``,
    the ``CaptionOntology`` shape the labeling worker consumes), so a bare dict
    is ambiguous on the wire. The list form is explicit about which side is
    which, is normalized by the backend for every endpoint, and is what the web
    app already sends.

    A plain iterable of class names is treated as ``{"cat": "cat"}`` — each
    class is its own prompt.
    """
    if ontology is None:
        return None
    if isinstance(ontology, dict):
        return [{"class": name, "prompt": prompt} for name, prompt in ontology.items()]
    return [{"class": name, "prompt": name} for name in ontology]


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
