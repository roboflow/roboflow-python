"""Helpers shared by the SDK and CLI for hosted auto-label requests."""

from __future__ import annotations

import base64
import binascii
import os
from typing import Any, Dict, Iterable, Optional, Tuple, Union

MODEL_TYPES = ("foundational", "roboflow")


def image_payload(image: str) -> Dict[str, str]:
    """Build the ``{type, value}`` image payload for the auto-label preview endpoint.

    Accepts an HTTP(S) URL, a local file path (``~`` is expanded; the file is
    read and base64-encoded) or an already base64-encoded string. Anything
    else is treated as a mistyped path and rejected here, rather than being
    sent to the API as "base64" and failing there with a generic inference
    error.

    Raises:
        ValueError: ``image`` is neither a URL, an existing file nor base64.
        OSError: the file exists but cannot be read.
    """
    if image.startswith(("http://", "https://")):
        return {"type": "url", "value": image}
    path = os.path.expanduser(image)
    if os.path.isfile(path):
        with open(path, "rb") as handle:
            return {"type": "base64", "value": base64.b64encode(handle.read()).decode("ascii")}
    compact = "".join(image.split())
    if _is_base64(compact):
        return {"type": "base64", "value": compact}
    raise ValueError(f"Image file not found: {image} (expected an HTTPS URL, an existing file path or base64 data)")


def _is_base64(value: str) -> bool:
    if not value:
        return False
    try:
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


Ontology = Union[Dict[str, str], Iterable[str]]


def ontology_payload(ontology: Optional[Ontology]) -> Optional[Dict[str, str]]:
    """Normalize an ontology into the API's ``{prompt: class name}`` object.

    Note the direction: the **key is the text prompt** sent to the model and the
    **value is the class name** written onto the annotations. It reads backwards
    at first, but it is the shape that lets several prompts collapse onto one
    output class, which is what the ontology is for::

        {"kitten": "cat", "tabby": "cat", "puppy": "dog"}

    A class-keyed object could not express that, since its keys would have to be
    unique. This is also the ``CaptionOntology`` shape the labeling worker
    consumes, so nothing is translated on the way out.

    A plain iterable of class names is expanded to ``{"cat": "cat"}``, each class
    prompted with its own name.
    """
    if ontology is None:
        return None
    if isinstance(ontology, str):
        raise ValueError(f"ontology must be a mapping or a list of classes, not a bare string {ontology!r}")
    if isinstance(ontology, dict):
        return dict(ontology)
    return {name: name for name in ontology}


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
