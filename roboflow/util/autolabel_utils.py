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


OntologyEntry = Dict[str, str]
Ontology = Union[Dict[str, str], Iterable[Union[str, OntologyEntry]]]


def ontology_payload(ontology: Optional[Ontology]) -> Optional[List[OntologyEntry]]:
    """Serialize an ontology into the ``[{"class", "prompt"}]`` wire form.

    Accepts, in order of convenience:

    * ``{"cat": "a cat"}`` — one prompt per class, the common case.
    * ``["cat", "dog"]`` — each class is its own prompt.
    * ``[{"class": "cat", "prompt": "kitten"}, {"class": "cat", "prompt": "tabby"}]``
      — several prompts for one class, which a class-keyed dict cannot express
      because its keys have to be unique. ``prompt`` defaults to ``class``.

    The API's own object form is ``{prompt: class}``, the ``CaptionOntology``
    shape the labeling worker consumes, so a bare dict is ambiguous on the
    wire: the two sides are both strings and only key order says which is
    which. The list form names them, and the backend normalizes it on both the
    preview and the start path.
    """
    if ontology is None:
        return None
    if isinstance(ontology, str):
        raise ValueError(f"ontology must be a mapping or a list of classes, not a bare string {ontology!r}")
    if isinstance(ontology, dict):
        entries = [{"class": name, "prompt": prompt} for name, prompt in ontology.items()]
    else:
        entries = [_ontology_entry(item) for item in ontology]
    _reject_ambiguous_prompts(entries)
    return entries


def _ontology_entry(item: Union[str, OntologyEntry]) -> OntologyEntry:
    if isinstance(item, str):
        return {"class": item, "prompt": item}
    if isinstance(item, dict) and "class" in item:
        return {"class": item["class"], "prompt": item.get("prompt", item["class"])}
    raise ValueError(
        f"ontology entries must be a class name or a {{'class': ..., 'prompt': ...}} mapping, got {item!r}"
    )


def _reject_ambiguous_prompts(entries: List[OntologyEntry]) -> None:
    """Refuse a prompt claimed by two classes.

    The backend keys its ontology by prompt, so it would keep whichever class
    came last and silently drop the other, leaving that class unlabeled for the
    whole job with nothing in the response to explain why.
    """
    by_prompt: Dict[str, str] = {}
    for entry in entries:
        claimed = by_prompt.setdefault(entry["prompt"], entry["class"])
        if claimed != entry["class"]:
            raise ValueError(
                f"ontology maps the prompt {entry['prompt']!r} to both {claimed!r} and "
                f"{entry['class']!r}. The API keys its ontology by prompt, so one of the two "
                "classes would be dropped. Give each class a distinct prompt."
            )


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
