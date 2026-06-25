"""DNA-style Training / TrainedModel objects for MMPV (multiple-models-per-version).

A Version owns many Trainings; each Training owns one or more Models (a NAS run
owns many). These objects couple to the v2 trainings adapter (``rfapi``), which
mirrors the platform's DNA operations 1:1 — the legacy-vs-MMPV branch lives on
the backend, never here.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.parse
from typing import List

import requests

from roboflow.adapters import rfapi
from roboflow.config import OBJECT_DETECTION_MODEL, OBJECT_DETECTION_URL
from roboflow.util.prediction import PredictionGroup


class TrainedModel:
    """A single trained model produced by a Training.

    Wraps an inference-style model id of either form — ``<dataset>/<version>``
    (SMPV) or ``<workspace>/<model-slug>`` (MMPV). Inference goes to the
    serverless host by that id (which the server resolves to the model and its
    task); weights download keys off the id's addressable segment.
    """

    def __init__(self, api_key, workspace, project, model_id, model_type=None, metrics=None):
        self.__api_key = api_key
        self.workspace = workspace
        self.project = project
        self.model_id = model_id
        self.model_type = model_type
        self.metrics = metrics
        # The second segment addresses the model on /ptFile: a model slug for
        # MMPV, a version number for SMPV.
        self._weights_id = model_id.split("/", 1)[1] if "/" in str(model_id) else model_id

    def predict(self, image_path, hosted=False, confidence=40, overlap=30, format="json", **kwargs):
        """Run hosted inference on an image by this model's id.

        The id is passed straight to serverless, which resolves the model and
        its task. Returns a ``PredictionGroup``. Set ``hosted=True`` when
        ``image_path`` is a public URL.
        """
        base = OBJECT_DETECTION_URL if OBJECT_DETECTION_URL.endswith("/") else OBJECT_DETECTION_URL + "/"
        params = {"api_key": self.__api_key, "confidence": confidence, "overlap": overlap, "format": format}
        params.update(kwargs)
        api_url = f"{base}{self.model_id}?{urllib.parse.urlencode(params)}"

        if hosted:
            api_url += "&image=" + urllib.parse.quote_plus(image_path)
            resp = requests.post(api_url)
            image_dims = {"width": "0", "height": "0"}
        else:
            from PIL import Image

            image = Image.open(image_path).convert("RGB")
            dimensions = image.size
            image_dims = {"width": str(dimensions[0]), "height": str(dimensions[1])}
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            encoded = base64.b64encode(buffered.getvalue()).decode("ascii")
            resp = requests.post(api_url, data=encoded, headers={"Content-Type": "application/x-www-form-urlencoded"})

        resp.raise_for_status()
        return PredictionGroup.create_prediction_group(
            resp.json(),
            image_path=image_path,
            prediction_type=OBJECT_DETECTION_MODEL,
            image_dims=image_dims,
            colors={},
        )

    def download(self, format="pt", location="."):
        """Download this model's PyTorch weights to ``location/weights.pt``."""
        weights_url = rfapi.get_model_weights_url(
            self.__api_key, self.workspace, self.project, self._weights_id, model_format=format
        )
        os.makedirs(location, exist_ok=True)
        out_path = os.path.join(location, "weights.pt")
        response = requests.get(weights_url, stream=True)
        response.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return out_path

    def __str__(self):
        return json.dumps(
            {"model_id": self.model_id, "model_type": self.model_type, "metrics": self.metrics},
            indent=2,
        )


class Training:
    """One training run on a dataset version.

    A version may own many trainings; a NAS run produces many models. Couples to
    the v2 trainings adapter — ``.models`` resolves the run's produced models via
    ``trainings.get``.
    """

    def __init__(self, api_key, workspace, project, version, raw):
        self.__api_key = api_key
        self.workspace = workspace
        self.project = project
        self.version = version
        self._raw = raw or {}
        self.training_id = self._raw.get("trainingId") or self._raw.get("id")
        self.status = self._raw.get("status")
        self.model_type = self._raw.get("modelType")
        self.model_group = self._raw.get("modelGroup")
        self.model_ids = self._raw.get("modelIds", []) or []

    @property
    def models(self) -> List["TrainedModel"]:
        """The models this run produced (DNA ``trainings.get`` → ``models[]``)."""
        bundle = rfapi.get_training(
            self.__api_key, self.workspace, self.project, self.version, training_id=self.training_id
        )
        models = []
        for entry in bundle.get("models", []) or []:
            model_id = entry.get("modelId")
            if not model_id:
                continue
            models.append(
                TrainedModel(
                    self.__api_key,
                    self.workspace,
                    self.project,
                    model_id,
                    model_type=entry.get("modelType"),
                    metrics=entry.get("metrics"),
                )
            )
        return models

    def refresh(self) -> "Training":
        """Re-read this run's status/results from the backend in place."""
        bundle = rfapi.get_training(
            self.__api_key, self.workspace, self.project, self.version, training_id=self.training_id
        )
        self._raw.update(bundle)
        self.status = bundle.get("status", self.status)
        return self

    def cancel(self, continue_if_no_refund: bool = False):
        """Cancel this run immediately (DNA ``trainings.cancel``)."""
        return rfapi.cancel_training_v2(
            self.__api_key,
            self.workspace,
            self.project,
            self.version,
            training_id=self.training_id,
            continue_if_no_refund=continue_if_no_refund,
        )

    def stop(self):
        """Request a graceful early stop on this run (DNA ``trainings.stop``)."""
        return rfapi.stop_training_v2(
            self.__api_key, self.workspace, self.project, self.version, training_id=self.training_id
        )

    def __str__(self):
        return json.dumps(
            {
                "training_id": self.training_id,
                "status": self.status,
                "model_type": self.model_type,
                "model_group": self.model_group,
            },
            indent=2,
        )
