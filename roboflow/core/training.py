"""DNA-style Training / TrainedModel objects for MMPV (multiple-models-per-version).

A Version owns many Trainings; each Training owns one or more Models (a NAS run
owns many). These objects couple to the v2 trainings adapter (``rfapi``), which
mirrors the platform's DNA operations 1:1 — the legacy-vs-MMPV branch lives on
the backend, never here.
"""

from __future__ import annotations

import json
import os
from typing import List

import requests

from roboflow.adapters import rfapi
from roboflow.config import (
    CLASSIFICATION_MODEL,
    INSTANCE_SEGMENTATION_MODEL,
    KEYPOINT_DETECTION_MODEL,
    OBJECT_DETECTION_MODEL,
    OBJECT_DETECTION_URL,
    SEMANTIC_SEGMENTATION_MODEL,
    SEMANTIC_SEGMENTATION_URL,
    TASK_CLS,
    TASK_OBB,
    TASK_POSE,
    TASK_SEG,
    TASK_SEM,
)
from roboflow.models.inference import InferenceModel
from roboflow.util.model_processor import task_of_model_type


def _serverless_base_url_for_task(task: str) -> str:
    if task == TASK_SEM:
        return SEMANTIC_SEGMENTATION_URL
    return OBJECT_DETECTION_URL


def _prediction_type_for_task(task: str) -> str:
    if task == TASK_CLS:
        return CLASSIFICATION_MODEL
    elif task == TASK_SEG:
        return INSTANCE_SEGMENTATION_MODEL
    elif task == TASK_SEM:
        return SEMANTIC_SEGMENTATION_MODEL
    elif task == TASK_POSE:
        return KEYPOINT_DETECTION_MODEL
    elif task == TASK_OBB:
        return OBJECT_DETECTION_MODEL
    else:
        return OBJECT_DETECTION_MODEL


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
        self._video_model_cache = None

    def predict(self, image_path, hosted=False, confidence=40, overlap=30, format="json", **kwargs):
        """Run hosted inference on an image by this model's id.

        The id is passed straight to serverless, which resolves the model and
        its task. Returns a ``PredictionGroup``. Set ``hosted=True`` when
        ``image_path`` is a public URL.
        """
        task = task_of_model_type(self.model_type or "")
        prediction_type = _prediction_type_for_task(task)
        base_url = _serverless_base_url_for_task(task).rstrip("/")
        model = InferenceModel(self.__api_key, "BASE_MODEL")
        model.api_url = f"{base_url}/{str(self.model_id).strip('/')}"
        model.colors = {}

        params = {"confidence": confidence, "overlap": overlap, "format": format}
        params.update(kwargs)
        return model.predict(image_path, prediction_type=prediction_type, **params)

    def _video_model(self):
        """Build (and cache) the legacy inference model used for video inference.

        Video upload and result polling still flow through the legacy
        ``/videoinfer`` endpoints, which the task-specific models implement.
        Caching keeps ``predict_video`` and the poll methods on one underlying
        object, so a job started here can be polled without re-passing its id.
        """
        if self._video_model_cache is not None:
            return self._video_model_cache

        from roboflow.models.classification import ClassificationModel
        from roboflow.models.instance_segmentation import InstanceSegmentationModel
        from roboflow.models.keypoint_detection import KeypointDetectionModel
        from roboflow.models.object_detection import ObjectDetectionModel
        from roboflow.models.semantic_segmentation import SemanticSegmentationModel

        task = task_of_model_type(self.model_type or "")
        legacy_class = {
            TASK_CLS: ClassificationModel,
            TASK_SEG: InstanceSegmentationModel,
            TASK_SEM: SemanticSegmentationModel,
            TASK_POSE: KeypointDetectionModel,
        }.get(task, ObjectDetectionModel)

        legacy_id = f"{self.workspace}/{self.project}/{self._weights_id}"
        self._video_model_cache = legacy_class(self.__api_key, legacy_id)
        return self._video_model_cache

    def predict_video(self, video_path, fps=5, additional_models=None, prediction_type="batch-video"):
        """Run hosted video inference for this model (DNA-era equivalent of the
        legacy ``version.model.predict_video``).

        Delegates to the task-appropriate legacy inference model built from this
        model's id, so a ``TrainedModel`` can do everything the old
        ``version.model`` could. Returns ``(job_id, signed_url, expires)``; poll
        with :meth:`poll_until_video_results` on the same object.

        NOTE: the legacy ``/videoinfer`` payload is keyed by ``<dataset>/<version>``.
        For MMPV models addressed by ``<workspace>/<model-slug>`` this routes the
        slug through as the version segment; verify against staging before relying
        on it for slug-addressed models.
        """
        return self._video_model().predict_video(
            video_path, fps=fps, additional_models=additional_models, prediction_type=prediction_type
        )

    def poll_for_video_results(self, job_id=None) -> dict:
        """Check once for this model's video inference results (DNA-era equivalent
        of the legacy ``version.model.poll_for_video_results``).

        Returns ``{}`` while the job is still running. Defaults to the job started
        by the most recent :meth:`predict_video` call on this object.
        """
        return self._video_model().poll_for_video_results(job_id)

    def poll_until_video_results(self, job_id=None) -> dict:
        """Block until this model's video inference job completes, returning the
        results (DNA-era equivalent of the legacy
        ``version.model.poll_until_video_results``).

        Defaults to the job started by the most recent :meth:`predict_video` call
        on this object.
        """
        return self._video_model().poll_until_video_results(job_id)

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
        self._models_cache = None

    @property
    def models(self) -> List["TrainedModel"]:
        """The models this run produced (DNA ``trainings.get`` → ``models[]``)."""
        if self._models_cache is not None:
            return self._models_cache

        bundle = rfapi.get_training(
            self.__api_key, self.workspace, self.project, self.version, training_id=self.training_id
        )
        bundle_model_type = bundle.get("modelType") or self.model_type
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
                    model_type=entry.get("modelType") or bundle_model_type,
                    metrics=entry.get("metrics"),
                )
            )
        self._models_cache = models
        return self._models_cache

    def refresh(self) -> "Training":
        """Re-read this run's status/results from the backend in place."""
        bundle = rfapi.get_training(
            self.__api_key, self.workspace, self.project, self.version, training_id=self.training_id
        )
        self._raw.update(bundle)
        self.training_id = bundle.get("trainingId") or bundle.get("id") or self.training_id
        self.status = bundle.get("status", self.status)
        self.model_type = bundle.get("modelType", self.model_type)
        self.model_group = bundle.get("modelGroup", self.model_group)
        self.model_ids = bundle.get("modelIds", self.model_ids)
        self._models_cache = None
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
