"""Model evaluation results — wraps the public ``/model-evals`` REST surface.

A :class:`ModelEval` is a thin lazy wrapper around a single evaluation run.
The constructor accepts the eval id (and optional cached metadata from a list
response); each panel (``map_results``, ``confusion_matrix``, etc.) is fetched
on demand and returned as the raw JSON dict the server emits.

The shape mirrors the REST endpoints documented at
``docs.roboflow.com/api-reference/model-evaluations``. Errors surface as
typed :mod:`roboflow.adapters.rfapi` subclasses so callers can distinguish
"eval doesn't exist" from "eval still running" without parsing strings.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from roboflow.adapters import rfapi


class ModelEval:
    """A single model-evaluation run.

    Construct via :meth:`roboflow.core.workspace.Workspace.eval` or list via
    :meth:`roboflow.core.workspace.Workspace.evals`. Direct construction is
    supported when you already hold an eval id::

        from roboflow.core.model_eval import ModelEval
        ev = ModelEval(api_key, "lee-sandbox", "huUF720inUcymARwqAGK")
        ev.refresh()  # populates .status, .summary, .config, etc.
    """

    def __init__(
        self,
        api_key: str,
        workspace_url: str,
        eval_id: str,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._api_key = api_key
        self._workspace_url = workspace_url
        self.id = eval_id
        # Populate metadata from a cached list/get response when available; the
        # caller can still refresh() to re-fetch from the server.
        self._apply(info or {})

    # -- internal -----------------------------------------------------------

    def _apply(self, info: Dict[str, Any]) -> None:
        self.status: Optional[str] = info.get("status")
        self.project_id: Optional[str] = info.get("projectId")
        self.version_id: Optional[str] = info.get("versionId")
        self.model_id: Optional[str] = info.get("modelId")
        self.created_at: Optional[str] = info.get("createdAt")
        self.config: Dict[str, Any] = info.get("config", {}) or {}
        self.summary: Optional[Dict[str, Any]] = info.get("summary")
        self._raw: Dict[str, Any] = info

    # -- core ---------------------------------------------------------------

    def refresh(self) -> "ModelEval":
        """Re-fetch the eval header (status, summary, config) from the server."""
        info = rfapi.get_model_eval(self._api_key, self._workspace_url, self.id)
        self._apply(info)
        return self

    # -- panel accessors ----------------------------------------------------

    def map_results(self) -> Dict[str, Any]:
        """Per-split mAP results (mAP50, mAP50-95, mAP75, by object size, per class)."""
        return rfapi.get_model_eval_map_results(self._api_key, self._workspace_url, self.id)

    def confidence_sweep(self) -> Dict[str, Any]:
        """Confidence-threshold sweep (precision/recall/F1) for the test split."""
        return rfapi.get_model_eval_confidence_sweep(self._api_key, self._workspace_url, self.id)

    def performance_by_class(self, split: Optional[str] = None) -> Dict[str, Any]:
        """Per-class precision / recall / F1 / mAP for the chosen split.

        ``split`` defaults to ``"test"`` server-side. Passing ``"all"`` raises
        :class:`rfapi.InvalidSplitError` — this panel does not support an
        aggregate view.
        """
        return rfapi.get_model_eval_performance_by_class(self._api_key, self._workspace_url, self.id, split=split)

    def confusion_matrix(
        self,
        split: Optional[str] = None,
        confidence: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Confusion matrix (classes + matrix) for *split* at integer *confidence* (0-100)."""
        return rfapi.get_model_eval_confusion_matrix(
            self._api_key, self._workspace_url, self.id, split=split, confidence=confidence
        )

    def vector_analysis(self, confidence: Optional[int] = None) -> Dict[str, Any]:
        """Embedding-cluster diagnostics (per-cluster sample images + metrics)."""
        return rfapi.get_model_eval_vector_analysis(self._api_key, self._workspace_url, self.id, confidence=confidence)

    def image_predictions(
        self,
        split: Optional[str] = None,
        confidence: Optional[int] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Paginated per-image stats (TP/FP/FN counts, augmentations, cluster id)."""
        return rfapi.get_model_eval_image_predictions(
            self._api_key,
            self._workspace_url,
            self.id,
            split=split,
            confidence=confidence,
            limit=limit,
            offset=offset,
        )

    def recommendations(self) -> Dict[str, Any]:
        """Server-generated suggestions for improving the model."""
        return rfapi.get_model_eval_recommendations(self._api_key, self._workspace_url, self.id)

    # -- helpers ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return the cached eval metadata as a plain dict (id + last header fetch)."""
        data: Dict[str, Any] = {"id": self.id}
        # Prefer raw payload (preserves keys we don't surface as attrs); fall
        # back to attributes when only the constructor was called with no info.
        if self._raw:
            return {**self._raw, "id": self.id}
        for key in ("status", "projectId", "versionId", "modelId", "createdAt", "config", "summary"):
            attr = (
                key
                if key in {"status", "config", "summary"}
                else {
                    "projectId": "project_id",
                    "versionId": "version_id",
                    "modelId": "model_id",
                    "createdAt": "created_at",
                }[key]
            )
            value = getattr(self, attr, None)
            if value is not None:
                data[key] = value
        return data

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"ModelEval(id={self.id!r}, status={self.status!r}, project={self.project_id!r})"


__all__: List[str] = ["ModelEval"]
