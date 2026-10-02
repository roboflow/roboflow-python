"""Transport for the public Workflow Evals API.

Every route lives under ``{API_URL}/workspaces/{workspace}/workflow-evals``.
This module owns the HTTP mechanics shared by all of them — bearer auth,
``Idempotency-Key`` / ``If-Match`` / ``Deletion-Key`` headers, the
Workflow Evals error envelope, SSE parsing for AI drafting and signed
uploads. The per-endpoint surface lives in
:class:`roboflow.core.workflow_evals.WorkflowEvals`.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterator, Optional
from urllib.parse import quote

import requests

from roboflow.adapters.rfapi import RoboflowError
from roboflow.config import API_URL

DEFAULT_TIMEOUT = 60


class WorkflowEvalError(RoboflowError):
    """Error returned by the Workflow Evals API.

    The server responds with ``{"error": {code, category, retryable,
    requestId, message, details?, resource?}}``. Authentication failures may
    use the platform ``{"error": {message, type, hint?}}`` envelope instead,
    in which case only ``message`` (and ``hint``) are populated.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        *,
        code: Optional[str] = None,
        category: Optional[str] = None,
        retryable: Optional[bool] = None,
        request_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        resource: Optional[Dict[str, Any]] = None,
        hint: Optional[str] = None,
        body: Any = None,
    ) -> None:
        super().__init__(message, status_code=status_code)
        self.message = message
        self.code = code
        self.category = category
        self.retryable = retryable
        self.request_id = request_id
        self.details = details or {}
        self.resource = resource
        self.hint = hint
        self.body = body

    def to_dict(self) -> Dict[str, Any]:
        """Return the error as the CLI's ``{"message": ..., ...}`` payload."""
        payload: Dict[str, Any] = {"message": self.message}
        for key, value in (
            ("status", self.status_code),
            ("code", self.code),
            ("category", self.category),
            ("retryable", self.retryable),
            ("requestId", self.request_id),
            ("resource", self.resource),
            ("hint", self.hint),
        ):
            if value is not None:
                payload[key] = value
        if self.details:
            payload["details"] = self.details
        return payload


def error_from_response(response: requests.Response) -> WorkflowEvalError:
    """Translate a non-2xx response into a :class:`WorkflowEvalError`."""
    body: Any = None
    try:
        body = response.json()
    except ValueError:
        pass
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(error, dict):
        return WorkflowEvalError(
            str(error.get("message") or error.get("code") or response.reason or "Request failed"),
            status_code=response.status_code,
            code=error.get("code"),
            category=error.get("category") or error.get("type"),
            retryable=error.get("retryable"),
            request_id=error.get("requestId"),
            details=error.get("details") if isinstance(error.get("details"), dict) else None,
            resource=error.get("resource"),
            hint=error.get("hint"),
            body=body,
        )
    if isinstance(error, str):
        message = body.get("message") or error
        return WorkflowEvalError(str(message), status_code=response.status_code, code=error, body=body)
    text = (response.text or "").strip() or response.reason or "Request failed"
    return WorkflowEvalError(text, status_code=response.status_code, body=body)


def new_idempotency_key() -> str:
    """Return a fresh UUID v4, the format the API requires for ``Idempotency-Key``."""
    return str(uuid.uuid4())


def base_url(workspace_url: str) -> str:
    return f"{API_URL}/workspaces/{quote(workspace_url, safe='')}/workflow-evals"


def segment(value: Any) -> str:
    """Percent-encode one path segment (resource IDs like ``skill:create-eval``)."""
    return quote(str(value), safe="")


def _headers(
    api_key: str,
    *,
    idempotency_key: Optional[str] = None,
    revision: Optional[Any] = None,
    deletion_key: Optional[str] = None,
    accept: str = "application/json",
) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": accept}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    if revision is not None:
        headers["If-Match"] = str(revision)
    if deletion_key:
        headers["Deletion-Key"] = deletion_key
    return headers


def _clean_params(params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not params:
        return None
    cleaned: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            cleaned[key] = "true" if value else "false"
        elif isinstance(value, (list, tuple)):
            if value:
                cleaned[key] = list(value)
        else:
            cleaned[key] = value
    return cleaned or None


def request(
    api_key: str,
    workspace_url: str,
    method: str,
    path: str = "",
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Any = None,
    idempotency_key: Optional[str] = None,
    revision: Optional[Any] = None,
    deletion_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Any:
    """Call one Workflow Evals route and return the decoded JSON body.

    Returns ``None`` for ``204 No Content``. Raises :class:`WorkflowEvalError`
    for any non-2xx response.
    """
    response = requests.request(
        method.upper(),
        f"{base_url(workspace_url)}{path}",
        params=_clean_params(params),
        json=body,
        headers=_headers(api_key, idempotency_key=idempotency_key, revision=revision, deletion_key=deletion_key),
        timeout=timeout,
    )
    if not 200 <= response.status_code < 300:
        raise error_from_response(response)
    if response.status_code == 204 or not response.content:
        return None
    try:
        return response.json()
    except ValueError:
        raise WorkflowEvalError(
            f"Expected JSON from {method.upper()} {path or '/'}", status_code=response.status_code
        ) from None


def stream_events(
    api_key: str,
    workspace_url: str,
    path: str,
    *,
    body: Any,
    idempotency_key: Optional[str] = None,
    timeout: float = 600,
) -> Iterator[Dict[str, Any]]:
    """POST to an SSE route and yield ``{"event": name, "data": payload}`` dicts.

    ``data`` is decoded as JSON when possible, otherwise left as text. The
    stream ends after the server's terminal ``done`` event.
    """
    response = requests.post(
        f"{base_url(workspace_url)}{path}",
        json=body,
        headers=_headers(api_key, idempotency_key=idempotency_key, accept="text/event-stream"),
        stream=True,
        timeout=timeout,
    )
    if not 200 <= response.status_code < 300:
        raise error_from_response(response)
    try:
        yield from parse_sse(response.iter_lines(decode_unicode=True))
    finally:
        response.close()


def parse_sse(lines: Any) -> Iterator[Dict[str, Any]]:
    """Parse ``text/event-stream`` lines into event dicts."""
    event = "message"
    data_lines: list = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if line is None:
            continue
        if line == "":
            if data_lines:
                yield {"event": event, "data": _decode_data("\n".join(data_lines))}
            event, data_lines = "message", []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "event":
            event = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        yield {"event": event, "data": _decode_data("\n".join(data_lines))}


def _decode_data(data: str) -> Any:
    try:
        return json.loads(data)
    except ValueError:
        return data


def put_signed_upload(upload_url: str, data: bytes, headers: Dict[str, str], *, timeout: float = 300) -> None:
    """PUT bytes to a signed upload URL with exactly the headers the API returned."""
    response = requests.put(upload_url, data=data, headers=headers, timeout=timeout)
    if not 200 <= response.status_code < 300:
        raise WorkflowEvalError(
            f"Signed upload failed ({response.status_code}): {response.text}", status_code=response.status_code
        )
