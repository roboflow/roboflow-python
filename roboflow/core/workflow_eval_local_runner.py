"""Run a local Workflow Eval Run (``executionMode: "browser"``) without a browser tab.

The runner does what the web app's browser run does: it reads each Execution's
frozen Cases page by page, calls the user's Inference server
(``POST {url}/workflows/run``) with the same request body as the browser worker,
and submits the raw output through the existing ``capture`` route. Scoring stays
on the server. A Case whose inputs failed to resolve (``workflowInputFailures``)
is not sent to the Inference server; it is captured as failed with the first
failure's message, as the engine does.

Differences from the browser, all deliberate:

* A sliding window of ``concurrency`` inferences (no per-page lockstep), and
  capture submission outside the inference slot.
* Up to 3 attempts for network errors and HTTP 408/429/5xx from the Inference
  server, before anything is submitted.
* A connection failure (connection refused or reset, DNS failure, connect
  timeout, connection dropped mid-response) is infrastructure, not a Case
  failure: when it persists after the 3 attempts nothing is captured and the
  Case stays pending on the server, so the next run picks it up. A run that
  leaves Cases pending ends with :class:`InferenceServerUnreachable` (exit 1);
  10 consecutive connection failures stop the run at once. A read timeout
  (inference slower than ``case_timeout``) and HTTP 408/429/5xx that persist
  are captured as failures with ``retryable: true``.
* A local journal (``<state-dir>/journal.jsonl`` plus ``captures/<sha256>.json``)
  so that a restarted runner resubmits the exact captured bytes instead of
  running the inference again.

Capture responses and what the runner does:

| Response                                          | Action                                             |
| ------------------------------------------------- | -------------------------------------------------- |
| 200                                               | Submitted.                                         |
| 409 "Different capture already submitted"         | ``done_by_other`` (counted as already done).       |
| 409 "Capture is being processed" (retryable)      | Wait for the lease, resend the same bytes.         |
| 409 "Case failure is not retryable"               | ``skipped``.                                       |
| 409 "Run no longer accepts captures"              | Stop dispatching; the Run is closed.               |
| 412 "Run attempt changed"                         | Re-read the Run, restart the Execution.            |
| 409 "No retained capture" (on a ``resume``)       | Re-read the Run, restart the Execution.            |
| input-cases page with a different attemptCount   | Re-read the Run, restart the Execution.            |
| 413                                               | Submit a failure capture instead (too large).      |
| 404                                               | Exit 3.                                            |
| 400                                               | Exit 1, with the server's error.                   |

A restarted Execution reads its Cases again: Cases the server already has are
not run again, and journaled captures of the same attempt are resubmitted
byte for byte, so no Case gets a second, different capture in one attempt.

Usage::

    from roboflow.core.workflow_eval_local_runner import LocalRunner, install_signal_handlers

    runner = LocalRunner(evals, eval_id, run_id, concurrency=4)
    with install_signal_handlers(runner):
        result = runner.run()
    print(result.to_dict())

Errors raise :class:`LocalRunnerError`, which carries the CLI exit code
(1 error, 2 auth, 3 not found); an interruption raises
:class:`LocalRunnerInterrupted` (exit code 130).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import requests
import urllib3
from requests.adapters import HTTPAdapter

from roboflow.adapters import workflowevalsapi
from roboflow.adapters.workflowevalsapi import WorkflowEvalError
from roboflow.core.workflow_evals import RUN_TERMINAL_STATES, WorkflowEvals
from roboflow.util import http

logger = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 16
DEFAULT_CASE_TIMEOUT = 300
CAPTURE_TIMEOUT = 300
PREFLIGHT_TIMEOUT = 15
INFERENCE_ATTEMPTS = 3
INFERENCE_BACKOFF_CAP = 30
MAX_CONNECTION_FAILURES = 10
API_BACKOFF_CAP = 60
API_RETRY_BUDGET = 600
# Signed media URLs live 10 minutes; re-read a page older than that minus a 2 minute margin.
PAGE_MAX_AGE = 8 * 60
PROCESSING_RESENDS = 10
MAX_RELOADS = 5
ERROR_MAX_LENGTH = 2000

INFERENCE_HEADERS = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
# The messages Chrome gives the browser worker for the same failures.
TIMEOUT_MESSAGE = "The operation was aborted due to timeout"
FETCH_FAILED_MESSAGE = "Failed to fetch"
CAPTURE_TOO_LARGE_MESSAGE = "Workflow output exceeds the 10 MB capture limit"
PREFLIGHT_HINT = "Start the inference server or pass --inference-url."
RESUME_HINT = "The journal is saved; run the command again to resume."

_RETRYABLE_HTTP = (408, 429)


class LocalRunnerError(Exception):
    """A local run failed. ``exit_code`` is the CLI exit code (1 error, 2 auth, 3 not found).

    ``result`` holds the run summary when one is available (for example a
    cancelled Run), so the CLI can still print it.
    """

    def __init__(
        self,
        message: str,
        *,
        hint: Optional[str] = None,
        exit_code: int = 1,
        details: Optional[Dict[str, Any]] = None,
        result: Optional[LocalRunResult] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.exit_code = exit_code
        self.details = details
        self.result = result

    def to_dict(self) -> Dict[str, Any]:
        """Return ``{"message", "hint"?, "details"?}`` for the CLI's structured error."""
        payload: Dict[str, Any] = {"message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return payload


class LocalRunnerInterrupted(LocalRunnerError):
    """The run was stopped on request (SIGINT/SIGTERM). Exit code 130; the journal is kept."""

    def __init__(self, message: str = "Interrupted.", *, result: Optional[LocalRunResult] = None) -> None:
        super().__init__(message, hint=RESUME_HINT, exit_code=130, result=result)


class _Abandoned(LocalRunnerInterrupted):
    """A wait or retry was abandoned because the runner is stopping."""


class InferenceServerUnreachable(LocalRunnerError):
    """The run finished, but some Cases were not run because the inference server was unreachable.

    Nothing was captured for them, so they are still pending on the server and
    running again (with the inference server up) picks them up. Exit code 1;
    ``result`` holds the run summary.
    """


@dataclass
class ExecutionTally:
    """Per-Execution counts.

    ``failed`` counts submitted captures whose runtime has an error; ``pending``
    counts Cases left uncaptured because the inference server was unreachable.
    """

    id: str
    submitted: int = 0
    already_done: int = 0
    skipped: int = 0
    failed: int = 0
    pending: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "submitted": self.submitted,
            "alreadyDone": self.already_done,
            "skipped": self.skipped,
            "failed": self.failed,
            "pending": self.pending,
        }


@dataclass
class LocalRunResult:
    """Outcome of :meth:`LocalRunner.run`. ``to_dict()`` is the CLI's ``--json`` output."""

    run_id: str
    state: Optional[str]
    executions: List[ExecutionTally]
    wall_clock_seconds: float
    concurrency: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "runId": self.run_id,
            "state": self.state,
            "executions": [execution.to_dict() for execution in self.executions],
            "wallClockSeconds": self.wall_clock_seconds,
            "concurrency": self.concurrency,
        }


# -- inference -----------------------------------------------------------------


def materialize_media(value: Any, urls: Dict[str, Any]) -> Any:
    """Replace every ``{artifactId}`` with ``{..., "type": "url", "value": <signed url>}``.

    Mirrors the web app's ``materializeResultMedia``: records whose
    ``artifactId`` has no signed URL are kept and searched recursively.
    """
    if isinstance(value, list):
        return [materialize_media(item, urls) for item in value]
    if not isinstance(value, dict):
        return value
    artifact_id = value.get("artifactId")
    capability = urls.get(artifact_id) if isinstance(artifact_id, str) else None
    if isinstance(capability, dict):
        return {**value, "type": "url", "value": capability.get("url")}
    return {key: materialize_media(item, urls) for key, item in value.items()}


def case_inputs(item: Dict[str, Any], urls: Dict[str, Any]) -> Dict[str, Any]:
    """Return the Workflow inputs for one ``input-cases`` item, with media materialized."""
    return materialize_media(item.get("workflowInputs") or {}, urls)


def input_failure_message(item: Dict[str, Any]) -> Optional[str]:
    """Return the first ``workflowInputFailures`` message, or ``None`` when every input resolved.

    A Case with an input failure is not sent to the Inference server; it is
    captured as failed, like the engine does (``output: null, runtime: {error}``).
    """
    failures = item.get("workflowInputFailures") or []
    if not failures:
        return None
    first = failures[0]
    message = first.get("message") if isinstance(first, dict) else None
    return _truncate(message if isinstance(message, str) else "Workflow input could not be resolved")


def _dumps(value: Any) -> bytes:
    # JSON.stringify's output: no whitespace, non-ASCII kept, no NaN.
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def build_inference_body(api_key: str, specification: Any, inputs: Any, workflow_id: Any) -> bytes:
    """Serialize the ``/workflows/run`` body exactly as the browser worker does."""
    return _dumps(
        {
            "api_key": api_key,
            "specification": specification,
            "inputs": inputs,
            "workflow_id": workflow_id,
            "is_preview": True,
            "disable_sinks": True,
            "debug": False,
        }
    )


def _reject_constant(name: str) -> Any:
    raise ValueError(f"Invalid JSON constant {name}")


def _parse_json(content: bytes) -> Any:
    """Parse like ``response.json()`` in a browser: invalid JSON (including NaN) is ``None``."""
    try:
        return json.loads(content, parse_constant=_reject_constant)
    except ValueError:
        return None


def _truncate(message: str) -> str:
    # The server limits ``error`` to 2000 UTF-16 code units, as JavaScript counts them.
    encoded = message.encode("utf-16-le")
    if len(encoded) <= ERROR_MAX_LENGTH * 2:
        return message
    return encoded[: ERROR_MAX_LENGTH * 2].decode("utf-16-le", "ignore")


def _body_message(body: Optional[Dict[str, Any]], status: int) -> str:
    if body is not None and isinstance(body.get("message"), str):
        return body["message"]
    if body is not None and isinstance(body.get("errorMsg"), str):
        return body["errorMsg"]
    return f"Workflow returned HTTP {status}"


@dataclass
class InferenceOutcome:
    """One ``/workflows/run`` call, already shaped as the capture's ``output`` and ``runtime``."""

    output: Any
    runtime: Dict[str, Any]
    connection_failure: bool = False
    responded: bool = False

    @property
    def error(self) -> bool:
        return "error" in self.runtime

    @property
    def retryable(self) -> bool:
        return self.runtime.get("retryable") is True


def run_workflow_inference(
    session: requests.Session,
    url: str,
    *,
    api_key: str,
    specification: Any,
    workflow_id: Any,
    inputs: Any,
    timeout: Optional[float],
    monotonic: Callable[[], float] = time.monotonic,
) -> InferenceOutcome:
    """Call ``{url}/workflows/run`` once and classify the result like the browser worker.

    Never raises for transport or HTTP failures; they become an error runtime
    ``{latencyMs, error, timedOut, retryable}``. ``latencyMs`` covers the request
    and the parsing of its body.

    ``connection_failure`` marks connection-level failures: ``requests.ConnectionError``
    (connection refused or reset, DNS failure, ``ConnectTimeout``) and
    ``ChunkedEncodingError`` (connection dropped mid-response). The runner does
    not capture those. A read timeout (``ReadTimeout``, also when it hits while
    reading the body) is a ``timedOut`` failure, not a connection failure.
    """
    body = build_inference_body(api_key, specification, inputs, workflow_id)
    started = monotonic()
    state = {"retryable": False, "timedOut": False, "connection": False, "responded": False}

    def failure(message: str) -> InferenceOutcome:
        runtime = {
            "latencyMs": (monotonic() - started) * 1000,
            "error": _truncate(message),
            "timedOut": state["timedOut"],
            "retryable": state["retryable"],
        }
        return InferenceOutcome(None, runtime, state["connection"], state["responded"])

    try:
        response = session.post(
            f"{url.rstrip('/')}/workflows/run", data=body, headers=dict(INFERENCE_HEADERS), timeout=timeout
        )
        content = response.content
    except requests.Timeout as exc:
        # ConnectTimeout is also a ConnectionError: the server was never reached.
        state.update(retryable=True, timedOut=True, connection=isinstance(exc, requests.ConnectionError))
        return failure(TIMEOUT_MESSAGE)
    except requests.ConnectionError as exc:
        if exc.args and isinstance(exc.args[0], urllib3.exceptions.ReadTimeoutError):
            # requests reports a read timeout while reading the body as a ConnectionError.
            state.update(retryable=True, timedOut=True)
            return failure(TIMEOUT_MESSAGE)
        state.update(retryable=True, connection=True)
        return failure(FETCH_FAILED_MESSAGE)
    except requests.exceptions.ChunkedEncodingError:
        state.update(retryable=True, connection=True)
        return failure(FETCH_FAILED_MESSAGE)
    except requests.exceptions.ContentDecodingError:
        state["retryable"] = True
        return failure(FETCH_FAILED_MESSAGE)
    except requests.RequestException as exc:
        return failure(str(exc) or "Local workflow request failed")
    state["responded"] = True
    parsed = _parse_json(content)
    record = parsed if isinstance(parsed, dict) else None
    status = response.status_code
    if not 200 <= status < 300:
        state["retryable"] = status in _RETRYABLE_HTTP or status >= 500
        return failure(_body_message(record, status))
    if record is None:
        return failure("Workflow returned an invalid response")
    if record.get("success") is False:
        return failure(_body_message(record, status))
    if "outputs" not in record:
        return failure("Workflow returned no outputs")
    outputs = record["outputs"]
    return InferenceOutcome(
        outputs if isinstance(outputs, list) else [outputs],
        {"latencyMs": (monotonic() - started) * 1000},
        responded=True,
    )


def check_inference_server(
    session: requests.Session, url: str, api_key: Optional[str], *, timeout: float = PREFLIGHT_TIMEOUT
) -> None:
    """Preflight: ``POST {url}/workflows/blocks/describe``, the web app's local-server check.

    Raises :class:`LocalRunnerError` (exit code 1) when the server is unreachable
    or answers with a non-2xx status.
    """
    headers = dict(INFERENCE_HEADERS)
    if api_key:
        headers["Api-Key"] = api_key
    try:
        response = session.post(
            f"{url}/workflows/blocks/describe",
            params={"api_key": api_key} if api_key else None,
            data=_dumps({"dynamic_blocks_definitions": []}),
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout:
        raise LocalRunnerError(
            f"The inference server at {url} did not respond within {timeout:g} seconds.", hint=PREFLIGHT_HINT
        ) from None
    except requests.RequestException:
        raise LocalRunnerError(f"Could not reach the inference server at {url}.", hint=PREFLIGHT_HINT) from None
    if not 200 <= response.status_code < 300:
        parsed = _parse_json(response.content)
        body = parsed if isinstance(parsed, dict) else {}
        message = next(
            (body[key] for key in ("message", "detail") if isinstance(body.get(key), str)),
            f"Server returned {response.status_code}.",
        )
        raise LocalRunnerError(
            f"The inference server at {url} failed the preflight check: {message}", hint=PREFLIGHT_HINT
        )


# -- journal -------------------------------------------------------------------


def default_state_dir(workspace: str, run_id: str) -> str:
    """``$XDG_CACHE_HOME`` (or ``~/.cache``) ``/roboflow/workflow-evals/<workspace>/<runId>``."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "roboflow", "workflow-evals", _path_segment(workspace), _path_segment(run_id))


def _path_segment(value: str) -> str:
    return str(value).replace("/", "_").replace(os.sep, "_") or "_"


def _write_atomic(path: str, data: bytes) -> None:
    temporary = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Journal:
    """Append-only record of captures, so a restarted run resubmits bytes instead of re-running inference.

    ``journal.jsonl`` holds ``{"t": "captured", attempt, executionId, caseId,
    shardIndex, sha256, path}`` and ``{"t": "submitted", attempt, executionId,
    caseId, status, outcome}`` lines, each fsynced. ``path`` is relative to the
    state directory. A later ``captured`` line for the same Case attempt
    supersedes earlier ones. Unreadable lines (a torn last write) are ignored.
    """

    def __init__(self, state_dir: str) -> None:
        self.state_dir = state_dir
        self.path = os.path.join(state_dir, "journal.jsonl")
        os.makedirs(os.path.join(state_dir, "captures"), exist_ok=True)
        self._lock = threading.Lock()
        self._captured: Dict[Tuple[int, str, str], Dict[str, Any]] = {}
        self._load()
        self._handle: Optional[Any] = open(self.path, "a", encoding="utf-8")  # noqa: SIM115
        if self._torn_tail():
            # Terminate the torn line so the next record is not glued to it (and lost on reload).
            self._handle.write("\n")
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def _torn_tail(self) -> bool:
        with open(self.path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return False
            handle.seek(-1, os.SEEK_END)
            return handle.read(1) != b"\n"

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except ValueError:
                    continue
                if isinstance(entry, dict) and entry.get("t") == "captured":
                    with contextlib.suppress(KeyError, TypeError, ValueError):
                        self._captured[(int(entry["attempt"]), str(entry["executionId"]), str(entry["caseId"]))] = entry

    def captured(self, attempt: int, execution_id: str, case_id: str) -> Optional[bytes]:
        """Return the stored capture bytes for one Case attempt, or ``None`` (missing or corrupt file)."""
        with self._lock:
            entry = self._captured.get((attempt, execution_id, case_id))
        if entry is None:
            return None
        try:
            with open(os.path.join(self.state_dir, str(entry["path"])), "rb") as handle:
                data = handle.read()
        except (OSError, KeyError):
            return None
        return data if hashlib.sha256(data).hexdigest() == entry.get("sha256") else None

    def record_capture(self, attempt: int, execution_id: str, case_id: str, shard_index: int, data: bytes) -> None:
        """Write the capture file (tmp + fsync + rename), then append its ``captured`` line."""
        digest = hashlib.sha256(data).hexdigest()
        relative = os.path.join("captures", f"{digest}.json")
        target = os.path.join(self.state_dir, relative)
        if not os.path.exists(target):
            _write_atomic(target, data)
        entry = {
            "t": "captured",
            "attempt": attempt,
            "executionId": execution_id,
            "caseId": case_id,
            "shardIndex": shard_index,
            "sha256": digest,
            "path": relative,
        }
        self._append(entry)
        with self._lock:
            self._captured[(attempt, execution_id, case_id)] = entry

    def record_submission(self, attempt: int, execution_id: str, case_id: str, status: int, outcome: str) -> None:
        self._append(
            {
                "t": "submitted",
                "attempt": attempt,
                "executionId": execution_id,
                "caseId": case_id,
                "status": status,
                "outcome": outcome,
            }
        )

    def _append(self, entry: Dict[str, Any]) -> None:
        line = json.dumps(entry, separators=(",", ":")) + "\n"
        with self._lock:
            if self._handle is None:
                return
            self._handle.write(line)
            self._handle.flush()
            os.fsync(self._handle.fileno())

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


# -- runner --------------------------------------------------------------------


def _field(resource: Any, key: str) -> Any:
    """Read a Run field from ``{...}`` or ``{"run": {...}}``."""
    if not isinstance(resource, dict):
        return None
    nested = resource.get("run")
    value = resource.get(key)
    if value is None and isinstance(nested, dict):
        value = nested.get(key)
    return value


def _parse_timestamp(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _already_done(checkpoint: Dict[str, Any], attempt: int) -> bool:
    # browserRunQueue.ts: completed, or failed and not retryable / already this attempt.
    state = checkpoint.get("state")
    return state == "completed" or (
        state == "failed" and (checkpoint.get("retryable") is not True or checkpoint.get("attemptCount") == attempt)
    )


def _api_error(exc: WorkflowEvalError) -> LocalRunnerError:
    status = exc.status_code
    hint = exc.hint
    if status == 401:
        hint = "Check your API key, or run 'roboflow auth login'."
    elif status == 403:
        hint = "The API key needs the workflow-evals:read and workflow-evals:run scopes and cannot be folder-scoped."
    elif status == 400:
        hint = "The server rejected the local runner's request; please report this with the error details."
    exit_code = 2 if status in (401, 403) else 3 if status == 404 else 1
    return LocalRunnerError(exc.message, hint=hint, exit_code=exit_code, details=exc.to_dict())


def _transient(exc: BaseException) -> bool:
    if isinstance(exc, WorkflowEvalError):
        status = exc.status_code
        return status is not None and (status in _RETRYABLE_HTTP or status >= 500)
    return isinstance(exc, requests.RequestException)


@dataclass
class _Pass:
    """One pass over an Execution's Cases at one Run attempt."""

    execution_id: str
    index: int
    total: int
    specification: Any
    workflow_id: Any
    selected: Optional[int]
    tally: ExecutionTally
    attempt: int = 0


@dataclass
class _Page:
    cursor: Optional[str]
    data: Dict[str, Any]
    fetched_at: float
    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.items = {str(item["case"]["id"]): item for item in self.data.get("items") or []}

    @property
    def attempt(self) -> int:
        return int(self.data.get("attemptCount") or 0)

    @property
    def urls(self) -> Dict[str, Any]:
        return self.data.get("artifactReadUrls") or {}

    @property
    def next_cursor(self) -> Optional[str]:
        return self.data.get("nextCursor") or None

    @property
    def closes_run(self) -> bool:
        return bool(self.data.get("cancellationRequested")) or self.data.get("runState") in RUN_TERMINAL_STATES


@dataclass
class _Work:
    ctx: _Pass
    case_id: str
    shard_index: int
    attempt: int
    cursor: Optional[str]


class LocalRunner:
    """Runs a local Workflow Eval Run against a local Inference server.

    Args:
        client: The workspace's :class:`WorkflowEvals` client.
        eval_id: The Eval id.
        run_id: The Run id; it must have ``executionMode: "browser"``.
        inference_url: Inference server URL. Defaults to the URL frozen in the Run.
        inference_api_key: API key sent to the Inference server. Defaults to the client's key.
        execution_ids: Only these Executions (default: all, in Run order).
        concurrency: Inferences in flight at once, 1 to 16.
        case_timeout: Seconds allowed to connect and between bytes read for each inference attempt
            (``requests`` timeout semantics), not a total deadline per Case; 0 means no limit.
        state_dir: Journal directory. Defaults to :func:`default_state_dir`.
        workspace: Workspace for the default ``state_dir``. Defaults to the client's.
        retry: Retry a terminal Run (when the server allows it) before running.
        on_progress: Receives a progress dict after every Case.
        warn: Receives warnings meant for stderr. Defaults to the module logger.
        clock: Wall clock in epoch seconds (leases, page age, API retry budget).
        monotonic: Monotonic clock in seconds (latency, wall-clock time).
        sleep: Sleeps for the given seconds. Defaults to a wait that a stop request interrupts.
        session: ``requests.Session`` for the Inference server.
    """

    def __init__(
        self,
        client: WorkflowEvals,
        eval_id: str,
        run_id: str,
        *,
        inference_url: Optional[str] = None,
        inference_api_key: Optional[str] = None,
        execution_ids: Optional[Sequence[str]] = None,
        concurrency: int = DEFAULT_CONCURRENCY,
        case_timeout: float = DEFAULT_CASE_TIMEOUT,
        state_dir: Optional[str] = None,
        workspace: Optional[str] = None,
        retry: bool = False,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        warn: Optional[Callable[[str], None]] = None,
        clock: Callable[[], float] = time.time,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Optional[Callable[[float], Any]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not 1 <= concurrency <= MAX_CONCURRENCY:
            raise ValueError(f"concurrency must be between 1 and {MAX_CONCURRENCY}")
        if case_timeout < 0:
            raise ValueError("case_timeout must be 0 (no limit) or positive")
        self._client = client
        self._eval_id = eval_id
        self._run_id = run_id
        self._inference_url = inference_url
        self._inference_api_key = inference_api_key or client._api_key
        self._execution_ids = list(dict.fromkeys(execution_ids)) if execution_ids else None
        self._concurrency = concurrency
        self._case_timeout = case_timeout or None
        self._state_dir = state_dir or default_state_dir(workspace or client.workspace_url, run_id)
        self._retry = retry
        self._on_progress = on_progress
        self._warn = warn or logger.warning
        self._clock = clock
        self._monotonic = monotonic
        self._halt = threading.Event()
        self._sleep: Callable[[float], Any] = sleep or self._halt.wait
        self._owns_session = session is None
        self._session = session or self._new_session()

        self._cond = threading.Condition(threading.RLock())
        self._stop_requested = False
        self._abort = False
        self._fatal: Optional[BaseException] = None
        self._run_closed = False
        self._reload = False
        self._in_flight = 0
        self._outstanding = 0
        self._submitting = 0
        self._connection_failures = 0
        self._last_latency_ms: Optional[float] = None
        self._threads: List[threading.Thread] = []
        self._tallies: List[ExecutionTally] = []
        self._journal: Optional[Journal] = None
        self._url = ""
        self._started = 0.0

    def _new_session(self) -> requests.Session:
        session = http.Session()
        adapter = HTTPAdapter(pool_connections=1, pool_maxsize=max(10, self._concurrency))
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    # -- control ---------------------------------------------------------------

    def request_stop(self) -> None:
        """Stop gracefully; a second call aborts at once.

        First call: no new inference is dispatched, in-flight inferences finish
        and are journaled, already queued captures are submitted, then
        :meth:`run` raises :class:`LocalRunnerInterrupted`. Second call:
        :meth:`run` raises immediately without waiting for in-flight requests.
        Safe to call from a signal handler or another thread.
        """
        with self._cond:
            if self._stop_requested:
                self._abort = True
            self._stop_requested = True
            self._halt.set()
            self._cond.notify_all()

    @property
    def state_dir(self) -> str:
        return self._state_dir

    @property
    def aborting(self) -> bool:
        """``True`` once :meth:`request_stop` was called twice."""
        return self._abort

    def _halting(self) -> bool:
        return self._stop_requested or self._abort or self._fatal is not None

    def _dispatch_stopped(self) -> bool:
        return self._halting() or self._run_closed or self._reload

    def _fail(self, exc: BaseException) -> None:
        with self._cond:
            if self._fatal is None:
                self._fatal = exc
            self._halt.set()
            self._cond.notify_all()

    def _wait(self, predicate: Callable[[], bool], then: Optional[Callable[[], None]] = None) -> bool:
        """Wait until ``predicate()`` (under the lock) or an abort; return ``False`` on abort.

        ``then`` runs under the same lock acquisition once the predicate holds.
        """
        with self._cond:
            while not predicate():
                if self._abort:
                    return False
                # A bounded wait keeps the main thread responsive to signals on every platform.
                self._cond.wait(0.5)
            if self._abort:
                return False
            if then is not None:
                then()
            return True

    def _sleep_until(self, deadline: Optional[float]) -> None:
        """Sleep until the wall-clock ``deadline``; raise :class:`_Abandoned` if the runner halts."""
        while deadline is not None and self._clock() < deadline:
            if self._dispatch_stopped():
                raise _Abandoned
            self._sleep(min(deadline - self._clock(), 5.0))

    def _backoff(self, seconds: float, abandon: Optional[Callable[[], bool]] = None) -> None:
        """Sleep before a retry; raise :class:`_Abandoned` if the runner halts (or ``abandon()``)."""

        def abandoned() -> bool:
            return self._halting() or (abandon is not None and abandon())

        if abandoned():
            raise _Abandoned
        self._sleep(seconds)
        if abandoned():
            raise _Abandoned

    # -- Roboflow API ----------------------------------------------------------

    def _api(
        self, fn: Callable[..., Any], *args: Any, abandon: Optional[Callable[[], bool]] = None, **kwargs: Any
    ) -> Any:
        """Call the Roboflow API, retrying timeouts, network errors, 408, 429 and 5xx.

        Backoff 1, 2, 4… up to 60 s; gives up (exit 1) after 10 minutes of
        consecutive failures. Other errors propagate unchanged. Retries stop with
        :class:`_Abandoned` when the runner halts or ``abandon()`` is true.
        """
        first_failure: Optional[float] = None
        delay = 1.0
        while True:
            try:
                return fn(*args, **kwargs)
            except (WorkflowEvalError, requests.RequestException) as exc:
                if not _transient(exc):
                    raise
                now = self._clock()
                first_failure = now if first_failure is None else first_failure
                if now - first_failure + delay > API_RETRY_BUDGET:
                    raise LocalRunnerError(
                        f"The Roboflow API kept failing for {API_RETRY_BUDGET // 60} minutes: {exc}",
                        hint=f"Check your network connection. {RESUME_HINT}",
                    ) from exc
                self._backoff(delay, abandon)
                delay = min(delay * 2, API_BACKOFF_CAP)

    def _call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """:meth:`_api`, mapping API errors to :class:`LocalRunnerError` exit codes."""
        try:
            return self._api(fn, *args, **kwargs)
        except WorkflowEvalError as exc:
            raise _api_error(exc) from exc

    # -- run -------------------------------------------------------------------

    def run(self) -> LocalRunResult:
        """Run every selected Execution to the end and return the summary.

        Returns when the work is done (or the Run was already terminal). Raises
        :class:`LocalRunnerError` on failure, including a cancelled Run (exit 1,
        with ``result`` set), and :class:`LocalRunnerInterrupted` after
        :meth:`request_stop`.
        """
        self._started = self._monotonic()
        try:
            result = self._run()
            if self._abort:
                # An abort never reports success, even if the work happened to finish.
                raise LocalRunnerInterrupted
            return result
        except KeyboardInterrupt:
            with self._cond:
                self._stop_requested = self._abort = True
                self._halt.set()
                self._cond.notify_all()
            raise LocalRunnerInterrupted(result=self._result(None)) from None
        except LocalRunnerInterrupted:
            self._raise_if_halted()
            raise
        finally:
            self._shutdown()

    def _run(self) -> LocalRunResult:
        client, eval_id, run_id = self._client, self._eval_id, self._run_id
        run = self._call(client.get_run, eval_id, run_id)
        if _field(run, "executionMode") != "browser":
            raise LocalRunnerError(
                f"Run {run_id} does not run on a local inference server.",
                hint="This Run executes on Roboflow; use `run get --wait`.",
            )
        if _field(run, "state") in RUN_TERMINAL_STATES:
            retry = _field(run, "retry")
            if not (self._retry and isinstance(retry, dict) and retry.get("allowed")):
                if self._retry:
                    self._warn(f"Run {run_id} is {_field(run, 'state')} and cannot be retried.")
                return self._finish(run)
            self._call(client.retry_run, eval_id, run_id, idempotency_key=workflowevalsapi.new_idempotency_key())

        configuration = self._call(client.get_run_configuration, eval_id, run_id)
        server = ((configuration or {}).get("sharedConfiguration") or {}).get("inferenceServer") or {}
        frozen_url = str(server.get("url") or "").rstrip("/")
        self._url = (self._inference_url or "").rstrip("/") or frozen_url
        if not self._url:
            raise LocalRunnerError(f"Run {run_id} has no inference server URL.", hint="Pass --inference-url.")
        if self._inference_url and frozen_url and self._url != frozen_url:
            self._warn(
                f"Using the inference server {self._url} instead of {frozen_url}; the Run still records {frozen_url}."
            )

        executions = self._select_executions((self._call(client.list_executions, eval_id, run_id) or {}).get("items"))
        check_inference_server(self._session, self._url, self._inference_api_key)
        self._journal = Journal(self._state_dir)
        for index, execution in enumerate(executions, 1):
            if self._run_closed:
                break
            self._run_execution(index, len(executions), execution)
        result = self._finish(self._call(client.get_run, eval_id, run_id))
        pending = sum(tally.pending for tally in self._tallies)
        if pending:
            cases = "1 Case was" if pending == 1 else f"{pending} Cases were"
            raise InferenceServerUnreachable(
                f"{cases} not run because the inference server at {self._url} was unreachable.",
                hint="Start the inference server and run the same command again; captured Cases are not re-run.",
                result=result,
            )
        return result

    def _select_executions(self, items: Any) -> List[Dict[str, Any]]:
        executions = [item for item in items or [] if isinstance(item, dict) and item.get("id") is not None]
        if self._execution_ids:
            by_id = {str(item["id"]): item for item in executions}
            unknown = [execution_id for execution_id in self._execution_ids if execution_id not in by_id]
            if unknown:
                raise LocalRunnerError(
                    f"Execution {', '.join(unknown)} is not in Run {self._run_id}.",
                    hint="List the Run's Executions with `roboflow workflow-eval execution list`.",
                    exit_code=3,
                )
            wanted = set(self._execution_ids)
            executions = [item for item in executions if str(item["id"]) in wanted]
        self._tallies = [ExecutionTally(str(item["id"])) for item in executions]
        return executions

    def _result(self, state: Optional[str]) -> LocalRunResult:
        return LocalRunResult(
            run_id=self._run_id,
            state=state,
            executions=list(self._tallies),
            wall_clock_seconds=round(self._monotonic() - self._started, 3),
            concurrency=self._concurrency,
        )

    def _finish(self, run: Dict[str, Any]) -> LocalRunResult:
        state = _field(run, "state")
        result = self._result(state)
        if state == "cancelled":
            raise LocalRunnerError(
                f"Run {self._run_id} was cancelled.", hint="Start a new Run to evaluate again.", result=result
            )
        return result

    def _raise_if_halted(self) -> None:
        if self._abort:
            raise LocalRunnerInterrupted(result=self._result(None))
        if self._fatal is not None:
            raise self._fatal
        if self._stop_requested:
            raise LocalRunnerInterrupted(result=self._result(None))

    def _shutdown(self) -> None:
        if not self._abort:
            for thread in self._threads:
                thread.join()
            if self._journal is not None:
                self._journal.close()
            if self._owns_session:
                self._session.close()

    # -- executions ------------------------------------------------------------

    def _run_execution(self, index: int, total: int, execution: Dict[str, Any]) -> None:
        execution_id = str(execution["id"])
        configuration = self._call(self._client.get_execution_configuration, self._eval_id, self._run_id, execution_id)
        subject = ((configuration or {}).get("linkedPlan") or {}).get("workflowSubject") or {}
        if subject.get("manifest") is None:
            raise LocalRunnerError(f"Execution {execution_id} has no frozen Workflow specification.")
        tally = self._tallies[index - 1]
        selected = execution.get("selectedCaseCount")
        reloads = 0
        while True:
            ctx = _Pass(
                execution_id=execution_id,
                index=index,
                total=total,
                specification=subject["manifest"],
                workflow_id=subject.get("id"),
                selected=int(selected) if isinstance(selected, (int, float)) else None,
                tally=tally,
            )
            if not self._run_pass(ctx):
                return
            reloads += 1
            if reloads > MAX_RELOADS:
                raise LocalRunnerError(f"The attempt of Run {self._run_id} kept changing; giving up.", hint=RESUME_HINT)
            # 412 "Run attempt changed", 409 "No retained capture" on a resume, or a page at
            # another attempt: reload the Run and restart this Execution at its current attempt.
            # Every capture of the old pass has been answered (the pass drained), so the new pass
            # sees them in the checkpoints; journal entries of an old attempt are ignored.
            run = self._call(self._client.get_run, self._eval_id, self._run_id)
            if _field(run, "state") in RUN_TERMINAL_STATES:
                self._run_closed = True
                return
            tally.submitted = tally.already_done = tally.skipped = tally.failed = tally.pending = 0

    def _run_pass(self, ctx: _Pass) -> bool:
        """Dispatch one pass and drain it. Returns ``True`` when the Execution must restart."""
        with self._cond:
            self._reload = False
        try:
            self._dispatch(ctx)
        except BaseException as exc:
            if not isinstance(exc, LocalRunnerInterrupted):
                self._fail(exc)
            raise
        finally:
            self._wait(lambda: self._outstanding == 0)
        self._raise_if_halted()
        return self._reload and not self._run_closed

    def _fetch_page(self, ctx: _Pass, cursor: Optional[str], abandon: Optional[Callable[[], bool]] = None) -> _Page:
        # Taken before the request: the server signs the URLs after this instant.
        fetched_at = self._clock()
        data = self._call(
            self._client.list_execution_input_cases,
            self._eval_id,
            self._run_id,
            ctx.execution_id,
            limit=self._concurrency,
            cursor=cursor,
            abandon=abandon,
        )
        if not isinstance(data, dict):
            raise LocalRunnerError("The input-cases response is not an object.")
        if any("workflowInputs" not in item for item in data.get("items") or []):
            raise LocalRunnerError(
                "The Roboflow server does not support local runner yet (input-cases has no workflowInputs).",
                hint="Run this Eval from the browser until the server is updated.",
            )
        return _Page(cursor, data, fetched_at)

    def _check_page(self, ctx: _Pass, page: _Page) -> bool:
        """Return ``False`` (and flag why) when dispatching from ``page`` must stop."""
        with self._cond:
            if page.closes_run:
                self._run_closed = True
            elif page.attempt != ctx.attempt:
                self._reload = True
            else:
                return True
            self._cond.notify_all()
            return False

    def _prefetch(self, ctx: _Pass, cursor: str) -> Callable[[], _Page]:
        """Fetch the next page in the background; the returned function waits for it."""
        box: Dict[str, Any] = {}

        def fetch() -> None:
            try:
                # Nobody waits for this page once dispatching stops.
                box["page"] = self._fetch_page(ctx, cursor, abandon=self._dispatch_stopped)
            except BaseException as exc:
                box["error"] = exc
            finally:
                with self._cond:
                    box["done"] = True
                    self._cond.notify_all()

        self._spawn(fetch, holds_outstanding=False)

        def result() -> _Page:
            if not self._wait(lambda: "done" in box):
                raise _Abandoned
            if "error" in box:
                raise box["error"]
            return box["page"]

        return result

    def _dispatch(self, ctx: _Pass) -> None:
        page = self._fetch_page(ctx, None)
        ctx.attempt = page.attempt
        while self._check_page(ctx, page):
            upcoming = self._prefetch(ctx, page.next_cursor) if page.next_cursor else None
            for case_id in list(page.items):
                if self._dispatch_stopped() or not self._dispatch_case(ctx, page, case_id):
                    return
            if upcoming is None or self._dispatch_stopped():
                return
            page = upcoming()

    def _refresh(self, ctx: _Pass, page: _Page) -> bool:
        """Re-read ``page`` (same cursor) if its signed URLs are about to expire."""
        if self._clock() - page.fetched_at <= PAGE_MAX_AGE:
            return True
        fresh = self._fetch_page(ctx, page.cursor)
        page.data, page.fetched_at, page.items = fresh.data, fresh.fetched_at, fresh.items
        return self._check_page(ctx, page)

    def _dispatch_case(self, ctx: _Pass, page: _Page, case_id: str) -> bool:
        """Decide and dispatch one Case. Returns ``False`` when dispatching must stop."""
        while True:
            if not self._refresh(ctx, page):
                return False
            item = page.items.get(case_id)
            if item is None:
                return True
            checkpoint = item.get("checkpoint") or {}
            if _already_done(checkpoint, page.attempt):
                self._count(ctx, "already_done")
                return True
            assert self._journal is not None
            stored = self._journal.captured(page.attempt, ctx.execution_id, case_id)
            resume = (
                stored is None and bool(checkpoint.get("hasCapture")) and checkpoint.get("attemptCount") == page.attempt
            )
            input_failure = input_failure_message(item) if stored is None and not resume else None
            needs_slot = stored is None and not resume and input_failure is None
            # Wait for room in the window (and, for a capture, in the submission backlog).
            if not self._wait(lambda: self._dispatch_stopped() or self._has_room(needs_slot)):
                return False
            if self._dispatch_stopped():
                return False
            if self._clock() - page.fetched_at > PAGE_MAX_AGE:
                continue
            work = _Work(ctx, case_id, int(item.get("shardIndex") or 0), page.attempt, page.cursor)
            with self._cond:
                self._outstanding += 1
                if needs_slot:
                    self._in_flight += 1
            if stored is not None:
                self._spawn(self._submit, work, stored, None)
            elif resume:
                self._spawn(self._submit, work, None, _parse_timestamp(checkpoint.get("leaseExpiresAt")))
            elif input_failure is not None:
                self._spawn(self._submit_input_failure, work, input_failure)
            else:
                self._spawn(self._infer_and_submit, work, case_inputs(item, page.urls))
            self._emit_progress(ctx)
            return True

    def _has_room(self, needs_slot: bool) -> bool:
        # In-flight inferences stay under N; captures waiting for or in submission
        # (N submitting plus a queue of N) bound the backlog, so dispatch blocks when
        # the Roboflow API is slow.
        if needs_slot and self._in_flight >= self._concurrency:
            return False
        return self._outstanding < 2 * self._concurrency

    def _spawn(self, target: Callable[..., None], *args: Any, holds_outstanding: bool = True) -> None:
        def body() -> None:
            try:
                target(*args)
            except LocalRunnerInterrupted:
                pass
            except BaseException as exc:
                self._fail(exc)
            finally:
                if holds_outstanding:
                    with self._cond:
                        self._outstanding -= 1
                        self._cond.notify_all()

        thread = threading.Thread(target=body, name="roboflow-workflow-eval-local", daemon=True)
        self._threads = [alive for alive in self._threads if alive.is_alive()]
        self._threads.append(thread)
        thread.start()

    # -- per Case --------------------------------------------------------------

    def _infer_and_submit(self, work: _Work, inputs: Dict[str, Any]) -> None:
        outcome: Optional[InferenceOutcome] = None
        data = b""
        try:
            outcome = self._infer(work.ctx, inputs)
            if outcome is None or outcome.connection_failure:
                # Unreachable inference server: infrastructure, not a Case failure. Nothing is
                # captured or journaled, so the Case stays pending on the server.
                return
            capture = {
                "attemptCount": work.attempt,
                "shardIndex": work.shard_index,
                "output": outcome.output,
                "runtime": outcome.runtime,
            }
            data = _dumps(capture)
            assert self._journal is not None
            self._journal.record_capture(work.attempt, work.ctx.execution_id, work.case_id, work.shard_index, data)
        finally:
            # The slot frees as soon as the inference result is journaled.
            with self._cond:
                self._in_flight -= 1
                if outcome is not None:
                    self._last_latency_ms = outcome.runtime.get("latencyMs")
                self._cond.notify_all()
            if outcome is not None and outcome.connection_failure:
                self._count(work.ctx, "pending")
        self._submit(work, data, None)

    def _submit_input_failure(self, work: _Work, message: str) -> None:
        # The engine's input failure capture: no inference, and a runtime with only ``error``.
        capture = {
            "attemptCount": work.attempt,
            "shardIndex": work.shard_index,
            "output": None,
            "runtime": {"error": message},
        }
        data = _dumps(capture)
        assert self._journal is not None
        self._journal.record_capture(work.attempt, work.ctx.execution_id, work.case_id, work.shard_index, data)
        self._submit(work, data, None)

    def _infer(self, ctx: _Pass, inputs: Dict[str, Any]) -> Optional[InferenceOutcome]:
        """Run the inference with up to 3 attempts; ``None`` means abandoned (nothing to capture).

        Network errors and HTTP 408/429/5xx are retried. Every connection-level
        failure counts toward the consecutive-connection-failure limit (any HTTP
        response resets it); the outcome returned for a Case whose last attempt
        failed to connect has ``connection_failure`` set and is not captured.
        """
        attempt = 1
        while True:
            outcome = run_workflow_inference(
                self._session,
                self._url,
                api_key=self._inference_api_key,
                specification=ctx.specification,
                workflow_id=ctx.workflow_id,
                inputs=inputs,
                timeout=self._case_timeout,
                monotonic=self._monotonic,
            )
            with self._cond:
                if outcome.responded:
                    self._connection_failures = 0
                elif outcome.connection_failure:
                    self._connection_failures += 1
                    if self._connection_failures >= MAX_CONNECTION_FAILURES:
                        self._fail(
                            LocalRunnerError(
                                f"The inference server at {self._url} failed {MAX_CONNECTION_FAILURES} connections "
                                "in a row.",
                                hint=f"{PREFLIGHT_HINT} {RESUME_HINT}",
                            )
                        )
                tripped = self._fatal is not None
            if outcome.connection_failure and tripped:
                return outcome
            if not outcome.error or not outcome.retryable or attempt >= INFERENCE_ATTEMPTS:
                return outcome
            try:
                self._backoff(min(2.0 ** (attempt - 1), INFERENCE_BACKOFF_CAP))
            except _Abandoned:
                return None
            attempt += 1

    def _submit(self, work: _Work, data: Optional[bytes], lease: Optional[float]) -> None:
        """Submit a capture (``data``) or a ``resume`` (``data is None``) and act on the answer.

        Every resend uses the same stored bytes.
        """
        ctx = work.ctx
        resends = 0
        replaced = False
        while True:
            self._sleep_until(lease)
            lease = None
            if data is not None:
                body = json.loads(data)
            else:
                body = {"attemptCount": work.attempt, "shardIndex": work.shard_index, "resume": True}
            error: Optional[WorkflowEvalError] = None
            if not self._wait(lambda: self._submitting < self._concurrency, self._take_submitter):
                return
            try:
                self._api(
                    self._client.submit_capture,
                    self._eval_id,
                    self._run_id,
                    ctx.execution_id,
                    work.case_id,
                    body,
                    timeout=CAPTURE_TIMEOUT,
                )
            except WorkflowEvalError as exc:
                error = exc
            finally:
                with self._cond:
                    self._submitting -= 1
                    self._cond.notify_all()
            if error is None:
                self._record(work, 200, "submitted")
                failed = data is not None and "error" in (body.get("runtime") or {})
                self._count(ctx, "submitted", failed=failed)
                return
            status, message = error.status_code, error.message or ""
            if status == 409 and message.startswith("Different capture already submitted"):
                # Another runner (a tab or another CLI) won this Case.
                self._record(work, 409, "done_by_other")
                self._count(ctx, "already_done")
                return
            if status == 409 and (
                error.details.get("reason") == "non_retryable_failure"
                or message.startswith("Case failure is not retryable")
            ):
                self._record(work, 409, "skipped")
                self._count(ctx, "skipped")
                return
            if status == 409 and message.startswith("Run no longer accepts captures"):
                self._record(work, 409, "run_closed")
                self._flag("_run_closed")
                return
            if status == 409 and (error.retryable is True or message.startswith("Capture is being processed")):
                resends += 1
                if resends > PROCESSING_RESENDS:
                    raise LocalRunnerError(
                        f"Case {work.case_id} is still being processed after {PROCESSING_RESENDS} resends.",
                        hint=RESUME_HINT,
                        details=error.to_dict(),
                    )
                lease = self._lease(work) or self._clock() + min(2.0**resends, API_BACKOFF_CAP)
                continue
            if status == 412 or (status == 409 and message.startswith("No retained capture")):
                # The Run attempt changed (or the retained capture is gone): restart the Execution.
                self._record(work, status, "reload")
                self._flag("_reload")
                return
            if status == 413 and data is not None and not replaced:
                data = self._too_large(work, body)
                replaced = True
                continue
            raise _api_error(error) from error

    def _take_submitter(self) -> None:
        self._submitting += 1

    def _too_large(self, work: _Work, body: Dict[str, Any]) -> bytes:
        latency = (body.get("runtime") or {}).get("latencyMs")
        self._warn(f"Case {work.case_id}: {CAPTURE_TOO_LARGE_MESSAGE}; submitting it as a failed Case.")
        runtime: Dict[str, Any] = {"latencyMs": latency, "error": CAPTURE_TOO_LARGE_MESSAGE, "retryable": False}
        if latency is None:
            del runtime["latencyMs"]
        data = _dumps(
            {"attemptCount": work.attempt, "shardIndex": work.shard_index, "output": None, "runtime": runtime}
        )
        assert self._journal is not None
        self._journal.record_capture(work.attempt, work.ctx.execution_id, work.case_id, work.shard_index, data)
        return data

    def _lease(self, work: _Work) -> Optional[float]:
        """Re-read the Case's page (same cursor) for its ``checkpoint.leaseExpiresAt``."""
        page = self._fetch_page(work.ctx, work.cursor)
        item = page.items.get(work.case_id) or {}
        return _parse_timestamp((item.get("checkpoint") or {}).get("leaseExpiresAt"))

    def _record(self, work: _Work, status: int, outcome: str) -> None:
        assert self._journal is not None
        self._journal.record_submission(work.attempt, work.ctx.execution_id, work.case_id, status, outcome)

    def _flag(self, name: str) -> None:
        with self._cond:
            setattr(self, name, True)
            self._cond.notify_all()

    # -- progress --------------------------------------------------------------

    def _count(self, ctx: _Pass, name: str, *, failed: bool = False) -> None:
        with self._cond:
            setattr(ctx.tally, name, getattr(ctx.tally, name) + 1)
            if failed:
                ctx.tally.failed += 1
        self._emit_progress(ctx)

    def _emit_progress(self, ctx: _Pass) -> None:
        if self._on_progress is None:
            return
        with self._cond:
            tally = ctx.tally
            done = tally.submitted + tally.already_done + tally.skipped + tally.pending
            progress = {
                "executionId": ctx.execution_id,
                "execution": ctx.index,
                "executions": ctx.total,
                "ok": tally.submitted - tally.failed,
                "failed": tally.failed,
                "alreadyDone": tally.already_done,
                "skipped": tally.skipped,
                "pending": tally.pending,
                "remaining": max(ctx.selected - done, 0) if ctx.selected is not None else None,
                "inFlight": self._in_flight,
                "lastLatencyMs": self._last_latency_ms,
            }
        self._on_progress(progress)


def run_local(client: WorkflowEvals, eval_id: str, run_id: str, **options: Any) -> LocalRunResult:
    """Shorthand for ``LocalRunner(client, eval_id, run_id, **options).run()``."""
    return LocalRunner(client, eval_id, run_id, **options).run()


@contextlib.contextmanager
def install_signal_handlers(
    runner: LocalRunner,
    signals: Sequence[int] = (signal.SIGINT, signal.SIGTERM),
    on_signal: Optional[Callable[[int], None]] = None,
) -> Iterator[None]:
    """Route SIGINT/SIGTERM to ``runner.request_stop()`` and restore the previous handlers afterwards.

    Must be used from the main thread. The first signal stops gracefully, the second aborts:
    it raises :class:`LocalRunnerInterrupted` from the handler, so a blocking API call on the
    main thread is interrupted at once (:meth:`LocalRunner.run` attaches the result).
    ``on_signal`` receives each signal number first (for example to tell the user what happens next).
    """
    previous: Dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        if on_signal is not None:
            on_signal(signum)
        runner.request_stop()
        if runner.aborting:
            raise LocalRunnerInterrupted

    for signum in signals:
        previous[signum] = signal.signal(signum, handler)
    try:
        yield
    finally:
        for signum, original in previous.items():
            signal.signal(signum, original)
