"""Tests for the local Workflow Eval runner (``roboflow.core.workflow_eval_local_runner``).

The Roboflow API and the Inference server are both faked with ``responses``
callbacks; time is a fake clock whose ``sleep`` only advances it, so nothing
here sleeps for real.
"""

from __future__ import annotations

import copy
import io
import json
import os
import re
import signal
import socket
import ssl
import threading
from datetime import datetime, timezone
from http.client import RemoteDisconnected
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

import pytest
import requests
import responses
import urllib3
from typer.testing import CliRunner

from roboflow.cli import app
from roboflow.config import API_URL
from roboflow.core import workflow_eval_local_runner as local
from roboflow.core.workflow_eval_local_runner import (
    CAPTURE_TOO_LARGE_MESSAGE,
    FETCH_FAILED_MESSAGE,
    PREFLIGHT_HINT,
    TIMEOUT_MESSAGE,
    InferenceServerUnreachable,
    Journal,
    LocalRunner,
    LocalRunnerError,
    LocalRunnerInterrupted,
    case_inputs,
    install_signal_handlers,
    materialize_media,
    run_workflow_inference,
)
from roboflow.core.workflow_evals import WorkflowEvals

WS, KEY = "test-ws", "test_key"
E, R, X = "eval-1", "run-1", "exec-1"
RUN = f"{API_URL}/workspaces/{WS}/workflow-evals/{E}/runs/{R}"
INF = "http://localhost:9001"
SPEC = {"version": "1.0", "inputs": [{"type": "WorkflowImage", "name": "image"}], "steps": [], "outputs": []}
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "workflow_eval_local_inference_request.json")
START = 1_800_000_000.0


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def err(message: str, *, retryable: bool = False, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    error: Dict[str, Any] = {"code": "CONFLICT", "category": "conflict", "retryable": retryable, "message": message}
    if details:
        error["details"] = details
    return {"error": error}


def case(case_id: str, *, checkpoint: Optional[Dict[str, Any]] = None, **fields: Any) -> Dict[str, Any]:
    item = {
        "case": {"id": case_id, "name": case_id},
        "shardIndex": 0,
        "workflowInputs": {"image": {"artifactId": f"art-{case_id}"}, "name": case_id},
        "workflowInputFailures": [],
        "checkpoint": checkpoint,
    }
    item.update(fields)
    return item


def ok(name: Optional[str] = None) -> Any:
    return (200, {}, json.dumps({"outputs": [{"case": name}]}))


def refused_error() -> requests.exceptions.ConnectionError:
    # What requests raises when nothing listens on the port: the server was never reached.
    reason = urllib3.exceptions.NewConnectionError(None, "Failed to establish a new connection: Connection refused")
    return requests.exceptions.ConnectionError(urllib3.exceptions.MaxRetryError(None, "/workflows/run", reason))


def refused(request: Any) -> Any:
    return refused_error()


def reset_error() -> requests.exceptions.ConnectionError:
    # Connected, then the server dropped the connection.
    return requests.exceptions.ConnectionError(
        urllib3.exceptions.ProtocolError("Connection aborted.", ConnectionResetError(104, "Connection reset by peer"))
    )


class BrokenBody(io.BufferedReader):
    """A response body whose socket fails on the first read, after the status line arrived."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(io.BytesIO(b"{}"))  # type: ignore[arg-type]
        self.error = error

    def read(self, *args: Any) -> bytes:
        raise self.error

    read1 = read

    def readinto(self, *args: Any) -> int:
        raise self.error


def tls_dropped(request: Any) -> Any:
    # Connected and answered, then TLS failed mid-body (e.g. behind ngrok); requests raises SSLError.
    return 200, {}, BrokenBody(ssl.SSLError("TLS connection dropped"))


# Waits of one Case whose inference server cannot be reached at all.
CONNECT_SLEEPS = [1.0, 2.0, 4.0, 8.0, 16.0]


class FakeClock:
    def __init__(self) -> None:
        self.now = START
        self.sleeps: List[float] = []
        self._lock = threading.Lock()

    def time(self) -> float:
        with self._lock:
            return self.now

    def sleep(self, seconds: float) -> None:
        with self._lock:
            self.sleeps.append(seconds)
            self.now += max(seconds, 0)

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds

    def set(self, value: float) -> None:
        with self._lock:
            self.now = value


class FakeInference:
    """``POST /workflows/run``. Behaviours are keyed by the Case name sent in ``inputs.name``."""

    def __init__(self) -> None:
        self.requests: List[Any] = []
        self.behaviors: Dict[Any, Any] = {}
        self.default: Callable[[Any], Any] = lambda request: ok(json.loads(request.body)["inputs"].get("name"))
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    @property
    def names(self) -> List[Any]:
        return [json.loads(request.body)["inputs"].get("name") for request in self.requests]

    def __call__(self, request: Any) -> Any:
        name = json.loads(request.body)["inputs"].get("name")
        with self._lock:
            self.requests.append(request)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            behavior = self.behaviors.get(name, self.default)
            if isinstance(behavior, list):
                behavior = behavior.pop(0) if len(behavior) > 1 else behavior[0]
        try:
            return behavior(request) if callable(behavior) else behavior
        finally:
            with self._lock:
                self.active -= 1


class FakeServer:
    """The Workflow Evals routes the runner calls, with a mutable Run and per-Case checkpoints."""

    def __init__(self, rsps: responses.RequestsMock, items: List[Dict[str, Any]], *, mode: str = "browser") -> None:
        self.lock = threading.Lock()
        self.items = [dict(item, checkpoint=None) for item in items]
        self.checkpoints = {item["case"]["id"]: item.get("checkpoint") for item in items}
        self.run: Dict[str, Any] = {"id": R, "state": "running", "executionMode": mode, "retry": {"allowed": False}}
        self.attempt = 1
        self.version = 0
        self.spec: Any = SPEC
        self.workflow_id = "wf-1"
        self.frozen_url = INF
        self.capture_script: Dict[str, List[Any]] = {}
        self.capture_default: Any = None
        self.captures: List[Any] = []
        self.page_requests: List[Any] = []
        self.page_seen: Dict[Any, threading.Event] = {}
        # Called (without the lock) before the page at this cursor is served.
        self.before_page: Dict[Any, Callable[[], None]] = {}
        # Mark a Case completed at the current attempt when its capture is accepted, like the server.
        self.complete_on_capture = False
        self.retried = 0
        rsps.add_callback(responses.GET, RUN, callback=lambda r: (200, {}, json.dumps(self.run)))
        rsps.add_callback(responses.GET, f"{RUN}/configuration", callback=self._configuration)
        rsps.add_callback(responses.POST, f"{RUN}/retry-failed", callback=self._retry)
        rsps.add_callback(
            responses.GET,
            f"{RUN}/executions",
            callback=lambda r: (200, {}, json.dumps({"items": [{"id": X, "selectedCaseCount": len(self.items)}]})),
        )
        rsps.add_callback(responses.GET, f"{RUN}/executions/{X}/configuration", callback=self._execution)
        rsps.add_callback(
            responses.GET, re.compile(re.escape(f"{RUN}/executions/") + r"[^/]+/input-cases.*"), callback=self._page
        )
        rsps.add_callback(
            responses.POST,
            re.compile(re.escape(f"{RUN}/executions/") + r"[^/]+/cases/([^/?]+)/capture"),
            callback=self._capture,
        )

    def checkpoint(self, case_id: str, value: Optional[Dict[str, Any]]) -> None:
        with self.lock:
            self.checkpoints[case_id] = value

    def seen(self, cursor: Any) -> threading.Event:
        with self.lock:
            return self.page_seen.setdefault(cursor, threading.Event())

    def capture_bodies(self, case_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return [json.loads(body) for cid, body in self.captures if case_id in (None, cid)]

    def _configuration(self, request: Any) -> Any:
        server = {"kind": "local", "url": self.frozen_url}
        return (
            200,
            {},
            json.dumps({"runId": R, "executionMode": "browser", "sharedConfiguration": {"inferenceServer": server}}),
        )

    def _execution(self, request: Any) -> Any:
        plan = {"workflowSubject": {"id": self.workflow_id, "manifest": self.spec}}
        return 200, {}, json.dumps({"execution": {"id": X}, "linkedPlan": plan})

    def _retry(self, request: Any) -> Any:
        with self.lock:
            self.retried += 1
            self.attempt += 1
            self.run["state"] = "running"
        return 202, {}, json.dumps({"runId": R})

    def _page(self, request: Any) -> Any:
        query = parse_qs(urlparse(request.url).query)
        cursor = query.get("cursor", [None])[0]
        limit = int(query.get("limit", ["50"])[0])
        hook = self.before_page.get(cursor)
        if hook is not None:
            hook()
        with self.lock:
            self.page_requests.append((cursor, limit))
            self.version += 1
            offset = int(cursor or 0)
            items = copy.deepcopy(self.items[offset : offset + limit])
            urls = {}
            for item in items:
                item["checkpoint"] = copy.deepcopy(self.checkpoints.get(item["case"]["id"]))
                for value in (item.get("workflowInputs") or {}).values():
                    if isinstance(value, dict) and "artifactId" in value:
                        urls[value["artifactId"]] = {
                            "url": f"https://storage.example/{value['artifactId']}?sig={self.version}",
                            "expiresInSeconds": 600,
                        }
            body = {
                "items": items,
                "nextCursor": str(offset + limit) if offset + limit < len(self.items) else None,
                "attemptCount": self.attempt,
                "runState": self.run["state"],
                "cancellationRequested": False,
                "artifactReadUrls": urls,
            }
        self.seen(cursor).set()
        return 200, {}, json.dumps(body)

    def _capture(self, request: Any) -> Any:
        case_id = unquote(re.search(r"/cases/([^/?]+)/capture", request.url).group(1))
        with self.lock:
            self.captures.append((case_id, request.body))
            script = self.capture_script.get(case_id)
            entry = script.pop(0) if script else self.capture_default
        if callable(entry):
            entry = entry()
        if entry is None:
            if self.complete_on_capture:
                self.checkpoint(
                    case_id, {"state": "completed", "attemptCount": json.loads(request.body)["attemptCount"]}
                )
            return 200, {}, json.dumps({"id": case_id, "state": "completed"})
        status, body = entry
        return status, {}, body if isinstance(body, str) else json.dumps(body)


class Env:
    def __init__(self, rsps: responses.RequestsMock, tmp_path: Any) -> None:
        self.rsps = rsps
        self.state_dir = str(tmp_path / "state")
        self.clock = FakeClock()
        self.inference = FakeInference()
        self.warnings: List[str] = []
        self.progress: List[Dict[str, Any]] = []
        self.describe: List[Any] = []
        self.describe_response: Any = (200, {}, json.dumps({"blocks": []}))
        self.server: Optional[FakeServer] = None
        rsps.add_callback(responses.POST, f"{INF}/workflows/run", callback=self.inference)
        rsps.add_callback(responses.POST, f"{INF}/workflows/blocks/describe", callback=self._describe)

    def _describe(self, request: Any) -> Any:
        self.describe.append(request)
        return self.describe_response

    def serve(self, items: List[Dict[str, Any]], **kwargs: Any) -> FakeServer:
        self.server = FakeServer(self.rsps, items, **kwargs)
        return self.server

    def runner(self, **kwargs: Any) -> LocalRunner:
        options: Dict[str, Any] = {
            "state_dir": self.state_dir,
            "clock": self.clock.time,
            "monotonic": self.clock.time,
            "sleep": self.clock.sleep,
            "warn": self.warnings.append,
            "on_progress": self.progress.append,
        }
        options.update(kwargs)
        return LocalRunner(WorkflowEvals(KEY, WS), E, R, **options)

    def journal(self) -> List[Dict[str, Any]]:
        with open(os.path.join(self.state_dir, "journal.jsonl"), encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def stored(self, entry: Dict[str, Any]) -> bytes:
        with open(os.path.join(self.state_dir, entry["path"]), "rb") as handle:
            return handle.read()


@pytest.fixture
def env(tmp_path: Any) -> Any:
    with responses.RequestsMock(assert_all_requests_are_fired=False) as rsps:
        yield Env(rsps, tmp_path)


def captured(env: Env) -> List[Dict[str, Any]]:
    return [entry for entry in env.journal() if entry["t"] == "captured"]


def outcomes(env: Env) -> List[Any]:
    return [
        (entry["caseId"], entry["status"], entry["outcome"]) for entry in env.journal() if entry["t"] == "submitted"
    ]


# -- 1. request contract -----------------------------------------------------------


def test_inference_request_matches_browser_worker_fixture(env: Env) -> None:
    with open(FIXTURE, encoding="utf-8") as handle:
        fixture = json.load(handle)
    expected = fixture["body"]
    server = env.serve(
        [case("c1", workflowInputs={"image": {"artifactId": "art_1"}, "confidence": 0.4, "label": "ñandú"})]
    )
    server.spec, server.workflow_id = expected["specification"], expected["workflow_id"]
    server.version = 0  # the first page signs ?sig=1

    result = env.runner(inference_api_key="sample_api_key").run()

    assert result.to_dict()["executions"] == [
        {"id": X, "submitted": 1, "alreadyDone": 0, "skipped": 0, "failed": 0, "pending": 0}
    ]
    (request,) = env.inference.requests
    assert request.url == fixture["url"]
    # Byte-for-byte what JSON.stringify produces for the worker's object literal.
    assert request.body == json.dumps(expected, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    body = json.loads(request.body)
    assert body == expected and list(body) == list(expected)
    assert body["is_preview"] is True and body["disable_sinks"] is True and body["debug"] is False
    for name, value in fixture["headers"].items():
        assert request.headers[name] == value

    # Preflight mirrors the web app's local server check.
    (describe,) = env.describe
    assert parse_qs(urlparse(describe.url).query) == {"api_key": ["sample_api_key"]}
    assert describe.headers["Api-Key"] == "sample_api_key"
    assert describe.headers["ngrok-skip-browser-warning"] == "true"
    assert json.loads(describe.body) == {"dynamic_blocks_definitions": []}


def test_capture_is_the_worker_message_with_attempt_and_shard(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.default = lambda request: (200, {}, json.dumps({"outputs": {"single": 1}}))
    env.runner().run()
    (body,) = server.capture_bodies()
    assert list(body) == ["attemptCount", "shardIndex", "output", "runtime"]
    assert body["attemptCount"] == 1 and body["shardIndex"] == 0
    assert body["output"] == [{"single": 1}]
    assert set(body["runtime"]) == {"latencyMs"}


# -- 2. media ----------------------------------------------------------------------


def test_materialize_media_nested_and_in_lists() -> None:
    urls = {"a1": {"url": "https://u/a1", "expiresInSeconds": 600}, "a2": {"url": "https://u/a2"}}
    value = {
        "image": {"artifactId": "a1", "role": "primary"},
        "nested": {"frames": [{"artifactId": "a2"}, {"artifactId": "unsigned"}, 3]},
        "plain": "x",
    }
    assert materialize_media(value, urls) == {
        "image": {"artifactId": "a1", "role": "primary", "type": "url", "value": "https://u/a1"},
        "nested": {
            "frames": [{"artifactId": "a2", "type": "url", "value": "https://u/a2"}, {"artifactId": "unsigned"}, 3]
        },
        "plain": "x",
    }


def test_case_inputs_materializes_media() -> None:
    item = {"workflowInputs": {"image": {"artifactId": "a1"}, "confidence": 0.4}, "workflowInputFailures": []}
    inputs = case_inputs(item, {"a1": {"url": "https://u/a1"}})
    assert inputs == {"image": {"artifactId": "a1", "type": "url", "value": "https://u/a1"}, "confidence": 0.4}


def test_runner_sends_materialized_inputs(env: Env) -> None:
    env.serve([case("c1", workflowInputs={"image": {"artifactId": "art-c1"}})])
    env.runner().run()
    inputs = json.loads(env.inference.requests[0].body)["inputs"]
    assert inputs == {"image": {"artifactId": "art-c1", "type": "url", "value": "https://storage.example/art-c1?sig=1"}}


def test_input_failure_is_captured_as_failed_without_inference(env: Env) -> None:
    message = 'Workflow input "mask" did not resolve'
    server = env.serve(
        [
            case("c1"),
            case(
                "c2",
                shardIndex=3,
                workflowInputFailures=[{"input": "mask", "message": message}, {"input": "other", "message": "x"}],
            ),
            case("c3"),
        ]
    )
    result = env.runner().run()

    assert sorted(env.inference.names) == ["c1", "c3"]
    (body,) = server.capture_bodies("c2")
    assert body == {"attemptCount": 1, "shardIndex": 3, "output": None, "runtime": {"error": message}}
    assert set(body["runtime"]) == {"error"}
    assert all(set(b["runtime"]) == {"latencyMs"} for cid in ("c1", "c3") for b in server.capture_bodies(cid))
    (entry,) = [entry for entry in captured(env) if entry["caseId"] == "c2"]
    assert json.loads(env.stored(entry)) == body and entry["shardIndex"] == 3
    assert ("c2", 200, "submitted") in outcomes(env)
    assert result.executions[0].to_dict() == {
        "id": X,
        "submitted": 3,
        "alreadyDone": 0,
        "skipped": 0,
        "failed": 1,
        "pending": 0,
    }


def test_input_failure_message_is_truncated(env: Env) -> None:
    server = env.serve([case("c1", workflowInputFailures=[{"input": "mask", "message": "x" * 3000}])])
    env.runner().run()
    assert env.inference.requests == []
    assert server.capture_bodies()[0]["runtime"] == {"error": "x" * 2000}


# -- 3. sliding window ----------------------------------------------------------------


class VirtualTime:
    """Holds every inference until the test releases the one that ends first, then jumps the clock there."""

    def __init__(self, clock: FakeClock, durations: Dict[str, float]) -> None:
        self.clock = clock
        self.durations = durations
        self.cond = threading.Condition()
        self.waiting: Dict[str, Any] = {}

    def __call__(self, request: Any) -> Any:
        name = json.loads(request.body)["inputs"]["name"]
        release = threading.Event()
        with self.cond:
            self.waiting[name] = (self.clock.time() + self.durations[name], release)
            self.cond.notify_all()
        assert release.wait(5)
        return ok(name)

    def drive(self, window: int) -> None:
        total = len(self.durations)
        for done in range(total):
            with self.cond:
                assert self.cond.wait_for(lambda: len(self.waiting) >= min(window, total - done), timeout=5)
                name = min(self.waiting, key=lambda key: self.waiting[key][0])
                end, release = self.waiting.pop(name)
            self.clock.set(end)
            release.set()


def test_sliding_window_beats_lockstep_and_respects_concurrency(env: Env) -> None:
    durations = {"c1": 5.0, "c2": 1.0, "c3": 1.0, "c4": 1.0, "c5": 1.0}
    server = env.serve([case(name) for name in durations])
    virtual = VirtualTime(env.clock, durations)
    env.inference.default = virtual
    runner = env.runner(concurrency=2)
    outcome: Dict[str, Any] = {}
    thread = threading.Thread(target=lambda: outcome.update(result=runner.run()))
    thread.start()
    virtual.drive(window=2)
    thread.join(5)

    assert outcome["result"].executions[0].submitted == 5
    # Lockstep pages of 2 would take max(5, 1) + max(1, 1) + 1 = 7 s; the window finishes with c1.
    assert env.clock.now - START == 5.0
    assert env.inference.max_active == 2
    assert {cid for cid, _ in server.captures} == set(durations)
    latency = {cid: json.loads(body)["runtime"]["latencyMs"] for cid, body in server.captures}
    assert latency["c1"] == 5000.0 and latency["c2"] == 1000.0
    assert all(cursor_limit[1] == 2 for cursor_limit in server.page_requests)


# -- 4. refresh ---------------------------------------------------------------------


def test_stale_page_is_reread_with_the_same_cursor_before_dispatch(env: Env) -> None:
    server = env.serve([case("c1"), case("c2")])

    def slow(request: Any) -> Any:
        assert server.seen("1").wait(5)  # the next page was prefetched at t=0
        env.clock.advance(9 * 60)
        return ok("c1")

    env.inference.behaviors["c1"] = slow
    env.runner(concurrency=1).run()

    assert server.page_requests == [(None, 1), ("1", 1), ("1", 1)]
    image = json.loads(env.inference.requests[1].body)["inputs"]["image"]
    assert image["value"] == "https://storage.example/art-c2?sig=3"


def test_retry_after_a_slow_attempt_rereads_the_page_for_fresh_urls(env: Env) -> None:
    server = env.serve([case("c1"), case("c2")])

    def slow_503(request: Any) -> Any:
        env.clock.advance(9 * 60)  # the signed URLs of c2's page are about to expire
        return 503, {}, "Service Unavailable"

    env.inference.behaviors["c2"] = [slow_503, ok("c2")]
    result = env.runner(concurrency=1).run()

    first, second = [json.loads(r.body)["inputs"]["image"]["value"] for r in env.inference.requests[1:]]
    assert first == "https://storage.example/art-c2?sig=2"
    assert second == "https://storage.example/art-c2?sig=3"
    # c2's page (cursor "1") is read again before the retry, with the same cursor.
    assert server.page_requests == [(None, 1), ("1", 1), ("1", 1)]
    assert server.capture_bodies("c2")[0]["output"] == [{"case": "c2"}]
    assert result.executions[0].submitted == 2 and result.executions[0].failed == 0


def test_retry_within_the_page_max_age_reuses_the_inputs(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.behaviors["c1"] = [(503, {}, "Service Unavailable"), ok("c1")]
    env.runner().run()

    assert len(env.inference.requests) == 2
    assert env.inference.requests[0].body == env.inference.requests[1].body
    assert server.page_requests == [(None, 4)]


def test_case_that_left_the_reread_page_restarts_the_execution(env: Env) -> None:
    server = env.serve([case("c1")])

    def slow_503_and_move(request: Any) -> Any:
        env.clock.advance(9 * 60)
        with server.lock:
            server.items = [case("c0"), case("c1")]  # c1 moves to the second page
        return 503, {}, "Service Unavailable"

    env.inference.behaviors["c1"] = [slow_503_and_move, ok("c1")]
    result = env.runner(concurrency=1).run()

    # The re-read page no longer has c1: the attempt is abandoned uncaptured and the
    # Execution restarts from the first page, where the new pass finds c1 on page "1".
    assert server.page_requests == [(None, 1), (None, 1), (None, 1), ("1", 1)]
    assert env.inference.names == ["c1", "c0", "c1"]
    assert sorted(cid for cid, _ in server.captures) == ["c0", "c1"]
    assert server.capture_bodies("c1")[0]["output"] == [{"case": "c1"}]
    assert result.executions[0].to_dict()["pending"] == 0
    assert result.executions[0].submitted == 2 and result.executions[0].failed == 0


def test_reread_page_at_another_attempt_restarts_the_execution(env: Env) -> None:
    server = env.serve([case("c1")])

    def slow_503(request: Any) -> Any:
        env.clock.advance(9 * 60)
        with server.lock:
            server.attempt = 2
        return 503, {}, "Service Unavailable"

    env.inference.behaviors["c1"] = [slow_503, ok("c1")]
    result = env.runner().run()

    # The attempt-1 inference is not retried nor captured; the restarted pass runs c1 at attempt 2.
    assert len(env.inference.requests) == 2
    assert [body["attemptCount"] for body in server.capture_bodies()] == [2]
    assert result.executions[0].submitted == 1


def test_stop_requested_during_the_reread_abandons_the_retry(env: Env) -> None:
    server = env.serve([case("c1")])
    runner = env.runner()
    page_calls: List[int] = []

    def slow_503(request: Any) -> Any:
        env.clock.advance(9 * 60)
        return 503, {}, "Service Unavailable"

    def stop_during_reread() -> None:
        page_calls.append(1)
        if len(page_calls) == 2:
            runner.request_stop()  # the re-read itself still returns a valid page

    env.inference.behaviors["c1"] = [slow_503, ok("c1")]
    server.before_page[None] = stop_during_reread
    with pytest.raises(LocalRunnerInterrupted) as raised:
        runner.run()

    assert raised.value.exit_code == 130
    assert len(env.inference.requests) == 1  # no inference after the stop
    assert server.page_requests == [(None, 4), (None, 4)]
    assert server.captures == [] and captured(env) == []


def test_reload_while_the_reread_is_in_backoff_abandons_the_case(env: Env) -> None:
    server = env.serve([case("c1"), case("c2")])
    runner = env.runner(concurrency=2)
    rereading = threading.Event()
    page_calls: List[int] = []

    def slow_503(request: Any) -> Any:
        env.clock.advance(9 * 60)
        return 503, {}, "Service Unavailable"

    def after_reread_started(request: Any) -> Any:
        assert rereading.wait(5)
        return ok("c1")

    def attempt_changed() -> Any:
        server.attempt = 2
        return 412, {"error": {"code": "PRECONDITION_FAILED", "message": "Run attempt changed"}}

    def flaky_reread() -> None:
        page_calls.append(1)
        if len(page_calls) == 2:
            rereading.set()
            # Fails only after the 412, so its backoff is abandoned because of the reload.
            with runner._cond:
                assert runner._cond.wait_for(lambda: runner._reload, timeout=5)
            raise requests.exceptions.ConnectionError("reset")

    env.inference.behaviors.update(c1=[after_reread_started, ok("c1")], c2=[slow_503, ok("c2")])
    server.capture_script["c1"] = [attempt_changed]
    server.before_page[None] = flaky_reread
    result = runner.run()

    # The failed re-read is not retried: the next page read is the restarted pass.
    assert server.page_requests == [(None, 2), (None, 2)]
    assert [body["attemptCount"] for body in server.capture_bodies("c2")] == [2]
    assert [body["attemptCount"] for body in server.capture_bodies("c1")] == [1, 2]
    assert result.executions[0].submitted == 2


# -- 5. journal ---------------------------------------------------------------------


def test_journal_resubmits_stored_bytes_without_running_inference(env: Env) -> None:
    server = env.serve([case("c1", checkpoint={"state": "completed", "attemptCount": 1}), case("c2")])
    original = Journal.record_capture

    def crash_after_capture(self: Journal, *args: Any) -> None:
        original(self, *args)
        raise RuntimeError("killed after captured")

    with patch.object(Journal, "record_capture", crash_after_capture), pytest.raises(RuntimeError):
        env.runner().run()
    assert env.inference.names == ["c2"]
    assert server.captures == []
    (entry,) = captured(env)
    stored = env.stored(entry)

    result = env.runner().run()

    assert env.inference.names == ["c2"]  # no inference for c2 (journaled) nor c1 (completed)
    assert [cid for cid, _ in server.captures] == ["c2"]
    assert json.loads(server.captures[0][1]) == json.loads(stored)
    assert env.stored(entry) == stored
    assert result.executions[0].to_dict() == {
        "id": X,
        "submitted": 1,
        "alreadyDone": 1,
        "skipped": 0,
        "failed": 0,
        "pending": 0,
    }
    assert outcomes(env) == [("c2", 200, "submitted")]


def test_journal_entries_of_another_attempt_are_ignored(env: Env) -> None:
    server = env.serve([case("c1")])
    Journal(env.state_dir).record_capture(7, X, "c1", 0, b'{"attemptCount":7}')
    env.runner().run()
    assert env.inference.names == ["c1"]
    assert server.capture_bodies()[0]["attemptCount"] == 1


def test_journal_torn_last_line_does_not_swallow_the_next_record(env: Env) -> None:
    os.makedirs(env.state_dir)
    with open(os.path.join(env.state_dir, "journal.jsonl"), "w", encoding="utf-8") as handle:
        handle.write('{"t":"captured"')
    journal = Journal(env.state_dir)
    journal.record_capture(1, X, "c1", 0, b'{"attemptCount":1}')
    journal.close()
    reopened = Journal(env.state_dir)
    reopened.close()
    assert reopened.captured(1, X, "c1") == b'{"attemptCount":1}'


def test_case_with_retained_capture_waits_for_lease_and_resumes(env: Env) -> None:
    lease = START + 42
    checkpoint = {"state": "running", "attemptCount": 1, "hasCapture": True, "leaseExpiresAt": iso(lease)}
    server = env.serve([case("c1", checkpoint=checkpoint)])
    env.runner().run()
    assert env.inference.requests == []
    assert server.capture_bodies() == [{"attemptCount": 1, "shardIndex": 0, "resume": True}]
    assert env.clock.now >= lease


# -- 6. capture responses ---------------------------------------------------------------


def test_capture_200_is_submitted(env: Env) -> None:
    env.serve([case("c1")])
    result = env.runner().run()
    assert result.to_dict() == {
        "runId": R,
        "state": "running",
        "executions": [{"id": X, "submitted": 1, "alreadyDone": 0, "skipped": 0, "failed": 0, "pending": 0}],
        "wallClockSeconds": 0.0,
        "concurrency": 4,
    }
    assert [entry["t"] for entry in env.journal()] == ["captured", "submitted"]
    assert env.progress[-1]["ok"] == 1 and env.progress[-1]["remaining"] == 0


def test_capture_409_different_capture_counts_as_done_by_other(env: Env) -> None:
    server = env.serve([case("c1")])
    server.capture_script["c1"] = [(409, err("Different capture already submitted for this Case attempt"))]
    result = env.runner().run()
    assert len(server.captures) == 1
    assert result.executions[0].already_done == 1 and result.executions[0].submitted == 0
    assert outcomes(env) == [("c1", 409, "done_by_other")]


def test_capture_409_processing_waits_for_lease_and_resends_same_bytes(env: Env) -> None:
    server = env.serve([case("c1")])
    lease = START + 30

    def processing() -> Any:
        server.checkpoint(
            "c1", {"state": "running", "attemptCount": 1, "hasCapture": True, "leaseExpiresAt": iso(lease)}
        )
        message = "Capture is being processed or the attempt changed; retry the same payload"
        return 409, err(message, retryable=True)

    server.capture_script["c1"] = [processing]
    result = env.runner().run()

    assert len(server.captures) == 2
    assert server.captures[0][1] == server.captures[1][1]
    assert server.page_requests == [(None, 4), (None, 4)]  # the lease came from re-reading the same page
    assert env.clock.now >= lease
    assert len(env.inference.requests) == 1
    assert result.executions[0].submitted == 1


def test_capture_409_non_retryable_failure_is_skipped(env: Env) -> None:
    server = env.serve([case("c1")])
    server.capture_script["c1"] = [
        (409, err("Case failure is not retryable", details={"reason": "non_retryable_failure"}))
    ]
    result = env.runner().run()
    assert result.executions[0].skipped == 1 and result.executions[0].submitted == 0
    assert outcomes(env) == [("c1", 409, "skipped")]


def test_capture_409_run_closed_stops_dispatch_and_cancelled_exits_1(env: Env) -> None:
    server = env.serve([case(f"c{i}") for i in range(1, 6)])
    closed = (409, err("Run no longer accepts captures"))

    c2_in_flight = threading.Event()

    def cancel() -> Any:
        # c1's slot frees before its capture is answered, so c2 is already running.
        assert c2_in_flight.wait(5)
        server.run["state"] = "cancelled"
        return closed

    server.capture_script["c1"] = [cancel]
    server.capture_default = closed
    runner = env.runner(concurrency=1)

    def wait_until_closed(request: Any) -> Any:
        c2_in_flight.set()
        # Finish only after the 409, so nothing else can be dispatched before it.
        with runner._cond:
            assert runner._cond.wait_for(lambda: runner._run_closed, timeout=5)
        return ok("c2")

    env.inference.behaviors["c2"] = wait_until_closed
    with pytest.raises(LocalRunnerError) as raised:
        runner.run()
    assert raised.value.exit_code == 1
    assert raised.value.result is not None and raised.value.result.state == "cancelled"
    assert env.inference.names == ["c1", "c2"]
    assert ("c1", 409, "run_closed") in outcomes(env)


def test_run_closed_while_cancellation_is_finalizing_exits_1(env: Env) -> None:
    server = env.serve([case("c1")])

    def cancel() -> Any:
        # The server keeps the Run running while it skips the remaining Cases.
        server.run["cancellationRequestedAt"] = iso(START)
        return 409, err("Run no longer accepts captures")

    server.capture_script["c1"] = [cancel]
    with pytest.raises(LocalRunnerError, match="was cancelled") as raised:
        env.runner().run()
    assert raised.value.exit_code == 1 and not isinstance(raised.value, LocalRunnerInterrupted)
    assert raised.value.resumable is False
    assert ("c1", 409, "run_closed") in outcomes(env)


def test_capture_412_restarts_the_execution_at_the_new_attempt(env: Env) -> None:
    server = env.serve([case("c1")])

    def attempt_changed() -> Any:
        server.attempt = 2
        return 412, {
            "error": {"code": "PRECONDITION_FAILED", "message": "Run attempt changed; reload before submitting"}
        }

    server.capture_script["c1"] = [attempt_changed]
    result = env.runner().run()

    assert env.inference.names == ["c1", "c1"]  # no output reused across attempts
    assert [body["attemptCount"] for body in server.capture_bodies()] == [1, 2]
    assert [entry["attempt"] for entry in captured(env)] == [1, 2]
    assert result.executions[0].submitted == 1


def test_412_while_the_next_page_prefetch_is_in_backoff_restarts_the_execution(env: Env) -> None:
    server = env.serve([case("c1"), case("c2")])
    runner = env.runner(concurrency=1)
    waiting_for_page = threading.Event()
    original_prefetch = runner._prefetch

    def prefetch(ctx: Any, cursor: str) -> Any:
        result = original_prefetch(ctx, cursor)

        def wait() -> Any:
            waiting_for_page.set()
            return result()

        return wait

    runner._prefetch = prefetch  # type: ignore[method-assign]

    def attempt_changed() -> Any:
        assert waiting_for_page.wait(5)  # the dispatcher is waiting for the prefetched page
        server.attempt = 2
        return 412, {"error": {"code": "PRECONDITION_FAILED", "message": "Run attempt changed"}}

    page_calls: List[int] = []

    def flaky_page() -> None:
        page_calls.append(1)
        if len(page_calls) == 1:
            # Fails only after the 412, so its backoff is abandoned because of the reload.
            with runner._cond:
                assert runner._cond.wait_for(lambda: runner._reload, timeout=5)
            raise requests.exceptions.ConnectionError("reset")

    server.capture_script["c1"] = [attempt_changed]
    server.before_page["1"] = flaky_page
    result = runner.run()

    assert [body["attemptCount"] for body in server.capture_bodies("c1")] == [1, 2]
    assert [body["attemptCount"] for body in server.capture_bodies("c2")] == [2]
    assert result.executions[0].submitted == 2


def output_captures(server: FakeServer, case_id: str) -> List[Dict[str, Any]]:
    return [body for body in server.capture_bodies(case_id) if "output" in body]


def test_capture_409_no_retained_capture_on_resume_restarts_the_execution(env: Env) -> None:
    checkpoint = {"state": "running", "attemptCount": 1, "hasCapture": True}
    server = env.serve([case("c1", checkpoint=checkpoint), case("c2")])
    server.complete_on_capture = True

    def capture_lost() -> Any:
        server.checkpoint("c1", {"state": "running", "attemptCount": 1, "hasCapture": False})
        return 409, err("No retained capture for this Case attempt")

    server.capture_script["c1"] = [capture_lost]
    result = env.runner(concurrency=1).run()

    resume, capture = server.capture_bodies("c1")
    assert resume == {"attemptCount": 1, "shardIndex": 0, "resume": True}
    assert len(output_captures(server, "c1")) == 1 and capture["output"] == [{"case": "c1"}]
    assert len(output_captures(server, "c2")) == 1  # c2 is not captured again by the restarted pass
    assert sorted(env.inference.names) == ["c1", "c2"]
    assert ("c1", 409, "reload") in outcomes(env)
    tally = result.executions[0]
    assert tally.submitted + tally.already_done == 2


def test_page_at_another_attempt_restarts_the_execution_without_double_captures(env: Env) -> None:
    server = env.serve([case(f"c{i}") for i in range(1, 5)])
    server.complete_on_capture = True

    def retried() -> None:
        with server.lock:
            server.attempt = 2

    server.before_page["2"] = retried  # the prefetched second page is at attempt 2
    result = env.runner(concurrency=2).run()

    attempts = [(cid, json.loads(body)["attemptCount"]) for cid, body in server.captures]
    assert len(attempts) == len(set(attempts))  # no Case captured twice within one attempt
    assert sorted(attempts) == [("c1", 1), ("c2", 1), ("c3", 2), ("c4", 2)]
    assert sorted(env.inference.names) == ["c1", "c2", "c3", "c4"]
    tally = result.executions[0]
    assert (tally.already_done, tally.submitted) == (2, 2)


def test_capture_413_is_replaced_by_a_failure_capture(env: Env) -> None:
    server = env.serve([case("c1")])
    server.capture_script["c1"] = [(413, "request entity too large")]
    result = env.runner().run()

    first, second = server.capture_bodies()
    assert second == {
        "attemptCount": 1,
        "shardIndex": 0,
        "output": None,
        "runtime": {"latencyMs": first["runtime"]["latencyMs"], "error": CAPTURE_TOO_LARGE_MESSAGE, "retryable": False},
    }
    entries = captured(env)
    assert len(entries) == 2 and json.loads(env.stored(entries[1])) == second
    assert any(CAPTURE_TOO_LARGE_MESSAGE in warning for warning in env.warnings)
    assert result.executions[0].submitted == 1 and result.executions[0].failed == 1


def test_capture_404_exits_3(env: Env) -> None:
    server = env.serve([case("c1")])
    server.capture_script["c1"] = [
        (404, {"error": {"code": "NOT_FOUND", "message": "Case is not in this frozen snapshot shard"}})
    ]
    with pytest.raises(LocalRunnerError) as raised:
        env.runner().run()
    assert raised.value.exit_code == 3


def test_capture_400_exits_1_with_the_server_error(env: Env) -> None:
    server = env.serve([case("c1")])
    server.capture_script["c1"] = [
        (400, {"error": {"code": "INVALID_REQUEST", "message": "Invalid browser Case capture"}})
    ]
    with pytest.raises(LocalRunnerError) as raised:
        env.runner().run()
    assert raised.value.exit_code == 1
    assert raised.value.details["message"] == "Invalid browser Case capture"
    assert raised.value.details["status"] == 400


# -- 7. inference errors ----------------------------------------------------------------


def test_inference_timeout_is_a_retryable_timed_out_failure_without_retries(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.default = lambda request: requests.exceptions.ReadTimeout("read timed out")
    result = env.runner().run()
    assert result.executions[0].pending == 0 and result.executions[0].failed == 1
    assert len(env.inference.requests) == 1  # one attempt, like the browser
    assert env.clock.sleeps == []
    (body,) = server.capture_bodies()
    assert body["output"] is None
    assert body["runtime"]["error"] == TIMEOUT_MESSAGE
    assert body["runtime"]["timedOut"] is True and body["runtime"]["retryable"] is True


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ((422, {}, json.dumps({"message": "Invalid input image"})), "Invalid input image"),
        ((400, {}, "<html>bad</html>"), "Workflow returned HTTP 400"),
        ((200, {}, json.dumps({"success": False, "errorMsg": "Step failed"})), "Step failed"),
        ((200, {}, json.dumps(["not", "an", "object"])), "Workflow returned an invalid response"),
        ((200, {}, json.dumps({"success": True})), "Workflow returned no outputs"),
    ],
)
def test_non_retryable_inference_failures_are_not_retried(env: Env, response: Any, message: str) -> None:
    server = env.serve([case("c1")])
    env.inference.default = lambda request: response
    result = env.runner().run()
    assert len(env.inference.requests) == 1
    (body,) = server.capture_bodies()
    assert body["runtime"]["error"] == message
    assert body["runtime"]["retryable"] is False and body["runtime"]["timedOut"] is False
    assert result.executions[0].failed == 1


def test_inference_503_is_retried_3_times_then_captured_as_retryable(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.default = lambda request: (503, {}, "Service Unavailable")
    result = env.runner().run()
    assert len(env.inference.requests) == 3
    assert result.executions[0].pending == 0
    (body,) = server.capture_bodies()  # exactly one capture
    assert body["runtime"]["error"] == "Workflow returned HTTP 503"
    assert body["runtime"]["retryable"] is True


def test_inference_retry_succeeds_after_a_429(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.behaviors["c1"] = [(429, {}, json.dumps({"message": "slow down"})), ok("c1")]
    env.runner().run()
    assert len(env.inference.requests) == 2
    assert server.capture_bodies()[0]["output"] == [{"case": "c1"}]


@pytest.mark.parametrize("outputs", ['[{"v":1e400}]', '[{"v":"\\ud800"}]'], ids=["non-finite-number", "lone-surrogate"])
def test_unserializable_output_is_captured_as_a_non_retryable_failure(env: Env, outputs: str) -> None:
    server = env.serve([case("c1")])
    env.inference.default = lambda request: (200, {}, '{"outputs":' + outputs + "}")
    result = env.runner().run()
    (body,) = server.capture_bodies()
    assert body["output"] is None
    runtime = body["runtime"]
    assert runtime["error"].startswith("Workflow output could not be serialized: ")
    assert runtime["timedOut"] is False and runtime["retryable"] is False and "latencyMs" in runtime
    (entry,) = captured(env)
    assert json.loads(env.stored(entry)) == body
    assert any("could not be serialized" in warning for warning in env.warnings)
    assert result.executions[0].submitted == 1 and result.executions[0].failed == 1


def test_run_workflow_inference_wraps_outputs_and_truncates_errors() -> None:
    session = requests.Session()
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, f"{INF}/workflows/run", json={"outputs": None})
        rsps.add(responses.POST, f"{INF}/workflows/run", json={"message": "x" * 3000}, status=500)
        options = {"api_key": "k", "specification": {}, "workflow_id": "w", "inputs": {}, "timeout": 1}
        success = run_workflow_inference(session, INF + "/", **options)
        failure = run_workflow_inference(session, INF, **options)
    assert success.output == [None] and not success.error
    assert len(failure.runtime["error"]) == 2000 and failure.retryable


def _name_resolution_error() -> requests.exceptions.ConnectionError:
    reason = urllib3.exceptions.NameResolutionError("nohost", None, socket.gaierror(-2, "Name or service not known"))
    return requests.exceptions.ConnectionError(urllib3.exceptions.MaxRetryError(None, "/", reason))


# urllib3 1.26 has no NameResolutionError; there a DNS failure is a plain NewConnectionError.
HAS_NAME_RESOLUTION_ERROR = hasattr(urllib3.exceptions, "NameResolutionError")


@pytest.mark.parametrize(
    ("error", "kind", "timed_out"),
    [
        (refused_error(), "connect", False),
        (requests.exceptions.ConnectionError(urllib3.exceptions.NewConnectionError(None, "refused")), "connect", False),
        *([(_name_resolution_error(), "connect", False)] if HAS_NAME_RESOLUTION_ERROR else []),
        (requests.exceptions.ConnectTimeout("connect timed out"), "connect", True),
        (requests.exceptions.SSLError("certificate verify failed"), "connect", False),
        (requests.exceptions.ProxyError("proxy refused"), "connect", False),
        (reset_error(), "dropped", False),
        (
            requests.exceptions.ConnectionError(
                urllib3.exceptions.ProtocolError("Connection aborted.", RemoteDisconnected("closed"))
            ),
            "dropped",
            False,
        ),
        (requests.exceptions.ConnectionError(BrokenPipeError(32, "Broken pipe")), "dropped", False),
        (requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"), "dropped", False),
        (requests.exceptions.ReadTimeout("read timed out"), None, True),
        # requests reports a timeout while reading the body as a ConnectionError.
        (requests.exceptions.ConnectionError(urllib3.exceptions.ReadTimeoutError(None, "/", "timed out")), None, True),
    ],
)
def test_run_workflow_inference_classifies_connection_failures(
    error: Exception, kind: Optional[str], timed_out: bool
) -> None:
    with responses.RequestsMock() as rsps:
        rsps.add(responses.POST, f"{INF}/workflows/run", body=error)
        outcome = run_workflow_inference(
            requests.Session(), INF, api_key="k", specification={}, workflow_id="w", inputs={}, timeout=1
        )
    assert outcome.connection_failure is (kind == "connect")
    assert outcome.dropped is (kind == "dropped")
    assert outcome.runtime["timedOut"] is timed_out and outcome.retryable and not outcome.responded


def test_tls_dropped_while_reading_the_body_is_a_dropped_connection() -> None:
    with responses.RequestsMock() as rsps:
        rsps.add_callback(responses.POST, f"{INF}/workflows/run", callback=tls_dropped)
        outcome = run_workflow_inference(
            requests.Session(), INF, api_key="k", specification={}, workflow_id="w", inputs={}, timeout=1
        )
    assert outcome.dropped and not outcome.connection_failure
    assert outcome.runtime["error"] == FETCH_FAILED_MESSAGE
    assert outcome.runtime["timedOut"] is False and outcome.retryable and not outcome.responded


class _StreamedResponse:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.closed = False

    @property
    def content(self) -> bytes:
        raise self.error

    def close(self) -> None:
        self.closed = True


class _StreamingSession:
    def __init__(self, response: _StreamedResponse) -> None:
        self.response = response
        self.kwargs: Dict[str, Any] = {}

    def post(self, url: str, **kwargs: Any) -> _StreamedResponse:
        self.kwargs = kwargs
        return self.response


@pytest.mark.parametrize(
    ("error", "kind", "timed_out"),
    [
        # While reading the body the connection was made: these are drops, not connection failures.
        (requests.exceptions.SSLError("EOF occurred in violation of protocol"), "dropped", False),
        (requests.exceptions.ProxyError("proxy closed the tunnel"), "dropped", False),
        (reset_error(), "dropped", False),
        (requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"), "dropped", False),
        (requests.exceptions.ConnectionError(urllib3.exceptions.ReadTimeoutError(None, "/", "timed out")), None, True),
        (requests.exceptions.ReadTimeout("read timed out"), None, True),
        (requests.exceptions.ContentDecodingError("bad gzip"), None, False),
    ],
)
def test_run_workflow_inference_classifies_failures_while_reading_the_body(
    error: Exception, kind: Optional[str], timed_out: bool
) -> None:
    response = _StreamedResponse(error)
    session = _StreamingSession(response)
    outcome = run_workflow_inference(
        session,  # type: ignore[arg-type]
        INF,
        api_key="k",
        specification={},
        workflow_id="w",
        inputs={},
        timeout=1,
    )
    assert session.kwargs["stream"] is True and response.closed
    assert outcome.connection_failure is False
    assert outcome.dropped is (kind == "dropped")
    assert outcome.runtime["timedOut"] is timed_out and outcome.retryable and not outcome.responded


# -- 8. connection failures ---------------------------------------------------------------


def test_stops_after_10_consecutive_cases_that_cannot_connect(env: Env) -> None:
    server = env.serve([case(f"c{i}") for i in range(1, 13)])
    env.inference.default = refused
    with pytest.raises(LocalRunnerError) as raised:
        env.runner(concurrency=1).run()
    assert not isinstance(raised.value, InferenceServerUnreachable)
    assert raised.value.exit_code == 1
    assert raised.value.message == f"The inference server at {INF} failed to connect for 10 Cases in a row."
    assert PREFLIGHT_HINT in (raised.value.hint or "")
    # Each of c1..c10 used its whole connect schedule (6 attempts); c11 was never dispatched.
    assert env.inference.names == [f"c{i}" for i in range(1, 11) for _ in range(6)]
    assert env.clock.sleeps == CONNECT_SLEEPS * 10
    # Connection failures are not Case failures: nothing is captured, the Cases stay pending on the server.
    assert server.captures == []
    assert captured(env) == [] and outcomes(env) == []


@pytest.mark.parametrize("tenth", ["ok", "dropped"])
def test_a_case_that_reaches_the_server_resets_the_consecutive_count(env: Env, tenth: str) -> None:
    names = [f"c{i}" for i in range(1, 20)]
    server = env.serve([case(name) for name in names])
    env.inference.default = refused
    # c10 reaches the server (an answer, or a connection dropped after it was made) between 9 + 9 Cases.
    env.inference.behaviors["c10"] = ok("c10") if tenth == "ok" else (lambda request: reset_error())
    with pytest.raises(InferenceServerUnreachable) as raised:
        env.runner(concurrency=1).run()

    assert raised.value.message == f"18 Cases were not run because the inference server at {INF} was unreachable."
    assert [cid for cid, _ in server.captures] == ["c10"]
    assert raised.value.result is not None
    (tally,) = raised.value.result.executions
    assert (tally.pending, tally.submitted, tally.failed) == (18, 1, 0 if tenth == "ok" else 1)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(refused_error(), id="refused"),
        *([pytest.param(_name_resolution_error(), id="dns")] if HAS_NAME_RESOLUTION_ERROR else []),
        pytest.param(requests.exceptions.ConnectTimeout("connect timed out"), id="connect-timeout"),
        pytest.param(requests.exceptions.SSLError("certificate verify failed"), id="tls"),
    ],
)
def test_unreachable_case_backs_off_1_2_4_8_16_then_is_left_pending(env: Env, error: Exception) -> None:
    server = env.serve([case("c1"), case("c2")])
    env.inference.behaviors["c1"] = lambda request: error
    with pytest.raises(LocalRunnerError) as raised:
        env.runner(concurrency=1).run()

    assert isinstance(raised.value, InferenceServerUnreachable)
    assert raised.value.exit_code == 1
    assert raised.value.message == f"1 Case was not run because the inference server at {INF} was unreachable."
    assert "run the same command again" in (raised.value.hint or "")
    assert env.inference.names == ["c1"] * 6 + ["c2"]
    assert env.clock.sleeps == CONNECT_SLEEPS
    assert [cid for cid, _ in server.captures] == ["c2"]  # no capture POST for c1
    assert [entry["caseId"] for entry in captured(env)] == ["c2"]
    assert raised.value.result is not None
    (tally,) = raised.value.result.executions
    assert (tally.pending, tally.submitted, tally.failed) == (1, 1, 0)
    assert tally.to_dict()["pending"] == 1
    assert env.progress[-1]["pending"] == 1 and env.progress[-1]["remaining"] == 0


@pytest.mark.parametrize(
    "error",
    [reset_error(), requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead"), None],
    ids=["reset", "dropped-mid-response", "tls-dropped-mid-body"],
)
def test_dropped_connection_is_retried_3_times_then_captured_as_retryable(env: Env, error: Optional[Exception]) -> None:
    server = env.serve([case("c1"), case("c2")])
    env.inference.behaviors["c1"] = tls_dropped if error is None else (lambda request: error)
    result = env.runner(concurrency=1).run()

    assert env.inference.names == ["c1"] * 3 + ["c2"]
    assert env.clock.sleeps == [1.0, 2.0]
    (body,) = server.capture_bodies("c1")
    assert body["output"] is None
    assert set(body["runtime"]) == {"latencyMs", "error", "timedOut", "retryable"}
    assert body["runtime"]["error"] == FETCH_FAILED_MESSAGE
    assert body["runtime"]["timedOut"] is False and body["runtime"]["retryable"] is True
    assert [entry["caseId"] for entry in captured(env)] == ["c1", "c2"]
    assert ("c1", 200, "submitted") in outcomes(env)
    assert result.executions[0].to_dict() == {
        "id": X,
        "submitted": 2,
        "alreadyDone": 0,
        "skipped": 0,
        "failed": 1,
        "pending": 0,
    }


def test_connected_and_connect_budgets_are_separate(env: Env) -> None:
    server = env.serve([case("c1")])
    unavailable = (503, {}, "Service Unavailable")
    # Two 503s use 2 of the 3 connected attempts; the refusals in between do not.
    env.inference.behaviors["c1"] = [unavailable, refused, refused, unavailable, refused, ok("c1")]
    result = env.runner().run()
    assert len(env.inference.requests) == 6
    assert env.clock.sleeps == [1.0, 1.0, 2.0, 2.0, 4.0]
    assert server.capture_bodies()[0]["output"] == [{"case": "c1"}]
    assert result.executions[0].failed == 0


def test_a_short_outage_does_not_fail_the_run(env: Env) -> None:
    names = [f"c{i}" for i in range(1, 17)]
    server = env.serve([case(name) for name in names])

    def down_for_3_seconds(request: Any) -> Any:
        if env.clock.time() < START + 3:
            return refused_error()
        return ok(json.loads(request.body)["inputs"]["name"])

    env.inference.default = down_for_3_seconds
    result = env.runner(concurrency=4).run()

    assert sorted(cid for cid, _ in server.captures) == sorted(names)
    assert all(body["output"] is not None for body in server.capture_bodies())
    assert any(sleep == 1.0 for sleep in env.clock.sleeps)  # the outage was hit and waited out
    assert result.executions[0].to_dict() == {
        "id": X,
        "submitted": 16,
        "alreadyDone": 0,
        "skipped": 0,
        "failed": 0,
        "pending": 0,
    }


def test_http_response_resets_the_connection_failure_count(env: Env) -> None:
    env.serve([case(f"c{i}") for i in range(1, 6)])
    for name in ("c1", "c2", "c3", "c4", "c5"):
        env.inference.behaviors[name] = [refused, refused, ok(name)]
    result = env.runner(concurrency=1).run()
    assert result.executions[0].submitted == 5 and result.executions[0].failed == 0


# -- 9. stop requests ---------------------------------------------------------------------


def test_stop_request_journals_in_flight_results_and_raises_130(env: Env) -> None:
    server = env.serve([case(f"c{i}") for i in range(1, 5)])
    runner = env.runner(concurrency=2)
    both_in_flight = threading.Barrier(2, timeout=5)
    stopped = threading.Event()

    def first(request: Any) -> Any:
        both_in_flight.wait()
        runner.request_stop()
        stopped.set()
        return ok("c1")

    def second(request: Any) -> Any:
        both_in_flight.wait()
        assert stopped.wait(5)
        return ok("c2")

    env.inference.behaviors.update(c1=first, c2=second)
    with pytest.raises(LocalRunnerInterrupted) as raised:
        runner.run()
    assert raised.value.exit_code == 130
    assert sorted(env.inference.names) == ["c1", "c2"]
    assert sorted(entry["caseId"] for entry in captured(env)) == ["c1", "c2"]
    assert sorted(cid for cid, _ in server.captures) == ["c1", "c2"]  # queued captures still submitted


def test_second_stop_request_aborts_without_waiting(env: Env) -> None:
    env.serve([case("c1")])
    runner = env.runner()
    release = threading.Event()

    def stuck(request: Any) -> Any:
        runner.request_stop()
        runner.request_stop()
        release.wait(5)
        return ok("c1")

    env.inference.behaviors["c1"] = stuck
    try:
        with pytest.raises(LocalRunnerInterrupted):
            runner.run()
        assert not release.is_set()
    finally:
        release.set()
        for thread in runner._threads:
            thread.join(5)


@pytest.mark.skipif(not hasattr(signal, "pthread_kill"), reason="needs pthread_kill")
def test_second_signal_interrupts_a_blocking_api_call_on_the_main_thread(env: Env) -> None:
    env.serve([case("c1")])
    runner = env.runner()
    entered, release = threading.Event(), threading.Event()

    def blocking_get_run(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        release.wait(2)
        return {"id": R, "state": "completed", "executionMode": "browser"}

    def send_two_signals() -> None:
        assert entered.wait(5)
        main = threading.main_thread().ident
        signal.pthread_kill(main, signal.SIGINT)
        while not runner._stop_requested:
            release.wait(0.01)
        signal.pthread_kill(main, signal.SIGINT)

    sender = threading.Thread(target=send_two_signals)
    try:
        with patch.object(WorkflowEvals, "get_run", blocking_get_run), install_signal_handlers(runner):
            sender.start()
            with pytest.raises(LocalRunnerInterrupted) as raised:
                runner.run()
        assert not release.is_set()  # interrupted inside get_run, not after it returned
        assert raised.value.exit_code == 130
        assert raised.value.result is not None
    finally:
        release.set()
        sender.join(5)
    assert runner._threads == []


def test_install_signal_handlers_routes_to_request_stop_and_restores(env: Env) -> None:
    runner = env.runner()
    before = signal.getsignal(signal.SIGINT)
    with install_signal_handlers(runner):
        handler = signal.getsignal(signal.SIGINT)
        assert callable(handler)
        handler(signal.SIGINT, None)
    assert runner._stop_requested
    assert signal.getsignal(signal.SIGINT) is before


# -- 10. startup checks ---------------------------------------------------------------------


def test_missing_workflow_inputs_means_the_server_is_too_old(env: Env) -> None:
    item = case("c1")
    del item["workflowInputs"]
    env.serve([item])
    with pytest.raises(LocalRunnerError, match="server does not support local runner yet") as raised:
        env.runner().run()
    assert raised.value.exit_code == 1
    assert env.inference.requests == []


def test_server_run_is_rejected_with_a_hint(env: Env) -> None:
    env.serve([case("c1")], mode="server")
    with pytest.raises(LocalRunnerError) as raised:
        env.runner().run()
    assert raised.value.exit_code == 1
    assert raised.value.hint == "This Run executes on Roboflow; use `run get --wait`."


@pytest.mark.parametrize(
    "describe",
    [(503, {}, json.dumps({"message": "warming up"})), requests.exceptions.ConnectionError("refused")],
)
def test_preflight_failure_exits_1_before_any_inference(env: Env, describe: Any) -> None:
    env.serve([case("c1")])
    env.describe_response = describe
    with pytest.raises(LocalRunnerError) as raised:
        env.runner().run()
    assert raised.value.exit_code == 1 and raised.value.hint == PREFLIGHT_HINT
    assert env.inference.requests == []


@pytest.mark.parametrize(("status", "exit_code"), [(401, 2), (403, 2), (404, 3)])
def test_api_errors_map_to_exit_codes(env: Env, status: int, exit_code: int) -> None:
    env.rsps.add(responses.GET, RUN, json={"error": {"code": "X", "message": "nope"}}, status=status)
    with pytest.raises(LocalRunnerError) as raised:
        env.runner().run()
    assert raised.value.exit_code == exit_code


def test_unknown_execution_exits_3(env: Env) -> None:
    env.serve([case("c1")])
    with pytest.raises(LocalRunnerError) as raised:
        env.runner(execution_ids=["exec-404"]).run()
    assert raised.value.exit_code == 3


def test_terminal_run_returns_without_work_unless_retry_is_allowed(env: Env) -> None:
    server = env.serve([case("c1")])
    server.run["state"] = "partially_completed"
    assert env.runner().run().state == "partially_completed"
    assert env.inference.requests == [] and env.describe == []

    server.run["retry"] = {"allowed": True}
    result = env.runner(retry=True).run()
    assert server.retried == 1
    assert env.inference.names == ["c1"]
    assert server.capture_bodies()[0]["attemptCount"] == 2
    assert result.executions[0].submitted == 1


def test_inference_url_override_warns(env: Env) -> None:
    server = env.serve([case("c1")])
    server.frozen_url = "http://gpu-box:9001/"
    env.runner(inference_url=INF + "/").run()
    assert len(env.inference.requests) == 1
    assert any("http://gpu-box:9001" in warning for warning in env.warnings)


def test_default_state_dir_respects_xdg_cache_home(tmp_path: Any) -> None:
    with patch.dict(os.environ, {"XDG_CACHE_HOME": str(tmp_path)}):
        assert local.default_state_dir("ws", "run-9") == os.path.join(
            str(tmp_path), "roboflow", "workflow-evals", "ws", "run-9"
        )


# -- 11. retry happens only before submission ------------------------------------------------


def test_capture_post_failure_never_reruns_inference(env: Env) -> None:
    server = env.serve([case("c1")])
    server.capture_script["c1"] = [(503, "unavailable")]
    result = env.runner().run()
    assert len(env.inference.requests) == 1
    assert len(server.captures) == 2 and server.captures[0][1] == server.captures[1][1]
    assert env.clock.sleeps == [1.0]
    assert result.executions[0].submitted == 1


def test_no_threads_are_left_behind(env: Env) -> None:
    env.serve([case(f"c{i}") for i in range(1, 7)])
    before = threading.active_count()
    env.runner(concurrency=3).run()
    assert threading.active_count() == before


# -- 12. inference server down, then back (end to end through the CLI) -------------------------


def test_inference_down_stops_then_the_same_command_completes_every_case_once(env: Env) -> None:
    names = [f"c{i}" for i in range(1, 14)]
    server = env.serve([case(name) for name in names])
    server.complete_on_capture = True
    env.inference.default = refused
    env.inference.behaviors.update(c1=ok("c1"), c2=ok("c2"))  # the server goes down after c2
    fake_time = {"clock": env.clock.time, "monotonic": env.clock.time, "sleep": env.clock.sleep}
    argv = ["--workspace", WS, "--api-key", KEY, "--json", "workflow-eval", "run", "local", R, "-e", E]
    argv += ["--concurrency", "1", "--state-dir", env.state_dir]

    def invoke() -> Any:
        before = threading.active_count()
        with patch.object(local, "LocalRunner", lambda *args, **kwargs: LocalRunner(*args, **kwargs, **fake_time)):
            result = CliRunner().invoke(app, argv)
        assert threading.active_count() == before
        return result

    first = invoke()
    assert first.exit_code == 1, first.output
    error = json.loads(first.stderr.strip().splitlines()[-1])["error"]
    assert "failed to connect for 10 Cases in a row" in error["message"] and PREFLIGHT_HINT in error["hint"]
    assert first.stdout == ""
    # c3..c12 each used their 6 connect attempts: 10 Cases in a row. Only c1 and c2 were captured.
    assert env.inference.names == ["c1", "c2"] + [f"c{i}" for i in range(3, 13) for _ in range(6)]
    assert [cid for cid, _ in server.captures] == ["c1", "c2"]
    assert [entry["caseId"] for entry in captured(env)] == ["c1", "c2"]  # the journal is kept

    env.inference.requests.clear()
    env.inference.behaviors.clear()
    env.inference.default = lambda request: ok(json.loads(request.body)["inputs"]["name"])
    second = invoke()

    assert second.exit_code == 0, second.output
    summary = json.loads(second.stdout)
    assert summary["executions"][0]["alreadyDone"] == 2 and summary["executions"][0]["submitted"] == 11
    assert summary["executions"][0]["pending"] == 0 and summary["warnings"] == []
    assert env.inference.names == names[2:]  # c1 and c2 are not inferred again
    assert sorted(cid for cid, _ in server.captures) == sorted(names)  # every Case captured exactly once
