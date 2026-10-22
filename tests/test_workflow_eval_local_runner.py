"""Tests for the local Workflow Eval runner (``roboflow.core.workflow_eval_local_runner``).

The Roboflow API and the Inference server are both faked with ``responses``
callbacks; time is a fake clock whose ``sleep`` only advances it, so nothing
here sleeps for real.
"""

from __future__ import annotations

import copy
import json
import os
import re
import signal
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

import pytest
import requests
import responses

from roboflow.config import API_URL
from roboflow.core import workflow_eval_local_runner as local
from roboflow.core.workflow_eval_local_runner import (
    CAPTURE_TOO_LARGE_MESSAGE,
    PREFLIGHT_HINT,
    TIMEOUT_MESSAGE,
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

    assert result.to_dict()["executions"] == [{"id": X, "submitted": 1, "alreadyDone": 0, "skipped": 0, "failed": 0}]
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
    assert result.executions[0].to_dict() == {"id": X, "submitted": 3, "alreadyDone": 0, "skipped": 0, "failed": 1}


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
    assert result.executions[0].to_dict() == {"id": X, "submitted": 1, "alreadyDone": 1, "skipped": 0, "failed": 0}
    assert outcomes(env) == [("c2", 200, "submitted")]


def test_journal_entries_of_another_attempt_are_ignored(env: Env) -> None:
    server = env.serve([case("c1")])
    Journal(env.state_dir).record_capture(7, X, "c1", 0, b'{"attemptCount":7}')
    env.runner().run()
    assert env.inference.names == ["c1"]
    assert server.capture_bodies()[0]["attemptCount"] == 1


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
        "executions": [{"id": X, "submitted": 1, "alreadyDone": 0, "skipped": 0, "failed": 0}],
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


def test_inference_timeout_is_a_retryable_timed_out_failure(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.default = lambda request: requests.exceptions.ReadTimeout("read timed out")
    env.runner().run()
    assert len(env.inference.requests) == 3
    assert env.clock.sleeps == [1.0, 2.0]
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
    env.runner().run()
    assert len(env.inference.requests) == 3
    (body,) = server.capture_bodies()
    assert body["runtime"]["error"] == "Workflow returned HTTP 503"
    assert body["runtime"]["retryable"] is True


def test_inference_retry_succeeds_after_a_429(env: Env) -> None:
    server = env.serve([case("c1")])
    env.inference.behaviors["c1"] = [(429, {}, json.dumps({"message": "slow down"})), ok("c1")]
    env.runner().run()
    assert len(env.inference.requests) == 2
    assert server.capture_bodies()[0]["output"] == [{"case": "c1"}]


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


# -- 8. connection failures ---------------------------------------------------------------


def test_stops_after_10_consecutive_connection_failures(env: Env) -> None:
    server = env.serve([case(f"c{i}") for i in range(1, 6)])
    env.inference.default = lambda request: requests.exceptions.ConnectionError("Connection refused")
    with pytest.raises(LocalRunnerError) as raised:
        env.runner(concurrency=1).run()
    assert raised.value.exit_code == 1
    assert PREFLIGHT_HINT in (raised.value.hint or "")
    assert len(env.inference.requests) == 10
    assert env.clock.sleeps == [1.0, 2.0] * 3
    # c1..c3 exhausted their 3 attempts and were captured as retryable failures; c4 was abandoned.
    assert [entry["caseId"] for entry in captured(env)] == ["c1", "c2", "c3"]
    assert sorted(outcomes(env)) == [(cid, 200, "submitted") for cid in ("c1", "c2", "c3")]
    assert all(body["runtime"]["retryable"] is True for body in server.capture_bodies())


def test_http_response_resets_the_connection_failure_count(env: Env) -> None:
    env.serve([case(f"c{i}") for i in range(1, 6)])
    refused = lambda request: requests.exceptions.ConnectionError("refused")  # noqa: E731
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
