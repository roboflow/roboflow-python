import json
import unittest
import uuid
from unittest.mock import patch

import responses

from roboflow.adapters.rfapi import RoboflowError
from roboflow.adapters.workflowdeploygatesapi import WorkflowDeployGateError
from roboflow.adapters.workflowevalsapi import WorkflowEvalError
from roboflow.config import API_URL
from roboflow.core.workflow_deploy_gates import WorkflowDeployGates

WS = "test-ws"
KEY = "test_key"
BASE = f"{API_URL}/workspaces/{WS}/workflows"
WF, V = "wf-1", "ver-1"
GATE = f"/{WF}/deploy-gates/workflow-eval"
APPROVAL = f"/{WF}/deploy-gates/manual-approval"

# (client method, args, kwargs, HTTP method, path relative to BASE, sends Idempotency-Key)
# One row per route in the server's publicDeployGatesRegistry.ts.
ROUTES = [
    ("openapi", (), {}, "GET", "/deploy-gates/openapi.json", False),
    ("list", (), {}, "GET", "/deploy-gates", False),
    ("list_for_workflow", (WF,), {}, "GET", f"/{WF}/deploy-gates", False),
    ("save_version", (WF, {"specification": {}}), {}, "POST", f"/{WF}/versions", True),
    ("get_version", (WF, V), {}, "GET", f"/{WF}/versions/{V}/deploy-gates", False),
    ("run", (WF, V), {}, "POST", f"/{WF}/versions/{V}/deploy-gates/run", True),
    ("deploy", (WF, V), {}, "POST", f"/{WF}/versions/{V}/deploy", False),
    ("get_eval_gate", (WF,), {}, "GET", GATE, False),
    ("set_eval_gate", (WF, "eval-1"), {}, "PUT", GATE, False),
    ("update_eval_gate", (WF, {"c1": 0.9}), {}, "PATCH", GATE, False),
    ("set_eval_gate_new_check_thresholds", (WF, {"c1": 0.9}), {}, "POST", f"{GATE}/new-check-thresholds", False),
    ("remove_eval_gate", (WF,), {}, "DELETE", GATE, False),
    ("get_eval_checks", (WF,), {}, "GET", f"{GATE}/checks", False),
    ("get_approval_gate", (WF,), {}, "GET", APPROVAL, False),
    ("set_approval_gate", (WF,), {"user_ids": ["u1"]}, "PUT", APPROVAL, False),
    ("remove_approval_gate", (WF,), {}, "DELETE", APPROVAL, False),
    ("list_approvals", (WF,), {}, "GET", f"{APPROVAL}/approvals", False),
    ("get_version_approval", (WF, V), {}, "GET", f"/{WF}/versions/{V}/deploy-gates/manual-approval", False),
]

NOT_PASSED = {
    "error": {
        "code": "deploy_check_not_passed",
        "category": "conflict",
        "retryable": False,
        "requestId": "req-1",
        "message": "The version has not passed its deploy gates",
        "details": {
            "gateType": "workflow_eval",
            "status": "failed",
            "current": True,
            "staleReason": None,
            "nextSteps": [
                {
                    "action": "run_deploy_checks",
                    "method": "POST",
                    "path": f"/workspaces/{WS}/workflows/{WF}/versions/{V}/deploy-gates/run",
                    "description": "Start every deploy gate of the workflow on the version",
                }
            ],
        },
    }
}


def _is_uuid4(value):
    try:
        return uuid.UUID(value).version == 4
    except (TypeError, ValueError):
        return False


def _verdicts(*statuses, publishable=False):
    gates = [{"type": "workflow_eval", "status": status, "reason": None} for status in statuses]
    return {"versionId": V, "gates": gates, "publishable": publishable}


class TestWorkflowDeployGatesRoutes(unittest.TestCase):
    def setUp(self):
        self.client = WorkflowDeployGates(KEY, WS)

    def test_covers_every_public_route(self):
        self.assertEqual(len({(row[3], row[4]) for row in ROUTES}), 18)

    @responses.activate
    def test_each_method_calls_its_route(self):
        for name, args, kwargs, method, path, idempotent in ROUTES:
            with self.subTest(method=name):
                responses.reset()
                responses.add(method, f"{BASE}{path}", json={"ok": True}, status=200)
                getattr(self.client, name)(*args, **kwargs)
                request = responses.calls[0].request
                self.assertEqual(request.method, method)
                self.assertEqual(request.url.split("?")[0], f"{BASE}{path}")
                self.assertEqual(request.headers["Authorization"], f"Bearer {KEY}")
                self.assertEqual(_is_uuid4(request.headers.get("Idempotency-Key")), idempotent)


class TestWorkflowDeployGatesClient(unittest.TestCase):
    def setUp(self):
        self.client = WorkflowDeployGates(KEY, WS)

    @responses.activate
    def test_save_version_wraps_a_bare_definition_and_keeps_the_key(self):
        responses.add(responses.POST, f"{BASE}/{WF}/versions", json={"replayed": False}, status=201)
        bare = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
        key = str(uuid.uuid4())
        self.client.save_version(WF, bare, version_name="candidate", idempotency_key=key)
        request = responses.calls[0].request
        body = json.loads(request.body)
        self.assertEqual(json.loads(body["config"]), {"specification": bare})
        self.assertEqual(body["versionName"], "candidate")
        self.assertEqual(request.headers["Idempotency-Key"], key)

    @responses.activate
    def test_set_eval_gate_sends_thresholds_and_the_stale_write_guard(self):
        responses.add(responses.PUT, f"{BASE}{GATE}", json={}, status=200)
        cases = [
            ({}, {"evalId": "e1"}),
            ({"expected_updated_at": None}, {"evalId": "e1", "expectedUpdatedAt": None}),
            (
                {"check_thresholds": {"code": 0.98}, "expected_updated_at": "2026-11-04T10:00:00.000Z"},
                {
                    "evalId": "e1",
                    "checkThresholds": [{"checkId": "code", "minimumPassRate": 0.98}],
                    "expectedUpdatedAt": "2026-11-04T10:00:00.000Z",
                },
            ),
            (
                {"check_thresholds": [{"checkId": "code", "minimumPassRate": 0.5}]},
                {"evalId": "e1", "checkThresholds": [{"checkId": "code", "minimumPassRate": 0.5}]},
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                responses.calls.reset()
                self.client.set_eval_gate(WF, "e1", **kwargs)
                self.assertEqual(json.loads(responses.calls[0].request.body), expected)

    @responses.activate
    def test_set_approval_gate_sends_rules_and_only_the_options_given(self):
        responses.add(responses.PUT, f"{BASE}{APPROVAL}", json={}, status=200)
        cases = [
            ({"user_ids": ["u1"]}, {"rules": [{"kind": "user", "userId": "u1"}]}),
            (
                {"user_ids": ["u1"], "roles": ["admin"], "allow_self_approval": False, "expected_revision": 3},
                {
                    "rules": [{"kind": "user", "userId": "u1"}, {"kind": "role", "role": "admin"}],
                    "allowSelfApproval": False,
                    "expectedRevision": 3,
                },
            ),
        ]
        for kwargs, expected in cases:
            with self.subTest(kwargs=kwargs):
                responses.calls.reset()
                self.client.set_approval_gate(WF, **kwargs)
                self.assertEqual(json.loads(responses.calls[0].request.body), expected)

    @responses.activate
    def test_get_eval_checks_filters_by_version(self):
        responses.add(responses.GET, f"{BASE}{GATE}/checks", json={}, status=200)
        self.client.get_eval_checks(WF, version_id=V)
        self.assertEqual(responses.calls[0].request.params, {"versionId": V})

    @responses.activate
    @patch("roboflow.core.workflow_deploy_gates.time.sleep")
    def test_wait_for_version_polls_until_no_gate_is_pending(self, mock_sleep):
        url = f"{BASE}/{WF}/versions/{V}/deploy-gates"
        responses.add(responses.GET, url, json=_verdicts("pending"), status=200)
        responses.add(responses.GET, url, json=_verdicts("pending"), status=200)
        responses.add(responses.GET, url, json=_verdicts("failed"), status=200)
        polled = []
        result = self.client.wait_for_version(WF, V, interval=0, on_poll=polled.append)
        self.assertEqual(result["gates"][0]["status"], "failed")
        self.assertEqual(len(polled), 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @responses.activate
    def test_wait_for_version_returns_at_once_when_nothing_is_pending(self):
        responses.add(
            responses.GET, f"{BASE}/{WF}/versions/{V}/deploy-gates", json=_verdicts("not_started"), status=200
        )
        self.assertFalse(self.client.wait_for_version(WF, V)["publishable"])
        self.assertEqual(len(responses.calls), 1)

    @responses.activate
    def test_wait_for_version_times_out(self):
        responses.add(responses.GET, f"{BASE}/{WF}/versions/{V}/deploy-gates", json=_verdicts("pending"), status=200)
        with self.assertRaises(TimeoutError):
            self.client.wait_for_version(WF, V, timeout=0)

    @responses.activate
    def test_refused_deploy_raises_the_typed_error(self):
        responses.add(responses.POST, f"{BASE}/{WF}/versions/{V}/deploy", json=NOT_PASSED, status=409)
        with self.assertRaises(WorkflowDeployGateError) as ctx:
            self.client.deploy(WF, V)
        error = ctx.exception
        self.assertIsInstance(error, WorkflowEvalError)
        self.assertIsInstance(error, RoboflowError)
        self.assertEqual(error.status_code, 409)
        self.assertEqual(error.code, "deploy_check_not_passed")
        self.assertEqual(error.gate_types, ["workflow_eval"])
        self.assertEqual(error.next_steps[0]["action"], "run_deploy_checks")


if __name__ == "__main__":
    unittest.main()
