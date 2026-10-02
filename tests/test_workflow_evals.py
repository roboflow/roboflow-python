import json
import os
import tempfile
import unittest
import uuid
from unittest.mock import patch

import responses

from roboflow.adapters import workflowevalsapi
from roboflow.adapters.workflowevalsapi import WorkflowEvalError
from roboflow.config import API_URL
from roboflow.core.workflow_evals import WorkflowEvals

WS = "test-ws"
KEY = "test_key"
BASE = f"{API_URL}/workspaces/{WS}/workflow-evals"

E, R, X, D, S, C = "eval-1", "run-1", "exec-1", "ds-1", "spec-1", "case-1"
SUBJECT = {"kind": "savedWorkflow", "workflowId": "wf", "subjectKey": "wf"}

# (client method, args, kwargs, HTTP method, path relative to BASE, sends Idempotency-Key, If-Match value)
# One row per public JSON route in the Workflow Evals OpenAPI registry. The 71st route,
# POST /specs/ai-draft, streams SSE and is covered by the ai_draft tests below.
ROUTES = [
    ("capabilities", (), {}, "GET", "/capabilities", False, None),
    ("openapi", (), {}, "GET", "/openapi.json", False, None),
    ("list_evaluators", (), {}, "GET", "/evaluators", False, None),
    ("get_evaluator", ("bool-match",), {}, "GET", "/evaluators/bool-match", False, None),
    ("get_schema", ("spec",), {}, "GET", "/schemas/spec", False, None),
    ("get_operation", ("op-1",), {}, "GET", "/operations/op-1", False, None),
    ("agent_manifest", (), {}, "GET", "/agent/manifest", False, None),
    ("list_agent_skills", (), {}, "GET", "/agent/skills", False, None),
    ("get_agent_skill", ("skill:create-eval",), {}, "GET", "/agent/skills/skill%3Acreate-eval", False, None),
    ("list_agent_bundles", (), {}, "GET", "/agent/bundles", False, None),
    ("get_agent_bundle", ("bundle:create-eval",), {}, "GET", "/agent/bundles/bundle%3Acreate-eval", False, None),
    ("get_agent_asset", ("schema:spec",), {}, "GET", "/agent/assets/schema%3Aspec", False, None),
    ("list", (), {}, "GET", "", False, None),
    ("create", (), {"name": "n"}, "POST", "", True, None),
    ("get", (E,), {}, "GET", f"/{E}", False, None),
    ("get_by_url", ("my-eval",), {}, "GET", "/by-url/my-eval", False, None),
    ("update", (E, {"name": "n"}), {"revision": 3}, "PATCH", f"/{E}", False, "3"),
    ("delete", (E,), {"deletion_key": "dk"}, "DELETE", f"/{E}", False, None),
    ("list_specs", (), {}, "GET", "/specs", False, None),
    ("create_spec", ({"spec": {}},), {}, "POST", "/specs", True, None),
    ("get_spec", (S,), {}, "GET", f"/specs/{S}", False, None),
    ("update_spec", (S, {"name": "n"}), {"revision": 2}, "PATCH", f"/specs/{S}", False, "2"),
    ("validate_spec", ({"checks": []},), {}, "POST", "/specs/validate", False, None),
    ("list_compatible_datasets", (S,), {}, "GET", f"/specs/{S}/compatible-eval-datasets", False, None),
    ("list_datasets", (), {}, "GET", "/eval-datasets", False, None),
    ("create_dataset", ({"name": "d"},), {}, "POST", "/eval-datasets", True, None),
    ("get_dataset", (D,), {}, "GET", f"/eval-datasets/{D}", False, None),
    ("update_dataset", (D, {"name": "d"}), {"revision": 1}, "PATCH", f"/eval-datasets/{D}", False, "1"),
    (
        "mutate_dataset_contract",
        (D, {"inputs": []}),
        {"revision": 1},
        "POST",
        f"/eval-datasets/{D}/contract-mutations",
        False,
        "1",
    ),
    ("list_compatible_specs", (D,), {}, "GET", f"/eval-datasets/{D}/compatible-specs", False, None),
    ("list_cases", (D,), {}, "GET", f"/eval-datasets/{D}/cases", False, None),
    ("add_case", (D, {"name": "c"}), {}, "POST", f"/eval-datasets/{D}/cases", True, None),
    ("get_case", (D, C), {}, "GET", f"/eval-datasets/{D}/cases/{C}", False, None),
    ("replace_case", (D, C, {"inputs": {}}), {"revision": 4}, "PUT", f"/eval-datasets/{D}/cases/{C}", False, "4"),
    ("delete_case", (D, C), {"revision": 4}, "DELETE", f"/eval-datasets/{D}/cases/{C}", False, "4"),
    ("import_cases", (D, {"items": []}), {}, "POST", f"/eval-datasets/{D}/cases/import", True, None),
    ("get_case_import", (D, "in-1"), {}, "GET", f"/eval-datasets/{D}/case-imports/in-1", False, None),
    (
        "prepare_case_asset_upload",
        (D,),
        {"asset_name": "a.png", "content_type": "image/png", "expected_size": 3},
        "POST",
        f"/eval-datasets/{D}/case-assets/uploads",
        True,
        None,
    ),
    (
        "complete_case_asset_upload",
        (D, "up-1"),
        {},
        "POST",
        f"/eval-datasets/{D}/case-assets/uploads/up-1/complete",
        False,
        None,
    ),
    ("suggest_bindings", (S, D, SUBJECT), {}, "POST", "/bindings/suggest", False, None),
    ("validate_bindings", (S, D, SUBJECT, {}), {}, "POST", "/bindings/validate", False, None),
    ("start_case_preparation", (E, {"runtime": {}}), {}, "POST", f"/{E}/case-preparations", True, None),
    ("get_current_case_preparation", (E,), {}, "GET", f"/{E}/case-preparations/current", False, None),
    ("get_case_preparation", (E, "p-1"), {}, "GET", f"/{E}/case-preparations/p-1", False, None),
    ("resume_case_preparation", (E, "p-1"), {}, "POST", f"/{E}/case-preparations/p-1/resume", False, None),
    ("list_case_preparation_cases", (E, "p-1"), {}, "GET", f"/{E}/case-preparations/p-1/cases", False, None),
    ("list_runs", (E,), {}, "GET", f"/{E}/runs", False, None),
    ("start_run", (E, {"executions": []}), {}, "POST", f"/{E}/runs", True, None),
    ("get_run", (E, R), {}, "GET", f"/{E}/runs/{R}", False, None),
    ("get_run_configuration", (E, R), {}, "GET", f"/{E}/runs/{R}/configuration", False, None),
    ("cancel_run", (E, R), {}, "POST", f"/{E}/runs/{R}/cancel", False, None),
    ("retry_run", (E, R), {}, "POST", f"/{E}/runs/{R}/retry-failed", True, None),
    ("replay_run", (E, R), {}, "POST", f"/{E}/runs/{R}/replay", True, None),
    ("rescore_run", (E, R, {"specId": S}), {}, "POST", f"/{E}/runs/{R}/rescore", True, None),
    ("delete_run", (E, R), {}, "DELETE", f"/{E}/runs/{R}", False, None),
    ("list_executions", (E, R), {}, "GET", f"/{E}/runs/{R}/executions", False, None),
    ("get_execution", (E, R, X), {}, "GET", f"/{E}/runs/{R}/executions/{X}", False, None),
    (
        "get_execution_configuration",
        (E, R, X),
        {},
        "GET",
        f"/{E}/runs/{R}/executions/{X}/configuration",
        False,
        None,
    ),
    ("list_execution_input_cases", (E, R, X), {}, "GET", f"/{E}/runs/{R}/executions/{X}/input-cases", False, None),
    (
        "submit_capture",
        (E, R, X, C, {"attemptCount": 1, "shardIndex": 0, "output": None}),
        {},
        "POST",
        f"/{E}/runs/{R}/executions/{X}/cases/{C}/capture",
        False,
        None,
    ),
    ("get_results_overview", (E, R, X), {}, "GET", f"/{E}/runs/{R}/executions/{X}/results/overview", False, None),
    ("list_case_results", (E, R, X), {}, "GET", f"/{E}/runs/{R}/executions/{X}/cases", False, None),
    ("get_case_result", (E, R, X, C), {}, "GET", f"/{E}/runs/{R}/executions/{X}/cases/{C}", False, None),
    ("compare", (E, {"executionIds": [X]}), {}, "POST", f"/{E}/comparisons", False, None),
    ("start_export", (E, R, {"format": "json"}), {}, "POST", f"/{E}/runs/{R}/exports", True, None),
    ("get_export_job", (E, R, "t-1"), {}, "GET", f"/{E}/runs/{R}/export-jobs/t-1", False, None),
    ("get_export_download", (E, R, "a-1"), {}, "GET", f"/{E}/runs/{R}/exports/a-1", False, None),
    (
        "start_embeddings",
        (E, R, X, "clip"),
        {},
        "POST",
        f"/{E}/runs/{R}/executions/{X}/analyses/embeddings",
        True,
        None,
    ),
    ("get_embeddings", (E, R, X), {}, "GET", f"/{E}/runs/{R}/executions/{X}/analyses/embeddings", False, None),
    (
        "cancel_embeddings",
        (E, R, X, "t-1"),
        {},
        "POST",
        f"/{E}/runs/{R}/executions/{X}/analyses/embeddings/t-1/cancel",
        False,
        None,
    ),
]


def _is_uuid4(value):
    try:
        return uuid.UUID(value).version == 4
    except (TypeError, ValueError):
        return False


class TestWorkflowEvalsRoutes(unittest.TestCase):
    def setUp(self):
        self.client = WorkflowEvals(KEY, WS)

    def test_covers_every_public_json_route(self):
        self.assertEqual(len({(row[3], row[4]) for row in ROUTES}), 70)

    @responses.activate
    def test_each_method_calls_its_route(self):
        for name, args, kwargs, method, path, idempotent, if_match in ROUTES:
            with self.subTest(method=name):
                responses.reset()
                responses.add(method, f"{BASE}{path}", json={"ok": True}, status=200)
                getattr(self.client, name)(*args, **kwargs)
                request = responses.calls[0].request
                self.assertEqual(request.method, method)
                self.assertEqual(request.url.split("?")[0], f"{BASE}{path}")
                self.assertEqual(request.headers["Authorization"], f"Bearer {KEY}")
                self.assertEqual(_is_uuid4(request.headers.get("Idempotency-Key")), idempotent)
                self.assertEqual(request.headers.get("If-Match"), if_match)


class TestWorkflowEvalsClient(unittest.TestCase):
    def setUp(self):
        self.client = WorkflowEvals(KEY, WS)

    @responses.activate
    def test_create_merges_body_and_keyword_fields(self):
        responses.add(responses.POST, BASE, json={"id": E}, status=201)
        key = str(uuid.uuid4())
        self.client.create({"subject": SUBJECT, "name": "old"}, name="new", spec_id=S, idempotency_key=key)
        request = responses.calls[0].request
        self.assertEqual(json.loads(request.body), {"subject": SUBJECT, "name": "new", "specId": S})
        self.assertEqual(request.headers["Idempotency-Key"], key)

    @responses.activate
    def test_list_query_parameters(self):
        responses.add(responses.GET, BASE, json={"items": []}, status=200)
        self.client.list(limit=10, cursor="c", include_spec_summary=True)
        self.assertEqual(responses.calls[0].request.params, {"limit": "10", "cursor": "c", "include": "specSummary"})

    @responses.activate
    def test_case_results_filters_repeat_slices_and_encode_booleans(self):
        url = f"{BASE}/{E}/runs/{R}/executions/{X}/cases"
        responses.add(responses.GET, url, json={"items": []}, status=200)
        self.client.list_case_results(E, R, X, slices=["a", "b"], failed_check=True, state="failed")
        query = responses.calls[0].request.url.split("?", 1)[1]
        self.assertIn("slice=a&slice=b", query)
        self.assertIn("failedCheck=true", query)
        self.assertIn("state=failed", query)

    @responses.activate
    def test_validate_spec_wraps_bare_spec(self):
        responses.add(responses.POST, f"{BASE}/specs/validate", json={"valid": True}, status=200)
        self.client.validate_spec({"checks": []})
        self.assertEqual(json.loads(responses.calls[0].request.body), {"spec": {"checks": []}})

    @responses.activate
    def test_no_content_returns_none(self):
        responses.add(responses.GET, f"{BASE}/{E}/case-preparations/current", status=204)
        self.assertIsNone(self.client.get_current_case_preparation(E))

    @responses.activate
    def test_error_envelope_is_parsed(self):
        body = {
            "error": {
                "code": "INVALID_SPEC",
                "category": "validation",
                "retryable": False,
                "requestId": "req-1",
                "message": "Spec is invalid",
                "details": {"diagnostics": [{"path": "$.checks"}]},
            }
        }
        responses.add(responses.POST, f"{BASE}/specs", json=body, status=422)
        with self.assertRaises(WorkflowEvalError) as ctx:
            self.client.create_spec({"spec": {}})
        error = ctx.exception
        self.assertEqual(error.status_code, 422)
        self.assertEqual(error.code, "INVALID_SPEC")
        self.assertFalse(error.retryable)
        self.assertEqual(error.request_id, "req-1")
        self.assertEqual(error.details["diagnostics"], [{"path": "$.checks"}])
        self.assertEqual(str(error), "Spec is invalid")

    @responses.activate
    def test_platform_auth_error_envelope(self):
        body = {"error": {"message": "Unauthorized", "type": "OAuthException", "hint": "check key"}}
        responses.add(responses.GET, BASE, json=body, status=401)
        with self.assertRaises(WorkflowEvalError) as ctx:
            self.client.list()
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertEqual(ctx.exception.category, "OAuthException")
        self.assertEqual(ctx.exception.hint, "check key")

    @responses.activate
    def test_delete_confirmed_resends_with_deletion_key(self):
        confirm = {
            "error": {
                "code": "deletion_confirmation_required",
                "category": "conflict",
                "retryable": False,
                "requestId": "r",
                "message": "Deletion confirmation is required",
                "details": {"impact": {"runs": 2}, "deletionKey": "dk-1"},
            }
        }
        responses.add(responses.DELETE, f"{BASE}/{E}", json=confirm, status=409)
        responses.add(responses.DELETE, f"{BASE}/{E}", json={"asyncTaskId": "t", "statusUrl": "/s"}, status=202)
        result = self.client.delete_confirmed(E)
        self.assertEqual(result["asyncTaskId"], "t")
        self.assertNotIn("Deletion-Key", responses.calls[0].request.headers)
        self.assertEqual(responses.calls[1].request.headers["Deletion-Key"], "dk-1")

    @responses.activate
    def test_delete_confirmed_reraises_other_conflicts(self):
        body = {"error": {"code": "CONFLICT", "message": "busy", "retryable": True, "requestId": "r"}}
        responses.add(responses.DELETE, f"{BASE}/{E}/runs/{R}", json=body, status=409)
        with self.assertRaises(WorkflowEvalError):
            self.client.delete_run_confirmed(E, R)
        self.assertEqual(len(responses.calls), 1)

    @responses.activate
    def test_ai_draft_streams_events_and_returns_draft(self):
        stream = (
            'event: status\ndata: {"message": "Reading Eval"}\n\n'
            'event: delta\ndata: {"delta": "..."}\n\n'
            ": keep-alive\n\n"
            'event: draft\ndata: {"spec": {"id": "s"}, "workflowBindings": {}}\n\n'
            "event: done\ndata: {}\n\n"
        )
        responses.add(
            responses.POST,
            f"{BASE}/specs/ai-draft",
            body=stream,
            status=200,
            content_type="text/event-stream",
        )
        events = []
        draft = self.client.ai_draft(E, "be strict", on_event=events.append)
        self.assertEqual(draft, {"spec": {"id": "s"}, "workflowBindings": {}})
        self.assertEqual([event["event"] for event in events], ["status", "delta", "draft", "done"])
        request = responses.calls[0].request
        self.assertEqual(json.loads(request.body), {"evalId": E, "instructions": "be strict"})
        self.assertTrue(_is_uuid4(request.headers["Idempotency-Key"]))
        self.assertEqual(request.headers["Accept"], "text/event-stream")

    @responses.activate
    def test_ai_draft_error_event_raises(self):
        stream = 'event: error\ndata: {"message": "No subject"}\n\nevent: done\ndata: {}\n\n'
        responses.add(responses.POST, f"{BASE}/specs/ai-draft", body=stream, status=200)
        with self.assertRaises(WorkflowEvalError) as ctx:
            self.client.ai_draft(E)
        self.assertIn("No subject", str(ctx.exception))

    @responses.activate
    def test_upload_case_asset_runs_the_three_steps(self):
        upload_url = "https://storage.googleapis.com/bucket/object?sig=1"
        responses.add(
            responses.POST,
            f"{BASE}/eval-datasets/{D}/case-assets/uploads",
            json={
                "uploadId": "up-1",
                "artifactId": "art-1",
                "uploadUrl": upload_url,
                "requiredHeaders": {"Content-Type": "image/png", "x-goog-meta": "1"},
                "expiresInSeconds": 600,
            },
            status=201,
        )
        responses.add(responses.PUT, upload_url, status=200)
        responses.add(
            responses.POST,
            f"{BASE}/eval-datasets/{D}/case-assets/uploads/up-1/complete",
            json={"artifactId": "art-1"},
            status=200,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pixel.png")
            with open(path, "wb") as handle:
                handle.write(b"png")
            asset = self.client.upload_case_asset(D, path)
        self.assertEqual(asset, {"artifactId": "art-1"})
        intent = json.loads(responses.calls[0].request.body)
        self.assertEqual(intent, {"assetName": "pixel.png", "contentType": "image/png", "expectedSize": 3})
        put = responses.calls[1].request
        self.assertEqual(put.body, b"png")
        self.assertEqual(put.headers["x-goog-meta"], "1")
        self.assertNotIn("Authorization", put.headers)

    @responses.activate
    @patch("roboflow.core.workflow_evals.time.sleep")
    def test_wait_for_run_polls_until_terminal(self, _sleep):
        url = f"{BASE}/{E}/runs/{R}"
        responses.add(responses.GET, url, json={"id": R, "state": "running"}, status=200)
        responses.add(responses.GET, url, json={"id": R, "state": "completed"}, status=200)
        run = self.client.wait_for_run(E, R, interval=0)
        self.assertEqual(run["state"], "completed")
        self.assertEqual(len(responses.calls), 2)


class TestSseParser(unittest.TestCase):
    def test_multiline_data_and_plain_text(self):
        lines = ["event: delta", "data: line one", "data: line two", "", 'data: {"a": 1}', ""]
        events = list(workflowevalsapi.parse_sse(lines))
        self.assertEqual(events[0], {"event": "delta", "data": "line one\nline two"})
        self.assertEqual(events[1], {"event": "message", "data": {"a": 1}})


class TestWorkspaceAccessor(unittest.TestCase):
    def test_workspace_returns_bound_client(self):
        from roboflow.core.workspace import Workspace

        info = {"workspace": {"name": "Test", "url": WS, "projects": [], "members": []}}
        workspace = Workspace(info, api_key=KEY, default_workspace=WS, model_format="yolov8")
        client = workspace.workflow_evals()
        self.assertIsInstance(client, WorkflowEvals)
        self.assertEqual(client.workspace_url, WS)


if __name__ == "__main__":
    unittest.main()
