"""Unit tests for Phase 2 rfapi functions."""

import json
import unittest
from unittest.mock import MagicMock, patch

from roboflow.adapters.rfapi import _normalize_workflow_config


class TestListBatches(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import list_batches

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"batches": [{"id": "b1"}]})
        result = list_batches("key", "ws", "proj")
        self.assertEqual(result, {"batches": [{"id": "b1"}]})
        mock_get.assert_called_once()
        self.assertIn("/ws/proj/batches", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, list_batches

        mock_get.return_value = MagicMock(status_code=404, text="Not found")
        with self.assertRaises(RoboflowError):
            list_batches("key", "ws", "proj")


class TestGetBatch(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_batch

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"batch": {"id": "b1"}})
        result = get_batch("key", "ws", "proj", "b1")
        self.assertEqual(result, {"batch": {"id": "b1"}})
        self.assertIn("/ws/proj/batches/b1", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_batch

        mock_get.return_value = MagicMock(status_code=500, text="Server error")
        with self.assertRaises(RoboflowError):
            get_batch("key", "ws", "proj", "b1")


class TestListAnnotationJobs(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import list_annotation_jobs

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"jobs": []})
        result = list_annotation_jobs("key", "ws", "proj")
        self.assertEqual(result, {"jobs": []})
        self.assertIn("/ws/proj/jobs", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, list_annotation_jobs

        mock_get.return_value = MagicMock(status_code=403, text="Forbidden")
        with self.assertRaises(RoboflowError):
            list_annotation_jobs("key", "ws", "proj")


class TestGetAnnotationJob(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_annotation_job

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"job": {"id": "j1", "name": "job1"}})
        result = get_annotation_job("key", "ws", "proj", "j1")
        self.assertEqual(result["job"]["id"], "j1")
        self.assertIn("/ws/proj/jobs/j1", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_annotation_job

        mock_get.return_value = MagicMock(status_code=404, text="Not found")
        with self.assertRaises(RoboflowError):
            get_annotation_job("key", "ws", "proj", "j1")


class TestCreateAnnotationJob(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import create_annotation_job

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"job": {"id": "j2"}})
        result = create_annotation_job("key", "ws", "proj", name="my-job", batch_id="b1")
        self.assertEqual(result["job"]["id"], "j2")
        # Verify URL and payload
        call_args = mock_post.call_args
        self.assertIn("/ws/proj/jobs", call_args[0][0])
        payload = call_args[1]["json"]
        self.assertEqual(payload["name"], "my-job")
        self.assertEqual(payload["batchId"], "b1")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success_200(self, mock_post):
        from roboflow.adapters.rfapi import create_annotation_job

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"job": {"id": "j3"}})
        result = create_annotation_job("key", "ws", "proj", name="my-job")
        self.assertEqual(result["job"]["id"], "j3")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_with_assignees(self, mock_post):
        from roboflow.adapters.rfapi import create_annotation_job

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"job": {"id": "j4"}})
        create_annotation_job("key", "ws", "proj", name="j", assignees=["a@b.com"])
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["assignees"], ["a@b.com"])

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, create_annotation_job

        mock_post.return_value = MagicMock(status_code=400, text="Bad request")
        with self.assertRaises(RoboflowError):
            create_annotation_job("key", "ws", "proj", name="j")


class TestListFolders(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import list_folders

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"groups": []})
        result = list_folders("key", "ws")
        self.assertEqual(result, {"groups": []})
        mock_get.assert_called_once()
        self.assertIn("/ws/groups", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, list_folders

        mock_get.return_value = MagicMock(status_code=404, text="Not found")
        with self.assertRaises(RoboflowError):
            list_folders("key", "ws")


class TestGetFolder(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_folder

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"group": {"id": "g1", "name": "Folder1"}})
        result = get_folder("key", "ws", "g1")
        self.assertEqual(result["group"]["id"], "g1")
        call_kwargs = mock_get.call_args[1]
        self.assertEqual(call_kwargs["params"]["groupId"], "g1")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_folder

        mock_get.return_value = MagicMock(status_code=404, text="Not found")
        with self.assertRaises(RoboflowError):
            get_folder("key", "ws", "g1")


class TestCreateFolder(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import create_folder

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"group": {"id": "g2"}})
        result = create_folder("key", "ws", "NewFolder")
        self.assertEqual(result["group"]["id"], "g2")
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["name"], "NewFolder")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_with_parent_and_projects(self, mock_post):
        from roboflow.adapters.rfapi import create_folder

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"group": {"id": "g3"}})
        create_folder("key", "ws", "Sub", parent_id="g1", project_ids=["p1", "p2"])
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["parent_id"], "g1")
        self.assertEqual(payload["projects"], ["p1", "p2"])

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, create_folder

        mock_post.return_value = MagicMock(status_code=400, text="Bad request")
        with self.assertRaises(RoboflowError):
            create_folder("key", "ws", "BadFolder")


class TestUpdateFolder(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import update_folder

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        result = update_folder("key", "ws", "g1", name="Renamed")
        self.assertEqual(result["status"], "ok")
        self.assertIn("/ws/groups/g1", mock_post.call_args[0][0])
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["name"], "Renamed")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, update_folder

        mock_post.return_value = MagicMock(status_code=500, text="Server error")
        with self.assertRaises(RoboflowError):
            update_folder("key", "ws", "g1", name="X")


class TestDeleteFolder(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.delete")
    def test_success(self, mock_delete):
        from roboflow.adapters.rfapi import delete_folder

        mock_delete.return_value = MagicMock(status_code=200, json=lambda: {"status": "deleted"})
        result = delete_folder("key", "ws", "g1")
        self.assertEqual(result["status"], "deleted")
        self.assertIn("/ws/groups/g1", mock_delete.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.delete")
    def test_error(self, mock_delete):
        from roboflow.adapters.rfapi import RoboflowError, delete_folder

        mock_delete.return_value = MagicMock(status_code=403, text="Forbidden")
        with self.assertRaises(RoboflowError):
            delete_folder("key", "ws", "g1")


class TestListWorkflows(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import list_workflows

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflows": [{"name": "wf1"}]})
        result = list_workflows("key", "ws")
        self.assertEqual(len(result["workflows"]), 1)
        self.assertIn("/ws/workflows", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, list_workflows

        mock_get.return_value = MagicMock(status_code=500, text="Error")
        with self.assertRaises(RoboflowError):
            list_workflows("key", "ws")


class TestGetWorkflow(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_workflow

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"workflow": {"url": "wf1"}})
        result = get_workflow("key", "ws", "wf1")
        self.assertEqual(result["workflow"]["url"], "wf1")
        self.assertIn("/ws/workflows/wf1", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_workflow

        mock_get.return_value = MagicMock(status_code=404, text="Not found")
        with self.assertRaises(RoboflowError):
            get_workflow("key", "ws", "wf1")


class TestCreateWorkflow(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "new-wf"}})
        result = create_workflow("key", "ws", name="New Workflow")
        self.assertEqual(result["workflow"]["url"], "new-wf")
        self.assertIn("/ws/createWorkflow", mock_post.call_args[0][0])
        # Params are passed as query-string params, not JSON body
        params = mock_post.call_args[1]["params"]
        self.assertEqual(params["name"], "New Workflow")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_auto_generates_url_slug(self, mock_post):
        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "my-workflow"}})
        create_workflow("key", "ws", name="My Workflow")
        params = mock_post.call_args[1]["params"]
        self.assertEqual(params["url"], "my-workflow")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_with_config_and_template(self, mock_post):
        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"workflow": {"url": "wf2"}})
        create_workflow("key", "ws", name="WF2", url="wf2", config='{"a":1}', template='{"b":2}')
        params = mock_post.call_args[1]["params"]
        self.assertEqual(params["url"], "wf2")
        self.assertEqual(params["config"], '{"a":1}')
        self.assertEqual(params["template"], '{"b":2}')

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_config_dict_serialized_to_string(self, mock_post):
        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"workflow": {"url": "wf3"}})
        create_workflow("key", "ws", name="WF3", config={"a": 1}, template={"b": 2})
        params = mock_post.call_args[1]["params"]
        # config and template must be strings per the API
        self.assertIsInstance(params["config"], str)
        self.assertIsInstance(params["template"], str)

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_defaults_config_and_template(self, mock_post):
        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "wf4"}})
        create_workflow("key", "ws", name="WF4")
        params = mock_post.call_args[1]["params"]
        self.assertEqual(params["config"], "{}")
        self.assertEqual(params["template"], "{}")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, create_workflow

        mock_post.return_value = MagicMock(status_code=400, text="Bad request")
        with self.assertRaises(RoboflowError):
            create_workflow("key", "ws", name="Bad")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_bare_spec_dict_is_auto_wrapped(self, mock_post):
        """Docs-shaped workflow definitions get wrapped in {"specification": ...}
        so they match the backend's stored format and the inference server's
        expectation. See `_normalize_workflow_config`."""
        import json as _json

        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "wf"}})
        bare = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
        create_workflow("key", "ws", name="WF", config=bare)
        sent_config = _json.loads(mock_post.call_args[1]["params"]["config"])
        self.assertEqual(sent_config, {"specification": bare})

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_already_wrapped_config_is_not_double_wrapped(self, mock_post):
        import json as _json

        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "wf"}})
        wrapped = {"specification": {"version": "1.0", "inputs": [], "steps": [], "outputs": []}}
        create_workflow("key", "ws", name="WF", config=wrapped)
        sent_config = _json.loads(mock_post.call_args[1]["params"]["config"])
        self.assertEqual(sent_config, wrapped)

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_bare_spec_json_string_is_auto_wrapped(self, mock_post):
        """JSON strings are parsed, wrapped if bare, and re-serialized."""
        import json as _json

        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "wf"}})
        bare_str = '{"version": "1.0", "steps": []}'
        create_workflow("key", "ws", name="WF", config=bare_str)
        sent_config = _json.loads(mock_post.call_args[1]["params"]["config"])
        self.assertEqual(sent_config, {"specification": {"version": "1.0", "steps": []}})

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_non_workflow_dict_is_not_wrapped(self, mock_post):
        """Dicts that don't look like a workflow spec (no version/inputs/steps/outputs)
        are passed through unchanged to avoid second-guessing custom payloads."""
        import json as _json

        from roboflow.adapters.rfapi import create_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "wf"}})
        create_workflow("key", "ws", name="WF", config={"a": 1})
        sent_config = _json.loads(mock_post.call_args[1]["params"]["config"])
        self.assertEqual(sent_config, {"a": 1})


class TestNormalizeWorkflowConfig(unittest.TestCase):
    """Direct unit tests for the private ``_normalize_workflow_config`` helper.

    Imported from the private API intentionally — whitebox tests lock the
    behavior contract that ``create_workflow``/``update_workflow`` rely on.
    """

    def test_none_returns_empty_object(self):
        self.assertEqual(_normalize_workflow_config(None), "{}")

    def test_empty_dict_serialized_to_empty_json(self):
        # Empty dict has no workflow keys, so it falls through the wrap check
        # and serializes to ``"{}"`` — coincidentally matching the legacy
        # ``None -> "{}"`` default.
        self.assertEqual(_normalize_workflow_config({}), "{}")

    def test_string_without_workflow_keys_preserved_byte_for_byte(self):
        self.assertEqual(_normalize_workflow_config('{"a":1}'), '{"a":1}')

    def test_non_json_string_passthrough(self):
        self.assertEqual(_normalize_workflow_config("not json"), "not json")

    def test_already_wrapped_json_string_preserved_byte_for_byte(self):
        wrapped = '{"specification": {"version": "1.0"}}'
        self.assertEqual(_normalize_workflow_config(wrapped), wrapped)

    def test_partial_workflow_dict_is_wrapped(self):
        # Single workflow-shaped key at top level is enough to classify as a
        # bare spec; users often build definitions incrementally.
        result = _normalize_workflow_config({"steps": [{"id": "s1"}]})
        self.assertEqual(json.loads(result), {"specification": {"steps": [{"id": "s1"}]}})

    def test_json_array_input_preserved(self):
        # ``isinstance(parsed, dict)`` guards against calling ``.keys()`` on
        # non-dict JSON; pinning the no-wrap behavior here protects that.
        self.assertEqual(_normalize_workflow_config("[1,2,3]"), "[1,2,3]")

    def test_json_scalar_inputs_preserved(self):
        self.assertEqual(_normalize_workflow_config("42"), "42")
        self.assertEqual(_normalize_workflow_config("true"), "true")
        self.assertEqual(_normalize_workflow_config("null"), "null")

    def test_utf8_bom_stripped_before_parse(self):
        # Windows editors frequently prepend a UTF-8 BOM. Without the strip,
        # ``json.loads`` raises and the raw (unwrapped) string would ship —
        # reproducing the exact 502 this PR is meant to fix.
        bom_str = '\ufeff{"version":"1.0","steps":[]}'
        result = _normalize_workflow_config(bom_str)
        self.assertEqual(json.loads(result), {"specification": {"version": "1.0", "steps": []}})

    def test_utf8_bom_stripped_when_already_wrapped(self):
        # Already-wrapped JSON saved from a Windows editor would otherwise
        # ship the BOM through to the backend, where the inference server's
        # ``json.loads`` rejects it ("Unexpected UTF-8 BOM") \u2014 same 502 in
        # a different shape.
        bom_wrapped = '\ufeff{"specification": {"version": "1.0"}}'
        self.assertEqual(
            _normalize_workflow_config(bom_wrapped),
            '{"specification": {"version": "1.0"}}',
        )

    def test_utf8_bom_stripped_for_non_workflow_dict_string(self):
        # A custom JSON payload (not a workflow spec) with a leading BOM
        # also gets the BOM removed so the backend stores parseable JSON.
        bom_custom = '\ufeff{"a":1}'
        self.assertEqual(_normalize_workflow_config(bom_custom), '{"a":1}')

    def test_utf8_bom_stripped_for_non_json_string(self):
        # Non-JSON string with a BOM: still strip the BOM, since shipping
        # it verbatim has no upside and would only produce a downstream
        # decode error if anything ever tries to parse it.
        self.assertEqual(_normalize_workflow_config("\ufeffnot json"), "not json")

    def test_wrapped_output_uses_compact_separators(self):
        # Matches the shape the web UI writes via ``JSON.stringify``, so
        # Firestore audit/diff tooling sees SDK- and UI-written rows as
        # byte-identical when the logical content matches.
        result = _normalize_workflow_config({"version": "1.0", "steps": []})
        self.assertEqual(result, '{"specification":{"version":"1.0","steps":[]}}')


class TestUpdateWorkflow(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import update_workflow

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        result = update_workflow(
            "key", "ws", workflow_id="id-1", workflow_name="WF1", workflow_url="wf1", config={"steps": [1]}
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("/ws/updateWorkflow", mock_post.call_args[0][0])
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["id"], "id-1")
        self.assertEqual(payload["name"], "WF1")
        self.assertEqual(payload["url"], "wf1")
        # config dict should be serialized to string
        self.assertIsInstance(payload["config"], str)

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_config_string_passthrough(self, mock_post):
        from roboflow.adapters.rfapi import update_workflow

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        update_workflow("key", "ws", workflow_id="id-1", workflow_name="WF1", workflow_url="wf1", config='{"a":1}')
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["config"], '{"a":1}')

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_bare_spec_dict_is_auto_wrapped_on_update(self, mock_post):
        import json as _json

        from roboflow.adapters.rfapi import update_workflow

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"status": "ok"})
        bare = {"version": "1.0", "inputs": [], "steps": [], "outputs": []}
        update_workflow("key", "ws", workflow_id="id-1", workflow_name="WF1", workflow_url="wf1", config=bare)
        sent_config = _json.loads(mock_post.call_args[1]["json"]["config"])
        self.assertEqual(sent_config, {"specification": bare})

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, update_workflow

        mock_post.return_value = MagicMock(status_code=500, text="Server error")
        with self.assertRaises(RoboflowError):
            update_workflow("key", "ws", workflow_id="id-1", workflow_name="WF1", workflow_url="wf1", config="{}")


class TestListWorkflowVersions(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import list_workflow_versions

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"versions": [{"id": "v1"}]})
        result = list_workflow_versions("key", "ws", "wf1")
        self.assertEqual(len(result["versions"]), 1)
        self.assertIn("/ws/workflows/wf1/versions", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, list_workflow_versions

        mock_get.return_value = MagicMock(status_code=500, text="Error")
        with self.assertRaises(RoboflowError):
            list_workflow_versions("key", "ws", "wf1")


class TestForkWorkflow(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import fork_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "forked"}})
        result = fork_workflow("key", "target-ws", source_workspace="src-ws", source_workflow="wf1")
        self.assertEqual(result["workflow"]["url"], "forked")
        self.assertIn("/target-ws/forkWorkflow", mock_post.call_args[0][0])
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["source_workspace"], "src-ws")
        self.assertEqual(payload["source_workflow"], "wf1")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success_200(self, mock_post):
        from roboflow.adapters.rfapi import fork_workflow

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"workflow": {"url": "forked2"}})
        result = fork_workflow("key", "ws", source_workspace="src-ws", source_workflow="wf2")
        self.assertEqual(result["workflow"]["url"], "forked2")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_with_name_and_url(self, mock_post):
        from roboflow.adapters.rfapi import fork_workflow

        mock_post.return_value = MagicMock(status_code=201, json=lambda: {"workflow": {"url": "custom-fork"}})
        fork_workflow(
            "key", "ws", source_workspace="src-ws", source_workflow="wf1", name="Custom Fork", url="custom-fork"
        )
        payload = mock_post.call_args[1]["json"]
        self.assertEqual(payload["name"], "Custom Fork")
        self.assertEqual(payload["url"], "custom-fork")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, fork_workflow

        mock_post.return_value = MagicMock(status_code=403, text="Forbidden")
        with self.assertRaises(RoboflowError):
            fork_workflow("key", "ws", source_workspace="src-ws", source_workflow="wf1")


class TestGetBillingUsage(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.post")
    def test_success(self, mock_post):
        from roboflow.adapters.rfapi import get_billing_usage

        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"usage": {"credits": 100}})
        result = get_billing_usage("key", "ws")
        self.assertEqual(result["usage"]["credits"], 100)
        self.assertIn("/ws/billing-usage-report", mock_post.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_error(self, mock_post):
        from roboflow.adapters.rfapi import RoboflowError, get_billing_usage

        mock_post.return_value = MagicMock(status_code=403, text="Forbidden")
        with self.assertRaises(RoboflowError):
            get_billing_usage("key", "ws")


class TestGetPlanInfo(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_plan_info

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"plan": "starter", "limit": 1000})
        result = get_plan_info("key")
        self.assertEqual(result["plan"], "starter")
        self.assertIn("/usage/plan", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_plan_info

        mock_get.return_value = MagicMock(status_code=401, text="Unauthorized")
        with self.assertRaises(RoboflowError):
            get_plan_info("key")


class TestGetLabelingStats(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_labeling_stats

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"stats": {"labeled": 50}})
        result = get_labeling_stats("key", "ws")
        self.assertEqual(result["stats"]["labeled"], 50)
        self.assertIn("/ws/stats", mock_get.call_args[0][0])

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_labeling_stats

        mock_get.return_value = MagicMock(status_code=500, text="Error")
        with self.assertRaises(RoboflowError):
            get_labeling_stats("key", "ws")


class TestGetVideoJobStatus(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import get_video_job_status

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"status": "completed", "progress": 1.0})
        result = get_video_job_status("key", "job-123")
        self.assertEqual(result["status"], "completed")
        call_kwargs = mock_get.call_args[1]
        self.assertEqual(call_kwargs["params"]["job_id"], "job-123")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, get_video_job_status

        mock_get.return_value = MagicMock(status_code=404, text="Not found")
        with self.assertRaises(RoboflowError):
            get_video_job_status("key", "job-123")


class TestSearchUniverse(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        from roboflow.adapters.rfapi import search_universe

        mock_get.return_value = MagicMock(
            status_code=200, json=lambda: {"results": [{"name": "cats-dataset"}], "total": 1}
        )
        result = search_universe("cats")
        self.assertEqual(result["total"], 1)
        call_kwargs = mock_get.call_args[1]
        self.assertEqual(call_kwargs["params"]["q"], "cats")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_with_type_and_limit(self, mock_get):
        from roboflow.adapters.rfapi import search_universe

        mock_get.return_value = MagicMock(status_code=200, json=lambda: {"results": [], "total": 0})
        search_universe("dogs", project_type="model", limit=5, page=2)
        call_kwargs = mock_get.call_args[1]
        self.assertEqual(call_kwargs["params"]["type"], "model")
        self.assertEqual(call_kwargs["params"]["limit"], 5)
        self.assertEqual(call_kwargs["params"]["page"], 2)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_error(self, mock_get):
        from roboflow.adapters.rfapi import RoboflowError, search_universe

        mock_get.return_value = MagicMock(status_code=500, text="Server error")
        with self.assertRaises(RoboflowError):
            search_universe("query")


if __name__ == "__main__":
    unittest.main()
