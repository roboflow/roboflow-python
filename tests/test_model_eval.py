"""Unit tests for the ModelEval SDK class and Workspace.evals/eval accessors."""

from __future__ import annotations

import unittest
from unittest.mock import patch


def _make_workspace(api_key="k", url="lee-sandbox"):
    """Build a Workspace with the minimal info dict its constructor accepts."""
    from roboflow.core.workspace import Workspace

    info = {
        "workspace": {
            "name": "Test",
            "url": url,
            "projects": [],
            "members": [],
        }
    }
    return Workspace(info, api_key=api_key, default_workspace=url, model_format="yolov8")


class TestModelEvalConstruction(unittest.TestCase):
    def test_apply_info_populates_attributes(self):
        from roboflow.core.model_eval import ModelEval

        info = {
            "evalId": "e1",
            "status": "done",
            "projectId": "p1",
            "versionId": "3",
            "modelId": "m1",
            "createdAt": "2025-01-01",
            "summary": {"mAP": 0.9, "precision": 0.8, "recall": 0.85},
        }
        ev = ModelEval("k", "ws", "e1", info=info)

        self.assertEqual(ev.id, "e1")
        self.assertEqual(ev.status, "done")
        self.assertEqual(ev.project_id, "p1")
        self.assertEqual(ev.version_id, "3")
        self.assertEqual(ev.model_id, "m1")
        self.assertEqual(ev.created_at, "2025-01-01")
        self.assertEqual(ev.summary["mAP"], 0.9)

    def test_construction_without_info(self):
        from roboflow.core.model_eval import ModelEval

        ev = ModelEval("k", "ws", "e1")
        self.assertEqual(ev.id, "e1")
        self.assertIsNone(ev.status)
        self.assertIsNone(ev.summary)


class TestModelEvalRefresh(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_model_eval")
    def test_refresh_updates_status_and_summary(self, mock_get):
        from roboflow.core.model_eval import ModelEval

        mock_get.return_value = {
            "evalId": "e1",
            "status": "done",
            "summary": {"mAP": 0.95},
        }
        ev = ModelEval("k", "ws", "e1")
        result = ev.refresh()

        self.assertIs(result, ev)  # chainable
        self.assertEqual(ev.status, "done")
        self.assertEqual(ev.summary["mAP"], 0.95)
        mock_get.assert_called_once_with("k", "ws", "e1")


class TestModelEvalPanelAccessors(unittest.TestCase):
    """Each panel method delegates to the matching rfapi function with the right args."""

    @patch("roboflow.adapters.rfapi.get_model_eval_map_results")
    def test_map_results(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"splits": {}}
        ev = ModelEval("k", "ws", "e1")
        result = ev.map_results()

        self.assertEqual(result, {"splits": {}})
        mock_fn.assert_called_once_with("k", "ws", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval_confidence_sweep")
    def test_confidence_sweep(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"splits": {}}
        ModelEval("k", "ws", "e1").confidence_sweep()

        mock_fn.assert_called_once_with("k", "ws", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval_performance_by_class")
    def test_performance_by_class_default_split(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"classes": []}
        ModelEval("k", "ws", "e1").performance_by_class()
        mock_fn.assert_called_once_with("k", "ws", "e1", split=None)

    @patch("roboflow.adapters.rfapi.get_model_eval_performance_by_class")
    def test_performance_by_class_with_split(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"classes": []}
        ModelEval("k", "ws", "e1").performance_by_class(split="valid")
        mock_fn.assert_called_once_with("k", "ws", "e1", split="valid")

    @patch("roboflow.adapters.rfapi.get_model_eval_confusion_matrix")
    def test_confusion_matrix(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"matrix": []}
        ModelEval("k", "ws", "e1").confusion_matrix(split="test", confidence=30)
        mock_fn.assert_called_once_with("k", "ws", "e1", split="test", confidence=30)

    @patch("roboflow.adapters.rfapi.get_model_eval_vector_analysis")
    def test_vector_analysis(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"clusters": []}
        ModelEval("k", "ws", "e1").vector_analysis(confidence=40)
        mock_fn.assert_called_once_with("k", "ws", "e1", confidence=40)

    @patch("roboflow.adapters.rfapi.get_model_eval_image_predictions")
    def test_image_predictions(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"images": []}
        ModelEval("k", "ws", "e1").image_predictions(split="valid", confidence=20, limit=50, offset=100)
        mock_fn.assert_called_once_with("k", "ws", "e1", split="valid", confidence=20, limit=50, offset=100)

    @patch("roboflow.adapters.rfapi.get_model_eval_recommendations")
    def test_recommendations(self, mock_fn):
        from roboflow.core.model_eval import ModelEval

        mock_fn.return_value = {"recommendations": []}
        ModelEval("k", "ws", "e1").recommendations()
        mock_fn.assert_called_once_with("k", "ws", "e1")


class TestModelEvalErrors(unittest.TestCase):
    """Typed errors from the adapter propagate through the SDK accessors."""

    @patch("roboflow.adapters.rfapi.get_model_eval_map_results")
    def test_not_done_error_propagates(self, mock_fn):
        from roboflow.adapters import rfapi
        from roboflow.core.model_eval import ModelEval

        mock_fn.side_effect = rfapi.ModelEvalNotDoneError("Eval still running")
        ev = ModelEval("k", "ws", "e1")
        with self.assertRaises(rfapi.ModelEvalNotDoneError):
            ev.map_results()

    @patch("roboflow.adapters.rfapi.get_model_eval")
    def test_refresh_404_propagates(self, mock_fn):
        from roboflow.adapters import rfapi
        from roboflow.core.model_eval import ModelEval

        mock_fn.side_effect = rfapi.ModelEvalNotFoundError("nope")
        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            ModelEval("k", "ws", "e1").refresh()


class TestWorkspaceEvalAccessors(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_model_evals")
    def test_evals_returns_modeleval_instances(self, mock_list):
        from roboflow.core.model_eval import ModelEval

        mock_list.return_value = {
            "evals": [
                {"evalId": "e1", "status": "done", "projectId": "p1"},
                {"evalId": "e2", "status": "running", "projectId": "p1"},
            ]
        }
        ws = _make_workspace()
        result = ws.evals(status="done", limit=5)

        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(e, ModelEval) for e in result))
        self.assertEqual(result[0].id, "e1")
        self.assertEqual(result[0].status, "done")
        self.assertEqual(result[1].id, "e2")
        # Workspace forwards filters to the adapter
        mock_list.assert_called_once_with(
            "k", "lee-sandbox", project=None, version=None, model=None, status="done", limit=5
        )

    @patch("roboflow.adapters.rfapi.list_model_evals")
    def test_evals_passes_all_filters(self, mock_list):
        mock_list.return_value = {"evals": []}

        ws = _make_workspace()
        ws.evals(project="p1", version="3", model="m1", status="failed", limit=200)

        mock_list.assert_called_once_with(
            "k", "lee-sandbox", project="p1", version="3", model="m1", status="failed", limit=200
        )

    @patch("roboflow.adapters.rfapi.list_model_evals")
    def test_evals_empty_list(self, mock_list):
        mock_list.return_value = {"evals": []}
        ws = _make_workspace()
        self.assertEqual(ws.evals(), [])

    @patch("roboflow.adapters.rfapi.get_model_eval")
    def test_eval_returns_populated_modeleval(self, mock_get):
        from roboflow.core.model_eval import ModelEval

        mock_get.return_value = {
            "evalId": "e1",
            "status": "done",
            "summary": {"mAP": 0.91},
        }
        ws = _make_workspace()
        ev = ws.eval("e1")

        self.assertIsInstance(ev, ModelEval)
        self.assertEqual(ev.id, "e1")
        self.assertEqual(ev.status, "done")
        self.assertEqual(ev.summary["mAP"], 0.91)
        mock_get.assert_called_once_with("k", "lee-sandbox", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval")
    def test_eval_propagates_not_found(self, mock_get):
        from roboflow.adapters import rfapi

        mock_get.side_effect = rfapi.ModelEvalNotFoundError("nope")
        ws = _make_workspace()
        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            ws.eval("bad")


if __name__ == "__main__":
    unittest.main()
