"""Unit tests for the model-eval rfapi helpers (`/{ws}/model-evals/...`)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from roboflow.adapters import rfapi
from roboflow.config import API_URL


def _resp(status: int, body):
    """Build a mock requests.Response double for the given status + JSON body."""
    mock = MagicMock(status_code=status)
    mock.json.return_value = body
    mock.text = repr(body)
    return mock


class TestListModelEvals(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success_no_filters(self, mock_get):
        mock_get.return_value = _resp(200, {"evals": [{"id": "e1", "status": "done"}]})

        result = rfapi.list_model_evals("k", "ws")

        self.assertEqual(result, {"evals": [{"id": "e1", "status": "done"}]})
        url = mock_get.call_args[0][0]
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(url, f"{API_URL}/ws/model-evals")
        self.assertEqual(params, {"api_key": "k"})

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success_with_filters(self, mock_get):
        mock_get.return_value = _resp(200, {"evals": []})

        rfapi.list_model_evals("k", "ws", project="p1", version=3, model="m1", status="done", limit=10)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(
            params,
            {
                "api_key": "k",
                "project": "p1",
                "version": 3,
                "model": "m1",
                "status": "done",
                "limit": 10,
            },
        )

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_omits_none_filters(self, mock_get):
        mock_get.return_value = _resp(200, {"evals": []})

        rfapi.list_model_evals("k", "ws", status="done", limit=None)

        params = mock_get.call_args.kwargs["params"]
        self.assertNotIn("limit", params)
        self.assertEqual(params["status"], "done")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_404_raises_not_found(self, mock_get):
        mock_get.return_value = _resp(404, {"error": "model_eval_not_found", "message": "nope"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError) as ctx:
            rfapi.list_model_evals("k", "ws")
        self.assertIn("nope", str(ctx.exception))


class TestGetModelEval(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value = _resp(200, {"id": "e1", "status": "done", "summary": {"mAP": 0.9}})

        result = rfapi.get_model_eval("k", "ws", "e1")

        self.assertEqual(result["summary"]["mAP"], 0.9)
        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1")


class TestPanelEndpoints(unittest.TestCase):
    """Each panel endpoint forwards path + params correctly."""

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_map_results_url(self, mock_get):
        mock_get.return_value = _resp(200, {"splits": {}})

        rfapi.get_model_eval_map_results("k", "ws", "e1")

        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1/map-results")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_confidence_sweep_url(self, mock_get):
        mock_get.return_value = _resp(200, {"splits": {}})

        rfapi.get_model_eval_confidence_sweep("k", "ws", "e1")

        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1/confidence-sweep")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_performance_by_class_passes_split(self, mock_get):
        mock_get.return_value = _resp(200, {"split": "valid", "classes": []})

        rfapi.get_model_eval_performance_by_class("k", "ws", "e1", split="valid")

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["split"], "valid")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_confusion_matrix_passes_params(self, mock_get):
        mock_get.return_value = _resp(200, {"matrix": []})

        rfapi.get_model_eval_confusion_matrix("k", "ws", "e1", split="test", confidence=30)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["split"], "test")
        self.assertEqual(params["confidence"], 30)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_image_predictions_pagination(self, mock_get):
        mock_get.return_value = _resp(200, {"images": []})

        rfapi.get_model_eval_image_predictions("k", "ws", "e1", split="test", confidence=20, limit=50, offset=100)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["limit"], 50)
        self.assertEqual(params["offset"], 100)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_recommendations_url(self, mock_get):
        mock_get.return_value = _resp(200, {"recommendations": []})

        rfapi.get_model_eval_recommendations("k", "ws", "e1")

        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1/recommendations")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_vector_analysis_passes_confidence(self, mock_get):
        mock_get.return_value = _resp(200, {"clusters": []})

        rfapi.get_model_eval_vector_analysis("k", "ws", "e1", confidence=25)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["confidence"], 25)


class TestErrorMapping(unittest.TestCase):
    """Typed errors are routed to the right exception subclass."""

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_404_flat_envelope(self, mock_get):
        # Server returns the flat shape: {"error": "code", "message": "..."}
        mock_get.return_value = _resp(404, {"error": "model_eval_not_found", "message": "Eval 'x' not found"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")
        self.assertIn("Eval 'x' not found", str(ctx.exception))

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_404_status_code_fallback(self, mock_get):
        # No `error` field at all — fall back to the status code mapping.
        mock_get.return_value = _resp(404, {"message": "something went wrong"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            rfapi.get_model_eval("k", "ws", "x")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_409_not_done(self, mock_get):
        mock_get.return_value = _resp(409, {"error": "model_eval_not_done", "message": "Eval still running"})

        with self.assertRaises(rfapi.ModelEvalNotDoneError):
            rfapi.get_model_eval_map_results("k", "ws", "x")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_400_invalid_split(self, mock_get):
        mock_get.return_value = _resp(400, {"error": "invalid_split", "message": "Invalid split"})

        with self.assertRaises(rfapi.InvalidSplitError):
            rfapi.get_model_eval_performance_by_class("k", "ws", "x", split="all")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_400_invalid_confidence(self, mock_get):
        mock_get.return_value = _resp(400, {"error": "invalid_confidence", "message": "out of range"})

        with self.assertRaises(rfapi.InvalidConfidenceError):
            rfapi.get_model_eval_confusion_matrix("k", "ws", "x", confidence=200)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_unknown_404_falls_back_to_not_found(self, mock_get):
        # 404 without a recognised code still maps by status code (forward-compat).
        mock_get.return_value = _resp(404, {"error": "some_new_code", "message": "?"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            rfapi.get_model_eval("k", "ws", "x")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_unknown_500_raises_generic_roboflow_error(self, mock_get):
        mock_get.return_value = _resp(500, {"error": "server_oops", "message": "boom"})

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")
        # Not one of the typed subclasses
        self.assertNotIsInstance(ctx.exception, rfapi.ModelEvalNotFoundError)
        self.assertNotIsInstance(ctx.exception, rfapi.ModelEvalNotDoneError)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_non_json_body_falls_back_to_text(self, mock_get):
        # Some misbehaving proxies return HTML 502s — make sure we don't crash.
        bad = MagicMock(status_code=502, text="<html>Bad Gateway</html>")
        bad.json.side_effect = ValueError("not JSON")
        mock_get.return_value = bad

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")
        self.assertIn("Bad Gateway", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
