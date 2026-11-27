"""Unit tests for the model-eval rfapi helpers (`/{ws}/model-evals/...`)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from roboflow.adapters import rfapi
from roboflow.config import API_URL
from roboflow.util.http import USER_AGENT


def _resp(status: int, body):
    """Build a mock requests.Response double for the given status + JSON body."""
    mock = MagicMock(status_code=status)
    mock.json.return_value = body
    mock.text = repr(body)
    return mock


class TestListModelEvals(unittest.TestCase):
    @patch("roboflow.util.http.requests.get")
    def test_success_no_filters(self, mock_get):
        mock_get.return_value = _resp(200, {"evals": [{"id": "e1", "status": "done"}]})

        result = rfapi.list_model_evals("k", "ws")

        self.assertEqual(result, {"evals": [{"id": "e1", "status": "done"}]})
        url = mock_get.call_args[0][0]
        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(url, f"{API_URL}/ws/model-evals")
        self.assertEqual(params, {"api_key": "k"})

    @patch("roboflow.util.http.requests.get")
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

    @patch("roboflow.util.http.requests.get")
    def test_omits_none_filters(self, mock_get):
        mock_get.return_value = _resp(200, {"evals": []})

        rfapi.list_model_evals("k", "ws", status="done", limit=None)

        params = mock_get.call_args.kwargs["params"]
        self.assertNotIn("limit", params)
        self.assertEqual(params["status"], "done")

    @patch("roboflow.util.http.requests.get")
    def test_404_raises_not_found(self, mock_get):
        mock_get.return_value = _resp(404, {"error": "model_eval_not_found", "message": "nope"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError) as ctx:
            rfapi.list_model_evals("k", "ws")
        self.assertIn("nope", str(ctx.exception))


class TestGetModelEval(unittest.TestCase):
    @patch("roboflow.util.http.requests.get")
    def test_success(self, mock_get):
        mock_get.return_value = _resp(200, {"id": "e1", "status": "done", "summary": {"mAP": 0.9}})

        result = rfapi.get_model_eval("k", "ws", "e1")

        self.assertEqual(result["summary"]["mAP"], 0.9)
        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1")


class TestPanelEndpoints(unittest.TestCase):
    """Each panel endpoint forwards path + params correctly."""

    @patch("roboflow.util.http.requests.get")
    def test_map_results_url(self, mock_get):
        mock_get.return_value = _resp(200, {"splits": {}})

        rfapi.get_model_eval_map_results("k", "ws", "e1")

        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1/map-results")

    @patch("roboflow.util.http.requests.get")
    def test_confidence_sweep_url(self, mock_get):
        mock_get.return_value = _resp(200, {"splits": {}})

        rfapi.get_model_eval_confidence_sweep("k", "ws", "e1")

        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1/confidence-sweep")

    @patch("roboflow.util.http.requests.get")
    def test_performance_by_class_passes_split(self, mock_get):
        mock_get.return_value = _resp(200, {"split": "valid", "classes": []})

        rfapi.get_model_eval_performance_by_class("k", "ws", "e1", split="valid")

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["split"], "valid")

    @patch("roboflow.util.http.requests.get")
    def test_confusion_matrix_passes_params(self, mock_get):
        mock_get.return_value = _resp(200, {"matrix": []})

        rfapi.get_model_eval_confusion_matrix("k", "ws", "e1", split="test", confidence=30)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["split"], "test")
        self.assertEqual(params["confidence"], 30)

    @patch("roboflow.util.http.requests.get")
    def test_image_predictions_pagination(self, mock_get):
        mock_get.return_value = _resp(200, {"images": []})

        rfapi.get_model_eval_image_predictions("k", "ws", "e1", split="test", confidence=20, limit=50, offset=100)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["limit"], 50)
        self.assertEqual(params["offset"], 100)

    @patch("roboflow.util.http.requests.get")
    def test_recommendations_url(self, mock_get):
        mock_get.return_value = _resp(200, {"recommendations": []})

        rfapi.get_model_eval_recommendations("k", "ws", "e1")

        url = mock_get.call_args[0][0]
        self.assertEqual(url, f"{API_URL}/ws/model-evals/e1/recommendations")

    @patch("roboflow.util.http.requests.get")
    def test_vector_analysis_passes_confidence(self, mock_get):
        mock_get.return_value = _resp(200, {"clusters": []})

        rfapi.get_model_eval_vector_analysis("k", "ws", "e1", confidence=25)

        params = mock_get.call_args.kwargs["params"]
        self.assertEqual(params["confidence"], 25)


class TestErrorMapping(unittest.TestCase):
    """Typed errors are routed to the right exception subclass."""

    @patch("roboflow.util.http.requests.get")
    def test_404_flat_envelope(self, mock_get):
        # Server returns the flat shape: {"error": "code", "message": "..."}
        mock_get.return_value = _resp(404, {"error": "model_eval_not_found", "message": "Eval 'x' not found"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")
        self.assertIn("Eval 'x' not found", str(ctx.exception))

    @patch("roboflow.util.http.requests.get")
    def test_404_status_code_fallback(self, mock_get):
        # No `error` field at all — fall back to the status code mapping.
        mock_get.return_value = _resp(404, {"message": "something went wrong"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            rfapi.get_model_eval("k", "ws", "x")

    @patch("roboflow.util.http.requests.get")
    def test_nested_error_object_keeps_the_raw_text(self, mock_get):
        body = {"error": {"type": "InvalidRequest", "message": "boom", "code": "invalid_split"}}
        mock_get.return_value = _resp(400, body)

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")

        self.assertEqual(str(ctx.exception), repr(body))
        self.assertNotIsInstance(ctx.exception, rfapi.InvalidSplitError)

    @patch("roboflow.util.http.requests.get")
    def test_409_not_done(self, mock_get):
        mock_get.return_value = _resp(409, {"error": "model_eval_not_done", "message": "Eval still running"})

        with self.assertRaises(rfapi.ModelEvalNotDoneError):
            rfapi.get_model_eval_map_results("k", "ws", "x")

    @patch("roboflow.util.http.requests.get")
    def test_400_invalid_split(self, mock_get):
        mock_get.return_value = _resp(400, {"error": "invalid_split", "message": "Invalid split"})

        with self.assertRaises(rfapi.InvalidSplitError):
            rfapi.get_model_eval_performance_by_class("k", "ws", "x", split="all")

    @patch("roboflow.util.http.requests.get")
    def test_400_invalid_confidence(self, mock_get):
        mock_get.return_value = _resp(400, {"error": "invalid_confidence", "message": "out of range"})

        with self.assertRaises(rfapi.InvalidConfidenceError):
            rfapi.get_model_eval_confusion_matrix("k", "ws", "x", confidence=200)

    @patch("roboflow.util.http.requests.get")
    def test_unknown_404_falls_back_to_not_found(self, mock_get):
        # 404 without a recognised code still maps by status code (forward-compat).
        mock_get.return_value = _resp(404, {"error": "some_new_code", "message": "?"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            rfapi.get_model_eval("k", "ws", "x")

    @patch("roboflow.util.http.requests.get")
    def test_unknown_500_raises_generic_roboflow_error(self, mock_get):
        mock_get.return_value = _resp(500, {"error": "server_oops", "message": "boom"})

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")
        # Not one of the typed subclasses
        self.assertNotIsInstance(ctx.exception, rfapi.ModelEvalNotFoundError)
        self.assertNotIsInstance(ctx.exception, rfapi.ModelEvalNotDoneError)

    @patch("roboflow.util.http.requests.get")
    def test_non_json_body_falls_back_to_text(self, mock_get):
        # Some misbehaving proxies return HTML 502s — make sure we don't crash.
        bad = MagicMock(status_code=502, text="<html>Bad Gateway</html>")
        bad.json.side_effect = ValueError("not JSON")
        mock_get.return_value = bad

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.get_model_eval("k", "ws", "x")
        self.assertIn("Bad Gateway", str(ctx.exception))


class TestCompareModelEvals(unittest.TestCase):
    @patch("roboflow.util.http.requests.get")
    def test_compare_returns_public_result_and_sends_frontier_metric(self, mock_get):
        comparison = {
            "project": "chess",
            "version": "131",
            "frontierMetric": "mAP5095",
            "availableMetrics": ["mAP", "mAP5095"],
            "models": [],
        }
        mock_get.return_value = _resp(200, comparison)

        result = rfapi.compare_model_evals("k", "ws", project="chess", version="131", frontier_metric="mAP5095")

        self.assertEqual(result, comparison)
        mock_get.assert_called_once_with(
            f"{API_URL}/ws/model-evals/compare",
            params={
                "api_key": "k",
                "project": "chess",
                "version": "131",
                "frontierMetric": "mAP5095",
            },
            headers={"User-Agent": USER_AGENT},
        )


CREATE_RESULT = {
    "evalId": "e9",
    "status": "running",
    "reused": False,
    "version": "5",
    "trainedOnVersion": "2",
    "leakage": {
        "status": "detected",
        "splits": {
            "test": {"total": 412, "leaked": 37, "fromTrain": 30, "fromValid": 9},
            "valid": {"total": 200, "leaked": 5, "fromTrain": 5},
        },
    },
    "classMismatch": {"targetOnly": ["washer"], "modelOnly": ["nut"]},
}
BLOCKS = [
    {"code": "preprocessing_mismatch", "message": "The preprocessing differs."},
    {"code": "different_project", "message": "The model does not belong to this project."},
]


class TestCreateModelEval(unittest.TestCase):
    @patch("roboflow.util.http.requests.post")
    def test_posts_body_and_returns_full_response(self, mock_post):
        mock_post.return_value = _resp(200, CREATE_RESULT)

        result = rfapi.create_model_eval("k", "ws", model="ws/my-model", project="chess", version=5)

        self.assertEqual(result, CREATE_RESULT)
        mock_post.assert_called_once_with(
            f"{API_URL}/ws/model-evals",
            params={"api_key": "k"},
            json={"modelId": "ws/my-model", "project": "chess", "version": "5"},
            headers={"User-Agent": USER_AGENT},
        )

    @patch("roboflow.util.http.requests.post")
    def test_omits_version_when_not_given(self, mock_post):
        mock_post.return_value = _resp(200, CREATE_RESULT)

        rfapi.create_model_eval("k", "ws", model="my-model", project="chess")

        self.assertEqual(mock_post.call_args.kwargs["json"], {"modelId": "my-model", "project": "chess"})

    @patch("roboflow.util.http.requests.post")
    def test_sends_rerun_only_when_true(self, mock_post):
        mock_post.return_value = _resp(200, CREATE_RESULT)

        rfapi.create_model_eval("k", "ws", model="m", project="chess", version=5, rerun=True)
        self.assertEqual(
            mock_post.call_args.kwargs["json"], {"modelId": "m", "project": "chess", "version": "5", "rerun": True}
        )

        rfapi.create_model_eval("k", "ws", model="m", project="chess", version=5, rerun=False)
        self.assertNotIn("rerun", mock_post.call_args.kwargs["json"])

    @patch("roboflow.util.http.requests.post")
    def test_keeps_null_tags_of_a_reused_eval(self, mock_post):
        body = {**CREATE_RESULT, "reused": True, "trainedOnVersion": None, "leakage": None, "classMismatch": None}
        mock_post.return_value = _resp(200, body)

        result = rfapi.create_model_eval("k", "ws", model="m", project="chess", version=5)

        self.assertTrue(result["reused"])
        self.assertIsNone(result["leakage"])

    @patch("roboflow.util.http.requests.post")
    def test_409_raises_blocked_error_with_blocks(self, mock_post):
        mock_post.return_value = _resp(
            409,
            {
                "error": BLOCKS[0]["message"],
                "code": BLOCKS[0]["code"],
                "details": {"blocks": BLOCKS},
                "retryable": False,
            },
        )

        with self.assertRaises(rfapi.ModelEvalBlockedError) as ctx:
            rfapi.create_model_eval("k", "ws", model="m", project="chess", version=5)

        self.assertEqual(ctx.exception.blocks, BLOCKS)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("The preprocessing differs.", str(ctx.exception))
        self.assertNotIsInstance(ctx.exception, rfapi.ModelEvalNotDoneError)

    @patch("roboflow.util.http.requests.post")
    def test_429_raises_concurrency_limit_error(self, mock_post):
        mock_post.return_value = _resp(
            429, {"error": {"code": "eval_concurrency_limit", "message": "10 evaluations are active."}}
        )

        with self.assertRaises(rfapi.ModelEvalConcurrencyLimitError) as ctx:
            rfapi.create_model_eval("k", "ws", model="m", project="chess", version=5)

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertIn("10 evaluations", str(ctx.exception))

    @patch("roboflow.util.http.requests.post")
    def test_429_with_another_code_is_a_generic_error(self, mock_post):
        mock_post.return_value = _resp(429, {"error": "rate_limited", "message": "Slow down."})

        with self.assertRaises(rfapi.RoboflowError) as ctx:
            rfapi.create_model_eval("k", "ws", model="m", project="chess")

        self.assertNotIsInstance(ctx.exception, rfapi.ModelEvalConcurrencyLimitError)

    @patch("roboflow.util.http.requests.post")
    def test_403_raises_access_error(self, mock_post):
        mock_post.return_value = _resp(403, {"error": "forbidden", "message": "Plan does not include evaluations."})

        with self.assertRaises(rfapi.ModelEvalAccessError):
            rfapi.create_model_eval("k", "ws", model="m", project="chess")

    @patch("roboflow.util.http.requests.post")
    def test_404_raises_not_found_error(self, mock_post):
        mock_post.return_value = _resp(404, {"error": "not_found", "message": "Model not found"})

        with self.assertRaises(rfapi.ModelEvalNotFoundError):
            rfapi.create_model_eval("k", "ws", model="m", project="chess")


class TestCheckModelEvalCompatibility(unittest.TestCase):
    @patch("roboflow.util.http.requests.post")
    def test_posts_to_check_compatibility_and_returns_full_response(self, mock_post):
        compatibility_check = {
            "allowed": True,
            "blocks": [],
            "trainedOnVersion": "2",
            "leakage": CREATE_RESULT["leakage"],
            "classMismatch": CREATE_RESULT["classMismatch"],
        }
        mock_post.return_value = _resp(200, compatibility_check)

        result = rfapi.check_model_eval_compatibility("k", "ws", model="ws/my-model", project="chess", version="5")

        self.assertEqual(result, compatibility_check)
        mock_post.assert_called_once_with(
            f"{API_URL}/ws/model-evals/check-compatibility",
            params={"api_key": "k"},
            json={"modelId": "ws/my-model", "project": "chess", "version": "5"},
            headers={"User-Agent": USER_AGENT},
        )

    @patch("roboflow.util.http.requests.post")
    def test_a_blocked_check_is_a_result_not_an_error(self, mock_post):
        blocked = {
            "allowed": False,
            "blocks": BLOCKS[:1],
            "trainedOnVersion": None,
            "leakage": None,
            "classMismatch": None,
        }
        mock_post.return_value = _resp(200, blocked)

        self.assertEqual(rfapi.check_model_eval_compatibility("k", "ws", model="m", project="chess"), blocked)

    @patch("roboflow.util.http.requests.post")
    def test_403_raises_access_error(self, mock_post):
        mock_post.return_value = _resp(403, {"error": "forbidden", "message": "No access."})

        with self.assertRaises(rfapi.ModelEvalAccessError):
            rfapi.check_model_eval_compatibility("k", "ws", model="m", project="chess")


if __name__ == "__main__":
    unittest.main()
