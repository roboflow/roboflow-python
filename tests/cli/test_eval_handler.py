"""Tests for the model-eval CLI handler (`roboflow eval ...`)."""

from __future__ import annotations

import json
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Registration / discoverability
# ---------------------------------------------------------------------------


class TestEvalRegistration(unittest.TestCase):
    """`roboflow eval ...` subcommands are registered with valid --help."""

    def test_eval_app_exists(self) -> None:
        from roboflow.cli.handlers.eval import eval_app

        self.assertIsNotNone(eval_app)

    def test_eval_root_help(self) -> None:
        result = runner.invoke(app, ["eval", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_each_subcommand_help(self) -> None:
        for cmd in [
            "list",
            "get",
            "map-results",
            "confidence-sweep",
            "performance-by-class",
            "confusion-matrix",
            "vector-analysis",
            "image-predictions",
            "recommendations",
        ]:
            with self.subTest(cmd=cmd):
                result = runner.invoke(app, ["eval", cmd, "--help"])
                self.assertEqual(result.exit_code, 0, f"{cmd} --help failed: {result.output}")


# ---------------------------------------------------------------------------
# Helpers — every test patches the workspace + key resolver so no IO happens.
# ---------------------------------------------------------------------------


def _args(**overrides):
    """Build a Namespace matching what ctx_to_args produces, with sane defaults."""
    base = {"json": False, "workspace": "lee-sandbox", "api_key": None, "quiet": False}
    base.update(overrides)
    return Namespace(**base)


# ---------------------------------------------------------------------------
# `eval list`
# ---------------------------------------------------------------------------


class TestEvalListHandler(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_model_evals")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_list_text_calls_adapter_with_filters(self, _key, _ws, mock_list):
        mock_list.return_value = {
            "evals": [
                {
                    "id": "e1",
                    "status": "done",
                    "project": "my-project-slug",
                    "versionId": "3",
                    "modelId": None,
                    "createdAt": "2025-01-01",
                }
            ]
        }
        args = _args(
            workspace=None,
            project="my-project-slug",
            version="3",
            model=None,
            status="done",
            limit=5,
        )

        from roboflow.cli.handlers.eval import _list_evals

        with patch("builtins.print") as mock_print:
            _list_evals(args)

        mock_list.assert_called_once_with(
            "key",
            "lee-sandbox",
            project="my-project-slug",
            version="3",
            model=None,
            status="done",
            limit=5,
        )
        printed = mock_print.call_args[0][0]
        self.assertIn("e1", printed)
        self.assertIn("done", printed)

    @patch("roboflow.adapters.rfapi.list_model_evals")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_list_json_emits_evals_array(self, _key, _ws, mock_list):
        mock_list.return_value = {"evals": [{"id": "e1", "status": "done"}]}
        args = _args(workspace=None, json=True, project=None, version=None, model=None, status=None, limit=None)

        from roboflow.cli.handlers.eval import _list_evals

        with patch("builtins.print") as mock_print:
            _list_evals(args)

        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data, [{"id": "e1", "status": "done"}])

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_list_no_workspace_exits_2(self, _ws):
        args = _args(workspace=None, project=None, version=None, model=None, status=None, limit=None)

        from roboflow.cli.handlers.eval import _list_evals

        with self.assertRaises(SystemExit) as ctx:
            _list_evals(args)
        self.assertEqual(ctx.exception.code, 2)


# ---------------------------------------------------------------------------
# `eval get`
# ---------------------------------------------------------------------------


class TestEvalGetHandler(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_model_eval")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_get_text(self, _key, _ws, mock_get):
        mock_get.return_value = {
            "id": "e1",
            "status": "done",
            "project": "my-project-slug",
            "versionId": "3",
            "modelId": "m1",
            "createdAt": "2025-01-01",
            "summary": {"mAP": 0.91, "precision": 0.85, "recall": 0.8},
        }
        args = _args(workspace=None, eval_id="e1")

        from roboflow.cli.handlers.eval import _get_eval

        with patch("builtins.print") as mock_print:
            _get_eval(args)

        printed = mock_print.call_args[0][0]
        self.assertIn("Eval: e1", printed)
        self.assertIn("Status:  done", printed)
        self.assertIn("mAP=0.91", printed)
        mock_get.assert_called_once_with("key", "lee-sandbox", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_get_404_exits_3(self, _key, _ws, mock_get):
        from roboflow.adapters import rfapi

        mock_get.side_effect = rfapi.ModelEvalNotFoundError("not found")
        args = _args(workspace=None, eval_id="bad")

        from roboflow.cli.handlers.eval import _get_eval

        with self.assertRaises(SystemExit) as ctx:
            _get_eval(args)
        self.assertEqual(ctx.exception.code, 3)


# ---------------------------------------------------------------------------
# Per-panel handlers — each forwards args to the right adapter function.
# ---------------------------------------------------------------------------


class TestPanelHandlers(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_model_eval_map_results")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_map_results_calls_adapter(self, _key, _ws, mock_fn):
        mock_fn.return_value = {"splits": {"test": {"map50": 0.9}}}
        args = _args(workspace=None, eval_id="e1")

        from roboflow.cli.handlers.eval import _map_results

        with patch("builtins.print"):
            _map_results(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval_confidence_sweep")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_confidence_sweep_calls_adapter(self, _key, _ws, mock_fn):
        mock_fn.return_value = {"splits": {}}
        args = _args(workspace=None, eval_id="e1")

        from roboflow.cli.handlers.eval import _confidence_sweep

        with patch("builtins.print"):
            _confidence_sweep(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval_performance_by_class")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_performance_by_class_passes_split(self, _key, _ws, mock_fn):
        mock_fn.return_value = {"split": "valid", "classes": [{"className": "car", "map50": 0.9}]}
        args = _args(workspace=None, eval_id="e1", split="valid")

        from roboflow.cli.handlers.eval import _performance_by_class

        with patch("builtins.print"):
            _performance_by_class(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1", split="valid")

    @patch("roboflow.adapters.rfapi.get_model_eval_performance_by_class")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_performance_by_class_invalid_split_exits_5(self, _key, _ws, mock_fn):
        from roboflow.adapters import rfapi

        mock_fn.side_effect = rfapi.InvalidSplitError("no")
        args = _args(workspace=None, eval_id="e1", split="all")

        from roboflow.cli.handlers.eval import _performance_by_class

        with self.assertRaises(SystemExit) as ctx:
            _performance_by_class(args)
        self.assertEqual(ctx.exception.code, 5)

    @patch("roboflow.adapters.rfapi.get_model_eval_confusion_matrix")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_confusion_matrix_passes_args(self, _key, _ws, mock_fn):
        mock_fn.return_value = {"matrix": []}
        args = _args(workspace=None, eval_id="e1", split="test", confidence=30)

        from roboflow.cli.handlers.eval import _confusion_matrix

        with patch("builtins.print"):
            _confusion_matrix(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1", split="test", confidence=30)

    @patch("roboflow.adapters.rfapi.get_model_eval_confusion_matrix")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_confusion_matrix_invalid_confidence_exits_5(self, _key, _ws, mock_fn):
        from roboflow.adapters import rfapi

        mock_fn.side_effect = rfapi.InvalidConfidenceError("bad")
        args = _args(workspace=None, eval_id="e1", split=None, confidence=999)

        from roboflow.cli.handlers.eval import _confusion_matrix

        with self.assertRaises(SystemExit) as ctx:
            _confusion_matrix(args)
        self.assertEqual(ctx.exception.code, 5)

    @patch("roboflow.adapters.rfapi.get_model_eval_vector_analysis")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_vector_analysis_calls_adapter(self, _key, _ws, mock_fn):
        mock_fn.return_value = {"clusters": []}
        args = _args(workspace=None, eval_id="e1", confidence=20)

        from roboflow.cli.handlers.eval import _vector_analysis

        with patch("builtins.print"):
            _vector_analysis(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1", confidence=20)

    @patch("roboflow.adapters.rfapi.get_model_eval_image_predictions")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_image_predictions_pagination(self, _key, _ws, mock_fn):
        mock_fn.return_value = {
            "split": "test",
            "confidenceThreshold": 30,
            "totalImages": 100,
            "offset": 50,
            "limit": 10,
            "images": [{"imageId": "i1", "imageName": "a.jpg", "split": "test", "stats": {}}],
        }
        args = _args(workspace=None, eval_id="e1", split="test", confidence=30, limit=10, offset=50)

        from roboflow.cli.handlers.eval import _image_predictions

        with patch("builtins.print") as mock_print:
            _image_predictions(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1", split="test", confidence=30, limit=10, offset=50)
        printed = mock_print.call_args[0][0]
        self.assertIn("a.jpg", printed)

    @patch("roboflow.adapters.rfapi.get_model_eval_image_predictions")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_image_predictions_table_renders_TP_FP_FN_from_camelCase_stats(self, _key, _ws, mock_fn):
        # Regression: the public API nests counts under `stats` with camelCase
        # keys (`truePositives`/`falsePositives`/`falseNegatives`). Earlier code
        # read `stats.tp`/`stats.fp`/`stats.fn` and silently rendered blanks.
        mock_fn.return_value = {
            "split": "test",
            "confidenceThreshold": 0.2,
            "totalImages": 1,
            "offset": 0,
            "limit": 1,
            "images": [
                {
                    "imageId": "i1",
                    "imageName": "abc.jpg",
                    "split": "test",
                    "augmentations": 2,
                    "stats": {
                        "truePositives": 7,
                        "falsePositives": 2,
                        "falseNegatives": 1,
                        "precision": 0.78,
                        "recall": 0.875,
                        "f1": 0.824,
                    },
                    # The cluster column previously stringified the whole dict;
                    # we only want the cluster id rendered.
                    "cluster": {"id": 4, "embedding2D": [1.5, -3.2]},
                }
            ],
        }
        args = _args(workspace=None, eval_id="e1", split="test", confidence=None, limit=None, offset=None)

        from roboflow.cli.handlers.eval import _image_predictions

        with patch("builtins.print") as mock_print:
            _image_predictions(args)
        printed = mock_print.call_args[0][0]
        # TP/FP/FN counts must appear in the rendered table.
        self.assertIn("7", printed)  # truePositives
        self.assertIn("2", printed)  # falsePositives + augmentations both = 2; either way it should appear
        self.assertIn("1", printed)  # falseNegatives
        # Cluster rendered as the bare id, not the embedding-bearing dict.
        self.assertIn(" 4 ", printed)
        self.assertNotIn("embedding2D", printed)

    @patch("roboflow.adapters.rfapi.get_model_eval_recommendations")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_recommendations_calls_adapter(self, _key, _ws, mock_fn):
        mock_fn.return_value = {"generated": False}
        args = _args(workspace=None, eval_id="e1")

        from roboflow.cli.handlers.eval import _recommendations

        with patch("builtins.print"):
            _recommendations(args)
        mock_fn.assert_called_once_with("key", "lee-sandbox", "e1")

    @patch("roboflow.adapters.rfapi.get_model_eval_map_results")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="lee-sandbox")
    @patch("roboflow.config.load_roboflow_api_key", return_value="key")
    def test_panel_409_not_done_exits_4(self, _key, _ws, mock_fn):
        from roboflow.adapters import rfapi

        mock_fn.side_effect = rfapi.ModelEvalNotDoneError("running")
        args = _args(workspace=None, eval_id="e1")

        from roboflow.cli.handlers.eval import _map_results

        with self.assertRaises(SystemExit) as ctx:
            _map_results(args)
        self.assertEqual(ctx.exception.code, 4)


# ---------------------------------------------------------------------------
# Exit-code mapping helper
# ---------------------------------------------------------------------------


class TestExitCodeMapping(unittest.TestCase):
    """The handler distinguishes 404/409/400 to give shell scripts useful exit codes."""

    def test_exit_codes(self) -> None:
        from roboflow.adapters import rfapi
        from roboflow.cli.handlers.eval import _eval_error_exit_code

        cases = {
            rfapi.ModelEvalAccessError("x"): 2,
            rfapi.ModelEvalNotFoundError("x"): 3,
            rfapi.ModelEvalNotDoneError("x"): 4,
            rfapi.InvalidSplitError("x"): 5,
            rfapi.InvalidConfidenceError("x"): 5,
            rfapi.RoboflowError("x"): 1,
            ValueError("x"): 1,
        }
        for exc, expected in cases.items():
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(_eval_error_exit_code(exc), expected)


class TestEvalCompareCommand(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_not_found_is_a_structured_error(self, mock_get):
        mock_get.return_value = MagicMock(status_code=404, text="Project not found")
        mock_get.return_value.json.return_value = {"error": "Project not found"}
        result = runner.invoke(
            app,
            ["--api-key", "k", "--workspace", "ws", "--json", "eval", "compare", "-p", "chess", "-v", "131"],
        )

        self.assertEqual(result.exit_code, 3)
        self.assertEqual(json.loads(result.stderr)["error"]["message"], "Project not found")
        self.assertEqual(
            json.loads(result.stderr)["error"]["hint"],
            "Check the project, version, frontier metric, and workspace access.",
        )
        self.assertEqual(result.stdout, "")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_null_frontier_metric_shows_exclusion_without_accuracy(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "frontierMetric": None,
            "models": [
                {
                    "modelId": "ws/chess-model",
                    "metrics": {"mAP": 0.9},
                    "medianLatencyMs": 10,
                    "onFrontier": False,
                    "exclusionReason": "unsupported_project_type",
                }
            ],
        }
        result = runner.invoke(
            app,
            ["--api-key", "k", "--workspace", "ws", "eval", "compare", "-p", "chess", "-v", "131"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        for text in ["ACCURACY", "ws/chess-model", "10.00", "unsupported_project_type"]:
            self.assertIn(text, result.stdout)
        self.assertNotIn("90.0%", result.stdout)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_list_access_error_includes_auth_and_entitlement_hint(self, mock_get):
        mock_get.return_value = MagicMock(status_code=403, text="Access denied")
        mock_get.return_value.json.return_value = {"error": "Access denied"}
        result = runner.invoke(app, ["--api-key", "k", "--workspace", "ws", "--json", "eval", "list"])

        self.assertEqual(result.exit_code, 2)
        hint = json.loads(result.stderr)["error"]["hint"]
        for text in ["API key", "model-eval:read", "Model Evaluation access"]:
            self.assertIn(text, hint)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_json_preserves_the_full_comparison(self, mock_get):
        comparison = json.loads((Path(__file__).parents[1] / "fixtures/model_eval_comparison.json").read_text())
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = comparison

        result = runner.invoke(
            app,
            [
                "--api-key",
                "k",
                "--workspace",
                "ws",
                "--json",
                "eval",
                "compare",
                "--project",
                "chess",
                "--version",
                "131",
                "--frontier-metric",
                "mAP5095",
            ],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(json.loads(result.stdout), comparison)
        self.assertEqual(mock_get.call_count, 1)
        self.assertEqual(mock_get.call_args.kwargs["params"]["frontierMetric"], "mAP5095")

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_text_shows_server_frontier_and_exclusion_with_zero_values(self, mock_get):
        mock_get.return_value = MagicMock(status_code=200)
        mock_get.return_value.json.return_value = {
            "project": "chess",
            "version": "131",
            "frontierMetric": "mAP",
            "availableMetrics": ["mAP"],
            "models": [
                {
                    "modelId": "ws/chess-fast",
                    "evaluationId": "eval-fast",
                    "metrics": {"mAP": 0},
                    "medianLatencyMs": 0,
                    "onFrontier": True,
                    "exclusionReason": None,
                },
                {
                    "modelId": "ws/chess-old",
                    "evaluationId": "eval-old",
                    "metrics": {"mAP": 0.9},
                    "medianLatencyMs": None,
                    "onFrontier": False,
                    "exclusionReason": "latency_unavailable",
                },
            ],
        }
        result = runner.invoke(
            app,
            ["--api-key", "k", "--workspace", "ws", "eval", "compare", "--project", "chess", "--version", "131"],
        )

        self.assertEqual(result.exit_code, 0, result.output)
        for text in [
            "MODEL",
            "mAP",
            "MEDIAN LATENCY (ms)",
            "FRONTIER",
            "EXCLUSION",
            "ws/chess-fast",
            "0.0%",
            "0.00",
            "Yes",
            "latency_unavailable",
        ]:
            self.assertIn(text, result.stdout)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_permission_failure_is_a_structured_auth_error(self, mock_get):
        mock_get.return_value = MagicMock(status_code=403, text="Comparison access denied")
        mock_get.return_value.json.return_value = {"error": "forbidden", "message": "Comparison access denied"}
        result = runner.invoke(
            app,
            [
                "--api-key",
                "k",
                "--workspace",
                "ws",
                "--json",
                "eval",
                "compare",
                "--project",
                "chess",
                "--version",
                "131",
            ],
        )

        self.assertEqual(result.exit_code, 2)
        self.assertEqual(json.loads(result.stderr)["error"]["message"], "Comparison access denied")
        hint = json.loads(result.stderr)["error"]["hint"]
        for text in ["API key", "model-eval:read", "Model Evaluation access"]:
            self.assertIn(text, hint)
        self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
