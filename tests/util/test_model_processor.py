import json
import os
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from roboflow.config import TASK_CLS, TASK_DET, TASK_OBB, TASK_POSE, TASK_SEG, TASK_SEM
from roboflow.util import model_processor
from roboflow.util.model_processor import (
    MissingFileError,
    ModelPackagingError,
    SizeMismatchError,
    TaskMismatchError,
    UnsupportedModelError,
    _detect_rfdetr_task,
    _detect_yolo_task,
    _infer_yolo_size,
    _legacy_yolo_args,
    _resolve_rfdetr_variant,
    _resolve_yolo_size,
    _rfdetr_checkpoint_pe_size,
    _write_rfdetr_class_names,
    get_classnames_txt_for_rfdetr,
    package_custom_weights,
    package_custom_weights_interactive,
    process,
    task_of_model_type,
    validate_model_type_for_project,
)


class _FakeModel:
    """Stand-in for an Ultralytics model_instance; only __class__.__name__ matters."""


def _make_fake(name: str):
    return type(name, (_FakeModel,), {})()


class TaskOfModelTypeTest(unittest.TestCase):
    def test_detect_defaults(self):
        self.assertEqual(task_of_model_type("yolov11"), TASK_DET)
        self.assertEqual(task_of_model_type("rfdetr-base"), TASK_DET)
        self.assertEqual(task_of_model_type("rfdetr-medium"), TASK_DET)
        self.assertEqual(task_of_model_type("yolov8"), TASK_DET)

    def test_segment(self):
        self.assertEqual(task_of_model_type("yolov11-seg"), TASK_SEG)
        self.assertEqual(task_of_model_type("rfdetr-seg-medium"), TASK_SEG)
        self.assertEqual(task_of_model_type("yolov7-seg"), TASK_SEG)

    def test_pose(self):
        self.assertEqual(task_of_model_type("yolov11-pose"), TASK_POSE)

    def test_classify(self):
        self.assertEqual(task_of_model_type("yolov11-cls"), TASK_CLS)

    def test_semantic(self):
        self.assertEqual(task_of_model_type("yolo26-sem"), TASK_SEM)

    def test_obb(self):
        self.assertEqual(task_of_model_type("yolov11-obb"), TASK_OBB)


class DetectYoloTaskTest(unittest.TestCase):
    def test_ultralytics_class_names(self):
        cases = {
            "SegmentationModel": TASK_SEG,
            "PoseModel": TASK_POSE,
            "ClassificationModel": TASK_CLS,
            "OBBModel": TASK_OBB,
            "DetectionModel": TASK_DET,
        }
        for cls_name, expected in cases.items():
            self.assertEqual(_detect_yolo_task(_make_fake(cls_name)), expected, cls_name)

    def test_semantic_segmentation_model(self):
        self.assertEqual(_detect_yolo_task(_make_fake("SemanticSegmentationModel")), TASK_SEM)

    def test_unrecognized_returns_none(self):
        self.assertIsNone(_detect_yolo_task(_make_fake("SomeOtherModel")))
        self.assertIsNone(_detect_yolo_task(None))


class DetectRfdetrTaskTest(unittest.TestCase):
    def test_segmentation_model_names(self):
        for name in ("RFDETRSegNano", "RFDETRSegSmall", "RFDETRSegMedium", "RFDETRSegLarge"):
            self.assertEqual(_detect_rfdetr_task({"model_name": name}), TASK_SEG, name)

    def test_detection_model_names(self):
        for name in ("RFDETRNano", "RFDETRSmall", "RFDETRMedium", "RFDETRLarge", "RFDETRXLarge"):
            self.assertEqual(_detect_rfdetr_task({"model_name": name}), TASK_DET, name)

    def test_segmentation_head_fallback(self):
        # Roboflow-hosted rf-detr .pt downloads lack `model_name` but always carry
        # `args.segmentation_head`. Cover both namespace and dict shapes.
        self.assertEqual(_detect_rfdetr_task({"args": SimpleNamespace(segmentation_head=True)}), TASK_SEG)
        self.assertEqual(_detect_rfdetr_task({"args": SimpleNamespace(segmentation_head=False)}), TASK_DET)
        self.assertEqual(_detect_rfdetr_task({"args": {"segmentation_head": True}}), TASK_SEG)
        self.assertEqual(_detect_rfdetr_task({"args": {"segmentation_head": False}}), TASK_DET)

    def test_model_name_preferred_over_args(self):
        # When both are present, model_name wins (matches rf-detr's loader).
        ckpt = {"model_name": "RFDETRNano", "args": SimpleNamespace(segmentation_head=True)}
        self.assertEqual(_detect_rfdetr_task(ckpt), TASK_DET)

    def test_unrecognized_returns_none(self):
        self.assertIsNone(_detect_rfdetr_task({}))
        self.assertIsNone(_detect_rfdetr_task({"model_name": None}))
        self.assertIsNone(_detect_rfdetr_task({"args": SimpleNamespace(other=1)}))


class GetClassnamesTxtForRfdetrTest(unittest.TestCase):
    def _classnames(self, args):
        with tempfile.TemporaryDirectory() as model_path:
            get_classnames_txt_for_rfdetr(model_path, "weights.pt", checkpoint={"args": args})
            with open(os.path.join(model_path, "class_names.txt")) as f:
                return f.read().splitlines()

    def test_dict_args(self):
        self.assertEqual(self._classnames({"class_names": ["cat", "dog"]}), ["background_class83422", "cat", "dog"])

    def test_namespace_args(self):
        self.assertEqual(
            self._classnames(SimpleNamespace(class_names=["cat", "dog"])),
            ["background_class83422", "cat", "dog"],
        )


class ValidateModelTypeForProjectTest(unittest.TestCase):
    def test_rejects_detection_for_classification(self):
        with self.assertRaises(TaskMismatchError) as ctx:
            validate_model_type_for_project("yolov8", "classification", "widgets")
        self.assertIn("classification", str(ctx.exception))
        self.assertIn("task 'cls'", str(ctx.exception))

    def test_task_mismatch_is_a_value_error(self):
        # Callers that caught the historical ValueError keep working.
        with self.assertRaises(ValueError):
            validate_model_type_for_project("yolov8", "classification", "widgets")

    def test_unknown_project_type_is_ignored(self):
        validate_model_type_for_project("yolov8", "some-new-type", "widgets")


class LegacyYoloArgsTest(unittest.TestCase):
    def test_reports_missing_batch_size(self):
        with self.assertRaises(ModelPackagingError) as ctx:
            _legacy_yolo_args({"imgsz": 640}, Path("opt.yaml"))
        self.assertIn("batch_size", str(ctx.exception))

    def test_reports_missing_image_size(self):
        with self.assertRaises(ModelPackagingError) as ctx:
            _legacy_yolo_args({"batch_size": 8}, Path("opt.yaml"))
        self.assertIn("imgsz", str(ctx.exception))

    def test_accepts_either_image_size_key(self):
        self.assertEqual(_legacy_yolo_args({"imgsz": 640, "batch_size": 8}, Path("x")), {"imgsz": 640, "batch": 8})
        self.assertEqual(
            _legacy_yolo_args({"img_size": 416, "batch_size": 4}, Path("x")), {"imgsz": 416, "batch": 4}
        )


class _FakeYoloModel:
    def __init__(self, yaml):
        self.yaml = yaml


class InferYoloSizeTest(unittest.TestCase):
    def test_from_depth_width_multiples(self):
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_infer_yolo_size(model), "n")

    def test_explicit_scale_letter_wins(self):
        model = _FakeYoloModel({"scale": "m", "depth_multiple": 0.67, "width_multiple": 0.75})
        self.assertEqual(_infer_yolo_size(model), "m")

    def test_unknown_returns_none(self):
        self.assertIsNone(_infer_yolo_size(_FakeYoloModel({})))


class ResolveYoloSizeTest(unittest.TestCase):
    def test_fills_bare_family_from_architecture(self):
        warnings: list = []
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_resolve_yolo_size("yolov8", model, warnings), "yolov8n")
        self.assertTrue(warnings and "Inferred model size 'yolov8n'" in warnings[0])

    def test_preserves_task_suffix_when_filling(self):
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.50})
        self.assertEqual(_resolve_yolo_size("yolov8-seg", model, []), "yolov8s-seg")

    def test_raises_when_size_cannot_be_inferred(self):
        with self.assertRaises(SizeMismatchError) as ctx:
            _resolve_yolo_size("yolov8", _FakeYoloModel({}), [])
        self.assertIn("could not be inferred", str(ctx.exception))
        self.assertIn("yolov8n", str(ctx.exception))

    def test_raises_on_declared_size_conflict(self):
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        with self.assertRaises(SizeMismatchError) as ctx:
            _resolve_yolo_size("yolov8m", model, [])
        self.assertIn("yolov8n", str(ctx.exception))
        self.assertIn("allow_size_mismatch=True", str(ctx.exception))
        self.assertEqual(ctx.exception.requested, "yolov8m")
        self.assertEqual(ctx.exception.detected, "yolov8n")

    def test_allow_mismatch_keeps_declared_size(self):
        warnings: list = []
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_resolve_yolo_size("yolov8m", model, warnings, allow_mismatch=True), "yolov8m")
        self.assertTrue(warnings and "as requested" in warnings[0])

    def test_keeps_user_size_when_not_inferable(self):
        warnings: list = []
        self.assertEqual(_resolve_yolo_size("yolov8m", _FakeYoloModel({}), warnings), "yolov8m")
        self.assertEqual(warnings, [])

    def test_keeps_matching_sized_type_without_warning(self):
        warnings: list = []
        model = _FakeYoloModel({"depth_multiple": 0.33, "width_multiple": 0.25})
        self.assertEqual(_resolve_yolo_size("yolov8n", model, warnings), "yolov8n")
        self.assertEqual(warnings, [])


class ResolveRfdetrVariantTest(unittest.TestCase):
    def test_raises_on_size_conflict_naming_the_fit(self):
        checkpoint = {"args": {"resolution": 384, "patch_size": 12}}
        with self.assertRaises(SizeMismatchError) as ctx:
            _resolve_rfdetr_variant("rfdetr-seg-nano", checkpoint, [])
        self.assertIn("rfdetr-seg-small", str(ctx.exception))
        self.assertIn("32x32", str(ctx.exception))
        self.assertIn("allow_size_mismatch=True", str(ctx.exception))

    def test_allow_mismatch_keeps_requested_variant(self):
        warnings: list = []
        checkpoint = {"args": {"resolution": 384, "patch_size": 12}}
        resolved = _resolve_rfdetr_variant("rfdetr-seg-nano", checkpoint, warnings, allow_mismatch=True)
        self.assertEqual(resolved, "rfdetr-seg-nano")
        self.assertTrue(warnings and "as requested" in warnings[0])

    def test_keeps_matching_grid_without_warning(self):
        warnings: list = []
        checkpoint = {"args": {"positional_encoding_size": 32}}
        self.assertEqual(_resolve_rfdetr_variant("rfdetr-seg-small", checkpoint, warnings), "rfdetr-seg-small")
        self.assertEqual(warnings, [])

    def test_warns_but_allows_custom_resolution(self):
        warnings: list = []
        resolved = _resolve_rfdetr_variant("rfdetr-seg-nano", {"args": {"positional_encoding_size": 99}}, warnings)
        self.assertEqual(resolved, "rfdetr-seg-nano")
        self.assertTrue(warnings and "matches no known RF-DETR variant" in warnings[0])

    def test_pe_size_derived_from_position_embeddings_tensor(self):
        class FakeTensor:
            shape = (1, 1025, 384)

        checkpoint = {"model": {"backbone.0.encoder.encoder.embeddings.position_embeddings": FakeTensor()}}
        self.assertEqual(_rfdetr_checkpoint_pe_size(checkpoint), 32)


class WriteRfdetrClassNamesTest(unittest.TestCase):
    def test_fails_cleanly_without_checkpoint_args(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(MissingFileError) as ctx:
                _write_rfdetr_class_names(Path(tmp), Path(tmp), checkpoint={})
        self.assertIn("does not include args with class_names", str(ctx.exception))

    def test_does_not_mutate_existing_class_names_file(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as build:
            source = Path(tmp) / "class_names.txt"
            source.write_text("cat\ndog\n")
            output = _write_rfdetr_class_names(Path(tmp), Path(build), checkpoint={})
            self.assertEqual(source.read_text(), "cat\ndog\n")
            self.assertEqual(
                output.read_text().splitlines(),
                ["background_class83422", "cat", "dog"],
            )


def _fake_torch(load_result, calls=None):
    module = types.ModuleType("torch")

    def load(path, **kwargs):
        if calls is not None:
            calls.append((Path(path), kwargs))
        return load_result

    def save(obj, path):
        Path(path).write_bytes(b"fake-state-dict")

    module.load = load
    module.save = save
    return module


def _import_patch(modules):
    def _import(module_name, install_hint):
        return modules[module_name]

    return mock.patch.object(model_processor, "_import_required_module", side_effect=_import)


def _write_yolonas_inputs(model_dir: Path):
    weights = model_dir / "weights" / "best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"checkpoint")
    (model_dir / "opt.yaml").write_text("imgsz: 640\nbatch_size: 8\narchitecture: yolo_nas_s\n")


class PackageCustomWeightsTest(unittest.TestCase):
    """Contract tests for the public non-interactive helper."""

    def _package_yolonas(self, model_dir: Path, **kwargs):
        calls: list = []
        torch = _fake_torch({"processing_params": {"class_names": ["widget"]}}, calls)
        with _import_patch({"torch": torch}):
            bundle = package_custom_weights("yolonas", str(model_dir), **kwargs)
        return bundle, calls

    def test_never_prompts_or_exits(self):
        prompt_guard = mock.patch(
            "builtins.input", side_effect=AssertionError("package_custom_weights must not prompt")
        )
        exit_guard = mock.patch.object(
            sys, "exit", side_effect=AssertionError("package_custom_weights must not exit")
        )
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            with prompt_guard, exit_guard:
                bundle, calls = self._package_yolonas(model_dir)
            try:
                self.assertTrue(bundle.archive_path.exists())
                self.assertEqual(calls[0][1], {"weights_only": False, "map_location": "cpu"})
            finally:
                bundle.cleanup()

    def test_does_not_write_into_model_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            before = sorted(path for path in model_dir.rglob("*"))
            bundle, _ = self._package_yolonas(model_dir)
            try:
                self.assertEqual(sorted(path for path in model_dir.rglob("*")), before)
                self.assertNotEqual(bundle.build_dir, model_dir)
                self.assertTrue(bundle.owns_build_dir)
            finally:
                bundle.cleanup()
            self.assertFalse(bundle.build_dir.exists())

    def test_explicit_build_dir_is_used_and_not_cleaned_up(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as build:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            bundle, _ = self._package_yolonas(model_dir, build_dir=build)
            self.assertEqual(bundle.build_dir, Path(build))
            self.assertFalse(bundle.owns_build_dir)
            bundle.cleanup()
            self.assertTrue(bundle.archive_path.exists())

    def test_owned_build_dir_is_removed_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as fake_build:
            with mock.patch.object(model_processor.tempfile, "mkdtemp", return_value=fake_build):
                with self.assertRaises(UnsupportedModelError):
                    package_custom_weights("not-a-model", tmp)
            self.assertFalse(Path(fake_build).exists())

    def test_missing_model_path_raises_missing_file(self):
        with self.assertRaises(MissingFileError):
            package_custom_weights("yolonas", "/nonexistent/path/for/test")

    def test_rfdetr_falls_back_to_discovered_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "other.pt").write_bytes(b"checkpoint")
            torch = _fake_torch({"args": {"class_names": ["widget"]}})
            with _import_patch({"torch": torch}):
                bundle = package_custom_weights("rfdetr-base", str(model_dir))
            try:
                self.assertTrue(any("other.pt" in warning for warning in bundle.warnings))
                with zipfile.ZipFile(bundle.archive_path) as archive:
                    self.assertIn("weights.pt", archive.namelist())
                    self.assertIn("class_names.txt", archive.namelist())
            finally:
                bundle.cleanup()

    def test_rfdetr_without_any_checkpoint_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            torch = _fake_torch({})
            with _import_patch({"torch": torch}):
                with self.assertRaises(MissingFileError):
                    package_custom_weights("rfdetr-base", tmp)

    def test_yolov8_full_flow_builds_artifacts(self):
        checkpoint_names = {1: "dog", 0: "cat"}

        class DetectionModel:
            names = checkpoint_names
            nc = 2
            yaml = {"nc": 2, "depth_multiple": 0.33, "width_multiple": 0.25}
            args = {"model": "yolov8n.yaml", "imgsz": 640, "batch": 16, "lr0": 0.01}

            def state_dict(self):
                return {"weight": b"w"}

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.__version__ = "8.0.196"
        fake_torch = _fake_torch({"model": DetectionModel()})

        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            weights = model_dir / "weights" / "best.pt"
            weights.parent.mkdir()
            weights.write_bytes(b"checkpoint")

            with (
                _import_patch({"torch": fake_torch, "ultralytics": fake_ultralytics}),
                mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}),
            ):
                bundle = package_custom_weights("yolov8", str(model_dir))
            try:
                self.assertEqual(bundle.model_type, "yolov8n")
                self.assertTrue(any("Inferred model size 'yolov8n'" in warning for warning in bundle.warnings))
                with zipfile.ZipFile(bundle.archive_path) as archive:
                    artifacts = json.loads(archive.read("model_artifacts.json"))
                    self.assertIn("state_dict.pt", archive.namelist())
                self.assertEqual(artifacts["names"], ["cat", "dog"])
                self.assertEqual(artifacts["model_type"], "yolov8n")
                self.assertEqual(artifacts["ultralytics_version"], "8.0.196")
                self.assertEqual(artifacts["args"], {"model": "yolov8n.yaml", "imgsz": 640, "batch": 16})
            finally:
                bundle.cleanup()


class ProcessCompatTest(unittest.TestCase):
    """The legacy process() entry point keeps its historical contract."""

    def test_packages_into_model_path_and_returns_tuple(self):
        calls: list = []
        torch = _fake_torch({"processing_params": {"class_names": ["widget"]}}, calls)
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            _write_yolonas_inputs(model_dir)
            with _import_patch({"torch": torch}):
                zip_file_name, model_type = process("yolonas", str(model_dir), "weights/best.pt")

            self.assertEqual(zip_file_name, "roboflow_deploy.zip")
            self.assertEqual(model_type, "yolonas")
            # Historical side effects: artifacts and archive land in model_path.
            self.assertTrue((model_dir / "roboflow_deploy.zip").exists())
            self.assertTrue((model_dir / "model_artifacts.json").exists())
            self.assertTrue((model_dir / "state_dict.pt").exists())


class PackageCustomWeightsInteractiveTest(unittest.TestCase):
    def _bundle(self):
        return model_processor.ModelUploadBundle(
            archive_path=Path("roboflow_deploy.zip"),
            build_dir=Path("."),
            model_type="yolov8n",
            warnings=("some warning",),
        )

    def test_retries_with_size_override_on_confirmation(self):
        error = SizeMismatchError("size conflict", requested="yolov8m", detected="yolov8n")
        outcomes = [error, self._bundle()]

        def fake_package(*args, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            self.assertTrue(kwargs["allow_size_mismatch"])
            return outcome

        with (
            mock.patch.object(model_processor, "package_custom_weights", side_effect=fake_package),
            mock.patch("builtins.input", return_value="y"),
            mock.patch("builtins.print"),
        ):
            bundle = package_custom_weights_interactive("yolov8m", "/models")
        self.assertEqual(bundle.model_type, "yolov8n")

    def test_reraises_when_user_declines(self):
        error = SizeMismatchError("size conflict", requested="yolov8m")
        with (
            mock.patch.object(model_processor, "package_custom_weights", side_effect=error),
            mock.patch("builtins.input", return_value="n"),
            mock.patch("builtins.print"),
        ):
            with self.assertRaises(SizeMismatchError):
                package_custom_weights_interactive("yolov8m", "/models")


if __name__ == "__main__":
    unittest.main()
