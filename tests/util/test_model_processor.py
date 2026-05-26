import unittest
from types import SimpleNamespace

from roboflow.config import TASK_CLS, TASK_DET, TASK_OBB, TASK_POSE, TASK_SEG
from roboflow.util.model_processor import (
    _detect_rfdetr_task,
    _detect_yolo_task,
    _validate_pose_kpt_shape,
    task_of_model_type,
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


class ValidatePoseKptShapeTest(unittest.TestCase):
    def test_non_pose_is_noop(self):
        # Detection model with no yaml at all must not raise.
        _validate_pose_kpt_shape("yolov11", SimpleNamespace(yaml=None), "/tmp/best.pt")
        _validate_pose_kpt_shape("yolov11-seg", SimpleNamespace(), "/tmp/best.pt")

    def test_pose_with_kpt_shape_ok(self):
        inst = SimpleNamespace(yaml={"nc": 1, "kpt_shape": [17, 3]})
        _validate_pose_kpt_shape("yolov11-pose", inst, "/tmp/best.pt")

    def test_pose_missing_kpt_shape_raises(self):
        inst = SimpleNamespace(yaml={"nc": 1})
        with self.assertRaises(ValueError) as ctx:
            _validate_pose_kpt_shape("yolov11-pose", inst, "/tmp/best.pt")
        msg = str(ctx.exception)
        self.assertIn("kpt_shape", msg)
        self.assertIn("/tmp/best.pt", msg)

    def test_pose_empty_kpt_shape_raises(self):
        inst = SimpleNamespace(yaml={"kpt_shape": []})
        with self.assertRaises(ValueError):
            _validate_pose_kpt_shape("yolov11-pose", inst, "/tmp/best.pt")

    def test_pose_no_yaml_raises(self):
        with self.assertRaises(ValueError):
            _validate_pose_kpt_shape("yolo26-pose", SimpleNamespace(yaml=None), "/tmp/best.pt")


if __name__ == "__main__":
    unittest.main()
