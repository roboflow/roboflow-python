import unittest
from types import SimpleNamespace

from roboflow.config import TASK_CLS, TASK_DET, TASK_OBB, TASK_POSE, TASK_SEG
from roboflow.util.model_processor import (
    _detect_rfdetr_task,
    _detect_yolo_task,
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


if __name__ == "__main__":
    unittest.main()
