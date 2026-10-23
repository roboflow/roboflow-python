"""Public Project wrapper coverage for annotation administration."""

from unittest.mock import patch

from tests import RoboflowTest


class TestProjectAnnotationAdministration(RoboflowTest):
    def test_batch_wrappers_delegate_to_rfapi(self):
        cases = [
            (
                "get_annotation_batches",
                (),
                {"limit": 10, "after": "cursor", "show_empty": True},
                "list_annotation_batches",
            ),
            ("get_annotation_batch", ("batch-1",), {}, "get_annotation_batch"),
            (
                "get_annotation_batch_images",
                ("batch-1",),
                {"limit": 5, "after": "next"},
                "list_annotation_batch_images",
            ),
            (
                "create_annotation_batch",
                ("source", ["image-1"]),
                {"name": "New batch"},
                "create_annotation_batch",
            ),
            ("merge_annotation_batches", (["source"], "target"), {}, "merge_annotation_batches"),
            ("delete_annotation_batch", ("batch-1",), {"permanent": True}, "delete_annotation_batch"),
        ]

        for method, args, kwargs, adapter in cases:
            with (
                self.subTest(method=method),
                patch(f"roboflow.adapters.rfapi.{adapter}", return_value={"ok": True}) as mock,
            ):
                self.assertEqual(getattr(self.project, method)(*args, **kwargs), {"ok": True})
                mock.assert_called_once()

    def test_move_images_to_batch_sends_one_selection_and_one_target(self):
        response = {"batchId": "b1", "batchName": "serial-42", "moved": 2, "skipped": 0}
        with patch("roboflow.adapters.rfapi.move_images_to_batch", return_value=response) as mock:
            self.assertEqual(
                self.project.move_images_to_batch(batch_name="serial-42", image_ids=["i1", "i2"]), response
            )
            self.assertEqual(
                mock.call_args.kwargs,
                {"batch_name": "serial-42", "batch_id": None, "image_ids": ["i1", "i2"], "tag": None},
            )
            self.project.move_images_to_batch(batch_id="b1", tag="serial-42")
            self.assertEqual(
                mock.call_args.kwargs,
                {"batch_name": None, "batch_id": "b1", "image_ids": None, "tag": "serial-42"},
            )

    def test_move_images_to_batch_rejects_ambiguous_arguments_before_calling_api(self):
        cases = [
            {"batch_name": "b"},
            {"batch_name": "b", "image_ids": ["i1"], "tag": "t"},
            {"batch_name": "b", "image_ids": []},
            {"batch_name": "b", "tag": " "},
            {"tag": "t"},
            {"batch_name": "b", "batch_id": "id", "tag": "t"},
            {"batch_name": "", "tag": "t"},
            {"batch_name": "b", "image_ids": [f"i{i}" for i in range(1001)]},
        ]
        with patch("roboflow.adapters.rfapi.move_images_to_batch") as mock:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.project.move_images_to_batch(**kwargs)
            mock.assert_not_called()

    def test_job_wrappers_delegate_to_rfapi(self):
        cases = [
            ("get_annotation_jobs", (), {}, "list_annotation_jobs"),
            (
                "get_annotation_jobs_admin",
                (),
                {"limit": 10, "after": "cursor", "show_empty": True},
                "list_annotation_jobs_admin",
            ),
            ("get_annotation_job", ("job-1",), {}, "get_annotation_job"),
            ("get_annotation_job_admin", ("job-1",), {}, "get_annotation_job_admin"),
            ("get_annotation_job_images", ("job-1",), {"limit": 5, "after": "next"}, "list_annotation_job_images"),
            (
                "create_annotation_job",
                (),
                {
                    "batch_id": "batch-1",
                    "labeler_email": "labeler@example.com",
                    "reviewer_email": "reviewer@example.com",
                    "instructions": "Guide",
                },
                "create_annotation_job_from_batch",
            ),
            (
                "create_annotation_job_admin",
                (),
                {
                    "batch_id": "batch-1",
                    "labeler_email": "labeler@example.com",
                    "reviewer_email": "reviewer@example.com",
                    "instructions": "Guide",
                },
                "create_annotation_job_admin",
            ),
            (
                "reassign_annotation_job_images",
                (["image-1"], "labeler@example.com"),
                {"reviewer_email": "reviewer@example.com"},
                "reassign_annotation_job_images",
            ),
            ("add_annotation_job_images", ("job-1", ["image-1"]), {}, "add_images_to_annotation_job"),
            ("update_annotation_job", ("job-1",), {"instructions": "New guide"}, "update_annotation_job"),
            ("submit_annotation_job_for_review", ("job-1",), {}, "submit_annotation_job_for_review"),
            (
                "return_annotation_job_for_edits",
                ("job-1",),
                {"new_labeler_email": "new@example.com"},
                "return_annotation_job_for_edits",
            ),
            ("review_annotation_job_image", ("job-1", "image-1", "approved"), {}, "review_annotation_job_image"),
            (
                "review_annotation_job_images",
                ("job-1", "approved", "annotated"),
                {},
                "review_annotation_job_images",
            ),
            (
                "accept_annotation_job_images",
                ("job-1", "split", ["approved"], 8, 1, 1),
                {"image_ids": ["image-1"]},
                "accept_annotation_job_images",
            ),
            ("move_annotation_job_to_unassigned", ("job-1",), {}, "move_annotation_job_to_unassigned"),
            ("delete_annotation_job_annotations", ("job-1",), {}, "delete_annotation_job_annotations"),
        ]

        for method, args, kwargs, adapter in cases:
            with (
                self.subTest(method=method),
                patch(f"roboflow.adapters.rfapi.{adapter}", return_value={"ok": True}) as mock,
            ):
                self.assertEqual(getattr(self.project, method)(*args, **kwargs), {"ok": True})
                mock.assert_called_once()

    def test_create_job_validates_required_assignments(self):
        with self.assertRaisesRegex(ValueError, "required"):
            self.project.create_annotation_job(batch_id="batch-1")
