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
