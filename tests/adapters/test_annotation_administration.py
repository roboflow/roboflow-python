"""HTTP contract tests for annotation administration adapters."""

import unittest
from unittest.mock import MagicMock, patch

from roboflow.adapters import rfapi


def _response(payload=None, status_code=200):
    return MagicMock(status_code=status_code, text="error", json=lambda: payload or {"success": True})


class TestAnnotationBatchAdministration(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_read_contracts(self, mock_get):
        mock_get.return_value = _response()

        rfapi.list_annotation_batches("key", "ws", "proj", limit=25, after="cursor", show_empty=True)
        self.assertTrue(mock_get.call_args.args[0].endswith("/ws/proj/annotation-batches"))
        self.assertEqual(
            mock_get.call_args.kwargs["params"],
            {"api_key": "key", "limit": 25, "after": "cursor", "showEmpty": "true"},
        )

        rfapi.get_annotation_batch("key", "ws", "proj", "batch-1")
        self.assertTrue(mock_get.call_args.args[0].endswith("/annotation-batches/batch-1"))

        rfapi.list_annotation_batch_images("key", "ws", "proj", "batch-1", limit=10, after="next")
        self.assertTrue(mock_get.call_args.args[0].endswith("/annotation-batches/batch-1/images"))
        self.assertEqual(
            mock_get.call_args.kwargs["params"],
            {"api_key": "key", "limit": 10, "after": "next"},
        )

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_create_and_merge_contracts(self, mock_post):
        mock_post.return_value = _response()

        rfapi.create_annotation_batch(
            "key",
            "ws",
            "proj",
            source_batch_id="source",
            image_ids=["image-1", "image-2"],
            name="Round two",
        )
        self.assertTrue(mock_post.call_args.args[0].endswith("/ws/proj/annotation-batches"))
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {"sourceBatchId": "source", "imageIds": ["image-1", "image-2"], "name": "Round two"},
        )

        rfapi.merge_annotation_batches(
            "key", "ws", "proj", source_batch_ids=["source-1", "source-2"], target_batch_id="target"
        )
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-batches/merge"))
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {"sourceBatchIds": ["source-1", "source-2"], "targetBatchId": "target"},
        )

    @patch("roboflow.adapters.rfapi.requests.delete")
    def test_delete_contract(self, mock_delete):
        mock_delete.return_value = _response()
        rfapi.delete_annotation_batch("key", "ws", "proj", "batch-1", permanent=True)
        self.assertTrue(mock_delete.call_args.args[0].endswith("/annotation-batches/batch-1"))
        self.assertEqual(mock_delete.call_args.kwargs["params"], {"api_key": "key", "permanent": "true"})


class TestAnnotationJobAdministration(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.requests.get")
    def test_read_contracts(self, mock_get):
        mock_get.return_value = _response()

        rfapi.list_annotation_jobs_admin("key", "ws", "proj", limit=20, after="cursor", show_empty=True)
        self.assertTrue(mock_get.call_args.args[0].endswith("/ws/proj/annotation-jobs"))
        self.assertEqual(
            mock_get.call_args.kwargs["params"],
            {"api_key": "key", "limit": 20, "after": "cursor", "showEmpty": "true"},
        )

        rfapi.get_annotation_job_admin("key", "ws", "proj", "job-1")
        self.assertTrue(mock_get.call_args.args[0].endswith("/annotation-jobs/job-1"))

        rfapi.list_annotation_job_images("key", "ws", "proj", "job-1", limit=5, after="next")
        self.assertTrue(mock_get.call_args.args[0].endswith("/annotation-jobs/job-1/images"))
        self.assertEqual(
            mock_get.call_args.kwargs["params"],
            {"api_key": "key", "limit": 5, "after": "next"},
        )

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_create_reassign_and_add_contracts(self, mock_post):
        mock_post.return_value = _response()

        rfapi.create_annotation_job_admin(
            "key",
            "ws",
            "proj",
            batch_id="batch-1",
            labeler_email="labeler@example.com",
            reviewer_email="reviewer@example.com",
            name="Round one",
            num_images=12,
            instructions="Follow the guide",
        )
        self.assertTrue(mock_post.call_args.args[0].endswith("/ws/proj/annotation-jobs"))
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {
                "batchId": "batch-1",
                "labelerEmail": "labeler@example.com",
                "reviewerEmail": "reviewer@example.com",
                "name": "Round one",
                "numImages": 12,
                "instructions": "Follow the guide",
            },
        )

        rfapi.reassign_annotation_job_images(
            "key",
            "ws",
            "proj",
            image_ids=["image-1"],
            labeler_email="labeler@example.com",
            reviewer_email="reviewer@example.com",
            name="Reassigned",
        )
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/reassign-images"))
        self.assertEqual(mock_post.call_args.kwargs["json"]["imageIds"], ["image-1"])
        self.assertNotIn("instructions", mock_post.call_args.kwargs["json"])

        rfapi.add_images_to_annotation_job("key", "ws", "proj", "job-1", image_ids=["image-2"])
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/images"))
        self.assertEqual(mock_post.call_args.kwargs["json"], {"imageIds": ["image-2"]})

    @patch("roboflow.adapters.rfapi.requests.patch")
    def test_update_contract_and_validation(self, mock_patch):
        mock_patch.return_value = _response()
        rfapi.update_annotation_job("key", "ws", "proj", "job-1", reviewer_email="new@example.com")
        self.assertEqual(mock_patch.call_args.kwargs["json"], {"reviewerEmail": "new@example.com"})

        with self.assertRaisesRegex(ValueError, "exactly one"):
            rfapi.update_annotation_job("key", "ws", "proj", "job-1")
        with self.assertRaisesRegex(ValueError, "exactly one"):
            rfapi.update_annotation_job("key", "ws", "proj", "job-1", labeler_email="a@example.com", instructions="new")

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_review_transition_contracts(self, mock_post):
        mock_post.return_value = _response()

        rfapi.submit_annotation_job_for_review("key", "ws", "proj", "job-1")
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/submit-review"))
        self.assertEqual(mock_post.call_args.kwargs["json"], {})

        rfapi.return_annotation_job_for_edits("key", "ws", "proj", "job-1", new_labeler_email="new@example.com")
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/return-edits"))
        self.assertEqual(mock_post.call_args.kwargs["json"], {"newLabelerEmail": "new@example.com"})

        rfapi.review_annotation_job_image("key", "ws", "proj", "job-1", "image-1", status="approved")
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/images/image-1/status"))
        self.assertEqual(mock_post.call_args.kwargs["json"], {"status": "approved"})

        rfapi.review_annotation_job_images("key", "ws", "proj", "job-1", status="rejected", current_status="annotated")
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/images/status"))
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {"status": "rejected", "currentStatus": "annotated"},
        )

    @patch("roboflow.adapters.rfapi.requests.post")
    def test_accept_and_move_contracts(self, mock_post):
        mock_post.return_value = _response()

        rfapi.accept_annotation_job_images(
            "key",
            "ws",
            "proj",
            "job-1",
            split_method="split",
            statuses_to_include=["approved", "annotated"],
            train_count=8,
            valid_count=1,
            test_count=1,
            image_ids=["image-1"],
        )
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/accept"))
        self.assertEqual(
            mock_post.call_args.kwargs["json"],
            {
                "splitMethod": "split",
                "statusesToInclude": ["approved", "annotated"],
                "trainCount": 8,
                "validCount": 1,
                "testCount": 1,
                "imageIds": ["image-1"],
            },
        )

        rfapi.move_annotation_job_to_unassigned("key", "ws", "proj", "job-1")
        self.assertTrue(mock_post.call_args.args[0].endswith("/annotation-jobs/job-1/move-to-unassigned"))
        self.assertEqual(mock_post.call_args.kwargs["json"], {})

    @patch("roboflow.adapters.rfapi.requests.delete")
    def test_delete_annotations_contract(self, mock_delete):
        mock_delete.return_value = _response()
        rfapi.delete_annotation_job_annotations("key", "ws", "proj", "job-1")
        self.assertTrue(mock_delete.call_args.args[0].endswith("/annotation-jobs/job-1/annotations"))

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_http_error_preserves_status_code(self, mock_get):
        mock_get.return_value = _response(status_code=403)
        with self.assertRaises(rfapi.RoboflowError) as context:
            rfapi.list_annotation_jobs("key", "ws", "proj")
        self.assertEqual(context.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
