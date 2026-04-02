"""Tests for the video CLI handler."""

import unittest


class TestVideoRegistration(unittest.TestCase):
    """Verify video handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.video import register

        self.assertTrue(callable(register))

    def test_video_infer_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["video", "infer", "-p", "my-project", "-v", "1", "-f", "vid.mp4"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.project, "my-project")
        self.assertEqual(args.version_number, 1)
        self.assertEqual(args.video_file, "vid.mp4")
        self.assertEqual(args.fps, 5)

    def test_video_infer_custom_fps(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["video", "infer", "-p", "proj", "-v", "2", "-f", "vid.mp4", "--fps", "10"])
        self.assertEqual(args.fps, 10)

    def test_video_status_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["video", "status", "job-123"])
        self.assertIsNotNone(args.func)
        self.assertEqual(args.job_id, "job-123")


class TestVideoStatus(unittest.TestCase):
    """Test video status handler."""

    def test_status_no_api_key(self) -> None:
        import io
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--json", "video", "status", "job-123"])
        from unittest.mock import patch

        captured = io.StringIO()
        old_stderr = sys.stderr
        sys.stderr = captured
        try:
            with patch("roboflow.config.load_roboflow_api_key", return_value=None):
                with self.assertRaises(SystemExit) as ctx:
                    args.func(args)
                self.assertEqual(ctx.exception.code, 2)
        finally:
            sys.stderr = old_stderr
        import json

        err = json.loads(captured.getvalue())
        self.assertIn("error", err)

    def test_status_success(self) -> None:
        import io
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["video", "status", "job-abc"])
        from unittest.mock import patch

        mock_data = {"status": "completed", "progress": "100%"}
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"):
                with patch("roboflow.adapters.rfapi.get_video_job_status", return_value=mock_data):
                    args.func(args)
        finally:
            sys.stdout = old_stdout
        out = captured.getvalue()
        self.assertIn("job-abc", out)
        self.assertIn("completed", out)

    def test_status_json_output(self) -> None:
        import io
        import json
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["--json", "video", "status", "job-abc"])
        from unittest.mock import patch

        mock_data = {"status": "processing", "progress": "50%"}
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"):
                with patch("roboflow.adapters.rfapi.get_video_job_status", return_value=mock_data):
                    args.func(args)
        finally:
            sys.stdout = old_stdout
        result = json.loads(captured.getvalue())
        self.assertEqual(result["status"], "processing")


    def test_status_passes_job_id_to_api(self) -> None:
        import io
        import sys

        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["video", "status", "my-unique-job-777"])
        from unittest.mock import patch

        mock_data = {"status": "completed"}
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured
        try:
            with patch("roboflow.config.load_roboflow_api_key", return_value="fake-key"):
                with patch("roboflow.adapters.rfapi.get_video_job_status", return_value=mock_data) as mock_api:
                    args.func(args)
        finally:
            sys.stdout = old_stdout
        mock_api.assert_called_once_with("fake-key", "my-unique-job-777")


if __name__ == "__main__":
    unittest.main()
