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


if __name__ == "__main__":
    unittest.main()
