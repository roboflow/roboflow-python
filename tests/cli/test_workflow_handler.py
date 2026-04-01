"""Tests for the workflow CLI handler."""

import unittest


class TestWorkflowRegistration(unittest.TestCase):
    """Verify workflow handler registers expected subcommands."""

    def test_register_callable(self) -> None:
        from roboflow.cli.handlers.workflow import register

        self.assertTrue(callable(register))

    def test_workflow_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "list"])
        self.assertIsNotNone(args.func)

    def test_workflow_get_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "get", "my-workflow"])
        self.assertEqual(args.workflow_url, "my-workflow")
        self.assertIsNotNone(args.func)

    def test_workflow_create_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "create", "--name", "test-wf"])
        self.assertEqual(args.name, "test-wf")
        self.assertIsNotNone(args.func)

    def test_workflow_update_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "update", "my-wf"])
        self.assertIsNotNone(args.func)

    def test_workflow_version_list_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "version", "list", "my-wf"])
        self.assertIsNotNone(args.func)

    def test_workflow_fork_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "fork", "my-wf"])
        self.assertIsNotNone(args.func)

    def test_workflow_build_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "build", "detect objects in a video"])
        self.assertEqual(args.prompt, "detect objects in a video")
        self.assertIsNotNone(args.func)

    def test_workflow_run_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "run", "my-wf", "--input", "image.jpg"])
        self.assertEqual(args.input, "image.jpg")
        self.assertIsNotNone(args.func)

    def test_workflow_deploy_exists(self) -> None:
        from roboflow.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["workflow", "deploy", "my-wf"])
        self.assertIsNotNone(args.func)


if __name__ == "__main__":
    unittest.main()
