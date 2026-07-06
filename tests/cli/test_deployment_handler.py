"""Tests for the deployment CLI handler."""

import argparse
import io
import json
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()


class TestDeploymentRegistration(unittest.TestCase):
    """Verify deployment handler registers expected subcommands."""

    def test_deployment_app_exists(self) -> None:
        from roboflow.cli.handlers.deployment import deployment_app

        self.assertIsNotNone(deployment_app)

    def test_deployment_subcommand_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "list", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_create_canonical(self) -> None:
        result = runner.invoke(app, ["deployment", "create", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_machine_type_canonical(self) -> None:
        result = runner.invoke(app, ["deployment", "machine-type", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_get_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "get", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_delete_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "delete", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_usage_canonical(self) -> None:
        result = runner.invoke(app, ["deployment", "usage", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_regions_exists(self) -> None:
        result = runner.invoke(app, ["deployment", "regions", "--help"])
        self.assertEqual(result.exit_code, 0)

    def test_deployment_create_accepts_region(self) -> None:
        result = runner.invoke(app, ["deployment", "create", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("--region", result.output)


class TestDeploymentRegionPassthrough(unittest.TestCase):
    """The region option must reach the /add request body (and be omitted otherwise)."""

    def test_add_deployment_sends_region(self) -> None:
        from roboflow.adapters import deploymentapi

        with patch.object(deploymentapi, "requests") as mock_requests:
            mock_requests.post.return_value.status_code = 200
            mock_requests.post.return_value.json.return_value = {}
            deploymentapi.add_deployment(
                "key", "me@roboflow.com", "dev-gpu", 3, True, "mydeploy", "latest", region="us-east-2"
            )
        body = mock_requests.post.call_args.kwargs["json"]
        self.assertEqual(body["region"], "us-east-2")

    def test_add_deployment_omits_region_when_unset(self) -> None:
        """No region flag -> no region key, so the API picks its default region."""
        from roboflow.adapters import deploymentapi

        with patch.object(deploymentapi, "requests") as mock_requests:
            mock_requests.post.return_value.status_code = 200
            mock_requests.post.return_value.json.return_value = {}
            deploymentapi.add_deployment("key", "me@roboflow.com", "dev-gpu", 3, True, "mydeploy", "latest")
        body = mock_requests.post.call_args.kwargs["json"]
        self.assertNotIn("region", body)

    def test_list_regions_calls_regions_endpoint(self) -> None:
        from roboflow.adapters import deploymentapi

        with patch.object(deploymentapi, "requests") as mock_requests:
            mock_requests.get.return_value.status_code = 200
            mock_requests.get.return_value.json.return_value = []
            status, _ = deploymentapi.list_regions("key")
        self.assertEqual(status, 200)
        url = mock_requests.get.call_args.args[0]
        self.assertIn("/regions?api_key=key", url)


class TestDeploymentErrorWrapping(unittest.TestCase):
    """Verify deployment errors produce structured output."""

    def test_wrapped_error_uses_structured_output(self) -> None:
        """Deployment errors should go through output_error, not bare print."""
        from roboflow.cli.handlers.deployment import _wrap

        def _fake_handler(args: object) -> None:
            print("401: Unauthorized (invalid api_key)")
            raise SystemExit(401)

        ns = argparse.Namespace(json=True, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap(_fake_handler)
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            with self.assertRaises(SystemExit) as ctx:
                wrapped(ns)
            self.assertLessEqual(ctx.exception.code, 3)
        err_output = stderr.getvalue().strip()
        parsed = json.loads(err_output)
        self.assertIn("error", parsed)

    def test_wrapped_success_prints_output(self) -> None:
        """On success, wrapped func should replay captured stdout."""
        from roboflow.cli.handlers.deployment import _wrap

        def _fake_handler(args: object) -> None:
            print('{"machines": []}')

        ns = argparse.Namespace(json=False, api_key=None, workspace=None, quiet=False)
        wrapped = _wrap(_fake_handler)
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            wrapped(ns)
        self.assertIn('{"machines": []}', captured.getvalue())


if __name__ == "__main__":
    unittest.main()
