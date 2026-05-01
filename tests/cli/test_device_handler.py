"""Tests for the device CLI handler."""

from __future__ import annotations

import json
import unittest
from argparse import Namespace
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.adapters.devicesapi import (
    DeviceAuthError,
    DeviceNotFoundError,
    DeviceRateLimitedError,
)
from roboflow.cli import app

runner = CliRunner()

WS = "test-ws"
KEY = "fake-key"


def _args(**kwargs) -> Namespace:
    defaults = {"json": False, "workspace": WS, "api_key": KEY, "quiet": False}
    defaults.update(kwargs)
    return Namespace(**defaults)


class TestDeviceRegistration(unittest.TestCase):
    """Subcommands are registered and `--help` works for each."""

    def test_top_level_help(self) -> None:
        result = runner.invoke(app, ["device", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("Manage RFDM devices", result.output)

    def test_subcommands(self) -> None:
        for verb in (
            "list",
            "get",
            "create",
            "config",
            "config-history",
            "streams",
            "stream",
            "logs",
            "telemetry",
            "events",
        ):
            with self.subTest(verb=verb):
                result = runner.invoke(app, ["device", verb, "--help"])
                self.assertEqual(result.exit_code, 0, msg=result.output)


class TestDeviceListHandler(unittest.TestCase):
    @patch("roboflow.adapters.devicesapi.list_devices")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_list_text(self, _mk, _mw, mock_list):
        mock_list.return_value = {
            "data": [
                {
                    "id": "a",
                    "name": "Cam A",
                    "status": "online",
                    "type": "edge",
                    "last_heartbeat": "2026-04-30T00:00:00Z",
                }
            ]
        }
        from roboflow.cli.handlers.device import _list

        with patch("builtins.print") as mock_print:
            _list(_args())
        mock_print.assert_called_once()
        printed = mock_print.call_args[0][0]
        self.assertIn("Cam A", printed)
        self.assertIn("online", printed)

    @patch("roboflow.adapters.devicesapi.list_devices")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_list_json(self, _mk, _mw, mock_list):
        mock_list.return_value = {"data": [{"id": "a", "name": "Cam A"}]}
        from roboflow.cli.handlers.device import _list

        with patch("builtins.print") as mock_print:
            _list(_args(json=True))
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["data"][0]["id"], "a")


class TestDeviceErrorMapping(unittest.TestCase):
    """Adapter exceptions map to documented exit codes."""

    @patch("roboflow.adapters.devicesapi.get_device")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_404_exits_3(self, _mk, _mw, mock_get):
        mock_get.side_effect = DeviceNotFoundError("not found", status_code=404)
        from roboflow.cli.handlers.device import _get

        with self.assertRaises(SystemExit) as ctx:
            _get(_args(device_id="missing"))
        self.assertEqual(ctx.exception.code, 3)

    @patch("roboflow.adapters.devicesapi.get_device")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_401_exits_2(self, _mk, _mw, mock_get):
        mock_get.side_effect = DeviceAuthError("nope", status_code=401)
        from roboflow.cli.handlers.device import _get

        with self.assertRaises(SystemExit) as ctx:
            _get(_args(device_id="x"))
        self.assertEqual(ctx.exception.code, 2)

    @patch("roboflow.adapters.devicesapi.get_device_logs")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_429_exits_1_with_hint(self, _mk, _mw, mock_logs):
        mock_logs.side_effect = DeviceRateLimitedError("slow down", status_code=429)
        from roboflow.cli.handlers.device import _logs

        args = _args(
            device_id="x",
            start_time=None,
            end_time=None,
            service=None,
            severity=None,
            limit=None,
            cursor=None,
            json=True,
        )
        with self.assertRaises(SystemExit) as ctx:
            _logs(args)
        self.assertEqual(ctx.exception.code, 1)


class TestDeviceCreateHandler(unittest.TestCase):
    @patch("roboflow.adapters.devicesapi.create_device")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_create_passes_args(self, _mk, _mw, mock_create):
        mock_create.return_value = {"deviceId": "d1", "installId": "i1"}
        from roboflow.cli.handlers.device import _create

        args = _args(
            device_name="Cam 1",
            device_type="edge",
            workflow_id="wf-1",
            tags=["a", "b"],
            offline_mode=None,
            source_device_id=None,
            json=True,
        )
        with patch("builtins.print") as mock_print:
            _create(args)
        kwargs = mock_create.call_args.kwargs
        self.assertEqual(kwargs["device_name"], "Cam 1")
        self.assertEqual(kwargs["device_type"], "edge")
        self.assertEqual(kwargs["tags"], ["a", "b"])
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["deviceId"], "d1")


class TestDeviceLogsCsvSerialization(unittest.TestCase):
    @patch("roboflow.adapters.devicesapi.get_device_logs")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=WS)
    @patch("roboflow.config.load_roboflow_api_key", return_value=KEY)
    def test_severity_passed_as_list(self, _mk, _mw, mock_logs):
        mock_logs.return_value = {"data": [], "pagination": {}}
        from roboflow.cli.handlers.device import _logs

        args = _args(
            device_id="x",
            start_time=None,
            end_time=None,
            service=["foo", "bar"],
            severity=["INFO"],
            limit=None,
            cursor=None,
        )
        _logs(args)
        kwargs = mock_logs.call_args.kwargs
        self.assertEqual(kwargs["service"], ["foo", "bar"])
        self.assertEqual(kwargs["severity"], ["INFO"])


if __name__ == "__main__":
    unittest.main()
