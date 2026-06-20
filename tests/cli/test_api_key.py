"""Tests for the api-key CLI handler."""

import json
import re
import unittest
from argparse import Namespace
from unittest.mock import patch

from typer.testing import CliRunner

from roboflow.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# ---------------------------------------------------------------------------
# Registration / --help tests
# ---------------------------------------------------------------------------


class TestApiKeyRegistration(unittest.TestCase):
    """Verify the api-key group and all subcommands are registered."""

    def test_api_key_app_exists(self) -> None:
        from roboflow.cli.handlers.api_key import api_key_app

        self.assertIsNotNone(api_key_app)

    def test_group_help(self) -> None:
        result = runner.invoke(app, ["api-key", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("api-key", _strip_ansi(result.output).lower())

    def test_list_help(self) -> None:
        result = runner.invoke(app, ["api-key", "list", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_get_help(self) -> None:
        result = runner.invoke(app, ["api-key", "get", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_publishable_help(self) -> None:
        result = runner.invoke(app, ["api-key", "publishable", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_create_help(self) -> None:
        result = runner.invoke(app, ["api-key", "create", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        output = _strip_ansi(result.output).lower()
        self.assertIn("scope", output)
        self.assertIn("folder", output)
        self.assertIn("metadata", output)
        self.assertIn("protected", output)

    def test_update_help(self) -> None:
        result = runner.invoke(app, ["api-key", "update", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        output = _strip_ansi(result.output).lower()
        self.assertIn("name", output)
        self.assertIn("scope", output)
        self.assertIn("metadata", output)

    def test_protect_help(self) -> None:
        result = runner.invoke(app, ["api-key", "protect", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_disable_help(self) -> None:
        result = runner.invoke(app, ["api-key", "disable", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_revoke_help(self) -> None:
        result = runner.invoke(app, ["api-key", "revoke", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestListApiKeys(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.list_api_keys")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_list_text(self, _mock_key, _mock_ws, mock_list) -> None:
        mock_list.return_value = {
            "apiKeys": [
                {
                    "keyId": "k1",
                    "name": "My Key",
                    "prefix": "abc",
                    "default": True,
                    "protected": False,
                    "disabled": False,
                }
            ]
        }
        args = Namespace(
            json=False, workspace=None, api_key=None, quiet=False, include_disabled=False, include_folders=False
        )
        from roboflow.cli.handlers.api_key import _list_keys

        with patch("builtins.print") as mock_print:
            _list_keys(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("My Key", printed)
        self.assertIn("k1", printed)

    @patch("roboflow.adapters.rfapi.list_api_keys")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_list_json_emits_full_envelope(self, _mock_key, _mock_ws, mock_list) -> None:
        keys = [
            {"keyId": "k1", "name": "My Key", "prefix": "abc", "default": True, "protected": False, "disabled": False}
        ]
        mock_list.return_value = {"apiKeys": keys, "publishableKey": "rf_myworkspace"}
        args = Namespace(
            json=True, workspace=None, api_key=None, quiet=False, include_disabled=False, include_folders=False
        )
        from roboflow.cli.handlers.api_key import _list_keys

        with patch("builtins.print") as mock_print:
            _list_keys(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        # Full server envelope, not a bare array — publishableKey must be preserved.
        self.assertIsInstance(data, dict)
        self.assertEqual(data["apiKeys"][0]["keyId"], "k1")
        self.assertEqual(data["publishableKey"], "rf_myworkspace")

    @patch("roboflow.adapters.rfapi.list_api_keys")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_list_passes_include_disabled(self, _mock_key, _mock_ws, mock_list) -> None:
        mock_list.return_value = {"apiKeys": []}
        args = Namespace(
            json=False, workspace=None, api_key=None, quiet=False, include_disabled=True, include_folders=False
        )
        from roboflow.cli.handlers.api_key import _list_keys

        with patch("builtins.print"):
            _list_keys(args)
        mock_list.assert_called_once_with("fake-key", "test-ws", include_disabled=True, include_folders=False)

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value=None)
    def test_list_no_workspace(self, _mock_ws) -> None:
        args = Namespace(
            json=True, workspace=None, api_key=None, quiet=False, include_disabled=False, include_folders=False
        )
        from roboflow.cli.handlers.api_key import _list_keys

        with self.assertRaises(SystemExit) as ctx:
            _list_keys(args)
        self.assertEqual(ctx.exception.code, 2)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGetApiKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_get_text(self, _mock_key, _mock_ws, mock_get) -> None:
        mock_get.return_value = {
            "apiKey": {
                "keyId": "k1",
                "name": "My Key",
                "prefix": "abc",
                "default": False,
                "protected": False,
                "disabled": False,
            }
        }
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1")
        from roboflow.cli.handlers.api_key import _get_key

        with patch("builtins.print") as mock_print:
            _get_key(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("k1", printed)
        self.assertIn("My Key", printed)

    @patch("roboflow.adapters.rfapi.get_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_get_json(self, _mock_key, _mock_ws, mock_get) -> None:
        payload = {"apiKey": {"keyId": "k1", "name": "My Key"}}
        mock_get.return_value = payload
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, key_id="k1")
        from roboflow.cli.handlers.api_key import _get_key

        with patch("builtins.print") as mock_print:
            _get_key(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertIn("apiKey", data)


# ---------------------------------------------------------------------------
# publishable
# ---------------------------------------------------------------------------


class TestPublishableKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.get_publishable_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_publishable_text(self, _mock_key, _mock_ws, mock_pub) -> None:
        mock_pub.return_value = {"publishableKey": "rf_myworkspace"}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False)
        from roboflow.cli.handlers.api_key import _get_publishable

        with patch("builtins.print") as mock_print:
            _get_publishable(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("rf_myworkspace", printed)

    @patch("roboflow.adapters.rfapi.get_publishable_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_publishable_json(self, _mock_key, _mock_ws, mock_pub) -> None:
        mock_pub.return_value = {"publishableKey": "rf_myworkspace"}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False)
        from roboflow.cli.handlers.api_key import _get_publishable

        with patch("builtins.print") as mock_print:
            _get_publishable(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["publishableKey"], "rf_myworkspace")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


class TestCreateApiKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.create_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_prints_secret_in_text_mode(self, _mock_key, _mock_ws, mock_create) -> None:
        mock_create.return_value = {"keyId": "k2", "key": "super-secret-value", "name": "New Key"}
        args = Namespace(
            json=False,
            workspace=None,
            api_key=None,
            quiet=False,
            name="New Key",
            scope=None,
            folder=None,
            protected=False,
        )
        from roboflow.cli.handlers.api_key import _create_key

        with patch("builtins.print") as mock_print:
            _create_key(args)
        printed = mock_print.call_args[0][0]
        self.assertIn("super-secret-value", printed)
        self.assertIn("WARNING", printed)

    @patch("roboflow.adapters.rfapi.create_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_json_contains_key(self, _mock_key, _mock_ws, mock_create) -> None:
        mock_create.return_value = {"keyId": "k2", "key": "super-secret-value", "name": "New Key"}
        args = Namespace(
            json=True,
            workspace=None,
            api_key=None,
            quiet=False,
            name="New Key",
            scope=None,
            folder=None,
            protected=False,
        )
        from roboflow.cli.handlers.api_key import _create_key

        with patch("builtins.print") as mock_print:
            _create_key(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["key"], "super-secret-value")
        self.assertEqual(data["keyId"], "k2")

    @patch("roboflow.adapters.rfapi.create_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_with_scopes_and_folders(self, _mock_key, _mock_ws, mock_create) -> None:
        mock_create.return_value = {"keyId": "k3", "key": "s3cr3t", "name": "Scoped Key"}
        args = Namespace(
            json=False,
            workspace=None,
            api_key=None,
            quiet=False,
            name="Scoped Key",
            scope=["read:images", "read:annotations"],
            folder=["f1", "f2"],
            metadata=None,
            protected=False,
        )
        from roboflow.cli.handlers.api_key import _create_key

        with patch("builtins.print"):
            _create_key(args)
        mock_create.assert_called_once_with(
            "fake-key",
            "test-ws",
            name="Scoped Key",
            scopes=["read:images", "read:annotations"],
            folder_ids=["f1", "f2"],
            custom_metadata=None,
            protected=False,
        )

    @patch("roboflow.adapters.rfapi.create_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_with_metadata(self, _mock_key, _mock_ws, mock_create) -> None:
        mock_create.return_value = {"keyId": "k5", "key": "m3ta", "name": "Meta Key"}
        args = Namespace(
            json=False,
            workspace=None,
            api_key=None,
            quiet=False,
            name="Meta Key",
            scope=None,
            folder=None,
            metadata=["team=vision", "env=prod"],
            protected=False,
        )
        from roboflow.cli.handlers.api_key import _create_key

        with patch("builtins.print"):
            _create_key(args)
        mock_create.assert_called_once_with(
            "fake-key",
            "test-ws",
            name="Meta Key",
            scopes=None,
            folder_ids=None,
            custom_metadata={"team": "vision", "env": "prod"},
            protected=False,
        )

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_invalid_metadata_exits(self, _mock_key, _mock_ws) -> None:
        args = Namespace(
            json=False,
            workspace=None,
            api_key=None,
            quiet=False,
            name="Bad Meta",
            scope=None,
            folder=None,
            metadata=["no-equals-sign"],
            protected=False,
        )
        from roboflow.cli.handlers.api_key import _create_key

        with self.assertRaises(SystemExit) as ctx:
            _create_key(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.adapters.rfapi.create_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_create_no_scopes_passes_none(self, _mock_key, _mock_ws, mock_create) -> None:
        mock_create.return_value = {"keyId": "k4", "key": "full-key", "name": "Full Key"}
        args = Namespace(
            json=False,
            workspace=None,
            api_key=None,
            quiet=False,
            name="Full Key",
            scope=None,
            folder=None,
            metadata=None,
            protected=False,
        )
        from roboflow.cli.handlers.api_key import _create_key

        with patch("builtins.print"):
            _create_key(args)
        mock_create.assert_called_once_with(
            "fake-key",
            "test-ws",
            name="Full Key",
            scopes=None,
            folder_ids=None,
            custom_metadata=None,
            protected=False,
        )


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


class TestUpdateApiKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_update_name(self, _mock_key, _mock_ws, mock_update) -> None:
        mock_update.return_value = {"apiKey": {"keyId": "k1", "name": "Renamed"}}
        args = Namespace(
            json=True, workspace=None, api_key=None, quiet=False, key_id="k1", name="Renamed", scope=None, metadata=None
        )
        from roboflow.cli.handlers.api_key import _update_key

        with patch("builtins.print") as mock_print:
            _update_key(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertIn("apiKey", data)
        mock_update.assert_called_once_with("fake-key", "test-ws", "k1", name="Renamed")

    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_update_scopes_and_metadata(self, _mock_key, _mock_ws, mock_update) -> None:
        mock_update.return_value = {"apiKey": {"keyId": "k1"}}
        args = Namespace(
            json=False,
            workspace=None,
            api_key=None,
            quiet=False,
            key_id="k1",
            name=None,
            scope=["read:images"],
            metadata=["team=vision"],
        )
        from roboflow.cli.handlers.api_key import _update_key

        with patch("builtins.print"):
            _update_key(args)
        mock_update.assert_called_once_with(
            "fake-key",
            "test-ws",
            "k1",
            scopes=["read:images"],
            custom_metadata={"team": "vision"},
        )

    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_update_missing_key_exits_3(self, _mock_key, _mock_ws, mock_update) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_update.side_effect = RoboflowError("not found", status_code=404)
        args = Namespace(
            json=False, workspace=None, api_key=None, quiet=False, key_id="nope", name="X", scope=None, metadata=None
        )
        from roboflow.cli.handlers.api_key import _update_key

        with self.assertRaises(SystemExit) as ctx:
            _update_key(args)
        self.assertEqual(ctx.exception.code, 3)


# ---------------------------------------------------------------------------
# protect
# ---------------------------------------------------------------------------


class TestProtectApiKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_protect_calls_patch_with_protected_true(self, _mock_key, _mock_ws, mock_update) -> None:
        mock_update.return_value = {"apiKey": {"keyId": "k1", "protected": True}}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1")
        from roboflow.cli.handlers.api_key import _protect_key

        with patch("builtins.print"):
            _protect_key(args)
        mock_update.assert_called_once_with("fake-key", "test-ws", "k1", protected=True)


# ---------------------------------------------------------------------------
# disable
# ---------------------------------------------------------------------------


class TestDisableApiKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_disable(self, _mock_key, _mock_ws, mock_update) -> None:
        mock_update.return_value = {"apiKey": {"keyId": "k1", "disabled": True}}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1", enable=False)
        from roboflow.cli.handlers.api_key import _disable_key

        with patch("builtins.print") as mock_print:
            _disable_key(args)
        mock_update.assert_called_once_with("fake-key", "test-ws", "k1", disabled=True)
        printed = mock_print.call_args[0][0]
        self.assertIn("Disabled", printed)

    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_enable(self, _mock_key, _mock_ws, mock_update) -> None:
        mock_update.return_value = {"apiKey": {"keyId": "k1", "disabled": False}}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1", enable=True)
        from roboflow.cli.handlers.api_key import _disable_key

        with patch("builtins.print") as mock_print:
            _disable_key(args)
        mock_update.assert_called_once_with("fake-key", "test-ws", "k1", disabled=False)
        printed = mock_print.call_args[0][0]
        self.assertIn("Enabled", printed)


# ---------------------------------------------------------------------------
# revoke
# ---------------------------------------------------------------------------


class TestRevokeApiKey(unittest.TestCase):
    @patch("roboflow.adapters.rfapi.revoke_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_revoke_with_yes(self, _mock_key, _mock_ws, mock_revoke) -> None:
        mock_revoke.return_value = {"status": "revoked", "keyId": "k1"}
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1", yes=True)
        from roboflow.cli.handlers.api_key import _revoke_key

        with patch("builtins.print") as mock_print:
            _revoke_key(args)
        mock_revoke.assert_called_once_with("fake-key", "test-ws", "k1")
        printed = mock_print.call_args[0][0]
        self.assertIn("Revoked", printed)

    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_revoke_without_yes_no_tty_exits(self, _mock_key, _mock_ws) -> None:
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1", yes=False)
        from roboflow.cli.handlers.api_key import _revoke_key

        with patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(SystemExit) as ctx:
                _revoke_key(args)
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.adapters.rfapi.revoke_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_revoke_json(self, _mock_key, _mock_ws, mock_revoke) -> None:
        mock_revoke.return_value = {"status": "revoked", "keyId": "k1"}
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, key_id="k1", yes=True)
        from roboflow.cli.handlers.api_key import _revoke_key

        with patch("builtins.print") as mock_print:
            _revoke_key(args)
        printed = mock_print.call_args[0][0]
        data = json.loads(printed)
        self.assertEqual(data["status"], "revoked")


# ---------------------------------------------------------------------------
# status-code propagation / exit-code consistency
# ---------------------------------------------------------------------------


class TestStatusCodePropagation(unittest.TestCase):
    """The rfapi wrappers must attach response.status_code so handlers can branch."""

    def _make_response(self, status_code, text="error body"):
        from unittest.mock import MagicMock

        resp = MagicMock()
        resp.ok = status_code < 400
        resp.status_code = status_code
        resp.text = text
        return resp

    @patch("roboflow.adapters.rfapi.requests.delete")
    def test_revoke_attaches_status_code(self, mock_delete) -> None:
        from roboflow.adapters.rfapi import RoboflowError, revoke_api_key

        mock_delete.return_value = self._make_response(409, "protected")
        with self.assertRaises(RoboflowError) as ctx:
            revoke_api_key("fake-key", "test-ws", "k1")
        self.assertEqual(ctx.exception.status_code, 409)

    @patch("roboflow.adapters.rfapi.requests.patch")
    def test_update_attaches_status_code(self, mock_patch) -> None:
        from roboflow.adapters.rfapi import RoboflowError, update_api_key

        mock_patch.return_value = self._make_response(403, "forbidden")
        with self.assertRaises(RoboflowError) as ctx:
            update_api_key("fake-key", "test-ws", "k1", protected=False)
        self.assertEqual(ctx.exception.status_code, 403)

    @patch("roboflow.adapters.rfapi.requests.get")
    def test_get_attaches_status_code(self, mock_get) -> None:
        from roboflow.adapters.rfapi import RoboflowError, get_api_key

        mock_get.return_value = self._make_response(404, "missing")
        with self.assertRaises(RoboflowError) as ctx:
            get_api_key("fake-key", "test-ws", "k1")
        self.assertEqual(ctx.exception.status_code, 404)


class TestErrorBranches(unittest.TestCase):
    """End-to-end: status_code now flows through to the right exit code / hint."""

    @patch("roboflow.adapters.rfapi.revoke_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_revoke_protected_409_hint(self, _mock_key, _mock_ws, mock_revoke) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_revoke.side_effect = RoboflowError("Key is protected", status_code=409)
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1", yes=True)
        from roboflow.cli.handlers.api_key import _revoke_key

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                _revoke_key(args)
        # 409 is a non-auth/non-notfound error → exit 1, with the protected-key hint.
        self.assertEqual(ctx.exception.code, 1)

    @patch("roboflow.adapters.rfapi.revoke_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_revoke_missing_exits_3(self, _mock_key, _mock_ws, mock_revoke) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_revoke.side_effect = RoboflowError("not found", status_code=404)
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="missing", yes=True)
        from roboflow.cli.handlers.api_key import _revoke_key

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                _revoke_key(args)
        # Consistent with `get <missing>` → exit 3, not 1.
        self.assertEqual(ctx.exception.code, 3)

    @patch("roboflow.adapters.rfapi.get_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_get_missing_exits_3(self, _mock_key, _mock_ws, mock_get) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_get.side_effect = RoboflowError("not found", status_code=404)
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="missing")
        from roboflow.cli.handlers.api_key import _get_key

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                _get_key(args)
        self.assertEqual(ctx.exception.code, 3)

    @patch("roboflow.adapters.rfapi.get_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_get_auth_error_exits_2(self, _mock_key, _mock_ws, mock_get) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_get.side_effect = RoboflowError("unauthorized", status_code=401)
        args = Namespace(json=False, workspace=None, api_key=None, quiet=False, key_id="k1")
        from roboflow.cli.handlers.api_key import _get_key

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit) as ctx:
                _get_key(args)
        self.assertEqual(ctx.exception.code, 2)

    @patch("roboflow.adapters.rfapi.update_api_key")
    @patch("roboflow.cli._resolver.resolve_default_workspace", return_value="test-ws")
    @patch("roboflow.config.load_roboflow_api_key", return_value="fake-key")
    def test_disable_protected_409_hint(self, _mock_key, _mock_ws, mock_update) -> None:
        from roboflow.adapters.rfapi import RoboflowError

        mock_update.side_effect = RoboflowError("Key is protected", status_code=409)
        args = Namespace(json=True, workspace=None, api_key=None, quiet=False, key_id="k1", enable=False)
        from roboflow.cli.handlers.api_key import _disable_key

        printed = {}

        def _capture(payload, *a, **k):
            printed["json"] = payload

        with patch("builtins.print", side_effect=_capture):
            with self.assertRaises(SystemExit) as ctx:
                _disable_key(args)
        self.assertEqual(ctx.exception.code, 1)
        data = json.loads(printed["json"])
        self.assertIn("settings/api", data["error"]["hint"])
        self.assertNotIn("settings/api-keys", data["error"]["hint"])


if __name__ == "__main__":
    unittest.main()
