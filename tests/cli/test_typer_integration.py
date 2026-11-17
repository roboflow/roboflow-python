import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import patch

import typer

from roboflow.cli import main
from roboflow.cli._compat import _sort_params


class TestTyperIntegration(unittest.TestCase):
    def test_json_usage_errors(self):
        cases = (
            (["project", "get"], "Missing argument"),
            (["--bad-option"], "--bad-option"),
            (["image", "search", "example", "--limit", "not-an-int"], "--limit"),
        )
        for args, message in cases:
            with self.subTest(args=args):
                stderr = io.StringIO()
                with patch.object(sys, "argv", ["roboflow", "--json", *args]):
                    with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                        main()
                self.assertEqual(error.exception.code, 2)
                payload = json.loads(stderr.getvalue())
                self.assertIn("hint", payload["error"])
                self.assertIn(message, payload["error"]["message"])

    def test_json_abort(self):
        with patch.object(sys, "argv", ["roboflow", "--json"]):
            with patch("roboflow.cli.app", side_effect=typer.Abort):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    main()
        self.assertEqual(error.exception.code, 1)
        # Preserve the existing silent cancellation behavior.
        self.assertEqual(stderr.getvalue(), "")

    def test_arguments_sort_before_required_options(self):
        argument = typer.core.TyperArgument(param_decls=["zulu"], required=True)
        option = typer.core.TyperOption(param_decls=["--alpha"], required=True)
        params = [option, argument]
        _sort_params(params)
        self.assertEqual(params, [argument, option])
