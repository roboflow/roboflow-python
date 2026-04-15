"""Roboflow CLI — computer vision at your fingertips.

Built on typer. Each command group is a separate Typer app in the
``handlers`` sub-package, registered via ``app.add_typer()``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Optional

import click
import typer

import roboflow
from roboflow.cli._compat import SortedGroup

# ---------------------------------------------------------------------------
# Root application
# ---------------------------------------------------------------------------

_DESCRIPTION = (
    "Build and deploy computer vision models with Roboflow. "
    "Manage datasets, train models, run inference, and deploy "
    "workflows \u2014 from the command line or via structured JSON for AI agents."
)

app = typer.Typer(
    name="roboflow",
    help=_DESCRIPTION,
    cls=SortedGroup,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        import sys

        if "--json" in sys.argv or "-j" in sys.argv:
            print(json.dumps({"version": roboflow.__version__}))
        else:
            print(roboflow.__version__)
        raise typer.Exit


@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="API key override (default: $ROBOFLOW_API_KEY or config file)"),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output results as JSON (stable schema, for agents and piping)"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output (progress bars, status messages)"),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version",
            "-v",
            help="Show package version and exit",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
    workspace: Annotated[
        Optional[str],
        typer.Option("--workspace", "-w", help="Workspace URL or ID override (default: configured default)"),
    ] = None,
) -> None:
    """Build and deploy computer vision models with Roboflow."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["api_key"] = api_key
    ctx.obj["workspace"] = workspace
    ctx.obj["quiet"] = quiet

    if ctx.invoked_subcommand is None:
        _print_flattened_help()
        raise typer.Exit(code=0)


def _print_flattened_help() -> None:
    """Print a Rich-formatted help screen with all commands flattened and alphabetized."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()

    click_app = typer.main.get_command(app)

    # Collect all visible commands, flattened
    commands: list[tuple[str, str]] = []

    def _walk(group: Any, prefix: str = "") -> None:
        for name in sorted(group.list_commands(None) or []):  # type: ignore[arg-type]
            cmd = group.get_command(None, name)  # type: ignore[arg-type]
            if cmd is None or getattr(cmd, "hidden", False):
                continue
            full = f"{prefix} {name}".strip() if prefix else name
            if hasattr(cmd, "list_commands") and cmd.list_commands(None):
                _walk(cmd, full)
            else:
                # Use the full help text: try help attr, then short_help, then docstring
                help_text = getattr(cmd, "help", None) or getattr(cmd, "short_help", None) or ""
                # Take only the first line/sentence
                help_text = help_text.split("\n")[0].strip()
                commands.append((full, help_text))

    _walk(click_app)
    commands.sort(key=lambda x: x[0])

    # Usage line
    console.print()
    console.print(" Usage: roboflow [OPTIONS] COMMAND [ARGS]...", highlight=False)
    console.print()
    console.print(f" {_DESCRIPTION}", highlight=False)
    console.print()

    # Options panel — match typer's color scheme
    options_data = [
        ("--api-key", "-k", "TEXT", "API key override (default: $ROBOFLOW_API_KEY or config file)"),
        ("--json", "-j", "", "Output results as JSON (stable schema, for agents and piping)"),
        ("--quiet", "-q", "", "Suppress non-essential output (progress bars, status messages)"),
        ("--version", "-v", "", "Show package version and exit"),
        ("--workspace", "-w", "TEXT", "Workspace URL or ID override (default: configured default)"),
        ("--help", "-h", "", "Show this message and exit."),
    ]
    opt_table = Table(show_header=False, box=None, padding=(0, 1))
    opt_table.add_column(no_wrap=True, style="bold cyan")  # long flag
    opt_table.add_column(no_wrap=True, style="bold green")  # short flag
    opt_table.add_column(no_wrap=True, style="bold yellow")  # metavar
    opt_table.add_column()  # description
    for long_flag, short_flag, metavar, desc in options_data:
        opt_table.add_row(long_flag, short_flag, metavar, desc)
    console.print(Panel(opt_table, title="Options", title_align="left", border_style="dim"))

    # Commands panel — group name in dim cyan, verb in bold
    cmd_table = Table(show_header=False, box=None, padding=(0, 1))
    cmd_table.add_column(no_wrap=True)  # command name
    cmd_table.add_column()  # description
    for cmd_name, help_text in commands:
        parts = cmd_name.split(" ", 1)
        styled_name = Text()
        if len(parts) == 1:
            # Top-level command (no group): just bold
            styled_name.append(parts[0], style="bold")
        else:
            # Group + verb: group in dim cyan, verb in bold
            styled_name.append(parts[0], style="cyan")
            styled_name.append(" ")
            styled_name.append(parts[1], style="bold")
        cmd_table.add_row(styled_name, help_text)
    console.print(Panel(cmd_table, title="Commands", title_align="left", border_style="dim"))
    console.print()


# ---------------------------------------------------------------------------
# Register command groups (explicit imports — no auto-discovery needed)
# ---------------------------------------------------------------------------

from roboflow.cli.handlers.annotation import annotation_app  # noqa: E402
from roboflow.cli.handlers.auth import auth_app  # noqa: E402
from roboflow.cli.handlers.batch import batch_app  # noqa: E402
from roboflow.cli.handlers.completion import completion_app  # noqa: E402
from roboflow.cli.handlers.deployment import deployment_app  # noqa: E402
from roboflow.cli.handlers.folder import folder_app  # noqa: E402
from roboflow.cli.handlers.image import image_app  # noqa: E402
from roboflow.cli.handlers.infer import infer_command  # noqa: E402
from roboflow.cli.handlers.model import model_app  # noqa: E402
from roboflow.cli.handlers.project import project_app  # noqa: E402
from roboflow.cli.handlers.search import search_command  # noqa: E402
from roboflow.cli.handlers.train import train_app  # noqa: E402
from roboflow.cli.handlers.universe import universe_app  # noqa: E402
from roboflow.cli.handlers.version import version_app  # noqa: E402
from roboflow.cli.handlers.video import video_app  # noqa: E402
from roboflow.cli.handlers.vision_events import vision_events_app  # noqa: E402
from roboflow.cli.handlers.workflow import workflow_app  # noqa: E402
from roboflow.cli.handlers.workspace import workspace_app  # noqa: E402

# Register ALL commands in alphabetical order for clean --help output
app.add_typer(annotation_app, name="annotation")
app.add_typer(auth_app, name="auth")
app.add_typer(batch_app, name="batch", hidden=True)  # All stubs — hidden until implemented
app.add_typer(completion_app, name="completion")
app.add_typer(deployment_app, name="deployment")
app.add_typer(folder_app, name="folder")
app.add_typer(image_app, name="image")

# "infer" — top-level command, registered alphabetically
infer_command(app)

app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")

# "search" — top-level command, registered alphabetically
search_command(app)

app.add_typer(train_app, name="train")
app.add_typer(universe_app, name="universe")
app.add_typer(version_app, name="version")
app.add_typer(video_app, name="video")
app.add_typer(vision_events_app, name="vision-events")
app.add_typer(workflow_app, name="workflow")
app.add_typer(workspace_app, name="workspace")

# Hidden aliases (loaded last — still functional but not in --help)
from roboflow.cli.handlers._aliases import register_hidden_aliases  # noqa: E402

register_hidden_aliases(app)


# "roboflow help" command
@app.command("help", hidden=True)
def help_command(ctx: typer.Context) -> None:  # noqa: ARG001
    """Show help information."""
    _print_flattened_help()


# ---------------------------------------------------------------------------
# Backwards-compat: build_parser returns None (argparse is gone)
# ---------------------------------------------------------------------------


class _LegacyParserShim:
    """Argparse-compatible shim wrapping the typer app.

    Supports ``parser.parse_args(argv)`` and ``parser.print_help()``.
    This keeps ``from roboflow.roboflowpy import _argparser`` working
    for the ~5M monthly installs that may depend on it.
    """

    def parse_args(self, argv: list[str] | None = None) -> object:  # noqa: ANN001
        """Parse *argv* and return an argparse-like namespace with ``func``.

        Does NOT execute the command — callers are expected to call
        ``args.func(args)`` themselves, matching the old argparse pattern.
        """
        import sys
        import types

        if argv is None:
            argv = sys.argv[1:]

        argv = _reorder_argv(list(argv))

        # Build a namespace with the parsed values by invoking the CLI
        # in a dry-run fashion: we intercept before execution.
        ns = types.SimpleNamespace(
            json=False,
            api_key=None,
            workspace=None,
            quiet=False,
            func=None,
        )

        # Extract global flags manually
        remaining = []
        i = 0
        while i < len(argv):
            if argv[i] in ("--json", "-j"):
                ns.json = True
            elif argv[i] in ("--quiet", "-q"):
                ns.quiet = True
            elif argv[i] in ("--api-key", "-k") and i + 1 < len(argv):
                i += 1
                ns.api_key = argv[i]
            elif argv[i] == "--workspace" and i + 1 < len(argv):
                i += 1
                ns.workspace = argv[i]
            else:
                remaining.append(argv[i])
            i += 1

        # Set func to a lambda that invokes the CLI with the original argv
        original_argv = list(argv)

        def _run_via_typer(_args: object) -> None:
            from typer.testing import CliRunner as _TyperRunner

            runner = _TyperRunner()
            result = runner.invoke(app, original_argv, catch_exceptions=False)
            if result.output:
                print(result.output, end="")  # noqa: T201
            if result.exit_code:
                sys.exit(result.exit_code)

        ns.func = _run_via_typer
        return ns

    def print_help(self) -> None:
        """Print the CLI help text."""
        from typer.testing import CliRunner as _TyperRunner

        runner = _TyperRunner()
        result = runner.invoke(app, ["--help"])
        if result.output:
            print(result.output, end="")  # noqa: T201


def build_parser() -> _LegacyParserShim:
    """Legacy compat: returns an argparse-like shim wrapping the typer app."""
    return _LegacyParserShim()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _reorder_argv(argv: list[str]) -> list[str]:
    """Move known global flags that appear after the subcommand to the front.

    Typer/Click only recognises parent-level options when they appear
    *before* the subcommand.  Many users (and AI agents) naturally write
    them at the end, e.g. ``roboflow project list --json``.  This helper
    transparently re-orders the argv so those flags are consumed by the
    root callback.
    """
    # Note: -w is intentionally excluded — it collides with deployment's
    # -w/--wait_on_pending (boolean).  --workspace (long form) is safe.
    global_flags_with_value = {"--api-key", "-k", "--workspace"}
    global_flags_bool = {"--json", "-j", "--quiet", "-q", "--version"}

    reordered: list[str] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in global_flags_bool:
            reordered.append(arg)
        elif arg in global_flags_with_value:
            if i + 1 < len(argv):
                reordered.append(arg)
                i += 1
                reordered.append(argv[i])
            else:
                # No value follows — leave in place so typer shows a proper error
                rest.append(arg)
        else:
            rest.append(arg)
        i += 1
    return reordered + rest


def main() -> None:
    """CLI entry point — called by ``roboflow`` console script."""
    import sys

    sys.argv[1:] = _reorder_argv(sys.argv[1:])

    # Intercept root-level --help/-h: show our flattened help instead of typer's grouped view.
    # Only for the ROOT command (not subcommands like 'roboflow project --help').
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        argv = sys.argv[1:]
        help_idx = next((i for i, a in enumerate(argv) if a in ("--help", "-h")), -1)
        pre_help = [a for a in argv[:help_idx] if not a.startswith("-")]
        if not pre_help:
            _print_flattened_help()
            sys.exit(0)

    # In --json mode, intercept Click/typer validation errors and emit
    # structured JSON on stderr instead of Rich-formatted text.
    json_mode = "--json" in sys.argv or "-j" in sys.argv
    if json_mode:
        try:
            app(standalone_mode=False)
        except SystemExit as exc:
            # Exit code 0 = success (already handled), just re-raise
            if exc.code == 0:
                raise
            # Exit code 2 = Click usage error (missing arg, bad option)
            # Other codes = our output_error already printed JSON
            raise
        except click.exceptions.UsageError as exc:
            # Click/typer validation error — emit JSON on stderr
            import json as _json

            payload = {"error": {"message": str(exc), "hint": "Run with --help for usage information."}}
            print(_json.dumps(payload), file=sys.stderr)
            sys.exit(2)
        except click.exceptions.Abort:
            sys.exit(1)
        except Exception as exc:
            import json as _json

            payload = {"error": {"message": str(exc)}}
            print(_json.dumps(payload), file=sys.stderr)
            sys.exit(1)
    else:
        app()
