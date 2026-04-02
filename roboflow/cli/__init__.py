"""Roboflow CLI — computer vision at your fingertips.

Built on typer. Each command group is a separate Typer app in the
``handlers`` sub-package, registered via ``app.add_typer()``.
"""

from __future__ import annotations

import json
from typing import Annotated, Optional

import typer

import roboflow

# ---------------------------------------------------------------------------
# Root application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="roboflow",
    help="Roboflow CLI: computer vision at your fingertips",
    no_args_is_help=True,
    pretty_exceptions_enable=False,  # We handle errors ourselves via output_error
)


def _version_callback(value: bool) -> None:
    if value:
        print(roboflow.__version__)
        raise typer.Exit()


def _json_version_callback(ctx: typer.Context, value: bool) -> None:
    """Handle --version with --json awareness."""
    if value:
        # Check if --json was also passed (it may or may not be parsed yet)
        json_mode = ctx.params.get("json_output", False)
        if json_mode:
            print(json.dumps({"version": roboflow.__version__}))
        else:
            print(roboflow.__version__)
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def _root_callback(
    ctx: typer.Context,
    json_output: Annotated[
        bool,
        typer.Option("--json", "-j", help="Output results as JSON (stable schema, for agents and piping)"),
    ] = False,
    api_key: Annotated[
        Optional[str],
        typer.Option("--api-key", "-k", help="API key override (default: $ROBOFLOW_API_KEY or config file)"),
    ] = None,
    workspace: Annotated[
        Optional[str],
        typer.Option("--workspace", "-w", help="Workspace URL or ID override (default: configured default)"),
    ] = None,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Suppress non-essential output (progress bars, status messages)"),
    ] = False,
    version: Annotated[
        Optional[bool],
        typer.Option("--version", help="Show package version and exit", callback=_version_callback, is_eager=True),
    ] = None,
) -> None:
    """Roboflow CLI: computer vision at your fingertips."""
    ctx.ensure_object(dict)
    ctx.obj["json"] = json_output
    ctx.obj["api_key"] = api_key
    ctx.obj["workspace"] = workspace
    ctx.obj["quiet"] = quiet


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
from roboflow.cli.handlers.workflow import workflow_app  # noqa: E402
from roboflow.cli.handlers.workspace import workspace_app  # noqa: E402

app.add_typer(annotation_app, name="annotation")
app.add_typer(auth_app, name="auth")
app.add_typer(batch_app, name="batch")
app.add_typer(completion_app, name="completion")
app.add_typer(deployment_app, name="deployment")
app.add_typer(folder_app, name="folder")
app.add_typer(image_app, name="image")
app.add_typer(model_app, name="model")
app.add_typer(project_app, name="project")
app.add_typer(train_app, name="train")
app.add_typer(universe_app, name="universe")
app.add_typer(version_app, name="version")
app.add_typer(video_app, name="video")
app.add_typer(workflow_app, name="workflow")
app.add_typer(workspace_app, name="workspace")

# Top-level commands (not nested under a group)
infer_command(app)
search_command(app)

# Backwards-compat aliases (loaded last)
from roboflow.cli.handlers._aliases import register_aliases  # noqa: E402

register_aliases(app)

# ---------------------------------------------------------------------------
# Backwards-compat: build_parser returns None (argparse is gone)
# ---------------------------------------------------------------------------


def build_parser():
    """Legacy compat. Returns None — use main() instead."""
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — called by ``roboflow`` console script."""
    app()
