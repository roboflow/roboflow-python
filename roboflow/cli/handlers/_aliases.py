"""Top-level backwards-compatibility aliases.

Split into three registration functions called at different points in
``__init__.py`` to control help ordering:

- ``register_download_alias(app)`` — visible ``download`` command (alphabetical slot)
- ``register_hidden_aliases(app)`` — all hidden aliases (loaded last)
"""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from roboflow.cli._compat import ctx_to_args


def register_hidden_aliases(app: typer.Typer) -> None:
    """Register all hidden backwards-compat aliases (not shown in --help)."""

    @app.command("download", hidden=True)
    def download_alias(
        ctx: typer.Context,
        url_or_id: Annotated[
            str, typer.Argument(metavar="datasetUrl", help="Dataset URL (e.g. workspace/project/version)")
        ],
        format: Annotated[str, typer.Option("-f", "--format", help="Export format")] = "voc",
        location: Annotated[Optional[str], typer.Option("-l", "--location", help="Download location")] = None,
    ) -> None:
        """Download a dataset version (alias for 'version download')."""
        from roboflow.cli.handlers.version import _download

        args = ctx_to_args(ctx, url_or_id=url_or_id, format=format, location=location)
        _download(args)

    @app.command("login", hidden=True)
    def login_alias(
        ctx: typer.Context,
        login_api_key: Annotated[
            Optional[str], typer.Option("--api-key", help="API key (skip interactive login)")
        ] = None,
        force: Annotated[bool, typer.Option("--force", "-f", help="Force re-login")] = False,
    ) -> None:
        """Log in to Roboflow (alias for 'auth login')."""
        from roboflow.cli.handlers.auth import _login

        args = ctx_to_args(ctx, login_api_key=login_api_key, force=force)
        _login(args)

    @app.command("whoami", hidden=True)
    def whoami_alias(ctx: typer.Context) -> None:
        """Show current user (alias for 'auth status')."""
        from roboflow.cli.handlers.auth import _status

        args = ctx_to_args(ctx)
        _status(args)

    @app.command("upload", hidden=True)
    def upload_alias(
        ctx: typer.Context,
        path: Annotated[str, typer.Argument(help="Path to image file or directory")],
        project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
        annotation: Annotated[Optional[str], typer.Option("-a", "--annotation", help="Annotation file")] = None,
        labelmap: Annotated[Optional[str], typer.Option("-m", "--labelmap", help="Labelmap file")] = None,
        split: Annotated[str, typer.Option("-s", "--split", help="Split (train/valid/test)")] = "train",
        num_retries: Annotated[int, typer.Option("-r", "--retries", help="Retry count")] = 0,
        batch: Annotated[Optional[str], typer.Option("-b", "--batch", help="Batch name")] = None,
        tag_names: Annotated[Optional[str], typer.Option("-t", "--tag", help="Tag names")] = None,
        metadata: Annotated[Optional[str], typer.Option("-M", "--metadata", help="JSON metadata")] = None,
        concurrency: Annotated[int, typer.Option("-c", "--concurrency", help="Concurrency")] = 10,
        is_prediction: Annotated[bool, typer.Option("--is-prediction", help="Mark as prediction")] = False,
    ) -> None:
        """Upload images to a project (alias for 'image upload')."""
        from roboflow.cli.handlers.image import _handle_upload

        args = ctx_to_args(
            ctx,
            path=path,
            project=project,
            annotation=annotation,
            labelmap=labelmap,
            split=split,
            num_retries=num_retries,
            batch=batch,
            tag_names=tag_names,
            metadata=metadata,
            concurrency=concurrency,
            is_prediction=is_prediction,
        )
        _handle_upload(args)

    @app.command("import", hidden=True)
    def import_alias(
        ctx: typer.Context,
        path: Annotated[str, typer.Argument(metavar="folder", help="Path to dataset folder")],
        project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
        concurrency: Annotated[int, typer.Option("-c", "--concurrency", help="Concurrency")] = 10,
        batch: Annotated[Optional[str], typer.Option("-n", "--batch-name", help="Batch name")] = None,
        num_retries: Annotated[int, typer.Option("-r", "--retries", help="Retry count")] = 0,
    ) -> None:
        """Import dataset from folder (alias for 'image upload')."""
        from roboflow.cli.handlers.image import _handle_upload

        args = ctx_to_args(
            ctx, path=path, project=project, concurrency=concurrency, batch=batch, num_retries=num_retries
        )
        _handle_upload(args)

    @app.command("search-export", hidden=True)
    def search_export_alias(
        ctx: typer.Context,
        query: Annotated[str, typer.Argument(help="Search query")],
        format: Annotated[str, typer.Option("-f", help="Format")] = "coco",
        location: Annotated[Optional[str], typer.Option("-l", help="Export location")] = None,
        dataset: Annotated[Optional[str], typer.Option("-d", help="Limit to dataset")] = None,
        annotation_group: Annotated[Optional[str], typer.Option("-g", help="Annotation group")] = None,
        name: Annotated[Optional[str], typer.Option("-n", help="Export name")] = None,
        no_extract: Annotated[bool, typer.Option("--no-extract", help="Keep zip")] = False,
    ) -> None:
        """Export search results as a dataset."""
        from roboflow.cli.handlers.search import _search

        args = ctx_to_args(
            ctx,
            query=query,
            format=format,
            location=location,
            dataset=dataset,
            annotation_group=annotation_group,
            name=name,
            no_extract=no_extract,
            export=True,
        )
        _search(args)

    @app.command("upload_model", hidden=True)
    def upload_model_alias(
        ctx: typer.Context,
        project: Annotated[Optional[list[str]], typer.Option("-p", help="Project ID (repeatable)")] = None,
        version_number: Annotated[Optional[int], typer.Option("-v", help="Version")] = None,
        model_type: Annotated[Optional[str], typer.Option("-t", help="Model type")] = None,
        model_path: Annotated[Optional[str], typer.Option("-m", help="Model path")] = None,
        filename: Annotated[str, typer.Option("-f", help="Filename")] = "weights/best.pt",
        model_name: Annotated[Optional[str], typer.Option("-n", help="Model name")] = None,
    ) -> None:
        """Upload a model (hidden legacy alias)."""
        from roboflow.cli.handlers.model import _upload_model

        args = ctx_to_args(
            ctx,
            project=project,
            version_number=version_number,
            model_type=model_type,
            model_path=model_path,
            filename=filename,
            model_name=model_name,
        )
        _upload_model(args)

    @app.command("get_workspace_info", hidden=True)
    def get_workspace_info_alias(
        ctx: typer.Context,
        project: Annotated[Optional[str], typer.Option("-p", help="Project ID")] = None,
        version_number: Annotated[Optional[int], typer.Option("-v", help="Version")] = None,
    ) -> None:
        """Get workspace info (hidden legacy alias)."""
        import roboflow as rf_mod

        args = ctx_to_args(ctx, project=project, version_number=version_number)
        rf_obj = rf_mod.Roboflow(args.api_key)
        workspace = rf_obj.workspace()
        print("workspace", workspace)  # noqa: T201
        proj = workspace.project(args.project)
        print("project", proj)  # noqa: T201
        ver = proj.version(args.version_number)
        print("version", ver)  # noqa: T201

    @app.command("run_video_inference_api", hidden=True)
    def run_video_inference_api_alias(
        ctx: typer.Context,
        project: Annotated[Optional[str], typer.Option("-p", help="Project ID")] = None,
        version_number: Annotated[Optional[int], typer.Option("-v", help="Version")] = None,
        video_file: Annotated[Optional[str], typer.Option("-f", help="Video file")] = None,
        fps: Annotated[int, typer.Option("-fps", help="FPS")] = 5,
    ) -> None:
        """Run video inference (hidden legacy alias)."""
        from roboflow.cli.handlers.video import _video_infer

        args = ctx_to_args(ctx, project=project, version_number=version_number, video_file=video_file, fps=fps)
        _video_infer(args)
