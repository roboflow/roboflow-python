
## Contributing

If you want to extend our Python library or if you find a bug, please open a PR!

Also be sure to test your code with the `unittest` command at the `/root` level directory.

## Installation for Contributors

Before starting your work on the project, set up your development environment:

1. Clone the repository:
```bash
git clone https://github.com/roboflow-ai/roboflow-python.git
cd roboflow-python
```

2. Create and activate a virtual environment:
```bash
python3 -m venv env
source env/bin/activate
```

3.Install the package in editable mode with development dependencies:
```bash
pip install -e ".[dev]"
```

### Devcontainer

This project comes with a [convenient devcontainer](https://www.loom.com/share/a183c4a351ed4700a79476fedf08ab9b) that makes it easier to run tests and has lint configured to run on save.

On rare occasions a full rebuild is needed, you can do it in VSCode by pressing `Ctrl+Shift+P` and running `Dev Containers: Rebuild Container`.

### Tests

```bash
python -m unittest
```

When creating new functions, please follow the [Google style for Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). See example below:

```python
def example_function(param1: int, param2: str) -> bool:
    """Example function that does something.

    Args:
        param1: The first parameter.
        param2: The second parameter.

    Returns:
        The return value. True for success, False otherwise.

    """
```

We provide a `Makefile` to format and ensure code quality. **Be sure to run them before creating a PR**.

```bash
# format code with `ruff`
make style

# check code with `ruff`
make check_code_quality
```

**Note** These tests will be run automatically when you commit thanks to git hooks.

### Docs

The docs can be built with `mkdocs serve`.

Before that, install the dependencies:

```python
python -m pip install mkdocs mkdocs-material mkdocstrings mkdocstrings[python]
```

### CLI Development

The CLI is built on [typer](https://typer.tiangolo.com/). Each command group is a separate `typer.Typer()` app registered in `roboflow/cli/__init__.py`. To add a new command:

1. Create `roboflow/cli/handlers/mycommand.py`:

```python
"""My command description."""
from __future__ import annotations
from typing import Annotated, Optional
import typer
from roboflow.cli._compat import ctx_to_args

mycommand_app = typer.Typer(help="Do something", no_args_is_help=True)

@mycommand_app.command("list")
def list_things(
    ctx: typer.Context,
    project: Annotated[str, typer.Option("-p", "--project", help="Project ID")],
) -> None:
    """List things in a project."""
    args = ctx_to_args(ctx, project=project)
    _list(args)

def _list(args) -> None:
    from roboflow.cli._output import output, output_error, suppress_sdk_output

    with suppress_sdk_output():
        try:
            # ... your logic here ...
            data = [{"id": "example"}]
        except Exception as exc:
            output_error(args, str(exc), hint="Check your project ID.", exit_code=3)
            return

    output(args, data, text="Found 1 result.")
```

2. Register in `roboflow/cli/__init__.py`:
```python
from roboflow.cli.handlers.mycommand import mycommand_app
app.add_typer(mycommand_app, name="mycommand")
```

3. Add tests using `typer.testing.CliRunner` in `tests/cli/test_mycommand_handler.py`
4. Run `make check_code_quality` and `python -m unittest`

**Agent experience checklist** (every command must satisfy):
- [ ] Supports `--json` via `output()` helper
- [ ] No interactive prompts when all required flags are provided
- [ ] Errors use `output_error(args, message, hint=..., exit_code=N)`
- [ ] SDK calls wrapped in `with suppress_sdk_output():`
- [ ] Exit codes: 0=success, 1=error, 2=auth, 3=not found

**Documentation policy:** `CLI-COMMANDS.md` in this repo is a quickstart only. The comprehensive command reference lives in [`roboflow-dev-reference`](https://github.com/roboflow/roboflow-dev-reference) and is published to docs.roboflow.com/developer/command-line-interface. When adding a new command, update both: add a quick example to `CLI-COMMANDS.md` and the full reference to the dev-reference CLI page.

### Pre-commit Hooks

To ensure code quality and consistency, we use pre-commit hooks. Follow these steps to set up pre-commit in your development environment:

1. Install pre-commit:
```bash
pip install pre-commit
```

2. Install the git hook scripts:
```bash
pre-commit install
```

After installation, `pre-commit` will automatically run on git commit. The hooks perform checks and corrections related to code formatting, linting, and other rules as defined in the `.pre-commit-config.yaml` file.

Note: If you need to bypass pre-commit hooks temporarily, you can use the `--no-verify` flag:

```bash
git commit --no-verify -m "Your commit message"
```
