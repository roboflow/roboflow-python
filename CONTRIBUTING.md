
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
