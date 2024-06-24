
## Contributing

If you want to extend our Python library or if you find a bug, please open a PR!

Also be sure to test your code with the `unittest` command at the `/root` level directory.

### Devcontainer

This project comes with a [convenient devcontainer](https://www.loom.com/share/a183c4a351ed4700a79476fedf08ab9b) that makes it easier to run tests and has black configured to run on save.

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
# format code with `black` and `isort`
make style

# check code with flake8
make check_code_quality
```

**Note** These tests will be run automatically when you commit thanks to git hooks.

### Docs

The docs can be built with `mkdocs serve`.

Before that, install the dependencies:

```python
python -m pip install mkdocs mkdocs-material mkdocstrings mkdocstrings[python]
```
