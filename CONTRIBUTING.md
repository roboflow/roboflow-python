
## Contributing

If you want to extend our Python library or if you find a bug, please open a PR!

Also be sure to test your code the `unittest` command at the `/root` level directory.

Run tests:

```bash
python -m unittest
```

When creating new functions, please follow the [Google style Python docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). See example below:

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
