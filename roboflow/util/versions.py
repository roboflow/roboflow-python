import sys
from importlib import import_module
from typing import List, Tuple

from packaging.version import Version


def get_wrong_dependencies_versions(
    dependencies_versions: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str, str]]:
    """
    Get a list of mismatching dependencies with current version installed.
    E.g., assuming we pass `get_wrong_dependencies_versions([("torch", "==", "1.2.0")]),
        we will check if the current version of `torch` is `==1.2.0`. If not,
        we will return `[("torch", "==", "1.2.0", "<current_installed_version>")]

    We support `<=`, `==`, `>=`

    Args:
        dependencies_versions (List[Tuple[str, str]]): List of dependencies
            we want to check, [("<package_name>", "<version_number_to_check")]

    Returns:
        List[Tuple[str, str, str]]: List of dependencies with wrong version,
            [("<package_name>", "<version_number_to_check", "<current_version>")]
    """
    wrong_dependencies_versions = []
    order_funcs = {
        "==": lambda x, y: x == y,
        ">=": lambda x, y: x >= y,
        "<=": lambda x, y: x <= y,
    }
    for dependency, order, version in dependencies_versions:
        module = import_module(dependency)
        module_version = module.__version__
        if order not in order_funcs:
            raise ValueError(f"order={order} not supported, please use" f" `{', '.join(order_funcs.keys())}`")

        is_okay = order_funcs[order](Version(module_version), Version(version))
        if not is_okay:
            wrong_dependencies_versions.append((dependency, order, version, module_version))
    return wrong_dependencies_versions


def print_warn_for_wrong_dependencies_versions(
    dependencies_versions: List[Tuple[str, str, str]], ask_to_continue: bool = False
):
    wrong_dependencies_versions = get_wrong_dependencies_versions(dependencies_versions)
    for dependency, order, version, module_version in wrong_dependencies_versions:
        print(
            f"Dependency {dependency}{order}{version} is required but found"
            f" version={module_version}, to fix: `pip install"
            f" {dependency}{order}{version}`"
        )
        if ask_to_continue:
            answer = input(f"Would you like to continue with the wrong version of {dependency}?" " y/n: ")
            if answer.lower() != "y":
                sys.exit(1)


def warn_for_wrong_dependencies_versions(dependencies_versions: List[Tuple[str, str, str]]):
    """
    Decorator to print a warning based on dependencies versions. E.g.

    ```python
    @warn_for_wrong_dependencies_versions([("torch", "==", "1.2.0")])
    def foo(x):
        # I only work with torch `1.2.0` but another one is installed
        print(f"foo {x}")
    ```

    prints:

    ```
    Dependency torch==1.2.0 is required but found version=1.13.1,
        to fix: `pip install torch==1.2.0`
    ```

    Args:
        dependencies_versions (List[Tuple[str, str]]): List of dependencies
            we want to check, [("<package_name>", "<version_number_to_check")]
    """

    def _inner(func):
        def _wrapper(*args, **kwargs):
            print_warn_for_wrong_dependencies_versions(dependencies_versions)
            func(*args, **kwargs)

        return _wrapper

    return _inner
