"""Handler modules for the Roboflow CLI.

Each module in this package that exposes a ``register(subparsers)`` function
is auto-discovered and loaded by ``roboflow.cli.build_parser()``.

Modules whose names start with ``_`` (e.g. ``_aliases.py``) are *not*
auto-discovered — they are loaded explicitly after all other handlers.
"""
