#!/usr/bin/env python3
"""Backwards-compatibility shim.

The CLI implementation has moved to :mod:`roboflow.cli`.  This module
re-exports ``main`` so that the ``setup.py`` entry-point
(``roboflow=roboflow.roboflowpy:main``) continues to work without changes.
"""

from roboflow.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
