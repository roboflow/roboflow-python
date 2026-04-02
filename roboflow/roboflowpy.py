#!/usr/bin/env python3
"""Backwards-compatibility shim.

The CLI implementation has moved to :mod:`roboflow.cli`.  This module
re-exports ``main`` so that the ``setup.py`` entry-point
(``roboflow=roboflow.roboflowpy:main``) continues to work without changes.

It also re-exports legacy function names so that existing scripts doing
``from roboflow.roboflowpy import _argparser`` (etc.) continue to work.
"""

from roboflow.cli import build_parser, main

# Legacy alias: some scripts import _argparser directly
_argparser = build_parser

__all__ = ["main", "_argparser"]

if __name__ == "__main__":
    main()
