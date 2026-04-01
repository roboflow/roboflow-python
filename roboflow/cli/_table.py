"""Simple table formatter for CLI list commands.

No external dependency — uses plain string formatting.  Respects terminal
width when available and truncates long fields.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional, Sequence


def format_table(
    rows: Sequence[Dict[str, Any]],
    columns: Sequence[str],
    headers: Optional[Sequence[str]] = None,
    max_width: Optional[int] = None,
) -> str:
    """Format a list of dicts as a columnar table.

    Parameters
    ----------
    rows:
        Each row is a dict whose keys match *columns*.
    columns:
        Ordered list of dict keys to include as columns.
    headers:
        Display names for each column.  Defaults to *columns* with
        title-casing and hyphens replaced by spaces.
    max_width:
        Terminal width cap.  ``None`` means auto-detect.

    Returns
    -------
    str
        The formatted table string (without trailing newline).
    """
    if not rows:
        return "(no results)"

    if headers is None:
        headers = [c.replace("_", " ").replace("-", " ").upper() for c in columns]

    # Stringify all cell values
    str_rows: List[List[str]] = []
    for row in rows:
        str_rows.append([str(row.get(c, "")) for c in columns])

    # Compute column widths
    col_widths = [len(h) for h in headers]
    for sr in str_rows:
        for i, cell in enumerate(sr):
            col_widths[i] = max(col_widths[i], len(cell))

    # Optionally clamp to terminal width
    if max_width is None:
        max_width = shutil.get_terminal_size((120, 24)).columns
    # Leave room for column separators (2 spaces between columns)
    total = sum(col_widths) + 2 * (len(columns) - 1)
    if total > max_width and len(columns) > 1:
        # Shrink the widest column proportionally
        excess = total - max_width
        widest_idx = col_widths.index(max(col_widths))
        col_widths[widest_idx] = max(col_widths[widest_idx] - excess, 10)

    def _truncate(s: str, width: int) -> str:
        return s if len(s) <= width else s[: width - 1] + "\u2026"

    # Build lines
    lines: list[str] = []
    header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    lines.append(header_line)
    lines.append("  ".join("-" * col_widths[i] for i in range(len(columns))))
    for sr in str_rows:
        line = "  ".join(_truncate(sr[i], col_widths[i]).ljust(col_widths[i]) for i in range(len(columns)))
        lines.append(line)

    return os.linesep.join(lines)
