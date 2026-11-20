"""Plain-text tables: aligned columns, a hairline under the header, nothing else."""

from __future__ import annotations

from collections.abc import Sequence


def format_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Return ``headers`` and ``rows`` as aligned text. Numbers right-align, text left-aligns."""
    cells = [[str(h) for h in headers]] + [[str(c) for c in row] for row in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    numeric = [all(_looks_numeric(r[i]) for r in cells[1:]) if len(cells) > 1 else False for i in range(len(headers))]

    def line(r: Sequence[str]) -> str:
        return "  ".join(c.rjust(widths[i]) if numeric[i] else c.ljust(widths[i]) for i, c in enumerate(r)).rstrip()

    rule = "  ".join("-" * w for w in widths)
    return "\n".join([line(cells[0]), rule] + [line(r) for r in cells[1:]])


def _looks_numeric(text: str) -> bool:
    """Return whether a cell is a number, an interval ``a [b, c]`` or ``n/a``."""
    if text == "n/a":
        return True
    head = text.split(" [", 1)[0].replace(",", "")
    try:
        float(head)
    except ValueError:
        return False
    return True
