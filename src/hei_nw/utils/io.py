"""I/O helpers for JSON and Markdown files."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def timestamp_slug(ts: datetime | None = None) -> str:
    """Return a filesystem-friendly timestamp slug."""
    ts = ts or datetime.now()
    return ts.strftime("%Y%m%d-%H%M%S")


def write_json(path: str | Path, data: Any) -> None:
    """Write *data* as JSON to *path*, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)


def read_json(path: str | Path) -> Any:
    """Read JSON data from *path*."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_markdown(path: str | Path, content: str) -> None:
    """Write Markdown *content* to *path*, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content)
