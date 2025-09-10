from datetime import datetime
from pathlib import Path

from hei_nw.utils.io import read_json, timestamp_slug, write_json, write_markdown


def test_write_and_read_json_roundtrip(tmp_path: Path) -> None:
    data = {"a": 1, "b": [1, 2]}
    path = tmp_path / "data.json"
    write_json(path, data)
    assert read_json(path) == data


def test_write_markdown_and_timestamp(tmp_path: Path) -> None:
    content = "# Title"
    path = tmp_path / "doc.md"
    write_markdown(path, content)
    assert path.read_text(encoding="utf-8") == content

    slug = timestamp_slug(datetime(2020, 1, 2, 3, 4, 5))
    assert slug == "20200102-030405"
