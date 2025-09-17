from __future__ import annotations

from hei_nw.models.base import _truncate_at_stop


def test_truncate_handles_none_stop() -> None:
    text, truncated = _truncate_at_stop("answer", None)
    assert text == "answer"
    assert truncated is False


def test_truncate_no_match_returns_original() -> None:
    text, truncated = _truncate_at_stop("alpha beta", "<stop>")
    assert text == "alpha beta"
    assert truncated is False


def test_truncate_at_start() -> None:
    text, truncated = _truncate_at_stop("END of line", "END")
    assert text == ""
    assert truncated is True


def test_truncate_multiple_occurrences() -> None:
    original = "first\nsecond\nthird"
    text, truncated = _truncate_at_stop(original, "\n")
    assert text == "first"
    assert truncated is True


def test_truncate_unicode_stop() -> None:
    text, truncated = _truncate_at_stop("Café ☕ break", "☕")
    assert text == "Café "
    assert truncated is True


def test_truncate_shorter_output_when_stop_present() -> None:
    original = "value<<STOP>>trailing"
    text, truncated = _truncate_at_stop(original, "<<STOP>>")
    assert truncated is True
    assert len(text) < len(original)
