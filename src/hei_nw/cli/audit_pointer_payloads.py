"""Audit gate trace samples and pointer payloads for banned text keys.

The evaluation harness emits optional ``gate.trace_samples`` entries in the
metrics JSON along with pointer-only payloads written by :class:`TraceWriter`.
This script scans those artifacts to ensure that raw text fields such as
``episode_text`` or ``snippet`` do not leak into persisted traces. It reports a
summary for each processed file and exits with a non-zero status when any
violations are detected.

Usage::

    python scripts/audit_pointer_payloads.py reports/m3-write-gate/A_B1_metrics.json

Multiple files or directories may be supplied. Directories are searched
recursively for ``*.json`` files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

_BANNED_TEXT_KEYS = {"episode_text", "raw_text", "snippet", "full_text", "text"}


@dataclass
class _AuditStats:
    """Aggregate statistics for a single JSON artifact."""

    path: Path
    samples_checked: int = 0
    pointer_records_checked: int = 0
    missing_pointer: int = 0
    banned_key_counts: Counter[str] = field(default_factory=Counter)
    pointer_check_missing: int = 0
    pointer_check_banned: set[str] = field(default_factory=set)
    pointer_check_flag: bool | None = None

    @property
    def pointer_only(self) -> bool:
        if self.pointer_check_flag is False:
            return False
        if self.pointer_check_missing:
            return False
        if self.pointer_check_banned:
            return False
        if self.missing_pointer:
            return False
        if self.banned_key_counts:
            return False
        return True


def _iter_input_paths(entries: Iterable[str]) -> Iterator[Path]:
    """Yield JSON files from *entries* (files or directories)."""

    seen: set[Path] = set()
    for raw in entries:
        path = Path(raw)
        if path.is_dir():
            for candidate in sorted(path.rglob("*.json")):
                if candidate.is_file() and candidate not in seen:
                    seen.add(candidate)
                    yield candidate
            continue
        if path.suffix.lower() == ".json" and path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                yield path


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("r", encoding="utf8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        print(f"Failed to parse JSON from {path}: {exc}", file=sys.stderr)
        return None
    if not isinstance(data, Mapping):
        print(f"Skipping {path}: JSON document is not an object", file=sys.stderr)
        return None
    return data


def _audit_pointer_check(data: Mapping[str, Any], stats: _AuditStats) -> None:
    gate = data.get("gate")
    if not isinstance(gate, Mapping):
        return
    pointer_check = gate.get("pointer_check")
    if not isinstance(pointer_check, Mapping):
        return
    pointer_only = pointer_check.get("pointer_only")
    if isinstance(pointer_only, bool):
        stats.pointer_check_flag = pointer_only
    missing = pointer_check.get("missing_pointer")
    if isinstance(missing, int) and missing > 0:
        stats.pointer_check_missing += missing
    banned = pointer_check.get("banned_keys")
    if isinstance(banned, list):
        for key in banned:
            if isinstance(key, str) and key in _BANNED_TEXT_KEYS:
                stats.pointer_check_banned.add(key)
    banned_counts = pointer_check.get("banned_key_counts")
    if isinstance(banned_counts, Mapping):
        for key, value in banned_counts.items():
            if isinstance(key, str) and key in _BANNED_TEXT_KEYS:
                stats.pointer_check_banned.add(key)
                if isinstance(value, int) and value > 0:
                    stats.banned_key_counts[key] += int(value)


def _audit_trace_samples(data: Mapping[str, Any], stats: _AuditStats) -> None:
    samples: list[Mapping[str, Any]] = []
    gate = data.get("gate")
    if isinstance(gate, Mapping):
        gate_samples = gate.get("trace_samples")
        if isinstance(gate_samples, list):
            samples.extend(sample for sample in gate_samples if isinstance(sample, Mapping))
    direct = data.get("trace_samples")
    if isinstance(direct, list):
        samples.extend(sample for sample in direct if isinstance(sample, Mapping))

    for sample in samples:
        stats.samples_checked += 1
        banned = sample.get("banned_keys")
        if isinstance(banned, list):
            for key in banned:
                if isinstance(key, str) and key in _BANNED_TEXT_KEYS:
                    stats.banned_key_counts[key] += 1
        has_pointer = sample.get("has_pointer")
        if has_pointer is False:
            stats.missing_pointer += 1


def _iter_pointer_records(obj: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(obj, Mapping):
        if "tokens_span_ref" in obj and "salience_tags" in obj:
            yield obj
        for value in obj.values():
            yield from _iter_pointer_records(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_pointer_records(item)


def _collect_banned_keys(obj: Mapping[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()

    def _walk(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if key == "banned_keys":
                    continue
                if key in _BANNED_TEXT_KEYS and nested:
                    counts[key] += 1
                _walk(nested)
        elif isinstance(value, list):
            for item in value:
                _walk(item)

    _walk(obj)
    return counts


def _pointer_is_valid(pointer: Any) -> bool:
    if not isinstance(pointer, Mapping):
        return False
    doc = pointer.get("doc")
    start_raw = pointer.get("start")
    end_raw = pointer.get("end")
    if not isinstance(doc, str) or not doc.strip():
        return False
    if start_raw is None or end_raw is None:
        return False
    try:
        start_val = int(cast(int | float | str, start_raw))
        end_val = int(cast(int | float | str, end_raw))
    except (TypeError, ValueError):
        return False
    if start_val < 0 or end_val <= start_val:
        return False
    return True


def _audit_pointer_records(data: Mapping[str, Any], stats: _AuditStats) -> None:
    for record in _iter_pointer_records(data):
        stats.pointer_records_checked += 1
        pointer = record.get("tokens_span_ref")
        if not _pointer_is_valid(pointer):
            stats.missing_pointer += 1
        counts = _collect_banned_keys(record)
        for key, value in counts.items():
            stats.banned_key_counts[key] += value


def _format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "{}"
    parts = [f"{key}={counter[key]}" for key in sorted(counter)]
    return "{" + ", ".join(parts) + "}"


def audit_path(path: Path) -> _AuditStats | None:
    data = _load_json(path)
    if data is None:
        return None
    stats = _AuditStats(path=path)
    _audit_pointer_check(data, stats)
    _audit_trace_samples(data, stats)
    _audit_pointer_records(data, stats)
    return stats


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit trace samples and pointer payloads for banned text keys",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Metrics JSON files, pointer payload dumps, or directories containing them",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    results: list[_AuditStats] = []
    for json_path in _iter_input_paths(args.paths):
        stats = audit_path(json_path)
        if stats is None:
            continue
        results.append(stats)
        pointer_only = str(stats.pointer_only).lower()
        details = (
            f"{json_path}: pointer_only={pointer_only}, "
            f"samples={stats.samples_checked}, "
            f"pointer_records={stats.pointer_records_checked}, "
            f"missing_pointer={stats.missing_pointer}, "
            f"banned_keys={_format_counter(stats.banned_key_counts)}, "
            f"pointer_check_banned={sorted(stats.pointer_check_banned)}"
        )
        print(details)

    if not results:
        print("No JSON artifacts found for auditing.")
        return 0

    total_samples = sum(stats.samples_checked for stats in results)
    total_records = sum(stats.pointer_records_checked for stats in results)
    total_missing = sum(stats.missing_pointer for stats in results)
    total_pointer_check_missing = sum(stats.pointer_check_missing for stats in results)
    aggregate_banned: Counter[str] = Counter()
    aggregate_pointer_check_banned: set[str] = set()
    for stats in results:
        aggregate_banned.update(stats.banned_key_counts)
        aggregate_pointer_check_banned.update(stats.pointer_check_banned)
    overall_pointer_only = all(stats.pointer_only for stats in results)

    summary_line = (
        "Summary pointer_only: "
        f"{str(overall_pointer_only).lower()}, samples={total_samples}, "
        f"pointer_records={total_records}, missing_pointer={total_missing}, "
        f"pointer_check_missing={total_pointer_check_missing}, "
        f"banned_keys={_format_counter(aggregate_banned)}, "
        f"pointer_check_banned={sorted(aggregate_pointer_check_banned)}"
    )
    print(summary_line)
    return 0 if overall_pointer_only else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
