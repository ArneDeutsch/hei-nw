"""Reporting utilities for evaluation results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hei_nw.utils.io import write_json, write_markdown


def bin_by_lag(records: Sequence[dict[str, Any]], bins: Sequence[int]) -> list[dict[str, Any]]:
    """Aggregate *records* into lag bins.

    Parameters
    ----------
    records:
        Sequence of evaluation record dictionaries each containing ``lag``,
        ``em``, ``f1`` and optional ``recall_at_k`` fields.
    bins:
        Monotonically increasing sequence of integer bin edges.

    Returns
    -------
    list of dicts
        Each dict contains ``lag_bin`` label, ``count`` of records in the bin,
        mean ``em``/``f1``/``recall_at_k``.
    """

    if len(bins) < 2:
        raise ValueError("bins must have at least two entries")
    results: list[dict[str, Any]] = []
    for start, end in zip(bins, bins[1:], strict=False):
        members = [r for r in records if start <= int(r.get("lag", 0)) < end]
        count = len(members)
        em = sum(float(r.get("em", 0.0)) for r in members) / count if count else 0.0
        f1 = sum(float(r.get("f1", 0.0)) for r in members) / count if count else 0.0
        recalls = [
            float(r.get("recall_at_k", 0.0)) for r in members if r.get("recall_at_k") is not None
        ]
        recall = sum(recalls) / len(recalls) if recalls else None
        label = f"{start}-{end}"
        results.append(
            {"lag_bin": label, "count": count, "em": em, "f1": f1, "recall_at_k": recall}
        )
    return results


def build_markdown_report(summary: dict[str, Any], scenario: str | None = None) -> str:
    """Build a Markdown report string from *summary* data."""

    agg = summary.get("aggregate", {})
    lines = ["# Evaluation Report", "", "## Aggregate Metrics", ""]
    lines.append(f"- EM: {agg.get('em', 0):.3f}")
    lines.append(f"- F1: {agg.get('f1', 0):.3f}")
    lines.append(f"- Latency: {agg.get('latency', 0):.3f}s")
    overhead = summary.get("adapter_latency_overhead_s")
    if overhead is not None:
        lines.append(f"- Adapter latency overhead: {overhead:.3f}s")
    lines.append("")
    lines.append("## Lag bins")
    lines.append("| Lag bin | count | EM | F1 | Recall@k |")
    lines.append("| ------- | ----- | --- | --- | -------- |")
    for bin_ in summary.get("lag_bins", []):
        r = bin_.get("recall_at_k")
        r_str = f"{r:.3f}" if isinstance(r, int | float) else "n/a"
        line = (
            f"| {bin_['lag_bin']} | {bin_['count']} | {bin_['em']:.3f} | "
            f"{bin_['f1']:.3f} | {r_str} |"
        )
        lines.append(line)
    lines.append("")
    lines.append("## Compute")
    comp = summary.get("compute", {})
    b0 = comp.get("b0", {})
    lines.append(
        f"B0 attention FLOPs: {b0.get('attention_flops', 'n/a')}\n"
        f"B0 KV cache bytes: {b0.get('kv_cache_bytes', 'n/a')}"
    )
    baseline = comp.get("baseline")
    if baseline:
        lines.append(
            f"\nBaseline attention FLOPs: {baseline.get('attention_flops', 'n/a')}\n"
            f"Baseline KV cache bytes: {baseline.get('kv_cache_bytes', 'n/a')}"
        )
    lines.append("")
    lines.append("## Retrieval")
    retrieval = summary.get("retrieval")
    if retrieval:
        lines.append(f"- P@1: {retrieval.get('p_at_1', 0):.3f}")
        lines.append(f"- MRR: {retrieval.get('mrr', 0):.3f}")
        lines.append(f"- Near-miss rate: {retrieval.get('near_miss_rate', 0):.3f}")
        lines.append(f"- Collision rate: {retrieval.get('collision_rate', 0):.3f}")
        lines.append(f"- Completion lift: {retrieval.get('completion_lift', 0):.3f}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("## Dataset notes")
    if scenario == "A":
        dataset = summary.get("dataset", {})
        ratio = dataset.get("hard_negative_ratio")
        ratio_str = f"{ratio:.2f}" if isinstance(ratio, int | float) else "unknown"
        lines.append(f"Hard negatives/confounders included (ratio {ratio_str})")
    else:
        lines.append("None")
    return "\n".join(lines)


def save_reports(
    outdir: Path, base: str, summary: dict[str, Any], scenario: str | None = None
) -> tuple[Path, Path]:
    """Write JSON and Markdown reports and return their paths."""

    json_path = outdir / f"{base}_metrics.json"
    md_path = outdir / f"{base}_report.md"
    write_json(json_path, summary)
    write_markdown(md_path, build_markdown_report(summary, scenario))
    return json_path, md_path
