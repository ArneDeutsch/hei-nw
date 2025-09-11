"""Reporting utilities for evaluation results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hei_nw.utils.io import write_json, write_markdown


def build_markdown_report(summary: dict[str, Any]) -> str:
    """Build a Markdown report string from *summary* data."""

    agg = summary.get("aggregate", {})
    lines = ["# Evaluation Report", "", "## Aggregate Metrics", ""]
    lines.append(f"- EM: {agg.get('em', 0):.3f}")
    lines.append(f"- F1: {agg.get('f1', 0):.3f}")
    lines.append(f"- Latency: {agg.get('latency', 0):.3f}s")
    lines.append("")
    lines.append("## Lag bins")
    lines.append("| lag | count | EM | F1 |")
    lines.append("| --- | ----- | --- | --- |")
    for bin_ in summary.get("lag_bins", []):
        lines.append(f"| {bin_['lag']} | {bin_['count']} | {bin_['em']:.3f} | {bin_['f1']:.3f} |")
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
    return "\n".join(lines)


def save_reports(outdir: Path, base: str, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write JSON and Markdown reports and return their paths."""

    json_path = outdir / f"{base}_metrics.json"
    md_path = outdir / f"{base}_report.md"
    write_json(json_path, summary)
    write_markdown(md_path, build_markdown_report(summary))
    return json_path, md_path
