"""Reporting utilities for evaluation results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from hei_nw.utils.io import write_json, write_markdown


def bin_by_lag(records: Sequence[dict[str, Any]], bins: Sequence[int]) -> list[dict[str, Any]]:
    """Aggregate *records* into lag bins.

    Parameters
    ----------
    records:
        Sequence of evaluation record dictionaries each containing ``lag``,
        ``em_relaxed``/``em_strict`` (or legacy ``em``), ``f1`` and optional
        ``recall_at_k`` fields.
    bins:
        Monotonically increasing sequence of integer bin edges.

    Returns
    -------
    list of dicts
        Each dict contains ``lag_bin`` label, ``count`` of records in the bin,
        mean ``em_relaxed``/``em_strict``/``f1``/``recall_at_k``.
    """

    if len(bins) < 2:
        raise ValueError("bins must have at least two entries")
    results: list[dict[str, Any]] = []
    for start, end in zip(bins, bins[1:], strict=False):
        members = [r for r in records if start <= int(r.get("lag", 0)) < end]
        count = len(members)
        em_relaxed_vals = [float(r.get("em_relaxed", r.get("em", 0.0))) for r in members]
        em_strict_vals = [float(r.get("em_strict", r.get("em", 0.0))) for r in members]
        em_relaxed = sum(em_relaxed_vals) / count if count else 0.0
        em_strict = sum(em_strict_vals) / count if count else 0.0
        f1 = sum(float(r.get("f1", 0.0)) for r in members) / count if count else 0.0
        recalls = [
            float(r.get("recall_at_k", 0.0)) for r in members if r.get("recall_at_k") is not None
        ]
        recall = sum(recalls) / len(recalls) if recalls else None
        label = f"{start}-{end}"
        results.append(
            {
                "lag_bin": label,
                "count": count,
                "em": em_relaxed,
                "em_relaxed": em_relaxed,
                "em_strict": em_strict,
                "f1": f1,
                "recall_at_k": recall,
            }
        )
    return results


def build_markdown_report(summary: dict[str, Any], scenario: str | None = None) -> str:
    """Build a Markdown report string from *summary* data."""

    def _format_value(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.3f}"
        if isinstance(value, str):
            if not value:
                return '""'
            if any(ch in value for ch in ("\n", "\r")):
                return repr(value)
            return value
        return str(value)

    agg = summary.get("aggregate", {})
    lines = ["# Evaluation Report", "", "## Aggregate Metrics", ""]
    em_relaxed = float(agg.get("em_relaxed", agg.get("em", 0)))
    em_strict = float(agg.get("em_strict", agg.get("em", 0)))
    lines.append(f"- EM (relaxed): {em_relaxed:.3f}")
    lines.append(f"- EM_strict: {em_strict:.3f}")
    lines.append(f"- F1: {agg.get('f1', 0):.3f}")
    non_empty_rate = float(agg.get("non_empty_rate", 0.0))
    lines.append(f"- Non-empty rate: {non_empty_rate:.3f}")
    lines.append(f"- Latency: {agg.get('latency', 0):.3f}s")
    overhead = summary.get("adapter_latency_overhead_s")
    if overhead is not None:
        lines.append(f"- Adapter latency overhead: {overhead:.3f}s")
    lines.append("")
    run_cfg = summary.get("run")
    if isinstance(run_cfg, dict):
        lines.append("## Run config")
        lines.append("")
        lines.append(f"- Mode: {_format_value(run_cfg.get('mode'))}")
        lines.append(f"- Scenario: {_format_value(run_cfg.get('scenario'))}")
        lines.append(f"- Seed: {_format_value(run_cfg.get('seed'))}")
        lines.append(f"- Requested N: {_format_value(run_cfg.get('requested_n'))}")
        lines.append(f"- Actual records: {_format_value(run_cfg.get('actual_records'))}")
        lines.append(f"- Model: {_format_value(run_cfg.get('model_id'))}")
        lines.append(f"- Baseline: {_format_value(run_cfg.get('baseline'))}")
        qa_cfg = run_cfg.get("qa") if isinstance(run_cfg.get("qa"), dict) else {}
        lines.append(f"- QA.prompt_style: {_format_value(qa_cfg.get('prompt_style'))}")
        lines.append(f"- QA.max_new_tokens: {_format_value(qa_cfg.get('max_new_tokens'))}")
        lines.append(f"- QA.stop: {_format_value(qa_cfg.get('stop'))}")
        lines.append(f"- QA.answer_hint: {_format_value(qa_cfg.get('answer_hint'))}")
        mem_cap = run_cfg.get("mem_max_tokens")
        lines.append(f"- mem_max_tokens: {_format_value(mem_cap)}")
        hop_cfg = run_cfg.get("hopfield") if isinstance(run_cfg.get("hopfield"), dict) else {}
        lines.append(
            "- Hopfield steps/temperature: "
            f"{_format_value(hop_cfg.get('steps'))} / {_format_value(hop_cfg.get('temperature'))}"
        )
        no_hopfield = run_cfg.get("no_hopfield")
        hop_enabled = None if no_hopfield is None else not bool(no_hopfield)
        lines.append(f"- Hopfield enabled: {_format_value(hop_enabled)}")
        lines.append("")
    lines.append("## Lag bins")
    lines.append("| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |")
    lines.append("| ------- | ----- | ------------- | --------- | --- | -------- |")
    for bin_ in summary.get("lag_bins", []):
        r = bin_.get("recall_at_k")
        r_str = f"{r:.3f}" if isinstance(r, int | float) else "n/a"
        em_relaxed_bin = float(bin_.get("em_relaxed", bin_.get("em", 0)))
        em_strict_bin = float(bin_.get("em_strict", bin_.get("em", 0)))
        line = (
            f"| {bin_['lag_bin']} | {bin_['count']} | {em_relaxed_bin:.3f} | "
            f"{em_strict_bin:.3f} | {bin_['f1']:.3f} | {r_str} |"
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
    lines.append("## Debug")
    debug = summary.get("debug")
    if debug:
        mem_len = debug.get("mem_len")
        if isinstance(mem_len, Sequence) and not isinstance(mem_len, str | bytes):
            mem_len_str = ", ".join(str(v) for v in mem_len)
            lines.append(f"- Memory token counts: [{mem_len_str}]")
        else:
            lines.append(f"- Memory token counts: {mem_len}")
        preview = debug.get("mem_preview") or []
        if isinstance(preview, Sequence) and not isinstance(preview, str | bytes):
            preview_str = ", ".join(str(tok) for tok in preview)
            lines.append(f"- Memory token preview: [{preview_str}]")
        else:
            lines.append(f"- Memory token preview: {preview}")
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


def save_completion_ablation_plot(
    outdir: Path,
    with_hopfield: dict[str, Any],
    without_hopfield: dict[str, Any],
) -> Path:
    """Save a bar plot comparing completion lift with and without Hopfield.

    Parameters
    ----------
    outdir:
        Directory where ``completion_ablation.png`` will be written.
    with_hopfield:
        Summary dictionary from a run with Hopfield enabled. The
        ``completion_lift`` value is read from ``retrieval``.
    without_hopfield:
        Summary dictionary from a run with Hopfield disabled.

    Returns
    -------
    Path
        Path to the written PNG file.
    """

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "completion_ablation.png"
    lift_with = float(with_hopfield.get("retrieval", {}).get("completion_lift", 0.0))
    lift_without = float(without_hopfield.get("retrieval", {}).get("completion_lift", 0.0))
    fig, ax = plt.subplots()
    ax.bar(["no-hopfield", "hopfield"], [lift_without, lift_with])
    ax.set_ylabel("Completion lift")

    max_val = max(lift_with, lift_without, 0.0)
    margin = max(max_val * 0.1, 0.1)
    ax.set_ylim(bottom=0, top=max_val + margin)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
