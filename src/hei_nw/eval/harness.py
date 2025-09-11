"""Command-line evaluation harness for HEI-NW baselines."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from hei_nw import datasets
from hei_nw.baselines.long_context import run_long_context
from hei_nw.baselines.rag import HFEmbedder, run_rag
from hei_nw.eval.report import bin_by_lag, build_markdown_report
from hei_nw.metrics import (
    ComputeRecord,
    estimate_attention_flops,
    estimate_kv_bytes,
    exact_match,
    time_block,
    token_f1,
)
from hei_nw.utils.cli import add_common_args
from hei_nw.utils.io import timestamp_slug, write_json, write_markdown
from hei_nw.utils.seed import set_global_seed

SCENARIOS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "A": datasets.scenario_a.generate,
    "B": datasets.scenario_b.generate,
    "C": datasets.scenario_c.generate,
    "D": datasets.scenario_d.generate,
    "E": datasets.scenario_e.generate,
}


@dataclass
class ModelGeometry:
    """Minimal geometry information used for compute estimates."""

    layers: int
    hidden: int
    heads: int
    dtype: str


def _model_geometry(model: Any) -> ModelGeometry:
    """Extract relevant model configuration fields."""

    cfg = model.config
    layers = int(getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)))
    hidden = int(getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)))
    heads = int(getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)))
    dtype_attr = getattr(cfg, "torch_dtype", "float32")
    dtype = str(dtype_attr).replace("torch.", "") if dtype_attr else "float32"
    return ModelGeometry(layers=layers, hidden=hidden, heads=heads, dtype=dtype)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="HEI-NW evaluation harness")
    parser.add_argument("--mode", choices=["B0", "B1", "B2", "B3"], required=True)
    parser.add_argument("--scenario", choices=list("ABCDE"), required=True)
    parser.add_argument("-n", type=int, required=True, help="Number of records")
    parser.add_argument(
        "--baseline",
        choices=["none", "long-context", "rag"],
        default="none",
        help="Optional baseline for compute measurements",
    )
    parser.add_argument("--model", type=str, default=None, help="Model identifier")
    add_common_args(parser)
    return parser.parse_args(args)


def _build_prompt(record: dict[str, Any]) -> tuple[str, str]:
    """Create prompt and expected answer from a record."""

    episode = record.get("episode_text", "")
    cues = record.get("cues", [])
    answers = record.get("answers", [])
    cue = cues[0] if cues else ""
    answer = answers[0] if answers else ""
    prompt = f"{episode}\n{cue}\n"
    return prompt, str(answer)


@dataclass
class EvalItem:
    """Per-record evaluation result."""

    prompt: str
    prediction: str
    truth: str
    em: float
    f1: float
    latency: float
    recall_at_k: float | None
    lag: int


def _evaluate_records(
    records: Sequence[dict[str, Any]], geom: ModelGeometry
) -> tuple[list[EvalItem], ComputeRecord]:
    """Run evaluation for *records* and return item metrics and compute."""
    from hei_nw.models.base import generate

    items: list[EvalItem] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
    for rec in records:
        prompt, truth = _build_prompt(rec)
        with time_block() as t:
            out = generate(prompt, max_new_tokens=32)
        pred = str(out["text"]).strip()
        em = exact_match(pred, truth)
        f1 = token_f1(pred, truth)
        items.append(
            EvalItem(
                prompt=prompt,
                prediction=pred,
                truth=truth,
                em=em,
                f1=f1,
                latency=t.elapsed,
                recall_at_k=None,
                lag=int(rec.get("lag", 0)),
            )
        )
        ptoks = int(out.get("prompt_tokens", 0))
        gtoks = int(out.get("generated_tokens", 0))
        compute.attention_flops = (compute.attention_flops or 0) + estimate_attention_flops(
            ptoks, gtoks, geom.layers, geom.hidden, geom.heads
        )
        compute.kv_cache_bytes = (compute.kv_cache_bytes or 0) + estimate_kv_bytes(
            ptoks + gtoks, geom.hidden, geom.dtype
        )
    return items, compute


def _aggregate_metrics(items: Sequence[EvalItem]) -> dict[str, float | None]:
    """Aggregate exact-match, F1, and latency over *items*."""

    if not items:
        return {"em": 0.0, "f1": 0.0, "latency": 0.0, "recall_at_k": None}
    n = len(items)
    recall_vals = [i.recall_at_k for i in items if i.recall_at_k is not None]
    recall_avg = sum(recall_vals) / len(recall_vals) if recall_vals else None
    return {
        "em": sum(i.em for i in items) / n,
        "f1": sum(i.f1 for i in items) / n,
        "latency": sum(i.latency for i in items) / n,
        "recall_at_k": recall_avg,
    }


class ToyEmbedder:
    """Deterministic hashed embedder used when HF model is unavailable."""

    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - simple
        vecs: list[np.ndarray] = []
        for text in texts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % self.dim
            vec = np.zeros(self.dim, dtype="float32")
            vec[h] = 1.0
            vecs.append(vec)
        return np.stack(vecs)


def _get_embedder() -> Any:
    """Return an HF embedder, falling back to :class:`ToyEmbedder`."""

    try:
        return HFEmbedder()
    except Exception:  # pragma: no cover - network failures
        return ToyEmbedder()


def _prepare_long_context_records(
    gen_records: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert generic records into long-context baseline format."""

    lc_records: list[dict[str, str]] = []
    for r in gen_records:
        if {"context", "query", "expected"} <= r.keys():
            record = {
                "context": str(r["context"]),
                "query": str(r["query"]),
                "expected": str(r["expected"]),
            }
        else:
            record = {
                "context": str(r.get("episode_text", "")),
                "query": str(r.get("cues", [""])[0]),
                "expected": str(r.get("answers", [""])[0]),
            }
        lc_records.append(record)
    return lc_records


def _run_long_context_baseline(
    gen_records: Sequence[dict[str, Any]], model: Any, tok: Any
) -> tuple[dict[str, Any], None]:
    """Run the long-context baseline and return compute metrics."""

    lc_records = _prepare_long_context_records(gen_records)
    out = run_long_context(model, tok, lc_records, {"max_new_tokens": 32})
    return cast(dict[str, Any], out["compute"].model_dump()), None


def _prepare_rag_records(gen_records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert generic records into RAG baseline format."""

    rag_records: list[dict[str, Any]] = []
    for r in gen_records:
        if {"documents", "query", "answers"} <= r.keys():
            docs = list(cast(Sequence[str], r["documents"]))
            query = cast(str, r["query"])
            answers = list(cast(Sequence[str], r["answers"]))
            expected = r.get("expected", answers[0] if answers else "")
        else:
            docs = [cast(str, r.get("context", r.get("episode_text", "")))]
            query = cast(str, r.get("query", r.get("cues", [""])[0]))
            ans = r.get("answers") or [r.get("expected", "")]
            answers = list(cast(Sequence[str], ans))
            expected = r.get("expected", answers[0] if answers else "")
        rag_records.append(
            {
                "documents": docs,
                "query": query,
                "answers": answers,
                "expected": expected,
            }
        )
    return rag_records


def _run_rag_baseline(
    gen_records: Sequence[dict[str, Any]], model: Any, tok: Any
) -> tuple[dict[str, Any], list[float]]:
    """Run the RAG baseline and return compute and recall metrics."""

    embedder = _get_embedder()
    rag_records = _prepare_rag_records(gen_records)
    out = run_rag(model, tok, rag_records, embedder=embedder, k=5, gen_cfg={"max_new_tokens": 32})
    recalls = [cast(float, resp.get("recall_at_k")) for resp in out["responses"]]
    compute = cast(dict[str, Any], out["compute"].model_dump())
    return compute, recalls


def _run_baseline(
    baseline: str,
    gen_records: Sequence[dict[str, Any]],
    model: Any,
    tok: Any,
) -> tuple[dict[str, Any] | None, list[float] | None]:
    """Dispatch to baseline runners based on *baseline* type."""

    if baseline == "long-context":
        return _run_long_context_baseline(gen_records, model, tok)
    if baseline == "rag":
        return _run_rag_baseline(gen_records, model, tok)
    return None, None


def _save_reports(outdir: Path, scenario: str, mode: str, summary: dict[str, Any]) -> None:
    """Persist JSON and Markdown reports to *outdir*."""

    ts = timestamp_slug()
    base = f"{ts}_{scenario}_{mode}"
    json_path = outdir / f"{base}_metrics.json"
    md_path = outdir / f"{base}_report.md"
    write_json(json_path, summary)
    md_content = build_markdown_report(summary)
    write_markdown(md_path, md_content)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the evaluation harness CLI."""

    args = parse_args(argv)
    if args.mode != "B0":
        print("Only B0 mode is supported in M0", file=sys.stderr)
        return 64

    set_global_seed(args.seed)
    gen_records = SCENARIOS[args.scenario](n=args.n, seed=args.seed)

    if gen_records:
        from hei_nw.models.base import load_base

        tok, model, _ = load_base(model_id=args.model, quant_4bit=False)
        geom = _model_geometry(model)
        items, compute = _evaluate_records(gen_records, geom)
        baseline_compute, recalls = _run_baseline(args.baseline, gen_records, model, tok)
        if recalls is not None:
            for itm, r in zip(items, recalls, strict=False):
                itm.recall_at_k = r
    else:
        items = []
        compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
        baseline_compute = None
        recalls = None

    record_dicts = [asdict(it) for it in items]
    summary = {
        "records": record_dicts,
        "aggregate": _aggregate_metrics(items),
        "lag_bins": bin_by_lag(record_dicts, [0, 1, 3, 7, 30]),
        "compute": {
            "b0": compute.model_dump(),
            "baseline": baseline_compute,
        },
    }

    _save_reports(args.outdir, args.scenario, args.mode, summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
