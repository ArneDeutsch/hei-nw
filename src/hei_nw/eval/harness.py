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
    collision_rate,
    completion_lift,
    estimate_attention_flops,
    estimate_kv_bytes,
    exact_match,
    mrr,
    near_miss_rate,
    precision_at_k,
    time_block,
    token_f1,
)
from hei_nw.pack import pack_trace
from hei_nw.recall import RecallService
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

# Default model used when none is specified on the command line. Mirrors the
# fallback in :func:`hei_nw.models.base.load_base`.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"


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
    dtype_attr = getattr(cfg, "dtype", None)
    if dtype_attr is None:
        dtype_attr = vars(cfg).get("torch_dtype")
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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model identifier",
    )
    parser.add_argument(
        "--no-hopfield",
        action="store_true",
        help="Disable Hopfield readout and use raw ANN candidates",
    )
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
    records: Sequence[dict[str, Any]],
    geom: ModelGeometry,
    adapter: Any | None = None,
    mem_tokens: list[int] | None = None,
) -> tuple[list[EvalItem], ComputeRecord]:
    """Run evaluation for *records* and return item metrics and compute."""
    from hei_nw.models.base import generate

    items: list[EvalItem] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
    for rec in records:
        prompt, truth = _build_prompt(rec)
        with time_block() as t:
            out = generate(
                prompt,
                max_new_tokens=32,
                adapter=adapter,
                mem_tokens=mem_tokens,
            )
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
    md_content = build_markdown_report(summary, scenario)
    write_markdown(md_path, md_content)


ModeResult = tuple[
    list[EvalItem],
    ComputeRecord,
    dict[str, Any] | None,
    dict[str, Any],
]
ModeHandler = Callable[
    [Sequence[dict[str, Any]], str, Any, Any, ModelGeometry, bool],
    ModeResult,
]


def _hard_negative_ratio(scenario: str, records: Sequence[dict[str, Any]]) -> float | None:
    """Return the hard-negative ratio for scenario ``A``."""

    if scenario != "A":
        return None
    pos = sum(1 for r in records if r.get("should_remember"))
    neg = sum(1 for r in records if not r.get("should_remember"))
    return neg / pos if pos else None


def _evaluate_mode_b0(
    records: Sequence[dict[str, Any]],
    baseline: str,
    model: Any,
    tok: Any,
    geom: ModelGeometry,
    _no_hopfield: bool = False,
) -> ModeResult:
    """Evaluate records in B0 mode."""

    items, compute = _evaluate_records(records, geom)
    baseline_compute, recalls = _run_baseline(baseline, records, model, tok)
    if recalls is not None:
        for itm, r in zip(items, recalls, strict=False):
            itm.recall_at_k = r
    return items, compute, baseline_compute, {}


def _evaluate_mode_b1(
    records: Sequence[dict[str, Any]],
    baseline: str,
    model: Any,
    tok: Any,
    geom: ModelGeometry,
    no_hopfield: bool = False,
) -> ModeResult:
    """Evaluate records in B1 mode using episodic recall."""

    from transformers import PreTrainedModel

    from hei_nw.models.base import build_default_adapter

    adapter = build_default_adapter(cast(PreTrainedModel, model))
    service = RecallService.build(records, tok, max_mem_tokens=64)
    b0_items, _ = _evaluate_records(records, geom)
    items: list[EvalItem] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
    cand_groups: list[list[int]] = []
    truths: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    hopfield_top1: list[bool] = []
    baseline_top1: list[bool] = []
    use_hopfield = not no_hopfield
    for rec in records:
        cue = rec.get("cues", [""])[0]
        group_id = int(rec.get("group_id", -1))
        should_remember = bool(rec.get("should_remember"))
        res_no = service.store.query(
            cue,
            return_m=service.return_m,
            use_hopfield=False,
            group_id=group_id,
            should_remember=should_remember,
        )
        if use_hopfield:
            res_h = service.store.query(
                cue,
                return_m=service.return_m,
                use_hopfield=True,
                group_id=group_id,
                should_remember=should_remember,
            )
        else:
            res_h = res_no
        cand_groups.append([c["group_id"] for c in res_no["candidates"]])
        truths.append(group_id)
        diagnostics.append(res_no["diagnostics"])
        baseline_top1.append(
            bool(res_no["selected"]) and res_no["selected"][0]["group_id"] == group_id
        )
        hopfield_top1.append(
            bool(res_h["selected"]) and res_h["selected"][0]["group_id"] == group_id
        )
        tokens: list[int] = []
        for trace in res_h["selected"]:
            answers = trace.get("answers", [])
            fields = {
                key: answers[i] if i < len(answers) else ""
                for i, key in enumerate(["who", "what", "where", "when"])
            }
            tokens.extend(pack_trace(fields, service.tokenizer, service.max_mem_tokens))
            if len(tokens) >= 128:
                break
        mem_tokens = tokens[:128]
        itm_list, comp = _evaluate_records([rec], geom, adapter=adapter, mem_tokens=mem_tokens)
        items.extend(itm_list)
        compute.attention_flops = (compute.attention_flops or 0) + (comp.attention_flops or 0)
        compute.kv_cache_bytes = (compute.kv_cache_bytes or 0) + (comp.kv_cache_bytes or 0)
    baseline_compute, recalls = _run_baseline(baseline, records, model, tok)
    if recalls is not None:
        for itm, r in zip(items, recalls, strict=False):
            itm.recall_at_k = r
    b0_latency = cast(float, _aggregate_metrics(b0_items)["latency"])
    b1_latency = cast(float, _aggregate_metrics(items)["latency"])
    retrieval = {
        "p_at_1": precision_at_k(cand_groups, truths, 1),
        "mrr": mrr(cand_groups, truths),
        "near_miss_rate": near_miss_rate(diagnostics),
        "collision_rate": collision_rate(diagnostics),
        "completion_lift": completion_lift(baseline_top1, hopfield_top1),
    }
    extra = {"adapter_latency_overhead_s": b1_latency - b0_latency, "retrieval": retrieval}
    return items, compute, baseline_compute, extra


MODE_HANDLERS: dict[str, ModeHandler] = {
    "B0": _evaluate_mode_b0,
    "B1": _evaluate_mode_b1,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the evaluation harness CLI."""

    args = parse_args(argv)
    handler = MODE_HANDLERS.get(args.mode)
    if handler is None:
        print(f"Mode {args.mode} is not supported in M1", file=sys.stderr)
        return 64
    if args.mode == "B1" and args.outdir == Path("reports/baseline"):
        args.outdir = Path("reports/m1-episodic-adapter")
        args.outdir.mkdir(parents=True, exist_ok=True)

    set_global_seed(args.seed)
    records = SCENARIOS[args.scenario](n=args.n, seed=args.seed)
    hard_neg_ratio = _hard_negative_ratio(args.scenario, records)

    if records:
        from hei_nw.models.base import load_base

        tok, model, _ = load_base(model_id=args.model, quant_4bit=False)
        geom = _model_geometry(model)
        items, compute, baseline_compute, extra = handler(
            records, args.baseline, model, tok, geom, args.no_hopfield
        )
    else:
        items = []
        compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
        baseline_compute = None
        extra = {}

    record_dicts = [asdict(it) for it in items]
    summary: dict[str, Any] = {
        "records": record_dicts,
        "aggregate": _aggregate_metrics(items),
        "lag_bins": bin_by_lag(record_dicts, [0, 1, 3, 7, 30]),
        "compute": {"b0": compute.model_dump(), "baseline": baseline_compute},
        "dataset": {"scenario": args.scenario},
    }
    if hard_neg_ratio is not None:
        summary["dataset"]["hard_negative_ratio"] = hard_neg_ratio
    summary.update(extra)

    _save_reports(args.outdir, args.scenario, args.mode, summary)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
