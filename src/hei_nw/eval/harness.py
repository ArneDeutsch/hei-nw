"""Command-line evaluation harness for HEI-NW baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from hei_nw import datasets
from hei_nw.baselines.long_context import run_long_context
from hei_nw.baselines.rag import HFEmbedder, run_rag
from hei_nw.eval.report import (
    bin_by_lag,
    save_completion_ablation_plot,
    save_reports,
)
from hei_nw.keyer import DGKeyer
from hei_nw.metrics import (
    ComputeRecord,
    collision_rate,
    completion_lift,
    estimate_attention_flops,
    estimate_kv_bytes,
    mrr,
    near_miss_rate,
    precision_at_k,
    relaxed_em,
    strict_em,
    time_block,
    token_f1,
)
from hei_nw.pack import pack_trace, truncate_memory_tokens
from hei_nw.recall import RecallService
from hei_nw.utils.cli import add_common_args
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
DEFAULT_DG_K = DGKeyer().k


@dataclass
class ModelGeometry:
    """Minimal geometry information used for compute estimates."""

    layers: int
    hidden: int
    heads: int
    dtype: str


@dataclass(frozen=True)
class QAPromptSettings:
    """Configuration for QA-style prompting and decoding."""

    prompt_style: str = "plain"
    max_new_tokens: int = 32
    stop: str | None = None
    answer_hint: bool = True

    def stop_value(self) -> str | None:
        """Return ``stop`` with empty strings normalized to ``None``."""

        if self.stop == "":
            return None
        return self.stop


@dataclass(frozen=True)
class HopfieldSettings:
    """Parameters controlling Hopfield readout refinement."""

    steps: int = 1
    temperature: float = 1.0


@dataclass(frozen=True)
class DevIsolationSettings:
    """Developer-only switches used to isolate failure modes."""

    retrieval_only: bool = False
    oracle_trace: bool = False


def _positive_int(value: str) -> int:
    """Return ``value`` parsed as a positive integer."""

    parsed = int(value)
    if parsed <= 0:
        msg = "Value must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    """Return ``value`` parsed as a positive float."""

    parsed = float(value)
    if parsed <= 0.0:
        msg = "Hopfield temperature must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


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
    parser.add_argument(
        "--hopfield.steps",
        dest="hopfield_steps",
        type=_positive_int,
        default=1,
        help="Number of refinement steps used by the Hopfield readout.",
    )
    parser.add_argument(
        "--hopfield.temperature",
        dest="hopfield_temperature",
        type=_positive_float,
        default=1.0,
        help="Softmax temperature applied inside the Hopfield readout.",
    )
    parser.add_argument(
        "--dg.k",
        dest="dg_k",
        type=_positive_int,
        default=DEFAULT_DG_K,
        help="Number of non-zero components retained by the DG keyer.",
    )
    parser.add_argument(
        "--qa.prompt_style",
        dest="qa_prompt_style",
        choices=["plain", "chat"],
        default=None,
        help="Prompt formatting used for QA episodes.",
    )
    parser.add_argument(
        "--qa.max_new_tokens",
        dest="qa_max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate for QA answers.",
    )
    parser.add_argument(
        "--qa.stop",
        dest="qa_stop",
        type=str,
        default=None,
        help="Substring that stops generation for QA answers.",
    )
    parser.add_argument(
        "--mem.max_tokens",
        dest="mem_max_tokens",
        type=_positive_int,
        default=128,
        help=(
            "Maximum number of episodic memory tokens concatenated per record. "
            "Applies to B1 mode only."
        ),
    )
    parser.add_argument(
        "--adapter.scale",
        dest="adapter_scale",
        type=float,
        default=0.2,
        help=(
            "Initial value for the learnable residual gate applied by the "
            "episodic adapter."
        ),
    )
    parser.add_argument(
        "--qa.answer_hint",
        dest="qa_answer_hint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include an instruction to answer with a single word or name.",
    )
    parser.add_argument(
        "--dev.retrieval_only",
        dest="dev_retrieval_only",
        action="store_true",
        help="Bypass generation and emit the top retrieved answer only.",
    )
    parser.add_argument(
        "--dev.oracle_trace",
        dest="dev_oracle_trace",
        action="store_true",
        help="Inject the ground-truth trace as the sole retrieved memory.",
    )
    add_common_args(parser)
    return parser.parse_args(args)


PromptInput = str | list[dict[str, str]]


def _build_prompt(
    record: dict[str, Any],
    *,
    prompt_style: str,
    answer_hint: bool,
) -> tuple[PromptInput, str]:
    """Create prompt data and expected answer from a record."""

    episode = str(record.get("episode_text", ""))
    cues = record.get("cues", [])
    answers = record.get("answers", [])
    cue = str(cues[0]) if cues else ""
    answer = str(answers[0]) if answers else ""

    hint_instruction = (
        "Answer the question using the episode. Reply with ONLY the single correct word or name."
        if answer_hint
        else "Answer the question using the episode."
    )
    episode_body = episode.strip()
    if not episode_body:
        episode_body = "(none)"
    cue_text = cue.strip()

    if prompt_style == "chat":
        system_message = (
            "You are a helpful assistant. "
            "Read the episode and answer the question accurately and concisely."
        )
        if answer_hint:
            system_message = (
                "You are a helpful assistant. Read the episode and answer with ONLY the single "
                "correct word or name."
            )
        user_lines = ["Episode:", episode_body]
        user_lines.extend(["", f"Question: {cue_text}" if cue_text else "Question:"])
        if answer_hint:
            user_lines.append("Respond with only the single correct word or name.")
        else:
            user_lines.append("Respond with a concise answer.")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n".join(user_lines).strip()},
        ]
        return messages, answer

    prompt_parts = [
        hint_instruction,
        "",
        f"Episode:\n{episode_body}",
        "",
        f"Question: {cue_text}",
        "Answer:",
    ]
    prompt = "\n".join(part for part in prompt_parts if part is not None)
    if not prompt.endswith(" "):
        prompt = f"{prompt} "
    return prompt, answer


@dataclass
class EvalItem:
    """Per-record evaluation result."""

    prompt: str
    prediction: str
    truth: str
    em_relaxed: float
    em_strict: float
    f1: float
    latency: float
    recall_at_k: float | None
    lag: int
    em: float = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "em", self.em_relaxed)


def _evaluate_records(
    records: Sequence[dict[str, Any]],
    geom: ModelGeometry,
    qa: QAPromptSettings,
    adapter: Any | None = None,
    mem_tokens: list[int] | None = None,
) -> tuple[list[EvalItem], ComputeRecord]:
    """Run evaluation for *records* and return item metrics and compute."""
    from hei_nw.models.base import generate

    items: list[EvalItem] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
    for rec in records:
        prompt, truth = _build_prompt(
            rec, prompt_style=qa.prompt_style, answer_hint=qa.answer_hint
        )
        with time_block() as t:
            out = generate(
                prompt,
                max_new_tokens=qa.max_new_tokens,
                adapter=adapter,
                mem_tokens=mem_tokens,
                stop=qa.stop_value(),
                prompt_style=qa.prompt_style,
            )
        pred = str(out["text"]).strip()
        em_rel = relaxed_em(pred, truth)
        em_str = strict_em(pred, truth)
        f1 = token_f1(pred, truth)
        items.append(
            EvalItem(
                prompt=prompt,
                prediction=pred,
                truth=truth,
                em_relaxed=em_rel,
                em_strict=em_str,
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
    """Aggregate exact-match, F1, latency, and decode health over *items*."""

    if not items:
        return {
            "em": 0.0,
            "em_relaxed": 0.0,
            "em_strict": 0.0,
            "f1": 0.0,
            "latency": 0.0,
            "recall_at_k": None,
            "non_empty_rate": 0.0,
        }
    n = len(items)
    recall_vals = [i.recall_at_k for i in items if i.recall_at_k is not None]
    recall_avg = sum(recall_vals) / len(recall_vals) if recall_vals else None
    em_relaxed = sum(i.em_relaxed for i in items) / n
    em_strict = sum(i.em_strict for i in items) / n
    non_empty = sum(1 for item in items if item.prediction.strip())
    non_empty_rate = non_empty / n
    return {
        "em": em_relaxed,
        "em_relaxed": em_relaxed,
        "em_strict": em_strict,
        "f1": sum(i.f1 for i in items) / n,
        "latency": sum(i.latency for i in items) / n,
        "recall_at_k": recall_avg,
        "non_empty_rate": non_empty_rate,
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


def _save_reports(
    outdir: Path, scenario: str, mode: str, summary: dict[str, Any], no_hopfield: bool
) -> Path:
    """Persist JSON and Markdown reports to *outdir*.

    Parameters
    ----------
    outdir:
        Destination directory for the reports.
    scenario:
        Scenario identifier (A–E).
    mode:
        Benchmark mode (B0/B1/etc.).
    summary:
        Metrics summary to serialize.
    no_hopfield:
        Whether the run was executed without Hopfield readout.

    Returns
    -------
    Path
        Path to the written JSON metrics file.
    """

    base = f"{scenario}_{mode}"
    if no_hopfield:
        base += "_no-hopfield"
    json_path, _ = save_reports(outdir, base, summary, scenario)
    return json_path


ModeResult = tuple[
    list[EvalItem],
    ComputeRecord,
    dict[str, Any] | None,
    dict[str, Any],
]
ModeHandler = Callable[..., ModeResult]


def _hard_negative_ratio(scenario: str, records: Sequence[dict[str, Any]]) -> float | None:
    """Return the hard-negative ratio for scenario ``A``."""

    if scenario != "A":
        return None
    pos = sum(1 for r in records if r.get("should_remember"))
    neg = sum(1 for r in records if not r.get("should_remember"))
    return neg / pos if pos else None


def _decode_mem_preview(tokenizer: Any, token_ids: Sequence[int]) -> list[str]:
    """Return a human-readable preview of *token_ids* using *tokenizer*."""

    if not token_ids:
        return []
    convert_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(convert_tokens):
        tokens = convert_tokens(list(token_ids))
        return [str(token) for token in tokens]
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):  # pragma: no cover - tokenizer always exposes convert
        text = str(decode(list(token_ids)))
        if not text:
            return []
        return text.split()
    return [str(tid) for tid in token_ids]


def _scenario_default_qa_settings(scenario: str) -> QAPromptSettings:
    """Return scenario-specific default QA settings."""

    if scenario == "A":
        return QAPromptSettings(prompt_style="chat", max_new_tokens=16, stop=None, answer_hint=True)
    return QAPromptSettings()


def _qa_settings_from_args(args: argparse.Namespace) -> QAPromptSettings:
    """Construct :class:`QAPromptSettings` from CLI *args* with defaults."""

    defaults = _scenario_default_qa_settings(args.scenario)
    prompt_style = args.qa_prompt_style if args.qa_prompt_style is not None else defaults.prompt_style
    max_new_tokens = (
        args.qa_max_new_tokens if args.qa_max_new_tokens is not None else defaults.max_new_tokens
    )
    stop = args.qa_stop if args.qa_stop is not None else defaults.stop
    answer_hint = defaults.answer_hint if args.qa_answer_hint is None else args.qa_answer_hint
    return QAPromptSettings(
        prompt_style=prompt_style,
        max_new_tokens=max_new_tokens,
        stop=stop,
        answer_hint=answer_hint,
    )


def _resolve_qa_settings(qa: QAPromptSettings | None) -> QAPromptSettings:
    """Return concrete QA settings with defaults applied."""

    return qa if qa is not None else QAPromptSettings()


def _evaluate_b0_records(
    records: Sequence[dict[str, Any]],
    geom: ModelGeometry,
    qa_settings: QAPromptSettings,
) -> tuple[list[EvalItem], ComputeRecord]:
    """Evaluate plain QA records and return generated items and compute."""

    return _evaluate_records(records, geom, qa_settings)


def _apply_recall_metrics(
    items: Sequence[EvalItem], recalls: Sequence[float] | None
) -> None:
    """Attach recall@k metrics from *recalls* onto ``items`` in-place."""

    if recalls is None:
        return
    for itm, recall in zip(items, recalls, strict=False):
        itm.recall_at_k = recall


def _run_baseline_with_recalls(
    baseline: str,
    records: Sequence[dict[str, Any]],
    model: Any,
    tok: Any,
    items: Sequence[EvalItem],
) -> dict[str, Any] | None:
    """Run *baseline* and propagate recall metrics onto ``items``."""

    baseline_compute, recalls = _run_baseline(baseline, records, model, tok)
    _apply_recall_metrics(items, recalls)
    return baseline_compute


def _evaluate_mode_b0(
    records: Sequence[dict[str, Any]],
    baseline: str,
    model: Any,
    tok: Any,
    geom: ModelGeometry,
    _no_hopfield: bool = False,
    _dg_keyer: DGKeyer | None = None,
    qa: QAPromptSettings | None = None,
    _hopfield: HopfieldSettings | None = None,
    _dev: DevIsolationSettings | None = None,
) -> ModeResult:
    """Evaluate records in B0 mode."""

    qa_settings = _resolve_qa_settings(qa)
    items, compute = _evaluate_b0_records(records, geom, qa_settings)
    baseline_compute = _run_baseline_with_recalls(
        baseline, records, model, tok, items
    )
    return items, compute, baseline_compute, {}


def _evaluate_mode_b1(
    records: Sequence[dict[str, Any]],
    baseline: str,
    model: Any,
    tok: Any,
    geom: ModelGeometry,
    no_hopfield: bool = False,
    dg_keyer: DGKeyer | None = None,
    qa: QAPromptSettings | None = None,
    hopfield: HopfieldSettings | None = None,
    dev: DevIsolationSettings | None = None,
    *,
    mem_max_tokens: int = 128,
    adapter_scale: float = 0.2,
) -> ModeResult:
    """Evaluate records in B1 mode using episodic recall."""

    from transformers import PreTrainedModel

    from hei_nw.models.base import build_default_adapter

    qa_settings = _resolve_qa_settings(qa)
    hopfield_settings = hopfield or HopfieldSettings()
    dev_settings = dev or DevIsolationSettings()
    adapter = build_default_adapter(cast(PreTrainedModel, model), scale=adapter_scale)
    service = RecallService.build(
        records,
        tok,
        max_mem_tokens=mem_max_tokens,
        hopfield_steps=hopfield_settings.steps,
        hopfield_temperature=hopfield_settings.temperature,
        keyer=dg_keyer,
    )
    b0_items, _ = _evaluate_b0_records(records, geom, qa_settings)
    items: list[EvalItem] = []
    compute = ComputeRecord(attention_flops=0, kv_cache_bytes=0)
    cand_groups: list[list[int]] = []
    truths: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    hopfield_top1: list[bool] = []
    baseline_top1: list[bool] = []
    use_hopfield = not no_hopfield
    mem_lengths: list[int] = []
    preview_tokens: list[str] | None = None
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
        res_h = res_no
        if use_hopfield:
            res_h = service.store.query(
                cue,
                return_m=service.return_m,
                use_hopfield=True,
                group_id=group_id,
                should_remember=should_remember,
            )
        if dev_settings.oracle_trace:
            oracle_answers = [str(ans) for ans in rec.get("answers", [])]
            res_h = {
                "selected": [
                    {
                        "answers": oracle_answers,
                        "group_id": group_id,
                    }
                ],
                "candidates": res_no.get("candidates", []),
                "diagnostics": res_no.get("diagnostics", {}),
            }
        selected_traces = cast(list[dict[str, Any]], res_h.get("selected", []))
        cand_groups.append([c["group_id"] for c in res_no["candidates"]])
        truths.append(group_id)
        diagnostics.append(res_no["diagnostics"])
        baseline_top1.append(
            bool(res_no["selected"]) and res_no["selected"][0]["group_id"] == group_id
        )
        hopfield_top1.append(
            bool(selected_traces) and selected_traces[0].get("group_id") == group_id
        )
        tokens: list[int] = []
        for trace in selected_traces:
            answers = trace.get("answers", [])
            fields = {
                key: answers[i] if i < len(answers) else ""
                for i, key in enumerate(["who", "what", "where", "when"])
            }
            tokens.extend(pack_trace(fields, service.tokenizer, service.max_mem_tokens))
            if len(tokens) >= mem_max_tokens:
                break
        mem_tokens = truncate_memory_tokens(tokens, mem_max_tokens)
        mem_lengths.append(len(mem_tokens))
        if preview_tokens is None and mem_tokens:
            preview_tokens = _decode_mem_preview(
                service.tokenizer, mem_tokens[:8]
            )
        if dev_settings.retrieval_only:
            prompt, truth = _build_prompt(
                rec,
                prompt_style=qa_settings.prompt_style,
                answer_hint=qa_settings.answer_hint,
            )
            pred = ""
            if selected_traces:
                top_answers = selected_traces[0].get("answers", [])
                if top_answers:
                    pred = str(top_answers[0])
            em_rel = relaxed_em(pred, truth)
            em_str = strict_em(pred, truth)
            f1 = token_f1(pred, truth)
            items.append(
                EvalItem(
                    prompt=prompt,
                    prediction=pred,
                    truth=truth,
                    em_relaxed=em_rel,
                    em_strict=em_str,
                    f1=f1,
                    latency=0.0,
                    recall_at_k=None,
                    lag=int(rec.get("lag", 0)),
                )
            )
            continue
        itm_list, comp = _evaluate_records(
            [rec],
            geom,
            qa_settings,
            adapter=adapter,
            mem_tokens=mem_tokens,
        )
        items.extend(itm_list)
        compute.attention_flops = (compute.attention_flops or 0) + (comp.attention_flops or 0)
        compute.kv_cache_bytes = (compute.kv_cache_bytes or 0) + (comp.kv_cache_bytes or 0)
    baseline_compute = _run_baseline_with_recalls(
        baseline, records, model, tok, items
    )
    b0_latency = cast(float, _aggregate_metrics(b0_items)["latency"])
    b1_latency = cast(float, _aggregate_metrics(items)["latency"])
    retrieval = {
        "p_at_1": precision_at_k(cand_groups, truths, 1),
        "mrr": mrr(cand_groups, truths),
        "near_miss_rate": near_miss_rate(diagnostics),
        "collision_rate": collision_rate(diagnostics),
        "completion_lift": completion_lift(baseline_top1, hopfield_top1),
    }
    extra = {
        "adapter_latency_overhead_s": b1_latency - b0_latency,
        "retrieval": retrieval,
        "debug": {
            "mem_len": mem_lengths,
            "mem_preview": preview_tokens or [],
            "dev_modes": {
                "retrieval_only": dev_settings.retrieval_only,
                "oracle_trace": dev_settings.oracle_trace,
            },
        },
    }
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
    qa_settings = _qa_settings_from_args(args)
    hopfield_settings = HopfieldSettings(
        steps=args.hopfield_steps, temperature=args.hopfield_temperature
    )

    if records:
        from hei_nw.models.base import load_base

        tok, model, _ = load_base(model_id=args.model, quant_4bit=False)
        geom = _model_geometry(model)
        dg_keyer = DGKeyer(k=args.dg_k) if args.mode == "B1" else None
        dev_settings = DevIsolationSettings(
            retrieval_only=args.dev_retrieval_only,
            oracle_trace=args.dev_oracle_trace,
        )
        handler_kwargs: dict[str, Any] = {}
        if args.mode == "B1":
            handler_kwargs["mem_max_tokens"] = args.mem_max_tokens
            handler_kwargs["adapter_scale"] = args.adapter_scale
        items, compute, baseline_compute, extra = handler(
            records,
            args.baseline,
            model,
            tok,
            geom,
            args.no_hopfield,
            dg_keyer,
            qa_settings,
            hopfield_settings,
            dev_settings,
            **handler_kwargs,
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
    summary["run_config"] = {
        "seed": args.seed,
        "requested_records": args.n,
        "actual_records": len(record_dicts),
        "qa": {
            "prompt_style": qa_settings.prompt_style,
            "max_new_tokens": qa_settings.max_new_tokens,
            "stop": qa_settings.stop,
            "answer_hint": qa_settings.answer_hint,
        },
        "memory": {"max_tokens": args.mem_max_tokens if args.mode == "B1" else None},
        "adapter": {"scale": args.adapter_scale if args.mode == "B1" else None},
        "hopfield": {
            "enabled": not args.no_hopfield,
            "steps": hopfield_settings.steps,
            "temperature": hopfield_settings.temperature,
        },
    }

    _save_reports(args.outdir, args.scenario, args.mode, summary, args.no_hopfield)
    if args.mode == "B1" and args.no_hopfield:
        with_hp_path = args.outdir / f"{args.scenario}_B1_metrics.json"
        if with_hp_path.exists():
            with with_hp_path.open("r", encoding="utf8") as fh:
                with_hp_summary = json.load(fh)
            save_completion_ablation_plot(args.outdir, with_hp_summary, summary)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
