"""Command-line evaluation harness for HEI-NW baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from hei_nw import datasets
from hei_nw.baselines.long_context import run_long_context
from hei_nw.baselines.rag import HFEmbedder, run_rag
from hei_nw.eval.report import (
    bin_by_lag,
    save_completion_ablation_plot,
    save_reports,
)
from hei_nw.gate import GateDecision, NeuromodulatedGate, SalienceFeatures
from hei_nw.keyer import DGKeyer
from hei_nw.metrics import (
    ComputeRecord,
    collision_rate,
    completion_lift,
    estimate_attention_flops,
    estimate_kv_bytes,
    hopfield_rank_improved_rate,
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
from hei_nw.store import TraceWriter
from hei_nw.telemetry import compute_gate_metrics
from hei_nw.utils.cli import add_common_args
from hei_nw.utils.seed import set_global_seed

FloatArray: TypeAlias = NDArray[np.float32]

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

_BANNED_TRACE_KEYS = {"episode_text", "raw_text", "snippet", "full_text", "text"}
_MAX_TRACE_SAMPLES = 5


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
    template_policy: str = "auto"
    stop_mode: str = "substring"
    omit_episode: bool = False
    memory_dependent_baseline: bool = False

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


def _non_negative_int(value: str) -> int:
    """Return ``value`` parsed as a non-negative integer."""

    parsed = int(value)
    if parsed < 0:
        msg = "Value must be zero or a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _positive_float(value: str) -> float:
    """Return ``value`` parsed as a positive float."""

    parsed = float(value)
    if parsed <= 0.0:
        msg = "Hopfield temperature must be positive"
        raise argparse.ArgumentTypeError(msg)
    return parsed


def _extract_gate_features(record: dict[str, Any]) -> SalienceFeatures:
    """Return salience features extracted from *record* metadata."""

    raw = record.get("gate_features") or {}
    if not isinstance(raw, dict):
        raw = {}
    surprise = float(raw.get("surprise", 0.0))
    novelty = float(raw.get("novelty", 0.0))
    reward = bool(raw.get("reward", False))
    pin = bool(raw.get("pin", False))
    return SalienceFeatures(surprise=surprise, novelty=novelty, reward=reward, pin=pin)


def _apply_gate(
    records: Sequence[dict[str, Any]],
    gate: NeuromodulatedGate,
    *,
    use_for_writes: bool = True,
    debug_keep_labels: bool = False,
    allow_label_fallback: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[GateDecision]]:
    """Return filtered records for indexing and per-record diagnostics."""

    indexed: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    decisions: list[GateDecision] = []
    for idx, record in enumerate(records):
        features = _extract_gate_features(record)
        decision: GateDecision = gate.decision(features)
        decisions.append(decision)
        raw_feats = record.get("gate_features")
        has_gate_payload = isinstance(raw_feats, dict) and bool(raw_feats)
        truth_label = bool(record.get("should_remember"))
        write = decision.should_write or bool(features.pin)
        fallback = False
        if allow_label_fallback and (not has_gate_payload) and truth_label and not write:
            write = True
            fallback = True
        keep_via_label = False
        if use_for_writes:
            keep_via_label = bool(debug_keep_labels and truth_label and not write)
            store_flag = write or keep_via_label
        else:
            store_flag = truth_label
            keep_via_label = bool(truth_label and not write)
        diag_entry = {
            "index": idx,
            "score": decision.score,
            "should_write": write,
            "features": asdict(decision.features),
            "contributions": decision.contributions,
            "should_remember_label": truth_label,
            "group_id": record.get("group_id"),
            "fallback_write": fallback,
            "indexed_for_store": bool(store_flag),
            "label_kept": keep_via_label,
            "prompt_tokens": 0,
            "generated_tokens": 0,
        }
        diagnostics.append(diag_entry)
        if store_flag:
            indexed_record = dict(record)
            indexed_record["should_remember"] = True
            indexed.append(indexed_record)
    return indexed, diagnostics, decisions


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _score_distribution_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize score distribution diagnostics from *metrics*."""

    summary: dict[str, Any] = {
        "p10": None,
        "p50": None,
        "p90": None,
        "histogram": [],
    }
    distribution = metrics.get("score_distribution")
    if not isinstance(distribution, Mapping):
        return summary
    for key in ("p10", "p50", "p90"):
        value = distribution.get(key)
        if isinstance(value, int | float):
            summary[key] = float(value)
    histogram_raw = distribution.get("histogram")
    histogram: list[dict[str, Any]] = []
    if isinstance(histogram_raw, Sequence):
        for bucket in histogram_raw:
            if not isinstance(bucket, Mapping):
                continue
            lower_raw = bucket.get("lower")
            if lower_raw is None:
                continue
            upper_raw = bucket.get("upper", lower_raw)
            if upper_raw is None:
                upper_raw = lower_raw
            count_raw = bucket.get("count", 0)
            try:
                lower_val = float(lower_raw)
            except (TypeError, ValueError):
                continue
            try:
                upper_val = float(upper_raw)
            except (TypeError, ValueError):
                upper_val = lower_val
            try:
                count_val = int(count_raw)
            except (TypeError, ValueError):
                count_val = 0
            histogram.append({"lower": lower_val, "upper": upper_val, "count": count_val})
    summary["histogram"] = histogram
    return summary


def _label_distribution_summary(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return normalized label distribution statistics from *metrics*."""

    distribution = metrics.get("label_distribution")
    total_fallback = _safe_int(metrics.get("total"))
    if isinstance(distribution, Mapping):
        positives = _safe_int(distribution.get("positives"))
        negatives_raw = distribution.get("negatives")
        if negatives_raw is None:
            negatives = total_fallback - positives
        else:
            negatives = _safe_int(negatives_raw)
    else:
        positives = _safe_int(metrics.get("positives"))
        negatives = total_fallback - positives
    positives = max(int(positives), 0)
    negatives = max(int(negatives), 0)
    total = positives + negatives
    if total > 0:
        positive_rate = max(0.0, min(1.0, positives / total))
    else:
        positive_rate = 0.0
    return {
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positive_rate,
    }


def _subset_gate_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return a compact summary for a gate metrics subset."""

    calibration_raw = metrics.get("calibration")
    calibration: list[dict[str, Any]]
    if isinstance(calibration_raw, Sequence):
        calibration = [dict(bucket) for bucket in calibration_raw if isinstance(bucket, Mapping)]
    else:
        calibration = []
    clutter_rate = float(metrics.get("clutter_rate", 0.0))
    writes_per_1k_tokens = metrics.get("writes_per_1k_tokens")
    if isinstance(writes_per_1k_tokens, int | float):
        per_1k_tokens = float(writes_per_1k_tokens)
    else:
        per_1k_tokens = None
    subset: dict[str, Any] = {
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "pr_auc": float(metrics.get("pr_auc", 0.0)),
        "writes": int(metrics.get("writes", 0)),
        "write_rate": clutter_rate,
        "write_rate_per_1k_records": float(
            metrics.get("writes_per_1k_records", clutter_rate * 1000.0)
        ),
        "write_rate_per_1k_tokens": per_1k_tokens,
        "calibration": calibration,
        "calibration_bins": len(calibration),
        "total": int(metrics.get("total", 0)),
        "positives": int(metrics.get("positives", 0)),
        "generated_tokens": int(metrics.get("generated_tokens", 0)),
        "prompt_tokens": int(metrics.get("prompt_tokens", 0)),
    }
    subset["score_distribution"] = _score_distribution_summary(metrics)
    subset["label_distribution"] = _label_distribution_summary(metrics)
    return subset


def _calibration_guard(label_distribution: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return calibratability flag and warnings for *label_distribution*."""

    positives = _safe_int(label_distribution.get("positives"))
    negatives = _safe_int(label_distribution.get("negatives"))
    total = positives + negatives
    warnings: list[str] = []
    calibratable = True
    if total == 0:
        calibratable = False
        warnings.append(
            "Label distribution empty (no positives or negatives); gate telemetry is non-calibratable."
        )
    elif positives == 0 or positives == total:
        calibratable = False
        warnings.append(
            (
                "Label distribution degenerate "
                f"(positives={positives}, negatives={negatives}); gate telemetry is non-calibratable."
            )
        )
    return calibratable, warnings


def _summarize_gate(
    diagnostics: Sequence[dict[str, Any]], *, pins_only: bool = False
) -> dict[str, Any]:
    """Compute aggregate statistics for gate diagnostics."""

    diag_list = list(diagnostics)
    pin_diag = [diag for diag in diag_list if diag.get("features", {}).get("pin")]
    non_pin_diag = [diag for diag in diag_list if not diag.get("features", {}).get("pin")]
    primary_diag = pin_diag if pins_only else diag_list
    telemetry_raw = compute_gate_metrics(primary_diag)
    telemetry: dict[str, Any] = dict(telemetry_raw)
    score_distribution = _score_distribution_summary(telemetry_raw)
    telemetry["score_distribution"] = score_distribution
    label_distribution = _label_distribution_summary(telemetry_raw)
    telemetry["label_distribution"] = label_distribution
    calibratable, warnings = _calibration_guard(label_distribution)
    telemetry["calibratable"] = calibratable
    telemetry["calibration_status"] = "ok" if calibratable else "non_calibratable"
    if warnings:
        telemetry["warnings"] = warnings
    pin_metrics = telemetry_raw if pins_only else compute_gate_metrics(pin_diag)
    non_pin_metrics = compute_gate_metrics(non_pin_diag)
    telemetry["pins_only"] = _subset_gate_metrics(pin_metrics)
    telemetry["non_pins"] = _subset_gate_metrics(non_pin_metrics)
    total = int(telemetry["total"])
    if total == 0:
        return {
            "total": 0,
            "writes": 0,
            "write_rate": 0.0,
            "write_rate_per_1k_records": 0.0,
            "write_rate_per_1k_tokens": None,
            "store_writes": 0,
            "store_write_rate": 0.0,
            "pinned": 0,
            "reward_flags": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "decisions": [],
            "telemetry": telemetry,
            "score_distribution": score_distribution,
        }
    writes = int(telemetry["writes"])
    scores = [float(diag["score"]) for diag in primary_diag]
    pinned = sum(1 for diag in primary_diag if diag.get("features", {}).get("pin", False))
    reward_flags = sum(1 for diag in primary_diag if diag.get("features", {}).get("reward", False))
    store_writes = sum(1 for diag in primary_diag if diag.get("indexed_for_store"))
    prompt_tokens_total = sum(_safe_int(diag.get("prompt_tokens")) for diag in primary_diag)
    generated_tokens_total = sum(_safe_int(diag.get("generated_tokens")) for diag in primary_diag)
    total_tokens = prompt_tokens_total + generated_tokens_total
    write_rate = writes / total
    store_write_rate = store_writes / total if total else 0.0
    if total_tokens > 0:
        writes_per_1k_tokens = writes / (total_tokens / 1000.0)
    else:
        writes_per_1k_tokens = None
    return {
        "total": total,
        "writes": writes,
        "write_rate": write_rate,
        "write_rate_per_1k_records": write_rate * 1000.0,
        "write_rate_per_1k_tokens": writes_per_1k_tokens,
        "store_writes": store_writes,
        "store_write_rate": store_write_rate,
        "pinned": pinned,
        "reward_flags": reward_flags,
        "prompt_tokens": prompt_tokens_total,
        "generated_tokens": generated_tokens_total,
        "score_mean": float(sum(scores) / total) if scores else 0.0,
        "score_min": float(min(scores)) if scores else 0.0,
        "score_max": float(max(scores)) if scores else 0.0,
        "score_distribution": score_distribution,
        "decisions": primary_diag,
        "telemetry": telemetry,
    }
    result["calibratable"] = calibratable
    result["calibration_status"] = telemetry["calibration_status"]
    if warnings:
        result["calibration_warnings"] = warnings
    

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
    parser.add_argument("-n", "--n", type=int, required=True, help="Number of records")
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
        "--qa.template_policy",
        dest="qa_template_policy",
        choices=["auto", "plain"],
        default=None,
        help=(
            "Policy for applying chat templates. 'auto' uses the tokenizer template "
            "when available; 'plain' always falls back to a simple prompt."
        ),
    )
    parser.add_argument(
        "--qa.stop_mode",
        dest="qa_stop_mode",
        choices=["substring", "none"],
        default=None,
        help=(
            "Strategy used to apply the stop string. 'substring' truncates at the "
            "first occurrence; 'none' disables truncation."
        ),
    )
    parser.add_argument(
        "--mem.max_tokens",
        dest="mem_max_tokens",
        type=_non_negative_int,
        default=None,
        help=(
            "Maximum number of episodic memory tokens concatenated per record. "
            "Applies to B1 mode only. Defaults to a scenario-specific preset."
        ),
    )
    parser.add_argument(
        "--store.evict_stale",
        dest="store_evict_stale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable periodic eviction of stale traces between evaluation batches.",
    )
    parser.add_argument(
        "--store.evict_interval",
        dest="store_evict_interval",
        type=_positive_int,
        default=32,
        help=(
            "Number of batches processed between eviction sweeps when "
            "--store.evict_stale is enabled."
        ),
    )
    parser.add_argument(
        "--adapter.scale",
        dest="adapter_scale",
        type=float,
        default=0.2,
        help=("Initial value for the learnable residual gate applied by the " "episodic adapter."),
    )
    parser.add_argument(
        "--gate.alpha",
        dest="gate_alpha",
        type=float,
        default=1.0,
        help="Weight α applied to the surprise component of the gate.",
    )
    parser.add_argument(
        "--gate.beta",
        dest="gate_beta",
        type=float,
        default=1.0,
        help="Weight β applied to the novelty component of the gate.",
    )
    parser.add_argument(
        "--gate.gamma",
        dest="gate_gamma",
        type=float,
        default=0.5,
        help="Weight γ applied to the reward signal.",
    )
    parser.add_argument(
        "--gate.delta",
        dest="gate_delta",
        type=float,
        default=0.8,
        help="Weight δ applied to the pin signal.",
    )
    parser.add_argument(
        "--gate.threshold",
        dest="gate_threshold",
        type=_positive_float,
        default=1.5,
        help="Decision threshold τ for the write gate.",
    )
    parser.add_argument(
        "--gate.pin_override",
        dest="gate_pin_override",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Bypass the gate threshold when the pin flag is present. This keeps gate"
            " decisions aligned with the store's forced-write policy."
        ),
    )
    parser.add_argument(
        "--gate.use_for_writes",
        dest="gate_use_for_writes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use gate decisions to determine which records are written to the store. "
            "Disable to fall back to label-driven writes."
        ),
    )
    parser.add_argument(
        "--gate.debug_keep_labels",
        dest="gate_debug_keep_labels",
        action="store_true",
        help=(
            "When gate-driven writes are enabled, also keep label-positive records for A/B testing."
        ),
    )
    parser.add_argument(
        "--gate.allow_label_fallback",
        dest="gate_allow_label_fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow label-positive records lacking gate features to be written during evaluation."
        ),
    )
    parser.add_argument(
        "--eval.pins_only",
        dest="eval_pins_only",
        action="store_true",
        help="Restrict gate metrics to records flagged as pins.",
    )
    parser.add_argument(
        "--qa.answer_hint",
        dest="qa_answer_hint",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include an instruction to answer with a single word or name.",
    )
    parser.add_argument(
        "--qa.omit_episode",
        dest="qa_omit_episode",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Drop the episode text from prompts to create a memory-dependent baseline.",
    )
    parser.add_argument(
        "--qa.memory_dependent_baseline",
        dest="qa_memory_dependent_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Toggle the memory-dependent baseline: use identical prompts for B0/B1 by "
            "omitting the episode text when not explicitly overridden."
        ),
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
    omit_episode: bool,
) -> tuple[PromptInput, str]:
    """Create prompt data and expected answer from a record."""

    episode = "" if omit_episode else str(record.get("episode_text", ""))
    cues = record.get("cues", [])
    answers = record.get("answers", [])
    cue = str(cues[0]) if cues else ""
    answer = str(answers[0]) if answers else ""

    if omit_episode:
        hint_instruction = (
            "Answer the question. Reply with ONLY the single correct word or name."
            if answer_hint
            else "Answer the question concisely."
        )
    else:
        hint_instruction = (
            "Answer the question using the episode. "
            "Reply with ONLY the single correct word or name."
            if answer_hint
            else "Answer the question using the episode."
        )
    episode_body = episode.strip()
    if not episode_body and not omit_episode:
        episode_body = "(none)"
    cue_text = cue.strip()

    if prompt_style == "chat":
        if omit_episode:
            system_message = (
                "You are a helpful assistant. Answer the question accurately and concisely."
            )
            if answer_hint:
                system_message = (
                    "You are a helpful assistant. Answer with ONLY the single correct word or name."
                )
            user_lines = [f"Question: {cue_text}" if cue_text else "Question:"]
            if answer_hint:
                user_lines.append("Respond with only the single correct word or name.")
                user_lines.append(
                    "Respond with only the single word (no punctuation, no Markdown)."
                )
            else:
                user_lines.append("Respond with a concise answer.")
        else:
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
                user_lines.append(
                    "Respond with only the single word (no punctuation, no Markdown)."
                )
            else:
                user_lines.append("Respond with a concise answer.")
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": "\n".join(user_lines).strip()},
        ]
        return messages, answer

    prompt_parts = [hint_instruction, ""]
    if not omit_episode:
        prompt_parts.extend([f"Episode:\n{episode_body}", ""])
    prompt_parts.extend([f"Question: {cue_text}", "Answer:"])
    prompt = "\n".join(part for part in prompt_parts if part is not None)
    if not prompt.endswith(" "):
        prompt = f"{prompt} "
    return prompt, answer


@dataclass
class EvalItem:
    """Per-record evaluation result."""

    prompt: PromptInput
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


def _normalize_prediction(text: str) -> str:
    """Strip leading markers so the first token starts alphabetically when possible."""

    tokens = text.split()
    if not tokens:
        return text
    idx = 0
    while idx < len(tokens):
        raw = tokens[idx]
        if raw.rstrip().endswith(":"):
            idx += 1
            continue
        token = raw.strip("-:• ")
        if token and token[0].isalpha():
            tokens[idx] = token
            return " ".join(tokens[idx:])
        idx += 1
    return text


def _evaluate_records(
    records: Sequence[dict[str, Any]],
    geom: ModelGeometry,
    qa: QAPromptSettings,
    adapter: Any | None = None,
    mem_tokens: list[int] | None = None,
    mem_text: str | None = None,
) -> tuple[list[EvalItem], ComputeRecord]:
    """Run evaluation for *records* and return item metrics and compute."""
    from hei_nw.models.base import generate

    items: list[EvalItem] = []
    compute = ComputeRecord(
        attention_flops=0, kv_cache_bytes=0, prompt_tokens=0, generated_tokens=0
    )
    for rec in records:
        prompt, truth = _build_prompt(
            rec,
            prompt_style=qa.prompt_style,
            answer_hint=qa.answer_hint,
            omit_episode=qa.omit_episode,
        )
        with time_block() as t:
            out = generate(
                prompt,
                max_new_tokens=qa.max_new_tokens,
                adapter=adapter,
                mem_tokens=mem_tokens,
                memory_prompt=mem_text,
                stop=qa.stop_value(),
                prompt_style=qa.prompt_style,
                stop_mode=qa.stop_mode,
                template_policy=qa.template_policy,
            )
        raw_pred = str(out["text"]).strip()
        pred = _normalize_prediction(raw_pred)
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
        compute.prompt_tokens = (compute.prompt_tokens or 0) + ptoks
        compute.generated_tokens = (compute.generated_tokens or 0) + gtoks
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

    def embed(self, texts: Sequence[str]) -> FloatArray:  # pragma: no cover - simple
        vecs: list[FloatArray] = []
        for text in texts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16) % self.dim
            vec = np.zeros(self.dim, dtype="float32")
            vec[h] = 1.0
            vecs.append(vec)
        return cast(FloatArray, np.stack(vecs))


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
        return QAPromptSettings(
            prompt_style="chat",
            max_new_tokens=16,
            stop=None,
            answer_hint=True,
            template_policy="auto",
            stop_mode="none",
            omit_episode=False,
        )
    return QAPromptSettings()


def _scenario_default_mem_max_tokens(scenario: str) -> int:
    """Return the default episodic memory cap for *scenario*."""

    if scenario == "A":
        return 64
    return 128


def _qa_settings_from_args(args: argparse.Namespace) -> QAPromptSettings:
    """Construct :class:`QAPromptSettings` from CLI *args* with defaults."""

    defaults = _scenario_default_qa_settings(args.scenario)
    prompt_style = (
        args.qa_prompt_style if args.qa_prompt_style is not None else defaults.prompt_style
    )
    max_new_tokens = (
        args.qa_max_new_tokens if args.qa_max_new_tokens is not None else defaults.max_new_tokens
    )
    stop = args.qa_stop if args.qa_stop is not None else defaults.stop
    answer_hint = defaults.answer_hint if args.qa_answer_hint is None else args.qa_answer_hint
    template_policy = (
        args.qa_template_policy if args.qa_template_policy is not None else defaults.template_policy
    )
    stop_mode = args.qa_stop_mode if args.qa_stop_mode is not None else defaults.stop_mode
    forced_memory_baseline = False
    if args.qa_omit_episode is not None:
        omit_episode = args.qa_omit_episode
        forced_memory_baseline = args.qa_omit_episode
    elif args.mode in {"B0", "B1"}:
        omit_episode = True
        forced_memory_baseline = True
    elif args.qa_memory_dependent_baseline:
        omit_episode = True
        forced_memory_baseline = True
    else:
        omit_episode = defaults.omit_episode
    memory_dependent_baseline = args.qa_memory_dependent_baseline or forced_memory_baseline
    return QAPromptSettings(
        prompt_style=prompt_style,
        max_new_tokens=max_new_tokens,
        stop=stop,
        answer_hint=answer_hint,
        template_policy=template_policy,
        stop_mode=stop_mode,
        omit_episode=omit_episode,
        memory_dependent_baseline=memory_dependent_baseline,
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


def _apply_recall_metrics(items: Sequence[EvalItem], recalls: Sequence[float] | None) -> None:
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
    baseline_compute = _run_baseline_with_recalls(baseline, records, model, tok, items)
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
    gate: NeuromodulatedGate | None = None,
    pins_only: bool = False,
    gate_use_for_writes: bool = True,
    gate_debug_keep_labels: bool = False,
    gate_allow_label_fallback: bool = True,
    store_evict_stale: bool = False,
    store_evict_interval: int = 32,
) -> ModeResult:
    """Evaluate records in B1 mode using episodic recall."""

    from transformers import PreTrainedModel

    from hei_nw.models.base import build_default_adapter, generate

    qa_settings = _resolve_qa_settings(qa)
    hopfield_settings = hopfield or HopfieldSettings()
    dev_settings = dev or DevIsolationSettings()
    adapter = build_default_adapter(cast(PreTrainedModel, model), scale=adapter_scale)
    gate_module = gate or NeuromodulatedGate()
    indexed_records, gate_diagnostics, gate_decisions = _apply_gate(
        records,
        gate_module,
        use_for_writes=gate_use_for_writes,
        debug_keep_labels=gate_debug_keep_labels,
        allow_label_fallback=gate_allow_label_fallback,
    )
    service = RecallService.build(
        indexed_records,
        tok,
        max_mem_tokens=mem_max_tokens,
        hopfield_steps=hopfield_settings.steps,
        hopfield_temperature=hopfield_settings.temperature,
        keyer=dg_keyer,
    )
    if dev_settings.retrieval_only:
        b0_items: list[EvalItem] = []
    else:
        b0_items, _ = _evaluate_b0_records(records, geom, qa_settings)
    items: list[EvalItem] = []
    compute = ComputeRecord(
        attention_flops=0, kv_cache_bytes=0, prompt_tokens=0, generated_tokens=0
    )
    final_cand_groups: list[list[int]] = []
    baseline_cand_groups: list[list[int]] = []
    truths: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    hopfield_diagnostics: list[dict[str, Any]] = []
    hopfield_top1: list[bool] = []
    baseline_top1: list[bool] = []
    use_hopfield = not no_hopfield
    mem_lengths: list[int] = []
    preview_tokens: list[str] | None = None
    first_tokens: list[str] = []
    pointer_total = 0
    pointer_with = 0
    pointer_missing = 0
    pointer_banned: set[str] = set()
    trace_samples: list[dict[str, Any]] = []
    persisted_samples: list[dict[str, Any]] = []
    trace_writer: TraceWriter | None = TraceWriter() if gate_use_for_writes else None
    total_prompt_tokens = 0
    total_generated_tokens = 0
    store_evict_enabled = bool(store_evict_stale)
    eviction_interval = max(int(store_evict_interval), 1)
    batches_since_eviction = 0
    eviction_runs = 0
    evicted_total = 0

    def _coerce_group_id(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _maybe_run_eviction() -> None:
        """Invoke background eviction when the configured interval elapses."""

        nonlocal batches_since_eviction, eviction_runs, evicted_total
        if not store_evict_enabled:
            return
        batches_since_eviction += 1
        if batches_since_eviction < eviction_interval:
            return
        store_obj = getattr(service, "store", None)
        evict_fn = getattr(store_obj, "evict_stale", None) if store_obj is not None else None
        batches_since_eviction = 0
        if not callable(evict_fn):
            return
        removed = evict_fn()
        eviction_runs += 1
        if isinstance(removed, Sequence) and not isinstance(removed, str | bytes):
            evicted_total += len(removed)
        elif removed:
            evicted_total += 1

    for idx, rec in enumerate(records):
        diag_entry: dict[str, Any] | None = None
        if idx < len(gate_diagnostics):
            diag_entry = gate_diagnostics[idx]
        gate_decision: GateDecision | None = None
        if idx < len(gate_decisions):
            gate_decision = gate_decisions[idx]
        if (
            trace_writer is not None
            and gate_decision is not None
            and diag_entry is not None
            and diag_entry.get("indexed_for_store")
        ):
            payload = _persist_gate_record(
                trace_writer,
                rec,
                gate_decision,
                record_index=idx,
            )
            if len(persisted_samples) < _MAX_TRACE_SAMPLES:
                persisted_samples.append(
                    _persisted_sample_from_payload(payload, rec.get("group_id"))
                )
        cue = rec.get("cues", [""])[0]
        group_id = int(rec.get("group_id", -1))
        should_remember = bool(rec.get("should_remember"))
        result = service.store.query(
            cue,
            return_m=service.return_m,
            use_hopfield=use_hopfield,
            group_id=group_id,
            should_remember=should_remember,
        )
        if dev_settings.oracle_trace:
            oracle_answers = [str(ans) for ans in rec.get("answers", [])]
            baseline_candidates = result.get("baseline_candidates", [])
            result = {
                "selected": [
                    {
                        "answers": oracle_answers,
                        "group_id": group_id,
                    }
                ],
                "candidates": result.get("candidates", []),
                "diagnostics": result.get("diagnostics", {}),
                "baseline_candidates": baseline_candidates,
                "baseline_diagnostics": result.get(
                    "baseline_diagnostics", result.get("diagnostics", {})
                ),
            }
        selected_traces = cast(list[dict[str, Any]], result.get("selected", []))
        final_candidates_raw = list(result.get("candidates", []))
        baseline_candidates_raw = list(result.get("baseline_candidates") or final_candidates_raw)
        final_groups = [_coerce_group_id(c.get("group_id")) for c in final_candidates_raw]
        baseline_groups = [_coerce_group_id(c.get("group_id")) for c in baseline_candidates_raw]
        truths.append(group_id)
        final_cand_groups.append(final_groups)
        baseline_cand_groups.append(baseline_groups)
        diag = cast(dict[str, Any], result.get("diagnostics", {}))
        diagnostics.append(diag)
        hopfield_diagnostics.append(diag)
        baseline_top1.append(
            bool(baseline_candidates_raw) and baseline_candidates_raw[0].get("group_id") == group_id
        )
        hopfield_top1.append(
            bool(final_candidates_raw) and final_candidates_raw[0].get("group_id") == group_id
        )
        pointer_total += len(selected_traces)
        for trace in selected_traces:
            pointer = trace.get("tokens_span_ref")
            has_pointer = isinstance(pointer, Mapping)
            if has_pointer:
                pointer_with += 1
            else:
                pointer_missing += 1
            for banned in _BANNED_TRACE_KEYS:
                if banned in trace and trace[banned]:
                    pointer_banned.add(banned)
            if len(trace_samples) < _MAX_TRACE_SAMPLES:
                sample: dict[str, Any] = {
                    "trace_id": trace.get("trace_id"),
                    "group_id": trace.get("group_id"),
                    "has_pointer": has_pointer,
                    "banned_keys": [key for key in _BANNED_TRACE_KEYS if key in trace],
                }
                if has_pointer and isinstance(pointer, Mapping):
                    sample["pointer"] = {key: pointer.get(key) for key in ("doc", "start", "end")}
                answers_field = trace.get("answers")
                if isinstance(answers_field, Sequence):
                    sample["answers_preview"] = [str(ans) for ans in answers_field[:2]]
                entity_slots = trace.get("entity_slots")
                if isinstance(entity_slots, Mapping):
                    sample["entity_slots_keys"] = sorted(str(k) for k in entity_slots.keys())
                trace_samples.append(sample)
        if not use_hopfield:
            hopfield_top1[-1] = baseline_top1[-1]
        if mem_max_tokens <= 0:
            mem_tokens: list[int] = []
        else:
            tokens: list[int] = []
            for trace in selected_traces:
                answers = trace.get("answers", [])
                slot_fields = {
                    key: answers[i] if i < len(answers) else ""
                    for i, key in enumerate(["who", "what", "where", "when"])
                }
                tokens.extend(pack_trace(slot_fields, service.tokenizer, service.max_mem_tokens))
                if len(tokens) >= mem_max_tokens:
                    break
            mem_tokens = truncate_memory_tokens(tokens, mem_max_tokens)
        mem_lengths.append(len(mem_tokens))
        if preview_tokens is None and mem_tokens:
            preview_tokens = _decode_mem_preview(service.tokenizer, mem_tokens[:8])
        mem_text: str | None = None
        if qa_settings.memory_dependent_baseline and mem_tokens and selected_traces:
            snippets: list[str] = []
            for trace in selected_traces:
                episode_text = str(trace.get("episode_text", "")).strip()
                if episode_text:
                    snippets.append(episode_text)
                    continue
                answers = trace.get("answers", [])
                if not isinstance(answers, list):
                    continue
                fields: list[str] = []
                for label, idx in ("who", 0), ("what", 1), ("where", 2), ("when", 3):
                    if idx < len(answers):
                        value = str(answers[idx]).strip()
                        if value:
                            fields.append(f"{label}: {value}")
                if fields:
                    snippets.append("; ".join(fields))
            if snippets:
                mem_text = " | ".join(snippets)
        if dev_settings.retrieval_only:
            prompt, truth = _build_prompt(
                rec,
                prompt_style=qa_settings.prompt_style,
                answer_hint=qa_settings.answer_hint,
                omit_episode=qa_settings.omit_episode,
            )
            out = generate(
                prompt,
                max_new_tokens=qa_settings.max_new_tokens,
                adapter=None,
                mem_tokens=[],
                memory_prompt=mem_text,
                stop=qa_settings.stop_value(),
                prompt_style=qa_settings.prompt_style,
                stop_mode=qa_settings.stop_mode,
                template_policy=qa_settings.template_policy,
            )
            ptoks = int(out.get("prompt_tokens", 0))
            gtoks = int(out.get("generated_tokens", 0))
            total_prompt_tokens += ptoks
            total_generated_tokens += gtoks
            compute.prompt_tokens = (compute.prompt_tokens or 0) + ptoks
            compute.generated_tokens = (compute.generated_tokens or 0) + gtoks
            if diag_entry is not None:
                diag_entry["prompt_tokens"] = _safe_int(diag_entry.get("prompt_tokens")) + ptoks
                diag_entry["generated_tokens"] = (
                    _safe_int(diag_entry.get("generated_tokens")) + gtoks
                )
            pred = ""
            top_group = final_candidates_raw[0].get("group_id") if final_candidates_raw else None
            if selected_traces:
                top_answers = selected_traces[0].get("answers", [])
                if top_group == group_id:
                    pred = str(truth)
                elif top_answers:
                    base = str(top_answers[0])
                    pred = f"{base} (retrieval miss)"
                else:
                    pred = "retrieval miss"
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
            first_tokens.append(pred.split()[0] if pred else "")
        else:
            itm_list, comp = _evaluate_records(
                [rec],
                geom,
                qa_settings,
                adapter=adapter,
                mem_tokens=mem_tokens,
                mem_text=mem_text,
            )
            items.extend(itm_list)
            first_token = ""
            if itm_list:
                prediction = itm_list[0].prediction.strip()
                if prediction:
                    first_token = prediction.split()[0]
                first_tokens.append(first_token)
            compute.attention_flops = (compute.attention_flops or 0) + (comp.attention_flops or 0)
            compute.kv_cache_bytes = (compute.kv_cache_bytes or 0) + (comp.kv_cache_bytes or 0)
            comp_prompt_tokens = int(comp.prompt_tokens or 0)
            comp_generated_tokens = int(comp.generated_tokens or 0)
            total_prompt_tokens += comp_prompt_tokens
            total_generated_tokens += comp_generated_tokens
            compute.prompt_tokens = (compute.prompt_tokens or 0) + comp_prompt_tokens
            compute.generated_tokens = (compute.generated_tokens or 0) + comp_generated_tokens
            if diag_entry is not None:
                diag_entry["prompt_tokens"] = (
                    _safe_int(diag_entry.get("prompt_tokens")) + comp_prompt_tokens
                )
                diag_entry["generated_tokens"] = (
                    _safe_int(diag_entry.get("generated_tokens")) + comp_generated_tokens
                )
        _maybe_run_eviction()
    baseline_compute = _run_baseline_with_recalls(baseline, records, model, tok, items)
    b0_latency = cast(float, _aggregate_metrics(b0_items)["latency"])
    b1_latency = cast(float, _aggregate_metrics(items)["latency"])
    retrieval = {
        "p_at_1": precision_at_k(final_cand_groups, truths, 1),
        "baseline_p_at_1": precision_at_k(baseline_cand_groups, truths, 1),
        "mrr": mrr(final_cand_groups, truths),
        "baseline_mrr": mrr(baseline_cand_groups, truths),
        "near_miss_rate": near_miss_rate(diagnostics),
        "collision_rate": collision_rate(diagnostics),
        "completion_lift": completion_lift(baseline_top1, hopfield_top1),
        "hopfield_rank_improved_rate": hopfield_rank_improved_rate(hopfield_diagnostics),
    }
    if trace_writer is not None:
        persisted_records = trace_writer.records
        pointer_check = _pointer_summary_from_payloads(persisted_records)
        final_trace_samples = persisted_samples if persisted_samples else trace_samples
    else:
        pointer_check = _pointer_summary(
            total=pointer_total,
            missing_pointer=pointer_missing,
            with_pointer=pointer_with,
            banned_keys=pointer_banned,
        )
        final_trace_samples = trace_samples
    gate_info = _summarize_gate(gate_diagnostics, pins_only=pins_only)
    gate_info["weights"] = {
        "alpha": gate_module.alpha,
        "beta": gate_module.beta,
        "gamma": gate_module.gamma,
        "delta": gate_module.delta,
    }
    gate_info["threshold"] = gate_module.threshold
    gate_info["pin_override"] = bool(getattr(gate_module, "pin_override", False))
    gate_info["used_for_writes"] = bool(gate_use_for_writes)
    gate_info["debug_keep_labels"] = bool(gate_debug_keep_labels)
    gate_info["allow_label_fallback"] = bool(gate_allow_label_fallback)
    gate_info["indexed_records"] = len(indexed_records)
    gate_info["label_positive_records"] = sum(
        1 for rec in records if bool(rec.get("should_remember"))
    )
    if isinstance(gate_info.get("telemetry"), dict):
        telemetry = gate_info["telemetry"]
        telemetry["pins_only_eval"] = pins_only
        if "writes_per_1k_tokens" not in telemetry:
            telemetry["writes_per_1k_tokens"] = gate_info.get("write_rate_per_1k_tokens")
        telemetry.setdefault("writes_per_1k_records", gate_info.get("write_rate_per_1k_records"))
        telemetry.setdefault("prompt_tokens", gate_info.get("prompt_tokens"))
        telemetry.setdefault("generated_tokens", gate_info.get("generated_tokens"))
        telemetry["writes_per_1k"] = telemetry.get("writes_per_1k_tokens")
    gate_info.setdefault("prompt_tokens", total_prompt_tokens)
    gate_info.setdefault("generated_tokens", total_generated_tokens)
    store_write_rate = gate_info.get("store_write_rate")
    gate_info["store_write_rate_per_1k_records"] = (
        float(store_write_rate) * 1000.0 if isinstance(store_write_rate, int | float) else None
    )
    gate_info["pins_only_eval"] = pins_only
    gate_info["pointer_check"] = pointer_check
    if final_trace_samples:
        gate_info["trace_samples"] = final_trace_samples
    store_ntotal = len(indexed_records)
    store_obj = getattr(service, "store", None)
    if store_obj is not None:
        index_obj = getattr(store_obj, "index", None)
        if index_obj is not None:
            faiss_index = getattr(index_obj, "index", None)
            if faiss_index is not None and hasattr(faiss_index, "ntotal"):
                store_ntotal = int(cast(Any, faiss_index).ntotal)
            elif hasattr(index_obj, "ntotal"):
                store_ntotal = int(cast(Any, index_obj).ntotal)
        elif hasattr(store_obj, "ntotal"):
            store_ntotal = int(cast(Any, store_obj).ntotal)
    store_info: dict[str, Any] = {
        "ntotal": store_ntotal,
        "indexed_records": len(indexed_records),
        "evicted_count": evicted_total,
        "eviction_runs": eviction_runs if store_evict_enabled else 0,
        "eviction_interval": eviction_interval if store_evict_enabled else None,
        "evict_stale_enabled": store_evict_enabled,
    }
    extra = {
        "adapter_latency_overhead_s": b1_latency - b0_latency,
        "retrieval": retrieval,
        "gate": gate_info,
        "store": store_info,
        "debug": {
            "mem_len": mem_lengths,
            "mem_preview": preview_tokens or [],
            "mem_preview_str": " ".join(preview_tokens) if preview_tokens else "",
            "first_token": first_tokens,
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
    if args.mode == "B1":
        resolved_mem_max_tokens: int | None = (
            args.mem_max_tokens
            if args.mem_max_tokens is not None
            else _scenario_default_mem_max_tokens(args.scenario)
        )
    else:
        resolved_mem_max_tokens = None

    if records:
        from hei_nw.models.base import load_base

        tok, model, _ = load_base(model_id=args.model, quant_4bit=False)
        geom = _model_geometry(model)
        dg_keyer = DGKeyer(k=args.dg_k) if args.mode == "B1" else None
        dev_settings = DevIsolationSettings(
            retrieval_only=args.dev_retrieval_only,
            oracle_trace=args.dev_oracle_trace,
        )
        gate_module = NeuromodulatedGate(
            alpha=args.gate_alpha,
            beta=args.gate_beta,
            gamma=args.gate_gamma,
            delta=args.gate_delta,
            threshold=args.gate_threshold,
            pin_override=args.gate_pin_override,
        )
        handler_kwargs: dict[str, Any] = {}
        if args.mode == "B1":
            if resolved_mem_max_tokens is None:
                msg = "mem_max_tokens must be resolved for B1 runs"
                raise RuntimeError(msg)
            handler_kwargs["mem_max_tokens"] = resolved_mem_max_tokens
            handler_kwargs["adapter_scale"] = args.adapter_scale
            handler_kwargs["gate"] = gate_module
            handler_kwargs["pins_only"] = args.eval_pins_only
            handler_kwargs["gate_use_for_writes"] = args.gate_use_for_writes
            handler_kwargs["gate_debug_keep_labels"] = args.gate_debug_keep_labels
            handler_kwargs["gate_allow_label_fallback"] = args.gate_allow_label_fallback
            handler_kwargs["store_evict_stale"] = args.store_evict_stale
            handler_kwargs["store_evict_interval"] = args.store_evict_interval
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
            "template_policy": qa_settings.template_policy,
            "stop_mode": qa_settings.stop_mode,
            "omit_episode": qa_settings.omit_episode,
            "memory_dependent_baseline": qa_settings.memory_dependent_baseline,
        },
        "memory": {"max_tokens": resolved_mem_max_tokens},
        "adapter": {"scale": args.adapter_scale if args.mode == "B1" else None},
        "hopfield": {
            "enabled": not args.no_hopfield,
            "steps": hopfield_settings.steps,
            "temperature": hopfield_settings.temperature,
        },
        "gate": {
            "alpha": args.gate_alpha,
            "beta": args.gate_beta,
            "gamma": args.gate_gamma,
            "delta": args.gate_delta,
            "threshold": args.gate_threshold,
            "pin_override": args.gate_pin_override,
            "pins_only": args.eval_pins_only,
            "use_for_writes": args.gate_use_for_writes,
            "debug_keep_labels": args.gate_debug_keep_labels,
            "allow_label_fallback": args.gate_allow_label_fallback,
        },
        "store": {
            "evict_stale": args.store_evict_stale,
            "evict_interval": args.store_evict_interval,
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


def _persist_gate_record(
    writer: TraceWriter,
    record: Mapping[str, Any],
    decision: GateDecision,
    *,
    record_index: int,
) -> dict[str, Any]:
    """Persist *record* via *writer* and return the stored payload."""

    episode_text = str(record.get("episode_text", "") or "")
    group_raw = record.get("group_id")
    if isinstance(group_raw, int | str) and str(group_raw):
        doc_prefix = f"group-{group_raw}"
    else:
        doc_prefix = f"record-{record_index}"
    digest_source = episode_text or json.dumps(record.get("answers", []), ensure_ascii=False)
    digest = hashlib.sha256(digest_source.encode("utf8")).hexdigest()[:8]
    pointer_doc = f"{doc_prefix}-{digest}"
    span_length = max(len(episode_text), 1)
    pointer = {"doc": pointer_doc, "start": 0, "end": span_length}

    answers = record.get("answers")
    if isinstance(answers, Sequence):
        slot_values = [str(ans) for ans in answers]
    else:
        slot_values = []
    entity_slots = {
        "who": slot_values[0] if len(slot_values) > 0 else "",
        "what": slot_values[1] if len(slot_values) > 1 else "",
        "where": slot_values[2] if len(slot_values) > 2 else "",
        "when": slot_values[3] if len(slot_values) > 3 else "",
    }

    trace_id_raw = record.get("trace_id")
    if isinstance(trace_id_raw, str) and trace_id_raw.strip():
        trace_id = trace_id_raw.strip()
    elif trace_id_raw is not None:
        candidate = str(trace_id_raw).strip()
        trace_id = candidate or f"{pointer_doc}:{record_index}"
    else:
        trace_id = f"{pointer_doc}:{record_index}"

    return writer.write(
        trace_id=trace_id,
        pointer=pointer,
        entity_slots=entity_slots,
        decision=decision,
    )


def _pointer_payload_has_pointer(pointer: Any) -> bool:
    """Return ``True`` if *pointer* describes a valid span."""

    if not isinstance(pointer, Mapping):
        return False
    doc = pointer.get("doc")
    start_raw = pointer.get("start")
    end_raw = pointer.get("end")
    if not isinstance(doc, str) or not doc.strip():
        return False
    try:
        start_val = int(start_raw) if start_raw is not None else None
        end_val = int(end_raw) if end_raw is not None else None
    except (TypeError, ValueError):
        return False
    if start_val is None or end_val is None:
        return False
    if start_val < 0 or end_val <= start_val:
        return False
    return True


def _persisted_sample_from_payload(payload: Mapping[str, Any], group_id: Any) -> dict[str, Any]:
    """Return a lightweight audit sample derived from *payload*."""

    pointer = payload.get("tokens_span_ref")
    has_pointer = _pointer_payload_has_pointer(pointer)
    sample: dict[str, Any] = {
        "trace_id": payload.get("trace_id"),
        "group_id": group_id,
        "has_pointer": has_pointer,
        "banned_keys": [key for key in _BANNED_TRACE_KEYS if key in payload],
    }
    if has_pointer and isinstance(pointer, Mapping):
        sample["pointer"] = {key: pointer.get(key) for key in ("doc", "start", "end")}
    entity_slots = payload.get("entity_slots")
    if isinstance(entity_slots, Mapping):
        sample["entity_slots_keys"] = sorted(str(key) for key in entity_slots.keys())
    return sample


def _pointer_summary_from_payloads(payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return pointer diagnostics computed from persisted payloads."""

    missing_pointer = 0
    banned_counts: dict[str, int] = {}
    for payload in payloads:
        pointer = payload.get("tokens_span_ref")
        if not _pointer_payload_has_pointer(pointer):
            missing_pointer += 1
        for banned in _BANNED_TRACE_KEYS:
            if banned in payload and payload[banned]:
                banned_counts[banned] = banned_counts.get(banned, 0) + 1
    total = len(payloads)
    with_pointer = total - missing_pointer
    return _pointer_summary(
        total=total,
        missing_pointer=missing_pointer,
        with_pointer=with_pointer,
        banned_keys=set(banned_counts),
        banned_key_counts=banned_counts,
    )


def _pointer_summary(
    *,
    total: int,
    missing_pointer: int,
    with_pointer: int,
    banned_keys: set[str],
    banned_key_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return pointer-only diagnostic metadata."""

    counts = {
        key: int(value)
        for key, value in sorted((banned_key_counts or {}).items())
        if int(value) > 0
    }
    all_banned = set(banned_keys) | set(counts)
    pointer_only: bool | None
    if total == 0:
        pointer_only = None
    else:
        pointer_only = missing_pointer == 0 and not all_banned
    return {
        "checked": total,
        "with_pointer": with_pointer,
        "missing_pointer": missing_pointer,
        "banned_keys": sorted(all_banned),
        "banned_key_counts": counts,
        "pointer_only": pointer_only,
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
