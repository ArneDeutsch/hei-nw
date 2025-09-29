"""Neuromodulated write gate calibration CLI.

The historical Bash ``scripts/run_m3_gate_calibration.sh`` grew large and
contained a mix of argument parsing, harness orchestration, calibration plot
rendering, sweep logic, and file bookkeeping. This Python module hosts the core
implementation so the public ``scripts`` entry point can remain a thin wrapper.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from hei_nw import datasets
from hei_nw.cli.plot_gate_calibration import render_plot
from hei_nw.cli.report_gate_write_rates import summarize_write_rates
from hei_nw.eval import harness

DEFAULT_THRESHOLD_SWEEP = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
TargetMetric = Literal["tokens", "records"]

__all__ = ["main", "GateCalibrationRunner", "CalibrationConfig"]


@dataclass(slots=True)
class CalibrationConfig:
    """Immutable configuration resolved from CLI arguments."""

    scenario: str
    n: int
    seed: int
    model: str
    threshold: float
    threshold_provided: bool
    threshold_mode: Literal["single", "manual", "auto"]
    threshold_values: list[float]
    target_band: tuple[float, float] | None
    target_per: TargetMetric
    target_value: float | None
    target_tolerance: float | None
    out_dir: Path
    plot_title: str | None
    pin_eval: bool

    def metrics_base(self) -> str:
        return f"{self.scenario}_B1"

    def summary_suffix(self) -> str:
        return "_pins" if self.pin_eval else ""

    def telemetry_suffix(self) -> str:
        return "_pins" if self.pin_eval else ""


@dataclass(slots=True)
class CalibrationResult:
    """Artifacts and summary statistics collected for a single τ run."""

    tau: float
    tau_text: str
    metrics_path: Path
    telemetry_path: Path
    calibration_plot: Path
    trace_samples_path: Path
    writes: float | None
    total: float | None
    write_rate: float | None
    writes_per_1k_tokens: float | None
    writes_per_1k_records: float | None
    pr_auc: float | None
    target_metric_name: str
    target_metric_value: float | None

    @property
    def run_dir(self) -> Path:
        return self.metrics_path.parent


def _format_tau(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _sanitize_tau(value: float) -> str:
    return _format_tau(value).replace("/", "_").replace(" ", "")


def _format_numeric(value: float | None) -> str:
    if value is None:
        return ""
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text or "0"


def _parse_threshold_sweep(raw: str | None) -> tuple[Literal["manual", "auto"], list[float]]:
    if raw is None:
        return "manual", []
    normalized = raw.strip().lower()
    if normalized == "auto":
        return "auto", []
    cleaned = raw.replace(",", " ")
    values: list[float] = []
    for token in cleaned.split():
        try:
            values.append(float(token))
        except ValueError as exc:  # pragma: no cover - argparse already validates floats
            raise argparse.ArgumentTypeError(f"Invalid τ value: {token}") from exc
    if not values:
        raise argparse.ArgumentTypeError("--threshold-sweep requires at least one τ")
    return "manual", values


def _parse_target_band(raw: str | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    cleaned = raw.replace(",", " ")
    tokens = [token for token in cleaned.split() if token]
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError("--target-band expects two values, e.g. '1,5'")
    low, high = (float(token) for token in tokens)
    if low > high:
        low, high = high, low
    return low, high


def _compute_target_tolerance(value: float | None, band: tuple[float, float] | None) -> float | None:
    if value is None:
        return None
    if band is not None:
        low, high = band
        return max(abs(value - low), abs(high - value))
    base = max(abs(value), 1.0)
    return max(0.1, 0.05 * base)


def _midpoint(a: float, b: float) -> float:
    return (a + b) / 2.0


def _increase_tau(current: float) -> float:
    if current <= 0.0:
        return 0.5
    return current * 2.0


def _decrease_tau(current: float) -> float:
    if current <= 0.0:
        return 0.25
    candidate = current / 2.0
    if candidate <= 0.0:
        candidate = 1e-4
    return candidate


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _extract_gate_summary(data: dict[str, Any]) -> dict[str, Any]:
    gate = data.get("gate") or {}
    telemetry = gate.get("telemetry") or {}
    summary: dict[str, Any] = {
        "writes": _as_float(gate.get("writes")),
        "total": _as_float(gate.get("total")),
        "write_rate": _as_float(gate.get("write_rate")),
        "writes_per_1k_tokens": _as_float(gate.get("write_rate_per_1k_tokens")),
        "writes_per_1k_records": _as_float(gate.get("write_rate_per_1k_records")),
        "pr_auc": _as_float(gate.get("pr_auc")),
        "prompt_tokens": _as_int(gate.get("prompt_tokens")),
        "generated_tokens": _as_int(gate.get("generated_tokens")),
    }
    if summary["writes_per_1k_tokens"] is None:
        tokens_val = telemetry.get("writes_per_1k_tokens")
        summary["writes_per_1k_tokens"] = _as_float(tokens_val)
    if summary["writes_per_1k_records"] is None:
        clutter = telemetry.get("writes_per_1k_records")
        value = _as_float(clutter)
        if value is not None:
            summary["writes_per_1k_records"] = value
        else:
            write_rate = summary["write_rate"]
            if write_rate is not None:
                summary["writes_per_1k_records"] = write_rate * 1000.0
    if summary["writes_per_1k_tokens"] is None:
        total_tokens = 0
        for component in (summary["prompt_tokens"], summary["generated_tokens"]):
            if component is not None:
                total_tokens += component
        if total_tokens > 0:
            writes = summary["writes"]
            if writes is None:
                writes = _as_float(telemetry.get("writes"))
            if writes is not None:
                summary["writes_per_1k_tokens"] = writes / (total_tokens / 1000.0)
    return summary


def _build_telemetry(
    metrics: dict[str, Any],
    *,
    model: str,
    n_value: int,
    seed_value: int,
    pins_only: bool,
) -> dict[str, Any]:
    gate = metrics.get("gate") or {}
    telemetry = dict(gate.get("telemetry") or {})
    dataset = metrics.get("dataset") or {}
    telemetry["scenario"] = dataset.get("scenario")
    telemetry["threshold"] = gate.get("threshold")
    telemetry["writes"] = gate.get("writes")
    telemetry["total"] = gate.get("total")
    telemetry["write_rate"] = gate.get("write_rate")
    telemetry["write_rate_per_1k_tokens"] = gate.get("write_rate_per_1k_tokens")
    telemetry["write_rate_per_1k_records"] = gate.get("write_rate_per_1k_records")
    telemetry["write_rate_per_1k"] = telemetry.get("write_rate_per_1k_tokens")
    telemetry["generated_tokens"] = gate.get("generated_tokens")
    telemetry["pinned"] = gate.get("pinned")
    telemetry["reward_flags"] = gate.get("reward_flags")
    telemetry["pins_only_eval"] = pins_only
    if model:
        telemetry["model"] = model
    telemetry["n"] = n_value
    telemetry["seed"] = seed_value
    return telemetry


def _write_trace_samples(gate_section: dict[str, Any], trace_path: Path) -> None:
    samples = gate_section.get("trace_samples")
    if isinstance(samples, list) and samples:
        trace_path.write_text(json.dumps({"trace_samples": samples}, indent=2), encoding="utf8")
    elif trace_path.exists():
        trace_path.unlink()


def _invoke_harness(arg_list: list[str]) -> int:
    """Invoke ``hei_nw.eval.harness.main`` while supporting stubbed entry points."""

    try:
        signature = inspect.signature(harness.main)
    except (TypeError, ValueError):  # pragma: no cover - extremely defensive
        signature = None

    if signature is not None and len(signature.parameters) == 0:
        argv_backup = sys.argv.copy()
        try:
            sys.argv = ["hei_nw.eval.harness", *arg_list]
            try:
                result = harness.main()
            except SystemExit as exc:  # pragma: no cover - harness may call sys.exit
                return int(exc.code or 0)
            return int(result or 0)
        finally:
            sys.argv = argv_backup

    try:
        result = harness.main(arg_list)
    except SystemExit as exc:  # pragma: no cover - harness may call sys.exit
        return int(exc.code or 0)
    return int(result or 0)


class GateCalibrationRunner:
    """Execute calibration runs for one or more τ values."""

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        self.results: list[CalibrationResult] = []

    # ----------------------------- public API -----------------------------

    def run(self) -> None:
        self.config.out_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_pin_records()
        if self.config.threshold_mode == "auto":
            auto_result = self._run_auto_sweep()
            self._persist_auto_selected(auto_result)
            self._generate_summaries(auto_result)
            return
        if self.config.threshold_values:
            print(
                f"[m3] Performing threshold sweep for scenario {self.config.scenario}: "
                f"{' '.join(_format_tau(t) for t in self.config.threshold_values)}"
            )
            for tau in self.config.threshold_values:
                run_dir = self.config.out_dir / f"tau_{_sanitize_tau(tau)}"
                self._run_single_threshold(
                    tau,
                    run_dir,
                    self._title_for_tau(tau),
                )
            self._generate_summaries(None)
            return
        # Single τ run
        self._run_single_threshold(
            self.config.threshold,
            self.config.out_dir,
            self.config.plot_title,
        )
        print(f"[m3] Calibration assets written to {self.config.out_dir}")

    # --------------------------- helper routines --------------------------

    def _title_for_tau(self, tau: float) -> str | None:
        if not self.config.plot_title:
            return None
        return f"{self.config.plot_title} (τ={_format_tau(tau)})"

    def _ensure_pin_records(self) -> None:
        if not self.config.pin_eval:
            return
        print(
            f"[m3] Checking for pinned records in scenario {self.config.scenario} "
            f"(n={self.config.n}, seed={self.config.seed})"
        )
        module = getattr(datasets, f"scenario_{self.config.scenario.lower()}", None)
        if module is None or not hasattr(module, "generate"):
            raise SystemExit(f"Unknown scenario '{self.config.scenario}' for pin evaluation")
        records = module.generate(self.config.n, seed=self.config.seed)
        has_pin = any(
            isinstance(record, dict)
            and isinstance(record.get("gate_features"), dict)
            and record["gate_features"].get("pin")
            for record in records
        )
        if not has_pin:
            print(
                f"[m3] Scenario {self.config.scenario} has no pinned records for the provided seed.",
                file=sys.stderr,
            )
            print("[m3] Try a different scenario/seed or omit --pin-eval.", file=sys.stderr)
            raise SystemExit(3)

    def _run_harness(self, tau: float, out_dir: Path) -> Path:
        args = [
            "--mode",
            "B1",
            "--scenario",
            self.config.scenario,
            "-n",
            str(self.config.n),
            "--seed",
            str(self.config.seed),
            "--model",
            self.config.model,
            "--outdir",
            str(out_dir),
            "--gate.threshold",
            _format_tau(tau),
            "--no-gate.allow_label_fallback",
        ]
        if self.config.pin_eval:
            args.append("--eval.pins_only")
        print(
            f"[m3] Running harness for scenario {self.config.scenario} at τ={_format_tau(tau)} "
            f"(n={self.config.n}, seed={self.config.seed})"
            + (" (pins-only)" if self.config.pin_eval else "")
        )
        exit_code = _invoke_harness(args)
        if exit_code != 0:
            raise SystemExit(exit_code)
        metrics_path = out_dir / f"{self.config.metrics_base()}_metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Expected metrics file not found: {metrics_path}")
        return metrics_path

    def _run_single_threshold(
        self,
        tau: float,
        out_dir: Path,
        plot_title: str | None,
        *,
        log_result: bool = True,
    ) -> CalibrationResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = self._run_harness(tau, out_dir)
        print(
            f"[m3] Extracting gate telemetry for τ={_format_tau(tau)}"
            + (" (pins-only)" if self.config.pin_eval else "")
        )
        metrics = _load_metrics(metrics_path)
        telemetry_path = out_dir / f"{self.config.scenario}_gate_telemetry{self.config.telemetry_suffix()}.json"
        trace_path = out_dir / f"{self.config.scenario}_trace_samples{self.config.telemetry_suffix()}.json"
        telemetry = _build_telemetry(
            metrics,
            model=self.config.model,
            n_value=self.config.n,
            seed_value=self.config.seed,
            pins_only=self.config.pin_eval,
        )
        telemetry_path.write_text(json.dumps(telemetry, indent=2), encoding="utf8")
        gate_section = metrics.get("gate") or {}
        _write_trace_samples(gate_section, trace_path)

        plot_path = out_dir / f"{self.config.scenario}_gate_calibration{self.config.telemetry_suffix()}.png"
        print(
            f"[m3] Rendering calibration curve for τ={_format_tau(tau)}"
            + (" (pins-only)" if self.config.pin_eval else "")
        )
        try:
            render_plot(
                telemetry,
                plot_path,
                telemetry_path=telemetry_path,
                metrics_path=metrics_path,
                title=plot_title,
                pins_only=self.config.pin_eval,
                overlay_nonpins=self.config.pin_eval,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

        summary = _extract_gate_summary(metrics)
        target_metric_name = (
            "writes_per_1k_records" if self.config.target_per == "records" else "writes_per_1k_tokens"
        )
        result = CalibrationResult(
            tau=tau,
            tau_text=_format_tau(tau),
            metrics_path=metrics_path,
            telemetry_path=telemetry_path,
            calibration_plot=plot_path,
            trace_samples_path=trace_path,
            writes=summary["writes"],
            total=summary["total"],
            write_rate=summary["write_rate"],
            writes_per_1k_tokens=summary["writes_per_1k_tokens"],
            writes_per_1k_records=summary["writes_per_1k_records"],
            pr_auc=summary["pr_auc"],
            target_metric_name=target_metric_name,
            target_metric_value=summary[target_metric_name],
        )
        self.results.append(result)
        if log_result:
            self._log_run_result(result)
        return result

    def _log_run_result(self, result: CalibrationResult) -> None:
        metric_label = "records" if self.config.target_per == "records" else "tokens"
        metric_value = result.target_metric_value
        text = "—" if metric_value is None else f"{metric_value:.6f}".rstrip("0").rstrip(".")
        print(
            f"[m3] τ={result.tau_text} yields {text} writes/1k {metric_label}"
        )

    def _evaluate_run(
        self,
        result: CalibrationResult,
    ) -> tuple[str, float | None]:
        metric_value = result.target_metric_value
        if metric_value is None:
            return "invalid", None
        if self.config.target_value is not None:
            diff = metric_value - self.config.target_value
            tolerance = self.config.target_tolerance
            if tolerance is not None and math.isfinite(tolerance) and abs(diff) <= tolerance:
                return "within", abs(diff)
            if diff > 0:
                return "above", abs(diff)
            if diff < 0:
                return "below", abs(diff)
            return "within", 0.0
        if self.config.target_band is not None:
            low, high = self.config.target_band
            if metric_value < low:
                return "below", low - metric_value
            if metric_value > high:
                return "above", metric_value - high
            return "within", 0.0
        return "unknown", None

    def _run_auto_sweep(self) -> CalibrationResult:
        if self.config.target_value is not None:
            if self.config.target_tolerance is not None:
                print(
                    f"[m3] Auto threshold sweep targeting {self.config.target_value} writes/1k {self.config.target_per} "
                    f"(tolerance ±{self.config.target_tolerance})"
                )
            else:
                print(
                    f"[m3] Auto threshold sweep targeting {self.config.target_value} writes/1k {self.config.target_per}"
                )
        elif self.config.target_band is not None:
            low, high = self.config.target_band
            print(
                f"[m3] Auto threshold sweep target band: [{low}, {high}] per 1k {self.config.target_per}"
            )
        else:
            print("[m3] Auto threshold sweep enabled")
        print(f"[m3] Starting search from τ={_format_tau(self.config.threshold)}")

        tau_low: float | None = None
        tau_high: float | None = None
        tau_current = self.config.threshold
        seen: set[str] = set()
        best: CalibrationResult | None = None
        best_diff: float | None = None

        for _ in range(20):
            tau_key = _format_tau(tau_current)
            if tau_key in seen:
                print(f"[m3] Auto sweep encountered repeated τ={tau_key}; stopping search", file=sys.stderr)
                break
            seen.add(tau_key)
            run_dir = self.config.out_dir / f"tau_{_sanitize_tau(tau_current)}"
            result = self._run_single_threshold(
                tau_current,
                run_dir,
                self._title_for_tau(tau_current),
                log_result=False,
            )
            direction, diff = self._evaluate_run(result)
            if direction == "invalid":
                raise SystemExit(
                    f"[m3] Unable to determine writes per 1k {self.config.target_per} for τ={result.tau_text}"
                )
            metric_label = "records" if self.config.target_per == "records" else "tokens"
            if diff is not None:
                diff_text = f"{diff:.6f}".rstrip("0").rstrip(".") or "0"
                if self.config.target_value is not None:
                    print(
                        f"[m3] τ={result.tau_text} yields {result.target_metric_value:.6f} writes/1k {metric_label} (Δ={diff_text}; {direction} target)"
                    )
                else:
                    print(
                        f"[m3] τ={result.tau_text} yields {result.target_metric_value:.6f} writes/1k {metric_label} ({direction})"
                    )
            else:
                print(
                    f"[m3] τ={result.tau_text} yields {result.target_metric_value} writes/1k {metric_label}"
                )

            update_best = False
            if direction == "within":
                if best is None:
                    update_best = True
                else:
                    assert diff is not None
                    assert best_diff is not None
                    if diff < best_diff:
                        update_best = True
                    elif math.isclose(diff, best_diff, rel_tol=1e-9, abs_tol=1e-9) and result.tau < best.tau:
                        update_best = True
            elif best is None and direction != "invalid":
                update_best = True

            if update_best and diff is not None:
                best = result
                best_diff = diff

            if direction == "within":
                tau_high = result.tau
                if tau_low is not None:
                    tau_current = _midpoint(tau_low, tau_high)
                else:
                    tau_current = _decrease_tau(result.tau)
                continue
            if direction == "above":
                tau_low = result.tau
                tau_current = _midpoint(tau_low, tau_high) if tau_high is not None else _increase_tau(result.tau)
                continue
            tau_high = result.tau
            tau_current = _midpoint(tau_low, tau_high) if tau_low is not None else _decrease_tau(result.tau)
        if best is None:
            raise SystemExit("Auto sweep failed to locate a suitable τ")
        print(
            f"[m3] Auto-selected τ={best.tau_text} with {best.target_metric_value:.6f} writes/1k {self.config.target_per}"
            + (" (Δ={:.6f})".format(best_diff) if best_diff is not None and self.config.target_value is not None else "")
        )
        return best

    def _persist_auto_selected(self, result: CalibrationResult) -> None:
        out_path = self.config.out_dir / f"{self.config.scenario}_auto_selected_tau{self.config.summary_suffix()}.json"
        direction, diff = self._evaluate_run(result)

        data: dict[str, Any] = {
            "scenario": self.config.scenario,
            "tau": float(result.tau),
            "metric": result.target_metric_name,
            "metric_value": result.target_metric_value,
            "pins_only": self.config.pin_eval or None,
            "target_metric": self.config.target_per,
        }
        if self.config.target_value is not None:
            data["target_value"] = self.config.target_value
        if self.config.target_tolerance is not None:
            data["tolerance"] = self.config.target_tolerance
        if self.config.target_band is not None:
            low, high = self.config.target_band
            data["target_band"] = {"lower": low, "upper": high}
        data["metrics_path"] = str(result.metrics_path)
        if result.write_rate is not None:
            data["write_rate"] = result.write_rate
        if result.writes_per_1k_tokens is not None:
            data["writes_per_1k_tokens"] = result.writes_per_1k_tokens
        if result.writes_per_1k_records is not None:
            data["writes_per_1k_records"] = result.writes_per_1k_records
        if diff is not None:
            data["delta_from_target"] = diff
        if direction in {"above", "below", "within"}:
            data["target_relation"] = direction
        out_path.write_text(json.dumps(data, indent=2), encoding="utf8")
        print(f"[m3] Auto-selected τ metadata written to {out_path}")

    def _generate_summaries(self, auto_result: CalibrationResult | None) -> None:
        metrics_paths = [result.metrics_path for result in self.results]
        if not metrics_paths:
            return
        summary_json = self.config.out_dir / f"{self.config.scenario}_sweep_summary{self.config.summary_suffix()}.json"
        summary_tsv = self.config.out_dir / f"{self.config.scenario}_sweep_summary{self.config.summary_suffix()}.tsv"
        auto_tau = auto_result.tau if auto_result is not None else None
        auto_metric_name = auto_result.target_metric_name if auto_result is not None else None
        auto_metric_value = auto_result.target_metric_value if auto_result is not None else None
        print(f"[m3] Generating sweep summary at {summary_json}")
        summarize_write_rates(
            metrics_paths,
            summary_json,
            target_band=self.config.target_band,
            target_per=self.config.target_per,
            auto_selected_tau=auto_tau,
            auto_selected_metric=auto_metric_name,
            auto_selected_metric_value=auto_metric_value,
            target_value=self.config.target_value,
        )
        self._write_summary_tsv(summary_tsv)
        self._write_threshold_index()

    def _write_summary_tsv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(
                [
                    "scenario",
                    "tau",
                    "writes",
                    "write_rate",
                    "writes_per_1k_tokens",
                    "writes_per_1k_records",
                    "pr_auc",
                ]
            )
            for result in self.results:
                writer.writerow(
                    [
                        self.config.scenario,
                        result.tau_text,
                        _format_numeric(result.writes),
                        _format_numeric(result.write_rate),
                        _format_numeric(result.writes_per_1k_tokens),
                        _format_numeric(result.writes_per_1k_records),
                        _format_numeric(result.pr_auc),
                    ]
                )

    def _write_threshold_index(self) -> None:
        if not self.results:
            return
        index_path = self.config.out_dir / f"{self.config.scenario}_threshold_sweep{self.config.summary_suffix()}.md"
        lines = [f"# {self.config.scenario} Threshold Sweep", ""]
        for result in sorted(self.results, key=lambda item: item.tau):
            metrics_rel = os.path.relpath(result.metrics_path, self.config.out_dir)
            telemetry_rel = os.path.relpath(result.telemetry_path, self.config.out_dir)
            lines.append(
                f"- τ={result.tau_text}: `{metrics_rel}` · `{telemetry_rel}`"
            )
        index_path.write_text("\n".join(lines) + "\n", encoding="utf8")


def _resolve_target_metric(args: argparse.Namespace) -> TargetMetric:
    metric = args.target_per
    if args.target_rate_per_1k is not None and args.target_rate_per_1k != metric:
        raise SystemExit("Conflicting target metric specifications")
    return metric


def parse_args(argv: Sequence[str] | None = None) -> CalibrationConfig:
    env_scenario = os.environ.get("SCENARIO", "A")
    env_n = os.environ.get("N")
    env_seed = os.environ.get("SEED")
    env_threshold = os.environ.get("THRESHOLD")
    env_out = os.environ.get("OUT", "reports/m3-write-gate")
    env_model = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    env_title = os.environ.get("TITLE") or os.environ.get("PLOT_TITLE")

    parser = argparse.ArgumentParser(
        description="Runs the B1 harness with gate telemetry enabled and produces calibration assets.",
    )
    parser.add_argument(
        "--scenario",
        default=env_scenario,
        help=f"Scenario identifier (default: {env_scenario})",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=int(env_n) if env_n is not None else 48,
        help="Number of records to evaluate (default: 48)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(env_seed) if env_seed is not None else 13,
        help="Random seed (default: 13)",
    )
    parser.add_argument(
        "--model",
        default=env_model,
        help=f"Model identifier (default: {env_model})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(env_threshold) if env_threshold is not None else None,
        help="Gate threshold τ (default: 1.5)",
    )
    parser.add_argument(
        "--threshold-sweep",
        type=str,
        default=None,
        help="Space/comma-separated list or 'auto' to search τ automatically",
    )
    parser.add_argument(
        "--target-band",
        type=str,
        default=None,
        help="Desired writes/1k tokens band for auto sweep (default: 1,5)",
    )
    parser.add_argument(
        "--target-per",
        choices=("tokens", "records"),
        default="tokens",
        help="Metric used for the target band evaluation (default: tokens)",
    )
    parser.add_argument(
        "--target-rate-per-1k",
        choices=("tokens", "records"),
        default=None,
        help="Metric used for the auto target evaluation (default: tokens)",
    )
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Desired writes per 1k metric value for auto τ selection",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(env_out),
        help="Output directory (default: reports/m3-write-gate)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=env_title,
        help="Custom title for the calibration plot",
    )
    parser.add_argument("--pin-eval", action="store_true", help="Evaluate pins-only slice")
    args = parser.parse_args(argv)

    scenario = args.scenario.upper()
    threshold_mode, sweep_values = _parse_threshold_sweep(args.threshold_sweep)
    target_band = _parse_target_band(args.target_band)
    band_provided = target_band is not None
    if target_band is None:
        target_band = (1.0, 5.0)
    target_per = _resolve_target_metric(args)

    threshold_env_provided = env_threshold is not None
    threshold_provided = threshold_env_provided or args.threshold is not None
    if args.threshold is not None:
        threshold_value = args.threshold
    elif env_threshold is not None:
        threshold_value = float(env_threshold)
    else:
        threshold_value = 1.5

    if args.target is not None and threshold_mode != "auto" and not sweep_values:
        print("[m3] --target provided without --threshold-sweep; enabling auto threshold search")
        threshold_mode = "auto"

    if threshold_mode != "auto" and not sweep_values and not threshold_provided:
        sweep_values = list(DEFAULT_THRESHOLD_SWEEP)
        print(
            "[m3] --threshold-sweep not provided; using default sweep: "
            + " ".join(_format_tau(value) for value in sweep_values)
        )

    target_tolerance = _compute_target_tolerance(
        args.target,
        target_band if args.target is not None and band_provided else None,
    )

    if threshold_mode == "auto":
        final_mode: Literal["single", "manual", "auto"] = "auto"
    elif sweep_values:
        final_mode = "manual"
    else:
        final_mode = "single"

    return CalibrationConfig(
        scenario=scenario,
        n=args.n,
        seed=args.seed,
        model=args.model,
        threshold=threshold_value,
        threshold_provided=threshold_provided,
        threshold_mode=final_mode,
        threshold_values=sweep_values,
        target_band=target_band,
        target_per=target_per,
        target_value=args.target,
        target_tolerance=target_tolerance,
        out_dir=args.out,
        plot_title=args.title,
        pin_eval=args.pin_eval,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    runner = GateCalibrationRunner(config)
    runner.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
