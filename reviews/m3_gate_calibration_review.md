# Review: `scripts/run_m3_gate_calibration.sh` — What’s implemented, what’s missing for M3, and how the parameters are meant to be used

## TL;DR

* The script **works for a single τ (threshold)** and produces the expected artifacts (metrics JSON, distilled telemetry JSON, calibration plot, optional trace samples).
* The options you saw in your shell are the **canonical interface** right now. There is **no `--threshold-sweep`** implemented in the script, and **no `--pin-eval`** either. Your CLI error is accurate.
* The harness already exposes the lower-level knobs (`--gate.alpha/beta/gamma/delta`, `--gate.threshold`), and the script forwards **only** `--gate.threshold` (via `--threshold`). A sweep would need to be added on top.
* Scenario C already emits **reward** and **pin** signals (see generator asserts), and the harness/telemetry already track **pinned counts**, but the script does **not** provide a “pins-only” evaluation mode or plot.

---

## What the script currently does (and where)

**Entrypoint & defaults.**
At the top of `scripts/run_m3_gate_calibration.sh` the script defines defaults:

* `SCENARIO=A`, `N=48`, `SEED=13`, `MODEL=Qwen/Qwen2.5-1.5B-Instruct`, `THRESHOLD=1.5`, `OUT=reports/m3-write-gate`, optional `PLOT_TITLE`.

**Harness invocation (single τ).**
It calls the evaluation harness once, passing the threshold into the harness’ gate arg:

```bash
python -m hei_nw.eval.harness \
  --mode B1 \
  --scenario "$SCENARIO" \
  -n "$N" \
  --seed "$SEED" \
  --model "$MODEL" \
  --outdir "$OUT" \
  --gate.threshold "$THRESHOLD"
```

This writes `${OUT}/${SCENARIO}_B1_metrics.json` (plus the usual per-run artifacts).

**Telemetry distillation.**
Immediately after, the script extracts a compact telemetry JSON from the metrics JSON. Keys include:

* `scenario`, `threshold`, `writes`, `total`, `write_rate`, `write_rate_per_1k`, and counts for `pinned` and `reward_flags`, plus whatever is in `gate.telemetry` (PR-AUC, calibration bins, etc.).
* Output paths:

  * `${OUT}/${SCENARIO}_gate_telemetry.json`
  * `${OUT}/${SCENARIO}_trace_samples.json` (only when samples are present)

**Calibration plot.**
Finally, it renders a reliability diagram via `scripts/plot_gate_calibration.py`, which can also back-write the plot path into the metrics JSON’s `gate.calibration_plot`.

**What it does *not* do.**

* There is **no loop** over multiple τ values (so no `--threshold-sweep`).
* There is **no “pins-only” evaluation** or overlay (`--pin-eval`).
* The script does not pass through the individual gate weights; if you need α/β/γ/δ ablations you must call the harness directly.

---

## Intended use of the current parameters

* `--scenario`: Dataset generator to run (`A` or `C` are the M3 targets).
* `--n`, `--seed`: Size and RNG for deterministic repro.
* `--model`: Base LLM identifier (defaults to Qwen 1.5B Instruct).
* `--threshold` (τ): Alias for `--gate.threshold`; adjusts write aggressiveness. Lower τ → more writes, higher τ → fewer writes. The design guidance targets **1–5 writes per 1k tokens**.
* `--out`: Output directory for metrics & plots.
* `--title`: Optional custom plot title.

### Related knobs (harness-level, not forwarded by the script)

* `--gate.alpha`, `--gate.beta`, `--gate.gamma`, `--gate.delta`: Weights for surprise/novelty/reward/pin (defaults match the design: α=1.0, β=1.0, γ=0.5, δ=0.8).
* `--gate.threshold`: Same as `--threshold` above (the script simply forwards this).

---

## Gaps vs. the M3 plan’s “Review & GPU Tasks”

1. **`--threshold-sweep` is not implemented.**
   The plan expects an automated τ sweep to study write-rate/precision/calibration behavior. Today you’d have to run the script multiple times manually and then summarize with `scripts/report_gate_write_rates.py` (which *does* exist and can produce a JSON/TSV summary across many `${SCENARIO}_B1_metrics.json` files).

2. **`--pin-eval` is not implemented.**
   Scenario C already produces `pin=True`/`reward=True` signals, and the harness telemetry includes counts, but there is no way to:

   * restrict metrics/calibration to **pins-only** episodes, or
   * overlay **pins vs. non-pins** on the calibration plot, or
   * **force-pin** a fraction of episodes to verify the “pins survive” behavior across τ.

3. **Minor reproducibility gap in telemetry.**
   The distilled telemetry currently stores `scenario` and τ, but not `n`, `seed`, or `model`. Including those would simplify provenance and plot titles.

---

## What’s already complete enough to validate M3 (single-τ path)

* **Run A (default τ=1.5)**

  ```bash
  bash scripts/run_m3_gate_calibration.sh --scenario A --n 48 --seed 13
  ```

  Expect in `reports/m3-write-gate/`:

  * `A_B1_metrics.json` with `gate` block (PR-AUC, write-rate, threshold, etc.)
  * `A_gate_telemetry.json` (distilled)
  * `A_gate_calibration.png`

* **Run C (tighter τ example)**

  ```bash
  bash scripts/run_m3_gate_calibration.sh --scenario C --n 64 --threshold 1.8
  ```

* **(Optional) Summarize multiple manual runs**

  ```bash
  python scripts/report_gate_write_rates.py reports/m3-write-gate/*_B1_metrics.json \
    --out reports/m3-write-gate/summary.json
  ```

This flow matches `documentation/write-gate.md` (which demonstrates single-τ calibration and interpretation). To meet the **exact** “Review & GPU” automation in the M3 plan, add the sweep + pin-eval below.

---

## \[CODEX] Tasks to finish the implementation (incl. tests & docs)

> Task format follows `prompts/HEI-NW_milestone_task_prompt_template.md`.

### M3-T6 — Add `--threshold-sweep` to `run_m3_gate_calibration.sh` **\[CODEX]**

* **Goal:** Automate calibration across a list of τ values and emit a consolidated summary and per-τ artifacts.
* **Key changes:**

  1. Extend `scripts/run_m3_gate_calibration.sh` parsing to accept:

     * `--threshold-sweep "1.2 1.4 1.5 1.6 1.8"` (space-separated floats).
     * If present, loop over τ values, set `THRESHOLD` each iteration, and write into `${OUT}/tau_${τ}/…`.
  2. After the loop, call `scripts/report_gate_write_rates.py ${OUT}/tau_*/${SCENARIO}_B1_metrics.json --out ${OUT}/${SCENARIO}_sweep_summary.json` and also write a TSV with `[scenario, tau, write_rate, writes_per_1k, pr_auc]`.
  3. Update the plot step to save one calibration PNG per τ under each subdir; optionally create a tiny “index.md” with a table of τ → plot.
* **Tests:**

  * `tests/utils/test_scripts.py::test_run_m3_gate_calibration_has_threshold_sweep_flag` (presence).
  * `tests/utils/test_scripts.py::test_threshold_sweep_creates_subdirs` (integration, mark as `slow` if it hits the harness; otherwise mock minimal JSONs).
  * Coverage for the small Python summarizer path (if any) ≥90% in changed lines.
* **Quality gates:** `black .`, `ruff .`, `mypy .`, `pytest -q`.
* **Acceptance check:**

  ```bash
  bash scripts/run_m3_gate_calibration.sh --scenario A --n 16 --threshold-sweep "1.3 1.5" --seed 7
  # Expect: reports/m3-write-gate/tau_1.3/*, tau_1.5/*, and a sweep_summary.json/tsv at the root
  ```

### M3-T7 — Pins-only evaluation mode (`--pin-eval`) **\[CODEX]**

* **Goal:** Provide a “pins-only” slice for telemetry and calibration, and an overlay comparison vs. non-pinned.
* **Key changes:**

  1. **Harness:** In `src/hei_nw/eval/harness.py`, add `--eval.pins_only` (boolean). When enabled, compute gate metrics on the subset of records with `gate_features.pin==True`. Ensure the `gate.telemetry` block carries a separate `pins_only` sub-section (`pr_auc`, `precision`, `recall`, `calibration_bins`, `writes`, `write_rate` for pins).
  2. **Plotter:** Extend `scripts/plot_gate_calibration.py` with an optional `--pins-only` flag to render the pins-only calibration (or a dual-series overlay if `--overlay-nonpins` is also set). Don’t hard-code colors; use default matplotlib per project rules.
  3. **Shell script:** Add `--pin-eval` to `run_m3_gate_calibration.sh`. If set, pass `--eval.pins_only` to the harness and suffix outputs with `_pins` (`${SCENARIO}_gate_calibration_pins.png`, `${SCENARIO}_gate_telemetry_pins.json`). If the scenario has **no pins**, fail fast with a helpful message.
* **Tests:**

  * `tests/eval/test_harness_gate_flow.py::test_pins_only_metrics_slice`.
  * `tests/scripts/test_plot_gate_calibration.py::test_pins_only_render_smoke` (headless).
  * `tests/utils/test_scripts.py::test_run_m3_gate_calibration_has_pin_eval_flag`.
* **Quality gates:** standard suite.
* **Acceptance check:**

  ```bash
  bash scripts/run_m3_gate_calibration.sh --scenario C --n 32 --pin-eval
  # Expect *_telemetry_pins.json and *_calibration_pins.png with non-empty pins metrics
  ```

### M3-T8 — Telemetry provenance & plot titles **\[CODEX]**

* **Goal:** Make single-file telemetry self-describing and plots easier to read.
* **Key changes:**

  1. In the telemetry distillation block of `run_m3_gate_calibration.sh`, append `model`, `n`, and `seed` to the JSON.
  2. In `scripts/plot_gate_calibration.py`, default the title to `"{scenario} — τ={threshold} — n={n}, seed={seed}, model={model}"` when available (still overridable by `--title`).
* **Tests:**

  * `tests/utils/test_scripts.py::test_telemetry_includes_provenance`.
  * `tests/scripts/test_plot_gate_calibration.py::test_default_title_falls_back_cleanly` (when fields missing).
* **Quality gates:** standard suite.
* **Acceptance check:**
  Run the script with no `--title` and verify the plot title contains scenario/τ/n/seed/model.

### M3-T9 — Documentation updates for M3 calibration **\[CODEX]**

* **Goal:** Align docs with the implemented CLI and new modes.
* **Key changes:**

  1. Expand `documentation/write-gate.md` with sections for:

     * Single-τ run (current examples),
     * `--threshold-sweep` usage and expected artifacts,
     * `--pin-eval` usage (Scenario C) and interpretation.
  2. Add a short “Quick Commands” block to `README.md` pointing to the above.
* **Tests:** Doc build lints (if any); otherwise ensure examples reference existing scripts and flags.
* **Quality gates:** standard suite.
* **Acceptance check:** Docs show the exact flags and paths, examples copy-paste cleanly.

---

## Minimal “do it now” commands (until sweep & pin-eval land)

```bash
# A) Two manual thresholds → calibration plots
bash scripts/run_m3_gate_calibration.sh --scenario A --n 64 --threshold 1.3
bash scripts/run_m3_gate_calibration.sh --scenario A --n 64 --threshold 1.6

# B) Summarize write rates across the two runs
python scripts/report_gate_write_rates.py reports/m3-write-gate/*/A_B1_metrics.json \
  --out reports/m3-write-gate/summary.json

# C) Scenario C pins present by default; inspect counts
jq '.gate.pinned' reports/m3-write-gate/C_B1_metrics.json
```

---

## Verdict re: milestone status

* **Implementation completeness for the calibration path:** ✅ (single-τ flow matches docs and produces artifacts).
* **Gaps vs. M3 plan’s “Review & GPU Tasks”:** ❌ `--threshold-sweep` and ❌ `--pin-eval` are not implemented and should be added to fully satisfy the plan’s automation and evaluation depth.
* **Risk note:** Without a sweep, τ tuning may be noisy; without pins-only slicing, it’s harder to verify the “pins survive” behavior quantitatively. The tasks above close both gaps cleanly.
