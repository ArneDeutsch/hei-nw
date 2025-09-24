# Neuromodulated Write Gate

The neuromodulated write gate decides which episodes are worth persisting in
HEI-NW. It combines predictive **surprise**, **novelty**, and explicit
**reward/pin** signals into a single salience score and compares that score to a
threshold. This page summarizes the formula, the default configuration, and the
calibration workflow introduced for Milestone 3.

## Salience signals

For each decoded record the harness derives a ``hei_nw.gate.SalienceFeatures`` object with the following channels:

| Feature  | Description |
| --- | --- |
| ``surprise`` | Negative log-probability of the ground-truth token emitted by the base model. Larger values indicate the model was more surprised by the observation. |
| ``novelty`` | Cosine-distance novelty relative to the sparse-key index. ``1.0`` means no similar key exists; ``0.0`` means the episode matches an existing trace. |
| ``reward`` | Boolean flag surfaced by the scenario generator when an external reward is observed. Reward-positive episodes should receive a salience boost even if they are not novel. |
| ``pin`` | Boolean flag that marks operator-specified episodes. Pins receive the largest salience boost and the harness persists them even when the score falls below τ. Enable ``--gate.pin_override`` (or ``NeuromodulatedGate(pin_override=True)``) to make the gate bypass the threshold directly. |

The helper utilities in ``hei_nw.gate`` clamp these inputs to safe ranges
before applying the gate weights.

## Default coefficients and threshold

Salience is computed as ``S = α·surprise + β·novelty + γ·reward + δ·pin``. The
Milestone 3 defaults match the design spec:

| Parameter | Default | Effect |
| --- | --- | --- |
| ``α`` (surprise weight) | ``1.0`` | Balances predictive surprise against other terms. |
| ``β`` (novelty weight) | ``1.0`` | Encourages storing episodes that expand coverage. |
| ``γ`` (reward weight) | ``0.5`` | Provides a moderate boost when reward annotations arrive. |
| ``δ`` (pin weight) | ``0.8`` | Keeps pinned episodes near the threshold even when surprise/novelty are low. Combine with ``pin_override`` to force writes via the gate itself. |
| ``τ`` (threshold) | ``1.5`` | Minimum salience required to persist an episode. |

Reduce ``τ`` to allow more writes per 1k tokens; increase it to tighten the
store. You can also override ``α``–``δ`` for ablation studies using the harness
flags (``--gate.alpha``, ``--gate.beta`` …), but the default mixture is tuned
for scenarios A and C.

Pins therefore affect the score twice: the ``δ`` weight lifts their salience and
the harness treats ``pin=True`` as a forced write when persisting traces. If you
need the gate decision itself to reflect that forced-write policy—e.g. when
embedding the gate in a different pipeline—enable ``--gate.pin_override`` or
instantiate ``NeuromodulatedGate(pin_override=True)`` to bypass the threshold.

### Store construction during Milestone 3

Milestone 3 keeps the **B1 episodic store label-driven by default**. The
evaluation harness always records gate telemetry, yet the persisted memories are
still sourced from the scenario-provided ``should_remember`` annotations. This
matches the behavior that shipped with the milestone artifacts and avoids
changing baseline metrics midstream. To experiment with gate-controlled writes,
run the harness with ``--gate.use_for_writes``; this optional flag switches the
indexer to honor gate decisions instead of the labels.

## Calibration workflows

Use ``scripts/run_m3_gate_calibration.sh`` to generate write-gate telemetry and
calibration plots. The script wraps the evaluation harness, extracts the gate
summary, and renders a reliability diagram. The sub-sections below cover the
supported operating modes.

### Default sweep

```bash
# Scenario A defaults (Qwen2.5-1.5B, n=48)
bash scripts/run_m3_gate_calibration.sh
```

When ``--threshold-sweep`` is omitted the script explores a broad set of τ
values by default: ``0.5 1.0 1.5 2.0 2.5 3.0 3.5``. Each τ is written to its
own ``reports/m3-write-gate/tau_<τ>/`` directory alongside aggregate sweep
summaries. Use this default mode to scan for the decision boundary quickly.

Artifacts are written to ``reports/m3-write-gate/`` by default:

* ``*_metrics.json`` – per-run harness metrics with an embedded ``gate`` block.
* ``*_gate_telemetry.json`` – distilled telemetry containing precision/recall,
  PR-AUC, clutter rate, provenance fields (scenario/τ/n/seed/model), and
  calibration bins.
* ``*_gate_calibration.png`` – scatter plot of mean score vs. empirical
  positive rate.
* ``*_trace_samples.json`` – pointer-only trace samples suitable for privacy
  spot checks (created when samples are emitted).

The goal for Milestone 3 is **1–5 writes per 1 000 tokens with high precision**.
Sweep ``--threshold`` until the ``write_rate_per_1k_tokens`` reported in the metrics
JSON sits inside that band. Lower thresholds increase the write rate; higher
thresholds reduce it. When the gate is too loose, expect clutter rate and
pointer-check warnings to rise.

### Single-τ run (``--threshold``)

```bash
# Scenario C with a tighter threshold (τ=1.8)
bash scripts/run_m3_gate_calibration.sh --scenario C --threshold 1.8 --n 64
```

Provide ``--threshold`` to bypass the sweep and focus on a single τ value. This
is useful once you have selected a target and want to regenerate the telemetry
for a final report.

### Threshold sweep (``--threshold-sweep``)

Provide a space or comma separated list of τ values to benchmark a custom set in
one pass:

```bash
bash scripts/run_m3_gate_calibration.sh \
  --scenario A \
  --n 64 \
  --threshold-sweep "1.3 1.4 1.5 1.6" \
  --seed 7
```

Each τ gets its own directory (``reports/m3-write-gate/tau_1.3/``, ``tau_1.4/``
…), containing the same artifacts described above. The script also emits:

* ``${SCENARIO}_sweep_summary.json`` – consolidated metrics from
  ``scripts/report_gate_write_rates.py``.
* ``${SCENARIO}_sweep_summary.tsv`` – tab-separated table with ``scenario``,
  ``tau``, ``writes``, ``write_rate``, ``writes_per_1k_tokens``,
  ``writes_per_1k_records``, and ``pr_auc`` for spreadsheet review.
* ``${SCENARIO}_threshold_sweep.md`` – Markdown index linking τ values to their
  calibration plots.

Use the TSV/JSON summaries to select the target τ band quickly, then open the
per-τ calibration plots to confirm reliability. Sweeps can be re-run with a
different seed to stress test stability.

### Pins-only evaluation (``--pin-eval``)

Scenario C produces ``pin=True`` records that should survive threshold tuning.
Enable ``--pin-eval`` to run a pins-only slice and compare it against the full
distribution:

```bash
bash scripts/run_m3_gate_calibration.sh --scenario C --n 64 --pin-eval
```

The script checks that the scenario/seed emit pinned examples and exits with a
helpful message if none are present. Pins-only runs suffix their outputs with
``_pins`` (for example ``C_gate_telemetry_pins.json`` and
``C_gate_calibration_pins.png``) and overlay the non-pin calibration curve for
side-by-side inspection. Combine ``--pin-eval`` with ``--threshold-sweep`` to
trace how pin precision changes across τ. Focus on matching the pins-only
``writes_per_1k_tokens`` (and the legacy ``writes_per_1k_records``) and PR-AUC
against the overall metrics—pins should stay inside the desired write band while
retaining higher precision than non-pinned episodes.

## Milestone 3 τ selection

The Milestone 3 calibration runs completed with ``n=256`` records, default gate
weights, and the Qwen/Qwen2.5-1.5B-Instruct checkpoint. Artifacts live under
``reports/m3-run0`` (scenario A auto sweep), ``reports/m3-run1`` (scenario A
manual grid), and ``reports/m3-run2`` (scenario C pins slice).

### Scenario A (base distribution)

* **Chosen τ:** ``1.5``. At this threshold the gate writes **3.27 episodes per
  1 000 tokens** with perfect precision/recall, and the score distribution keeps
  p50 well below τ while p90 remains safely above it, indicating a steep shoulder
  rather than a flat plateau.【F:reports/m3-run0/tau_1.5/A_gate_telemetry.json†L455-L470】【F:reports/m3-run0/tau_1.5/A_gate_telemetry.json†L451-L454】
* **Lower τ rejected:** Dropping to ``τ=0.375`` pushes the write rate close to
  the upper target band but precision collapses to ~0.66, signaling excessive
  clutter despite staying within the 1–5 writes/1 000 tokens envelope.【F:reports/m3-run0/tau_0.375/A_gate_telemetry.json†L455-L470】【F:reports/m3-run0/tau_0.375/A_gate_telemetry.json†L168-L175】【F:reports/m3-run0/tau_0.375/A_gate_telemetry.json†L392-L398】
* **Stability check:** A second seed sweeping τ∈[1.2, 2.0] keeps the write rate
  pinned at 3.21 writes/1 000 tokens, confirming that ``τ=1.5`` sits in the
  center of a broad precision plateau.【F:reports/m3-run1/A_sweep_summary.tsv†L1-L6】

**Recommendation:** Adopt ``τ_A = 1.5`` for scenario A and reuse the associated
telemetry JSON, calibration plot, and trace samples for future audits.

### Scenario C (pins and non-pins)

* **Chosen τ:** ``1.5``. The pins-only sweep shows that pinned episodes retain a
  high-salience cluster (p50≈3.05) far above τ with perfect precision, so the
  policy does not endanger forced writes.【F:reports/m3-run2/tau_1.5/C_gate_telemetry_pins.json†L360-L398】
* **Non-pin behavior:** Simulating the gate on the scenario C generator confirms
  that the same τ writes every ``should_remember`` record (pinned and
  non-pinned) while keeping all ``should_forget`` cases out of the store; raising
  τ toward ``1.8`` starts dropping non-pin recall.【efa83b†L1-L15】

**Recommendation:** Reuse ``τ_C = 1.5`` for scenario C. When running
``scripts/run_m3_gate_calibration.sh`` with ``--pin-eval``, expect the auto
search to keep doubling τ—pins are forced writes (see the gate harness code
below), so the write-rate metric never declines. Run a baseline sweep **without**
``--pin-eval`` first to bracket τ, then add the pins slice for validation.

> ``write = decision.should_write or bool(features.pin)`` — pinned records are
> always persisted by the harness, independently of τ.【F:src/hei_nw/eval/harness.py†L158-L188】

## Interpreting telemetry

The ``gate`` section in the harness metrics summarizes the run:

* ``writes`` / ``total`` and ``write_rate`` quantify how many records cleared the
  gate.
* ``pinned`` / ``reward_flags`` count how often the respective signals fired.
* ``write_rate_per_1k_tokens`` normalizes the write rate by the combined prompt
  and generated token budget, while ``write_rate_per_1k_records`` preserves the
  per-record view.
* ``telemetry`` embeds precision, recall, PR-AUC, clutter rate,
  ``writes_per_1k_tokens``, ``writes_per_1k_records``, and calibration histogram buckets.
* ``pointer_check`` confirms that stored traces remain pointer-only. Investigate
  any run that reports missing pointers or banned keys.

For calibration, compare the scatter plot against the diagonal “ideal” line. A
well-calibrated gate places most points near the diagonal, with dense buckets in
the lower-left (rare writes) and only a handful in the upper-right (high
salience). Deviations usually indicate that ``τ`` needs adjustment or that the
scenario is missing reward/pin annotations.

## Troubleshooting

* **Precision drifts low:** raise ``τ`` or inspect the scenario’s labels for
  noisy ``should_remember`` annotations.
* **Write rate collapses:** lower ``τ`` slightly or increase ``γ``/``δ`` if
  reward/pin signals are too weak.
* **Pointer check fails:** ensure upstream writers only persist pointer payloads
  (see ``TraceWriter`` in ``hei_nw.store``).
* **No calibration plot:** confirm the harness produced ``gate.telemetry``
  entries and that ``scripts/plot_gate_calibration.py`` succeeded.
