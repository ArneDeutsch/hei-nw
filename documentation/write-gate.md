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
| ``pin`` | Boolean flag that forces high salience when an operator explicitly pins an episode. Pins should survive decay/eviction and therefore receive the largest boost. |

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
| ``δ`` (pin weight) | ``0.8`` | Keeps pinned episodes near the threshold even when surprise/novelty are low. |
| ``τ`` (threshold) | ``1.5`` | Minimum salience required to persist an episode. |

Reduce ``τ`` to allow more writes per 1k tokens; increase it to tighten the
store. You can also override ``α``–``δ`` for ablation studies using the harness
flags (``--gate.alpha``, ``--gate.beta`` …), but the default mixture is tuned
for scenarios A and C.

## Calibration workflows

Use ``scripts/run_m3_gate_calibration.sh`` to generate write-gate telemetry and
calibration plots. The script wraps the evaluation harness, extracts the gate
summary, and renders a reliability diagram. The sub-sections below cover the
supported operating modes.

### Single-τ run

```bash
# Scenario A defaults (Qwen2.5-1.5B, n=48, τ=1.5)
bash scripts/run_m3_gate_calibration.sh

# Scenario C with a tighter threshold (τ=1.8)
bash scripts/run_m3_gate_calibration.sh --scenario C --threshold 1.8 --n 64
```

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

### Threshold sweep (``--threshold-sweep``)

Provide a space or comma separated list of τ values to benchmark multiple gate
settings in one pass:

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
  ``tau``, ``write_rate``, ``writes_per_1k_tokens``, ``writes_per_1k_records``, and ``pr_auc`` for spreadsheet
  review.
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

## Interpreting telemetry

The ``gate`` section in the harness metrics summarizes the run:

* ``writes`` / ``total`` and ``write_rate`` quantify how many records cleared the
  gate.
* ``pinned`` / ``reward_flags`` count how often the respective signals fired.
* ``write_rate_per_1k_tokens`` normalizes the write rate by generated tokens, while
  ``write_rate_per_1k_records`` preserves the per-record view.
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
