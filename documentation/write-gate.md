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

## Calibration workflow

Use ``scripts/run_m3_gate_calibration.sh`` to generate write-gate telemetry and
calibration plots. The script wraps the evaluation harness, extracts the gate
summary, and renders a reliability diagram.

```bash
# Scenario A defaults (Qwen2.5-1.5B, n=48, τ=1.5)
bash scripts/run_m3_gate_calibration.sh

# Scenario C with a tighter threshold (τ=1.8)
bash scripts/run_m3_gate_calibration.sh --scenario C --threshold 1.8 --n 64
```

Artifacts are written to ``reports/m3-write-gate/`` by default:

* ``*_metrics.json`` – per-run harness metrics with an embedded ``gate`` block.
* ``*_gate_telemetry.json`` – distilled telemetry containing precision/recall,
  PR-AUC, clutter rate, and calibration bins.
* ``*_gate_calibration.png`` – scatter plot of mean score vs. empirical
  positive rate.
* ``*_trace_samples.json`` – pointer-only trace samples suitable for privacy
  spot checks (created when samples are emitted).

The goal for Milestone 3 is **1–5 writes per 1 000 tokens with high precision**.
Sweep ``--threshold`` until the ``write_rate_per_1k`` reported in the metrics
JSON sits inside that band. Lower thresholds increase the write rate; higher
thresholds reduce it. When the gate is too loose, expect clutter rate and
pointer-check warnings to rise.

## Interpreting telemetry

The ``gate`` section in the harness metrics summarizes the run:

* ``writes`` / ``total`` and ``write_rate`` quantify how many records cleared the
  gate.
* ``pinned`` / ``reward_flags`` count how often the respective signals fired.
* ``write_rate_per_1k`` normalizes the write rate for long sequences.
* ``telemetry`` embeds precision, recall, PR-AUC, clutter rate, ``writes_per_1k``,
  and calibration histogram buckets.
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
