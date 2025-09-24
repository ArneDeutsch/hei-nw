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

### Milestone 3 τ policy

Milestone 3 locks in one τ per scenario using the sweeps committed under
``reports/m3-run*/``. The summary files make it easy to re-verify the choices
and compare future reruns.

* **Scenario A:** The automatic sweep (``reports/m3-run0``) shows that τ below
  roughly ``0.7`` floods the store with clutter—precision drops to ``0.65`` even
  though the write rate stays inside the 1–5/1 000 band.【F:reports/m3-run0/tau_0.369919/A_gate_telemetry.json†L392-L459】
  The manual confirmation sweep with a different seed (``reports/m3-run1``)
  keeps precision and PR-AUC at ``1.0`` for τ≥1.2 while holding the write rate at
  ``≈3.21`` writes/1 000 tokens.【F:reports/m3-run1/A_sweep_summary.tsv†L1-L6】【F:reports/m3-run1/tau_1.2/A_gate_telemetry.json†L171-L237】【F:reports/m3-run1/tau_1.2/A_gate_telemetry.json†L393-L470】
  Adopt **τ = 1.2** for production runs and reference the seed-17 sweep when
  validating future updates.

* **Scenario C:** The pins-only sweep (``reports/m3-run2``) reports both the pin
  slice and the non-pin overlay. τ = 1.5 keeps non-pin precision/recall at ``1.0``
  with ``≈4.40`` writes/1 000 tokens, well inside the policy band, while pins
  remain fully recalled.【F:reports/m3-run2/tau_1.5/C_gate_telemetry_pins.json†L86-L237】【F:reports/m3-run2/tau_1.5/C_gate_telemetry_pins.json†L322-L470】
  Increasing τ to 3.0 drops non-pin recall to ``≈0.18`` and reduces the write
  rate below target, so **τ = 1.5** is the highest setting that still preserves
  non-pin coverage.【F:reports/m3-run2/tau_3/C_gate_telemetry_pins.json†L160-L237】

Record future calibrations by appending to this section: list the chosen τ,
briefly justify the selection (write rate, precision/recall, and any pin
considerations), and link to the relevant ``reports/`` directory.

#### When to declare calibration failure

Mark the task as failed and escalate if **any** of the following occur:

* No τ tested lands in the requested write-rate band (``first_tau_within_target_band``
  remains ``null`` in the sweep summary).
* Every in-band τ drives precision or PR-AUC unacceptably low—for example,
  precision <0.5 implies heavy clutter even if the write rate is nominal.
* The telemetry histogram shows a flat score plateau around τ (little
  separation between the 50th and 90th percentiles), indicating the current
  feature mix cannot enforce the target policy without retuning the weights.

Capture the sweep TSV/JSON and a short explanation when logging a failure so
the next reviewer can reproduce the issue quickly.

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
