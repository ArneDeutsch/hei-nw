# M3 — Neuromodulated Write Gate + Trace Store — Milestone Review

### 1) Executive summary (2–4 short paragraphs)

Milestone 3 set out to implement a neuromodulated write gate (salience‐based) and a pointer-only trace store with eviction/decay, integrate both into the evaluation harness, and ship calibration/telemetry for scenarios A & C. The DoD also requires τ tuning to achieve **1–5 writes per 1k tokens**, pointer-only privacy guarantees, and CI/QA gates.

The repo contains a real gate (`src/hei_nw/gate.py`), a pointer-only `TraceWriter`, eviction/decay (`src/hei_nw/eviction.py`), harness integration (CLI flags + telemetry), calibration scripts, scenario C enhancements, and docs. Reports exist under `reports/m3-write-gate/` for A & C, including threshold sweeps, telemetry JSONs, and calibration PNGs.

However, acceptance **falls short on a key DoD item**: the provided sweep yields **\~28–549 writes per 1k tokens**, not the required **1–5**. This appears to stem from how the metric is computed (per **generated** tokens only) and from an insufficient τ range. Pointer-only enforcement is implemented and unit-tested, but the harness’s emitted “trace samples” are not clearly proving persisted payloads are pointer-only.

**Verdict: Partial.** Most deliverables exist and are wired; telemetry and docs are solid. To pass, fix the writes-per-1k-tokens metric, expand/auto-tune τ to reach the target band, and expose/store a verifiable pointer-only sample path from actual writes.

---

### 2) Evidence snapshot: repo & planning anchors

* **Repo tree (short)**

  * `src/hei_nw/`: `gate.py`, `eviction.py`, `store.py`, `pack.py`, `eval/harness.py`, `eval/report.py`, `telemetry/gate.py`, `datasets/scenario_{a,c}.py`
  * `scripts/`: `run_m3_gate_calibration.sh`, `plot_gate_calibration.py`, `report_gate_write_rates.py`, `audit_pointer_payloads.py`
  * `reports/m3-write-gate/`: `A_sweep_summary.{json,tsv}`, `C_sweep_summary.{json,tsv}`, `tau_*/*{metrics.json,report.md,gate_calibration.png}`
  * `tests/`: `test_gate.py`, `test_trace_writer.py`, `test_gate_metrics.py`, `test_eviction.py`, `eval/test_harness_gate_flow.py`, `scripts/test_plot_gate_calibration.py`
  * `documentation/`: `write-gate.md`, `quick-validate.md`

* **Planning anchors used**

  * `planning/milestone-3-plan.md` — **“M3 — Neuromodulated Write Gate + Trace Store”** (Tasks T1–T6, HUMAN tasks R1–R3, DoD)
  * `planning/design.md` — **“5.5 Trace Schema (Value Payload)”**, **“5.7 Neuromodulated Write Gate”**, **“5.9 Eviction / Decay / Protection”**
  * `planning/validation-plan.md` — gate telemetry (PR-AUC, clutter, calibration curve), scenarios **A** and **C**, modes **B0–B3**
  * `planning/project-plan.md` — milestone scopes/DoD alignment

* **Assumptions/limits**

  * I did not execute CI or GPU runs; conclusions are based on code/tests/scripts and the committed `reports/` artifacts.

---

### 3) DoD / Acceptance verification table

| Item                                                                      | Evidence (files/funcs/CLI)                                                                                                                                     | Status      | Notes                                                                                                                                                |
| ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Gate telemetry for A & C (PR-AUC, precision/recall, clutter, calibration) | `reports/m3-write-gate/{A,C}_sweep_summary.json`; per-τ files (e.g., `tau_1.5/A_gate_telemetry.json`), calibration PNGs via `scripts/plot_gate_calibration.py` | **Pass**    | Metrics present with calibration bins and PR-AUC.                                                                                                    |
| τ tuned to **1–5 writes / 1k tokens**                                     | `reports/m3-write-gate/A_sweep_summary.tsv` shows 28–549 writes/1k tokens                                                                                      | **Fail**    | Metric uses **generated tokens only**; τ range 0.5–3.5 insufficient.                                                                                 |
| Pointer-only persistence; packer enforces redaction; manual privacy check | `src/hei_nw/pack.py` enforces no text keys; `tests/test_trace_writer.py`; `scripts/audit_pointer_payloads.py`                                                  | **Partial** | Harness “trace\_samples” show `banned_keys: ["episode_text"]` and `has_pointer: false` in A; not a proof of **persisted** writes being pointer-only. |
| Eviction/decay with pin protection                                        | `src/hei_nw/eviction.py`; tests: `tests/test_eviction.py::{test_ttl_decay_removes_expired,test_pin_protection_blocks_eviction}`                                | **Pass**    | TTL and pin protection covered by unit tests.                                                                                                        |
| No stubs/mocks in production                                              | `scripts/grep_no_stubs.sh` (excludes tests/docs)                                                                                                               | **Pass**    | Scan excludes tests; code paths look real.                                                                                                           |
| Quality gates & tests                                                     | `pyproject.toml` (black/ruff/mypy/pytest), `.pre-commit-config.yaml`, GH **ci.yml**                                                                            | **Partial** | Config present; not executed here.                                                                                                                   |
| Calibration script emits artifacts                                        | `scripts/run_m3_gate_calibration.sh` (supports `--threshold-sweep`), artifacts in `reports/m3-write-gate/`                                                     | **Pass**    | Sweep summaries and per-τ assets exist.                                                                                                              |
| Documentation updated                                                     | `documentation/write-gate.md`; `documentation/quick-validate.md` includes M3 steps; README links                                                               | **Pass**    | Usage & tuning documented.                                                                                                                           |

---

### 4) Task-by-task review

#### M3-T1 \[CODEX] Neuromodulated Gate Core

* **Intent:** Compute salience (α·surprise + β·novelty + γ·reward + δ·pin) and threshold to decide writes; integrate in harness.
* **Findings:** `src/hei_nw/gate.py` defines `SalienceFeatures`, `NeuromodulatedGate`, helpers like `surprise_from_logits` and `novelty_from_similarity` (e.g., export list includes `"surprise_from_logits", "novelty_from_similarity"`). `harness.py` exposes flags `--gate.{alpha,beta,gamma,delta,threshold}` (e.g., `parser.add_argument("--gate.threshold", ...)`). Tests: `tests/test_gate.py::{test_gate_computes_weighted_salience,test_gate_threshold_controls_write_rate}`.
* **Gaps / Risks:** None functional; metric downstream (writes per 1k tokens) is miscomputed (see T4).
* **Status:** **Pass**.

#### M3-T2 \[CODEX] Pointer-only Trace Writer & Telemetry

* **Intent:** Persist gate-approved episodes in pointer-only form and compute telemetry.
* **Findings:** `store.py` defines `TraceWriter.write` building payload with `tokens_span_ref`, `entity_slots`, `salience_tags`, `eviction` (short excerpt: `"tokens_span_ref", "entity_slots", "salience_tags"`). `pack.py` rejects text keys (e.g., `raise ValueError("trace contains disallowed key '...'" )`). Telemetry in `telemetry/gate.py` computes precision/recall/PR-AUC/calibration. Tests: `tests/test_trace_writer.py::test_pointer_only_payload`, `tests/test_gate_metrics.py`.
* **Gaps / Risks:** Harness does not clearly persist real **write** samples for audit; “trace\_samples” seen are gate diagnostics not store payloads.
* **Status:** **Partial** (telemetry OK; pointer audit proof incomplete).

#### M3-T3 \[CODEX] Eviction, Decay, Pin Protection

* **Intent:** TTL/decay eviction with pins protected.
* **Findings:** `eviction.py` defines `DecayPolicy`, `PinProtector`, `TraceEvictionState`. Tests exist and assert TTL removal / pin protection.
* **Gaps / Risks:** None obvious; integration hooks in `store.py` set eviction metadata.
* **Status:** **Pass**.

#### M3-T4 \[CODEX] Harness Integration & Calibration Reports

* **Intent:** Gate telemetry via harness, CLI knobs, calibration artifacts.
* **Findings:** `eval/harness.py` includes gate flags (`--gate.use_for_writes`, `--gate.debug_keep_labels`, etc.), aggregates telemetry incl. `write_rate_per_1k_records` and `write_rate_per_1k_tokens`. Scripts: `scripts/run_m3_gate_calibration.sh` supports `--threshold-sweep "τ …"` (usage shows: `--threshold-sweep "τ …"`), writes `*_sweep_summary.{json,tsv}` and per-τ calibration PNGs via `scripts/plot_gate_calibration.py`. Test: `tests/eval/test_harness_gate_flow.py::test_gate_metrics_logged`; `tests/scripts/test_plot_gate_calibration.py`.
* **Gaps / Risks:** **Bug**: tokens denominator excludes **prompt tokens**, inflating writes/1k tokens. Sweep range may not reach 1–5.
* **Status:** **Partial**.

#### M3-T5 \[CODEX] Scenario C Reward & Pin Enhancements

* **Intent:** Provide reward/pin annotations for scenario C and fixtures.
* **Findings:** `datasets/scenario_c.py` emits `gate_features` incl. `reward` and `pin`; deterministic fixtures `tests/fixtures/scenario_c_gate.json`. Test present: `tests/test_scenario_c.py::test_reward_pin_annotations`.
* **Gaps / Risks:** None.
* **Status:** **Pass**.

#### M3-T6 \[CODEX] Documentation & MkDocs

* **Intent:** Document formula, defaults, τ tuning, and telemetry.
* **Findings:** `documentation/write-gate.md` explains signals and workflow; `documentation/quick-validate.md` includes steps “M3 gate calibration… sweeps τ=0.5…3.5”.
* **Gaps / Risks:** None.
* **Status:** **Pass**.

---

### 5) Design & validation alignment

* **Design mapping**

  * **Write gate** matches `design.md §5.7` scoring `S = α·surprise + β·novelty + γ·reward + δ·pin`; implemented in `src/hei_nw/gate.py`; harness flags wire α..δ and τ in `eval/harness.py`.
  * **Trace schema & pointer-only** per `§5.5`: `TraceWriter` stores `tokens_span_ref` + metadata; `pack.py` forbids raw text.
  * **Eviction/decay** per `§5.9`: `DecayPolicy`, TTL metadata in stored payloads; unit tests confirm.

* **Validation mapping**

  * **Scenarios A & C** telemetry present; calibration curves and PR-AUC computed (`telemetry/gate.py` + emitted JSON/PNG).
  * **Modes B0–B1** are the context for gate evaluation; reports under `reports/m3-write-gate/` include `A_B1_metrics.json` etc.
  * **Mismatch:** “writes per 1k tokens” deviates from the validation spec spirit (token budget of the run), because prompt tokens are ignored.

---

### 6) Quality & CI assessment

* **Tooling:** `black`, `ruff`, `mypy`, `pytest` configured in `pyproject.toml` and `.pre-commit-config.yaml`.
* **CI:** `.github/workflows/ci.yml` runs tests, diff-coverage ≥85%, parity guard, M2 smoke, gate non-empty predictions, pointer payload audit, and “No stubs”.
* **Testing depth:** Good unit coverage for gate math, telemetry, eviction; harness smoke tests for gate flow and plotting. Determinism via fixtures/seed. Risk: some tests are smoke-level for scripts; no end-to-end proof of **persisted** pointer-only writes.

---

### 7) Gaps and **Follow-up tasks**

````
### M3-F1 [CODEX] Fix writes-per-1k-tokens denominator

* **Goal:** Make DoD meaningful by computing writes per 1k **prompt+generated** tokens.
* **Key changes:**
  1) Update `src/hei_nw/eval/harness.py` to compute `writes_per_1k_tokens = writes / ((prompt_tokens_total + generated_tokens_total)/1000.0)`.
  2) Ensure `gate.telemetry` JSON includes both totals and the corrected metric.
  3) Update `scripts/report_gate_write_rates.py` to prefer the corrected field.
* **Tests:**
  - Add `tests/eval/test_harness_gate_flow.py::test_writes_per_1k_tokens_uses_prompt_and_generated`.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --n 32 --seed 3 --gate.use_for_writes \
    && jq '.gate.write_rate_per_1k_tokens' reports/m3-write-gate/tau_1.5/A_B1_metrics.json
````

```
```

### M3-F2 \[CODEX] Robust τ sweep to hit 1–5 writes / 1k tokens

* **Goal:** Ensure the calibration script can actually reach the target write budget.
* **Key changes:**

  1. Extend `scripts/run_m3_gate_calibration.sh` to accept `--threshold-sweep auto` which expands/searches τ until the target band \[1,5] is bracketed (e.g., exponential search + bisection).
  2. Add `--target-band "1,5"` knob and record the first τ meeting it in the sweep summary.
* **Tests:**

  * `tests/scripts/test_run_m3_gate_calibration_smoke.py::test_auto_sweep_brackets_target` (stub harness with fixed token totals).
* **Quality gates:** standard
* **Acceptance check:**

  ```bash
  scripts/run_m3_gate_calibration.sh --scenario A --n 256 --threshold-sweep auto
  # Verify A_sweep_summary.tsv contains a τ with writes_per_1k_tokens in [1,5]
  ```

```
```

### M3-F3 \[CODEX] Persist and audit real write payloads

* **Goal:** Prove pointer-only persistence by sampling actual `TraceWriter` outputs in metrics.
* **Key changes:**

  1. Wire `TraceWriter` into the B1 write path in `eval/harness.py` when `--gate.use_for_writes` is set.
  2. Emit a small sample of **persisted** payloads to `gate.trace_samples` and record `pointer_check` summary (pointer\_only flag, banned\_keys counts).
  3. Update `scripts/audit_pointer_payloads.py` to read the new summary if present.
* **Tests:**

  * `tests/eval/test_harness_gate_flow.py::test_trace_samples_are_pointer_only`
* **Quality gates:** standard
* **Acceptance check:**

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --n 64 --gate.use_for_writes \
    && python scripts/audit_pointer_payloads.py reports/m3-write-gate/tau_*/A_B1_metrics.json
  ```

```
```

### M3-F4 \[CODEX] Pins-only telemetry & reporting polish

* **Goal:** Ensure pins slice is explicitly computed and written with plots/links.
* **Key changes:**

  1. Confirm `pins_only_eval` fields in `A/C_gate_telemetry.json`; ensure `report.md` includes “Pins-only PR-AUC / clutter”.
  2. Add title override to `plot_gate_calibration.py` for pins-only (already supported; just ensure used).
* **Tests:**

  * `tests/eval/test_harness_gate_flow.py::test_pins_only_metrics_present`
* **Quality gates:** standard
* **Acceptance check:**

  ```bash
  scripts/run_m3_gate_calibration.sh --scenario C --n 128 --pin-eval
  # Confirm C_gate_telemetry.json has "pins_only" and report references the pins plot
  ```

```
```

### M3-F5H \[HUMAN/ChatGPT] Calibrate τ on GPU with corrected metric

* **Goal:** Produce evidence that τ achieves 1–5 writes/1k tokens after M3-F1/F2.
* **Steps:**

  1. Pull latest main; run `scripts/run_m3_gate_calibration.sh --threshold-sweep auto --n 512` on A and `--pin-eval` on C.
  2. Attach updated `*_sweep_summary.tsv` and top candidate τ.
* **Acceptance check:** Commit the TSVs and PNGs under `reports/m3-write-gate/` showing a τ within \[1,5] for A and C.

```

---

### 8) Final verdict
**Partial.** Deliverables exist and are mostly real; telemetry, scripts, and docs are in place. To meet DoD, complete: **M3-F1**, **M3-F2**, **M3-F3** (minimum), and then validate via **M3-F5H**. These address the write-budget target, provide rigorous pointer-only proof from persisted writes, and make calibration robust.
```
