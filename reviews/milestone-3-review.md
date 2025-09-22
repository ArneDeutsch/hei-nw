# M3 — Neuromodulated Write Gate + Trace Store — Milestone Review

### 1) Executive summary (2–4 short paragraphs)

**Intended delivery.** Milestone 3 sets up a neuromodulated write gate (surprise/novelty/reward/pin), pointer-only trace persistence with decay/eviction and pin protection, and evaluation/telemetry (PR-AUC, precision/recall, calibration, write-rate) on scenarios A & C, with privacy guarantees. DoD also requires a runnable calibration script and updated docs.

**What’s in the repo.** The codebase contains a real gate (`src/hei_nw/gate.py`), decay/eviction (`src/hei_nw/eviction.py`), pointer-only writer and enforcement in `store.py`/`pack.py`, harness integration and CLI flags with telemetry and calibration assets, plus unit/integration tests and CI. Scenario C is extended with reward/pin fields and fixtures. Docs include a dedicated write-gate guide and quick-validate steps.

**Top line verdict: Partial.** Implementation quality is strong and well-tested, and most artifacts exist and run. The **main gaps** are (a) **write-rate calibration is not actually tuned to the target (1–5 per 1k)** in the committed reports (Scenario A sweep is flat at 500/1k across τ), and (b) the **harness does not use the gate to drive writes into the store end-to-end** (the store is still built from `should_remember` labels), which undercuts the “HEI-NW can decide which episodes to store” claim. There’s also a **metrics mismatch**: “writes per 1k tokens” in the plan vs “per 1k records” implemented.

---

### 2) Evidence snapshot: repo & planning anchors

**Repo tree (short).**

* `src/hei_nw/{gate.py, eviction.py, store.py, pack.py, telemetry/gate.py, eval/harness.py, datasets/scenario_{a,c}.py}`
* `scripts/run_m3_gate_calibration.sh`, `scripts/plot_gate_calibration.py`, `scripts/report_gate_write_rates.py`
* `tests/{test_gate.py,test_trace_writer.py,test_eviction.py,eval/test_harness_gate_flow.py,scripts/test_plot_gate_calibration.py,utils/test_scripts.py}`
* `reports/m3-write-gate/{A_* , C_* , tau_*}`, plus `documentation/write-gate.md`, `documentation/quick-validate.md`, `pyproject.toml`, `.github/workflows/ci.yml`

**Planning anchors used.**

* `planning/milestone-3-plan.md` — **M3 — Neuromodulated Write Gate + Trace Store** (Tasks T1–T6; DoD checklist §6; Deliverables §5).
* `planning/design.md` — §5.5 **Trace schema**, §5.7 **Neuromodulated Write Gate**, memory packer.
* `planning/validation-plan.md` — modes B0–B3; scenarios A/C; gate metrics and protocol.
* `planning/project-plan.md` — M3 scope/DoD (gate + pointer-only traces + telemetry artifacts).

**Assumptions/limits.**

* I did not execute GPU runs; judgments on “passes/coverage” are based on code/tests/artifacts in repo.
* “Manual privacy audit” DoD item has no explicit artifact; I assessed code/tests and the harness pointer audit.

---

### 3) DoD / Acceptance verification table

| Item                                                             | Evidence (files/funcs/CLI)                                                                                                                                                                 | Status                    | Notes                                                                                                                        |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Gate telemetry (PR-AUC, precision/recall, calibration) for A & C | `telemetry/gate.py::compute_gate_metrics`; `reports/m3-write-gate/A_gate_telemetry.json`, `C_B1_metrics.json`’s `gate.telemetry`                                                           | **Pass**                  | Telemetry present incl. calibration bins and PR-AUC; pins-only slice exists for C.                                           |
| τ tuned to 1–5 writes/1k                                         | `reports/m3-write-gate/A_sweep_summary.tsv`                                                                                                                                                | **Fail**                  | A sweep shows **500/1k** constant for τ∈\[1.1,1.8]. No evidence of target band. Also currently “per 1k records”, not tokens. |
| Pointer-only persistence; packer redaction; manual privacy check | `store.TraceWriter.write` (enforces pointer/excludes text); `pack.py::_validate_pointer_payload`; tests `test_trace_writer.py`, `test_pack.py`; harness pointer check in `eval/harness.py` | **Partial**               | Code/tests enforce pointer-only and ban raw text keys. “Manual privacy audit” completion not evidenced as an artifact.       |
| Eviction/decay + pin protection                                  | `eviction.DecayPolicy`, `PinProtector`; `store.EpisodicStore.evict_stale`; tests `test_eviction.py`                                                                                        | **Pass**                  | TTL removal and pin protection tested.                                                                                       |
| No stubs                                                         | CI step `scripts/grep_no_stubs.sh`                                                                                                                                                         | **Pass**                  | Grep rule covers TODO/FIXME/NotImplementedError.                                                                             |
| Quality gates (`black`, `ruff`, `mypy`, `pytest ≥85% diff`)      | `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`                                                                                                                    | **Pass (config present)** | Tooling & workflow configured; actual thresholds not re-run here.                                                            |
| Calibration script produces artifacts                            | `scripts/run_m3_gate_calibration.sh` (supports `--threshold-sweep`), `tests/utils/test_scripts.py::test_run_m3_gate_calibration_smoke`; `reports/m3-write-gate/*`                          | **Pass**                  | Script exists, exercised by tests, artifacts present.                                                                        |
| Documentation updated                                            | `documentation/write-gate.md`, `documentation/quick-validate.md` (M3 steps)                                                                                                                | **Pass**                  | Gate usage/tuning documented.                                                                                                |

---

### 4) Task-by-task review (mirror the milestone plan order)

#### M3-T1 \[CODEX] Neuromodulated Gate Core

* **Intent.** Implement feature channels and thresholded decision with weights.
* **Findings.** `gate.py` defines `SalienceFeatures`, `NeuromodulatedGate`, helpers (`surprise_from_logits/prob`, `novelty_from_similarity`, `bool_to_signal`). Harness exposes flags `--gate.alpha/beta/gamma/delta/threshold`. Tests: `tests/test_gate.py` covers weighted sum, threshold control, helpers.
  Example: CLI in `eval/harness.py` adds `--gate.alpha`…`--gate.threshold` and passes them into `NeuromodulatedGate(...)`.
* **Gaps/Risks.** None functionally; see end-to-end wiring (T4).
* **Status.** **Pass**.

#### M3-T2 \[CODEX] Pointer-only Trace Writer & Telemetry

* **Intent.** Persist pointer-only traces with salience metadata; compute gate telemetry.
* **Findings.** `store.TraceWriter.write(...)` emits only `tokens_span_ref`, `entity_slots`, `salience_tags`, `eviction` (no raw text). Privacy bans are centralized via `_BANNED_TEXT_KEYS` (“episode\_text”, “full\_text”, …). `telemetry/gate.py::compute_gate_metrics` returns precision/recall/PR-AUC/clutter+calibration. Tests: `tests/test_trace_writer.py`, `tests/test_gate_metrics.py`.
  Short excerpt: `TraceWriter.write` returns payload keys including `"tokens_span_ref"` and `"salience_tags"`.
* **Gaps/Risks.** Writer is not used in the B1 harness path; writes are not gate-driven.
* **Status.** **Pass** for components; **gap** in usage (see T4).

#### M3-T3 \[CODEX] Eviction, Decay, Pin Protection

* **Intent.** TTL decay and pin/higher-salience protection.
* **Findings.** `eviction.DecayPolicy` computes TTL; `PinProtector.blocks_eviction` prevents removal; integrated in `EpisodicStore.evict_stale` (marks inactive in index). Tests: `tests/test_eviction.py`.
  Example: `PinProtector(...).blocks_eviction(state)` used inside `EpisodicStore.evict_stale`.
* **Gaps/Risks.** None found.
* **Status.** **Pass**.

#### M3-T4 \[CODEX] Harness Integration & Calibration Reports

* **Intent.** Compute gate decisions during streaming; log telemetry; provide calibration scripts/plots and sweep.
* **Findings.** Harness computes gate diagnostics and telemetry; outputs in metrics JSON (`gate.telemetry`, `write_rate`, `write_rate_per_1k`). Scripts: `run_m3_gate_calibration.sh` (supports `--threshold-sweep`, `--pin-eval`), `plot_gate_calibration.py`, `report_gate_write_rates.py`. Tests: `tests/eval/test_harness_gate_flow.py::test_gate_metrics_logged`, `tests/utils/test_scripts.py` validates `--threshold-sweep` end-to-end.
  Example: sweep creates `reports/m3-write-gate/tau_*` directories with `A_gate_calibration.png`.
* **Gaps/Risks.** **Gate does not drive writes into the store** (store is built from `should_remember` labels in `EpisodicStore.from_records`), so τ changes do not affect retrieval/write-rate in B1. Also write-rate is computed per **records**, not per **tokens** (plan says “1–5 / 1k tokens”).
* **Status.** **Partial**.

#### M3-T5 \[CODEX] Scenario C Reward & Pin Enhancements

* **Intent.** Provide reward/pin labels, novelty counters, deterministic fixtures.
* **Findings.** `datasets/scenario_c.py` emits `reward_annotation`, `pin_annotation`, `novelty_counters`. Fixture `tests/fixtures/scenario_c_gate.json`. Test: `tests/test_scenario_c.py`.
* **Gaps/Risks.** None found.
* **Status.** **Pass**.

#### M3-T6 \[CODEX] Documentation & MkDocs

* **Intent.** Document gate formula, defaults, τ tuning; quick-validate updates.
* **Findings.** `documentation/write-gate.md` (formula, defaults, single-τ & sweep & pins-only workflows); `documentation/quick-validate.md` includes M3 steps.
* **Gaps/Risks.** Docs still imply achieving the 1–5/1k target; committed sweep does not.
* **Status.** **Pass** for docs; **content mismatch** with artifacts.

#### M3-R1..R3 \[HUMAN/ChatGPT]

* **Intent.** Review implementation, perform threshold tuning on A/C, privacy/pointer audit.
* **Findings.** Reports exist (`reports/m3-write-gate/*`), but τ **not tuned** to target; **no explicit artifact** for the manual privacy audit.
* **Status.** **Partial**.

---

### 5) Design & validation alignment

* **Design mapping.** Gate module and weights mirror §5.7 in `design.md`. Trace schema matches §5.5 (pointer-only `tokens_span_ref`, `entity_slots`, `salience_tags`). Packer enforces the schema and bans text. Eviction/decay is aligned with “Forget” stage.
  Pointers: `src/hei_nw/gate.py` (S = α·surprise + β·novelty + γ·reward + δ·pin), `src/hei_nw/pack.py` (pointer enforcement), `src/hei_nw/store.py` (ANN + Hopfield, TTL hooks).
* **Validation mapping.** Harness produces gate telemetry and calibration for A/C per `validation-plan.md`. **Mismatch:** the plan’s acceptance mentions **writes per 1k tokens**, but the harness computes `write_rate_per_1k` as **per 1k records**. Also, because the store is still built from labels (`should_remember`) rather than gate outcomes, τ does not affect retrieval nor realized write-rate in B1.

---

### 6) Quality & CI assessment

* **Tooling.** `black`, `ruff`, `mypy`, `pytest` configured (`pyproject.toml`, `.pre-commit-config.yaml`). CI (`.github/workflows/ci.yml`) installs deps, runs tests, performs the **no-stubs** grep, and includes QA prompting smokes and gate non-empty checks.
* **Tests.** Good unit coverage for gate math, telemetry, writer, eviction, scripts; harness tests exercise gate logging and pins-only slicing. Determinism handled via seeds and fixtures. Integration is thoughtfully stubbed in script tests to keep CI light.

---

### 7) Gaps and **Follow-up tasks**

> IDs start at **M3-F#**. All are scoped within M3.

````
### M3-F1 [CODEX] Make gate decisions drive writes in B1 (streaming path)

* **Goal:** Ensure τ actually controls which episodes enter the store during B1 eval.
* **Key changes:**
  1) In `eval/harness.py::_evaluate_mode_b1`, build a gate-filtered list (`indexed`) and pass **only** those to `RecallService.build`/`EpisodicStore.from_records`.
  2) Add `--gate.use_for_writes` (default: on) and a `--gate.debug_keep_labels` escape hatch for A/B testing.
* **Tests:**
  - `tests/eval/test_harness_gate_flow.py::test_gate_controls_store_size` (vary τ and assert store.ntotal decreases).
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 64 \
    --gate.threshold 3.0 --outdir reports/tmp_f1
  # Expect reports/tmp_f1/* with gate.write_rate << 0.5 and store.ntotal shrinking vs τ=1.5
````

```
```

### M3-F2 \[CODEX] Correct write-rate metric to “per 1k tokens”

* **Goal:** Align telemetry with the plan (normalize writes by generated tokens).
* **Key changes:**

  1. Track generated token count in B1 loop (already tracked as `generated_tokens`); compute `writes_per_1k_tokens = writes / (gen_tokens/1000)`.
  2. Update `telemetry/gate.py` and harness summary fields; keep backward-compatible `writes_per_1k_records` if needed.
  3. Update `scripts/report_gate_write_rates.py` and docs to reflect tokens.
* **Tests:**

  * `tests/scripts/test_plot_gate_calibration.py` to read new field.
  * `tests/eval/test_harness_gate_flow.py::test_writes_per_1k_tokens_present`.
* **Quality gates:** standard.
* **Acceptance check:**

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 32 \
    --outdir reports/tmp_f2
  # Expect gate.telemetry to include writes_per_1k_tokens (finite, >0).
  ```

```
```

### M3-F3 \[CODEX] Expand τ sweep range & summary

* **Goal:** Produce a sweep that actually explores decision boundary.
* **Key changes:**

  1. In `run_m3_gate_calibration.sh`, if `--threshold-sweep` absent, default to a broader set (e.g., `0.5 1.0 1.5 2.0 2.5 3.0 3.5`).
  2. Add PR-AUC and write metrics to the sweep TSV for both records and tokens.
* **Tests:**

  * `tests/utils/test_scripts.py::test_threshold_sweep_generates_diverse_dirs` (already partial; extend assertions for summary TSV columns).
* **Quality gates:** standard.
* **Acceptance check:**

  ```bash
  bash scripts/run_m3_gate_calibration.sh --scenario A --n 64 --seed 7
  # Expect sweeping TSV with non-flat write rates across τ.
  ```

```
```

### M3-F4 \[CODEX] Add automated privacy audit for trace samples

* **Goal:** Turn the “manual privacy audit” into a reproducible check.
* **Key changes:**

  1. New `scripts/audit_pointer_payloads.py` scanning metrics `gate.trace_samples` and store payloads for banned text keys.
  2. Add CI step invoking it on recent artifacts or a small synthetic run.
* **Tests:**

  * `tests/scripts/test_audit_pointer_payloads.py::test_detects_banned_keys`.
* **Quality gates:** standard.
* **Acceptance check:**

  ```bash
  python scripts/audit_pointer_payloads.py reports/m3-write-gate/A_B1_metrics.json
  # Exit 0; print summary "pointer_only: true".
  ```

```
```

### M3-F5 \[CODEX] Clarify docs: B1 store is label-driven (M3), gate-driven optional

* **Goal:** Avoid confusion about “HEI-NW decides which episodes to store”.
* **Key changes:**

  1. Update `documentation/write-gate.md` and `README.md` to note that M3 uses labels for store construction by default; gate-driven writes become active with `--gate.use_for_writes` (from F1).
* **Tests:** n/a (docs).
* **Quality gates:** standard.
* **Acceptance check:**

  ```bash
  rg "use_for_writes" documentation/ README.md
  ```

```
```

### M3-F6H \[HUMAN/ChatGPT] Threshold tuning & artifacts (A and C)

* **Goal:** Produce runs demonstrating τ in the 1–5 writes/1k **tokens** band and attach artifacts.
* **Steps:**

  1. After F1–F3, run `run_m3_gate_calibration.sh --scenario A --n 256 --seed {13,17}` and `--scenario C`.
  2. Choose τ achieving ≈1–5/1k tokens; commit sweep TSV/JSON and plots.
* **Acceptance check:** Attach `reports/m3-write-gate/*_sweep_summary.tsv` showing writes\_per\_1k\_tokens ∈ \[1,5].

```

---

### 8) Final verdict
**Verdict: Partial.** The gate, writer, eviction, telemetry, scripts, tests, and docs are **implemented and real**, but **τ is not tuned to target in the committed artifacts**, the **metric normalization is off (per records vs per tokens)**, and **gate decisions don’t yet control writes into the store** in the harness.

**Minimum to reach DoD for M3:** **M3-F1**, **M3-F2**, **M3-F3**, **M3-F6H**.  
(Recommend **M3-F4** and **M3-F5** for robustness/clarity.)

---

#### Bonus note on your earlier issue (“`--threshold-sweep` not available”)
In this repo, `scripts/run_m3_gate_calibration.sh` **does** support `--threshold-sweep` and is exercised by `tests/utils/test_scripts.py::test_run_m3_gate_calibration_smoke`. If you still see the flag missing, you’re likely on an older branch/snapshot; pull this version and re-run.
```
