# M3 — Neuromodulated Write Gate + Trace Store — Milestone Review

### 1) Executive summary

**Intent.** Milestone 3 stands up a neuromodulated write gate that decides which episodes to persist; stores pointer-only traces with salience metadata and eviction/decay (pins protected); and validates gate quality (PR-AUC, calibration, write-rate) for Scenarios A & C.

**Reality.** The repo contains a real gate implementation (`src/hei_nw/gate.py`), pointer-only trace writer and packer (`store.py`, `pack.py`), eviction/decay and pin protection (`eviction.py`), harness integration with gate telemetry and plots, and calibration scripts that produced artifacts in `reports/m3-write-gate/` and additional sweeps under `reports/m3-gate-threshold*`.

**Verdict.** **Pass**, with caveats. All DoD/acceptance artifacts exist and are wired. Gate telemetry for A & C is present; τ tuned for A lands in the 1–5 writes/1k band. Two risks deserve follow-ups: (1) Scenario A shows a *flat* write-rate plateau across τ∈\[1.2, 2.0] → weak calibration leverage; (2) a small **doc↔code mismatch** around whether `pin=True` forces a write vs “just” boosts salience.

---

### 2) Evidence snapshot: repo & planning anchors

**Repo tree (short).**

* `src/hei_nw/` → `gate.py`, `store.py`, `eviction.py`, `recall.py`, `pack.py`, `eval/harness.py`, `telemetry/gate.py`
* `scripts/` → `run_m3_gate_calibration.sh`, `plot_gate_calibration.py`, `report_gate_write_rates.py`, `audit_pointer_payloads.py`
* `tests/` → `test_gate.py`, `test_trace_writer.py`, `test_eviction.py`, `eval/test_harness_gate_flow.py`, `scripts/test_run_m3_gate_calibration_smoke.py`
* `reports/` → `m3-write-gate/` (A & C sweeps/telemetry/plots); `m3-gate-threshold{12,15,20,25,30}/…`
* `planning/` → `milestone-3-plan.md`, `design.md`, `validation-plan.md`, `project-plan.md`
* `documentation/` → `write-gate.md`, `quick-validate.md`

**Planning anchors used.**

* `planning/milestone-3-plan.md`

  * **“M3 — Neuromodulated Write Gate + Trace Store”** (H1)
  * Task list **M3-T1…T6** and reviews **M3-R1…R3**
  * **“Deliverables & Artifacts”**, **“Definition of Done (DoD) Checklist”**
* `planning/design.md`

  * **“5.7 Neuromodulated Write Gate”** (signals, formula, defaults)
  * **“5.9 Eviction / Decay / Protection”**
  * **“5.10 Telemetry, Provenance, Safety”**
* `planning/validation-plan.md`

  * **Write-gate quality**: precision/recall, PR-AUC, clutter rate, calibration for S; Scenario A & C protocol.

**Assumptions/limits.**

* I did not execute tests/CI here; verdicts rely on code/tests presence and the produced `reports/` artifacts checked into the repo.

---

### 3) DoD / Acceptance verification table

| Item                                                                        | Evidence (files/funcs/CLI)                                                                                                                                                                                                      | Status   | Notes                                                                  |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | ---------------------------------------------------------------------- |
| Gate telemetry (PR-AUC, P/R, clutter, calibration) for A & C                | `reports/m3-write-gate/tau_1.5/A_gate_telemetry.json` → keys: `pr_auc`, `precision`, `recall`, `calibration`; `reports/m3-write-gate/tau_3.5/C_gate_telemetry_pins.json`                                                        | **Pass** | Both scenarios recorded; C includes `pins_only` and `non_pins` slices. |
| τ tuned to 1–5 writes/1k tokens; results logged                             | `reports/m3-write-gate/A_sweep_summary.json` → `"writes_per_1k_tokens": 3.275…`, `"first_tau_within_target_band": 1.5`                                                                                                          | **Pass** | A hits the target band at τ=1.5.                                       |
| Pointer-only trace store; packer enforces redaction; manual privacy check   | `src/hei_nw/pack.py` rejects `_BANNED_TEXT_KEYS`; `tests/test_trace_writer.py` asserts no `episode_text`; `scripts/audit_pointer_payloads.py` present; `reports/m3-write-gate/tau_1.5/A_trace_samples.json` shows only pointers | **Pass** | Sample shows `pointer` fields and no raw text.                         |
| Eviction/decay removes expired; pins protected                              | `src/hei_nw/eviction.py`; `tests/test_eviction.py::test_ttl_decay_removes_expired`, `::test_pin_protection_blocks_eviction`                                                                                                     | **Pass** | Verified by unit tests.                                                |
| No stubs / TODOs                                                            | `scripts/grep_no_stubs.sh`; quick grep over `src/` found none                                                                                                                                                                   | **Pass** | CI includes stub grep step.                                            |
| Quality gates present (`black`, `ruff`, `mypy`, `pytest`) and coverage gate | `pyproject.toml` contains `[tool.black]`, `[tool.ruff]`, `[tool.mypy]`, pytest config; `.github/workflows/ci.yml`                                                                                                               | **Pass** | Configs & CI present; not re-run here.                                 |
| Calibration script produces artifacts in `reports/m3-write-gate/`           | `scripts/run_m3_gate_calibration.sh`; artifacts exist under `reports/m3-write-gate/`                                                                                                                                            | **Pass** | Also `plot_gate_calibration.py` and `…_summary.json`.                  |
| Documentation updated                                                       | `documentation/write-gate.md`, `README.md` references                                                                                                                                                                           | **Pass** | Gate defaults + tuning described.                                      |

---

### 4) Task-by-task review

#### M3-T1 \[CODEX] Neuromodulated Gate Core

* **Intent.** Implement S=α·surprise+β·novelty+γ·reward+δ·pin and thresholded write decision; harness computes/records diagnostics and exposes `--gate.*` flags.
* **Findings.**

  * `src/hei_nw/gate.py`: class **`NeuromodulatedGate`** (“*writes an episode when `S > τ`*”) with defaults α=1.0, β=1.0, γ=0.5, δ=0.8, τ=1.5.
  * Harness flags: `--gate.alpha … --gate.threshold … --gate.use_for_writes` in `src/hei_nw/eval/harness.py`.
  * Tests: `tests/test_gate.py::test_gate_computes_weighted_salience`, `::test_gate_threshold_controls_write_rate`, plus surprise/novelty helpers.
* **Gaps/Risks.** Slight **doc↔code mismatch** around pin semantics (docstring hints “should be written regardless”; code still applies threshold).
* **Status.** **Pass**.

#### M3-T2 \[CODEX] Pointer-only Trace Writer & Telemetry

* **Intent.** Persist pointer-only traces with salience metadata; compute PR-AUC, P/R, calibration, clutter.
* **Findings.**

  * `src/hei_nw/store.py`: **`TraceWriter`** (“*Persist pointer-only episodic traces with salience and decay metadata.*”).
  * `src/hei_nw/telemetry/gate.py`: **`compute_gate_metrics`** emits `precision`, `recall`, `pr_auc`, `calibration`, `clutter_rate`.
  * `tests/test_trace_writer.py` verifies pointer-only payload and salience tags; `tests/test_gate_metrics.py` checks metrics math.
* **Gaps/Risks.** None blocking.
* **Status.** **Pass**.

#### M3-T3 \[CODEX] Eviction, Decay, Pin Protection

* **Intent.** TTL/decay, last-access timestamps, and pin-based protection.
* **Findings.**

  * `src/hei_nw/eviction.py`: `DecayPolicy`, `PinProtector`; `store.py` keeps an `_eviction_state`.
  * `tests/test_eviction.py` validates TTL expiry and pin protection.
* **Gaps/Risks.** Harness does not run periodic eviction (only unit/demo path), but DoD only requires correctness, not wiring into runs.
* **Status.** **Pass**.

#### M3-T4 \[CODEX] Harness Integration & Calibration Reports

* **Intent.** Gate in the harness; telemetry and plots; calibration script(s).
* **Findings.**

  * Harness computes and stores telemetry; `_summarize_gate(...)` uses `compute_gate_metrics`.
  * Scripts: `run_m3_gate_calibration.sh` (manual & auto sweep), `plot_gate_calibration.py` (Matplotlib, PNG), `report_gate_write_rates.py`.
  * Tests: `tests/eval/test_harness_gate_flow.py` (metrics logged), `tests/utils/test_scripts.py::test_run_m3_gate_calibration_smoke`.
* **Gaps/Risks.** Scenario A sweep shows **flat write-rate plateau** for τ∈{1.2,1.5,2.0} (0.5 write rate each), indicating weak lever-arm in S.
* **Status.** **Pass** (with caveat).

#### M3-T5 \[CODEX] Scenario C Reward & Pin Enhancements

* **Intent.** Scenario C emits reward/pin/novelty; deterministic fixtures; docs.
* **Findings.**

  * `src/hei_nw/datasets/scenario_c.py` emits `gate_features` incl. `reward`, `pin`, novelty counters.
  * `tests/test_scenario_c.py::test_reward_pin_annotations` and fixtures `tests/fixtures/scenario_c_gate.json`.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M3-T6 \[CODEX] Documentation & MkDocs updates

* **Intent.** Document formula, defaults, τ tuning flow; update README/quick-validate.
* **Findings.** `documentation/write-gate.md` (defaults & calibration); README references write gate.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M3-R1 \[HUMAN/ChatGPT] Gate Review & Threshold Tuning

* **Findings.** `reports/m3-write-gate/A_sweep_summary.json` shows τ=1.5 yields \~3.28 writes/1k tokens (within target).
* **Gaps/Risks.** Plateau across τ 1.2–2.0 suggests S distribution poorly separated; may reduce control over write budget.
* **Status.** **Pass** (tuned for A; see follow-ups).

#### M3-R2 \[HUMAN/ChatGPT] Scenario C Validation

* **Findings.** C pins-only telemetry shows write\_rate=1.0 across τ (by design); non-pins slice reported separately with lower write\_rate (\~0.149 at τ=3.5).
* **Gaps/Risks.** None; telemetry splits are correct.
* **Status.** **Pass**.

#### M3-R3 \[HUMAN/ChatGPT] Privacy & Pointer Audit

* **Findings.** `A_trace_samples.json` includes `pointer` and `entity_slots_keys` only; CI includes `audit_pointer_payloads.py` smoke.
* **Status.** **Pass**.

---

### 5) Design & validation alignment

* **Design mapping.**

  * **Gate (Design §5.7).** Implemented in `src/hei_nw/gate.py`; formula & defaults match docs; harness exposes `--gate.*` knobs.
  * **Trace schema/packer (Design §5.6, §5.10).** Packer enforces pointer-only via `_BANNED_TEXT_KEYS`; telemetry includes salience breakdown paths.
  * **Eviction/protection (Design §5.9).** `DecayPolicy` + `PinProtector` mirror the design; stored TTL/last-access fields.
* **Validation mapping.**

  * **Write-gate quality** (validation-plan): precision/recall, PR-AUC, clutter, calibration all present in telemetry JSON; Scenario A tuning within 1–5 writes/1k; Scenario C pins and non-pins slices both reported.
  * **Modes B0–B3.** Harness retains B0/B1 infra; M3 focuses on B1 write decisions and telemetry (consistent with plan).
  * **No metrics mismatch observed.**

---

### 6) Quality & CI assessment

* **Tooling.** `black`, `ruff`, `mypy` configured in `pyproject.toml`; pytest markers present; coverage configured; CI workflow (`.github/workflows/ci.yml`) installs, lints, runs tests, and includes a pointer-audit smoke step.
* **Testing depth.** Unit tests for gate math, metrics, writer privacy, eviction, harness logging, and scripts; scenario fixtures ensure determinism. Some script tests are “smoke” (expected) and could be extended with stronger monotonicity assertions (see follow-ups).

---

### 7) Gaps and **Follow-up tasks**

#### M3-F1 \[CODEX] Clarify and enforce pin semantics

* **Goal:** Align code/docs: either (a) `pin=True` **forces** a write, or (b) docstrings/docs clearly state “pin boosts salience but still compares to τ”.
* **Key changes:**

  1. Update `src/hei_nw/gate.py` docstrings & `documentation/write-gate.md`.
  2. Optionally add `--gate.pin_override` flag; if set, bypass threshold when `pin=True`.
* **Tests:**

  * `tests/test_gate.py::test_pin_override_forces_write` (new).
  * `tests/test_harness_gate_flow.py::test_pin_semantics_documented`.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  PYTHONPATH=src python - <<'PY'
  from hei_nw.gate import NeuromodulatedGate,SalienceFeatures
  g=NeuromodulatedGate(threshold=3.0); d=g.decision(SalienceFeatures(0,0,False,True))
  print(d.should_write)  # expected per chosen policy
  PY
  ```

#### M3-F2 \[CODEX] Add S-distribution diagnostics to calibration

* **Goal:** Explain the plateau by logging S quantiles/histograms per scenario and split (pins/non-pins).
* **Key changes:**

  1. Extend `telemetry/gate.py` to compute `{p10,p50,p90}` and histogram bins of S.
  2. Modify `eval/harness.py` to persist `gate.score_distribution.*` into telemetry JSON and annotate plots.
* **Tests:**

  * `tests/test_gate_metrics.py::test_score_distribution_fields_present`
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  bash scripts/run_m3_gate_calibration.sh --scenario A --n 64
  # Expect A_gate_telemetry.json to include score quantiles & histogram.
  ```

#### M3-F3 \[CODEX] Auto-tune τ for target write-rate (robust search)

* **Goal:** Make calibration less brittle; find minimal τ that yields target 1–5 writes/1k tokens (A) with binary search.
* **Key changes:**

  1. Enhance `run_m3_gate_calibration.sh` to support `--threshold-sweep auto --target-band "1 5" --target-per tokens`.
  2. Record the auto-selected τ in `*_sweep_summary.json`.
* **Tests:**

  * Extend `tests/scripts/test_run_m3_gate_calibration_smoke.py` to assert the auto mode emits `"first_tau_within_target_band"`.
* **Quality gates:** `shellcheck` (optional) · `pytest -q`
* **Acceptance check:**

  ```bash
  scripts/run_m3_gate_calibration.sh --scenario A --n 48 --threshold-sweep auto --target-band "1 5"
  cat reports/m3-write-gate/A_sweep_summary.json  # shows chosen τ
  ```

#### M3-F4 \[CODEX] Strengthen monotonicity test for τ→write-rate

* **Goal:** Ensure write counts are **non-increasing** as τ rises (at least on synthetic/stub data).
* **Key changes:**

  1. Add `tests/scripts/test_run_m3_gate_calibration.py` with a deterministic stub harness producing monotonic S.
  2. Assert monotonicity over a provided sweep.
* **Tests:** new test file above.
* **Quality gates:** `pytest -q`
* **Acceptance check:**

  ```bash
  pytest -q tests/scripts/test_run_m3_gate_calibration.py::test_monotonic_write_rate
  ```

#### M3-F5 \[CODEX] Wire optional background eviction into harness runs

* **Goal:** Demonstrate end-to-end eviction without manual script, behind a flag (off by default).
* **Key changes:**

  1. Add `--store.evict_stale` and interval to `eval/harness.py`; call `EpisodicStore.evict_stale()` between batches when enabled.
  2. Log `evicted_count` in run summary.
* **Tests:**

  * `tests/eval/test_harness_gate_flow.py::test_optional_eviction_path`
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A --n 32 --store.evict_stale 1
  # Summary logs include evicted_count
  ```

#### M3-F6 \[HUMAN/ChatGPT] Interpret A & C calibration and set τ policy

* **Goal:** Pick production τ (or per-scenario τs) using new score distributions to avoid flat regions.
* **Steps:** Run the enhanced calibration; inspect `score_distribution` quantiles; choose τ that lands within 1–5 writes/1k with best precision; record in `documentation/write-gate.md`.
* **Acceptance check:** Commit an updated “τ selection” note plus the new A/C telemetry JSONs and plots.

---

### 8) Final verdict

**Pass.** The gate, trace store, telemetry, scripts, reports, tests, and docs are present and aligned with the plan; DoD items are satisfied.
**Recommended follow-ups to harden M3:** **M3-F1**, **M3-F2**, **M3-F3**, **M3-F4** (and optionally **M3-F5**, **M3-F6**). These address the observed τ plateau, clarify pin semantics, and improve calibration robustness without expanding scope beyond M3.
