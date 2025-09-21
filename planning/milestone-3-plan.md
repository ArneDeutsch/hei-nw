# M3 — Neuromodulated Write Gate + Trace Store

## 1) Milestone Summary
1. Stand up the neuromodulated write gate so HEI-NW can decide which episodes to store.  
2. Persist pointer-only episodic traces with salience metadata, plus eviction/decay with pin protection.  
3. Validate gate quality (PR-AUC, calibration, write rate) on scenarios A & C while enforcing privacy guarantees.

## 2) Dependencies / Inputs
- Retrieval stack & recall service from M2 (`src/hei_nw/keyer.py`, `src/hei_nw/store.py`, `src/hei_nw/recall.py`).  
- Scenario generators with `should_remember` labels (`src/hei_nw/datasets/scenario_{a,c}.py`) and harness plumbing (`src/hei_nw/eval/harness.py`).  
- Design spec §5.5 (Trace schema), §6 (Neuromodulated write gate), §7 (Eviction/decay), plus validation-plan.md §3 (gate metrics) & §4 (protocol).

## 3) [CODEX] Implementation Tasks

1. **M3-T1 — Neuromodulated Gate Core**
   - **Goal:** Implement the salience feature computation and thresholded gate decision.  
   - **Key changes:**
     1. Create `src/hei_nw/gate.py` with `SalienceFeatures`, `NeuromodulatedGate`, and helpers for surprise/novelty/reward/pin features per design defaults.
     2. Extend `src/hei_nw/eval/harness.py` to compute gate decisions when streaming records (B1/B2/B3), capture salience diagnostics, and expose `--gate.*` CLI knobs.
     3. Update dataset utilities to surface required feature channels (novelty flags, reward tags, pins).
   - **Tests:**
     1. `tests/test_gate.py::test_gate_computes_weighted_salience` — verifies weighted sum matches spec.
     2. `tests/test_gate.py::test_gate_threshold_controls_write_rate` — ensures τ leads to expected write ratios on synthetic data.
     3. Coverage ≥ 90% in `tests/test_gate.py`.
   - **Quality gates:** `black .`, `ruff .`, `mypy .`, `pytest -q`, plus `pytest tests/test_gate.py`.
   - **Acceptance check:** `PYTHONPATH=src python -m hei_nw.eval.gate_sanity --scenario A --n 16 --seed 3` prints write rate in [1,5]/1k and salience stats.

2. **M3-T2 — Pointer-only Trace Writer & Telemetry**
   - **Goal:** Persist gate-approved episodes in pointer-only form and log telemetry.  
   - **Key changes:**
     1. Extend `src/hei_nw/store.py` with a `TraceWriter` storing pointer references, provenance, salience metadata.
     2. Add `src/hei_nw/telemetry/gate.py` for PR-AUC, precision/recall, calibration, clutter rate.
     3. Update `src/hei_nw/pack.py` to enforce pointer-only packing and pass privacy checks.
   - **Tests:**
     1. `tests/test_trace_writer.py::test_pointer_only_payload`.
     2. `tests/test_gate_metrics.py::test_precision_recall_computation`.
   - **Quality gates:** standard suite.
   - **Acceptance check:** `PYTHONPATH=src python -m hei_nw.telemetry.dump_gate_metrics --scenario A --out reports/m3-write-gate/telemetry.json` shows finite metrics & calibration bins.

3. **M3-T3 — Eviction, Decay, Pin Protection**
   - **Goal:** Implement decay/TTL eviction while safeguarding pinned traces.  
   - **Key changes:**
     1. Add `src/hei_nw/eviction.py` with `DecayPolicy`, `PinProtector`, TTL scheduling.
     2. Integrate eviction hooks into `EpisodicStore` maintenance loop (`src/hei_nw/store.py`).
     3. Emit last-access timestamps & TTL metadata for traces.
   - **Tests:**
     1. `tests/test_eviction.py::test_ttl_decay_removes_expired`.
     2. `tests/test_eviction.py::test_pin_protection_blocks_eviction`.
   - **Quality gates:** standard suite.
   - **Acceptance check:** `PYTHONPATH=src python -m tests.manual.eviction_demo` prints surviving traces verifying pins kept and expired removed.

4. **M3-T4 — Harness Integration & Calibration Reports**
   - **Goal:** Surface gate telemetry through harness CLI and produce calibration artifacts.  
   - **Key changes:**
     1. Update harness CLI (`src/hei_nw/eval/harness.py`) with gate-specific flags, integrate with the write flow, and record metrics per run.
     2. Extend `src/hei_nw/eval/report.py` to include gate telemetry, calibration PNG, clutter stats, pointer-only confirmation.
     3. Add scripts `scripts/run_m3_gate_calibration.sh` & `scripts/plot_gate_calibration.py` writing to `reports/m3-write-gate/`.
   - **Tests:**
     1. `tests/eval/test_harness_gate_flow.py::test_gate_metrics_logged`.
     2. `tests/utils/test_scripts.py::test_run_m3_gate_calibration_smoke`.
   - **Quality gates:** standard suite.
   - **Acceptance check:** `scripts/run_m3_gate_calibration.sh --scenario A --n 48 --seed 13` produces telemetry JSON+PNG in `reports/m3-write-gate/`.

5. **M3-T5 — Scenario C Reward & Pin Enhancements**
   - **Goal:** Supply gate labels and pin-worthy items for scenario C evaluation.  
   - **Key changes:**
     1. Extend `src/hei_nw/datasets/scenario_c.py` to emit reward annotations, pin flags, and novelty counters.
     2. Add fixture seeds under `tests/fixtures/scenario_c_gate.json` for deterministic tests.
     3. Ensure generator docs describe reward/pin semantics.
   - **Tests:** `tests/test_scenario_c.py::test_reward_pin_annotations`.
   - **Quality gates:** standard suite.
   - **Acceptance check:** `PYTHONPATH=src python - <<'PY' ...` verifying scenario C output fields (command included in task docstring).

6. **M3-T6 — Documentation & MkDocs updates**
   - **Goal:** Document gate usage, tuning, and telemetry interpretation.  
   - **Key changes:**
     1. Add `documentation/write-gate.md` summarizing salience formula, defaults, τ tuning flow.
     2. Update `README.md` & `documentation/quick-validate.md` with M3 calibration instructions.
     3. Ensure new public APIs include docstrings.
   - **Tests:** `pytest -q` (docs unaffected) and optional `python -m pydocstyle` if configured.
   - **Quality gates:** standard suite.
   - **Acceptance check:** `rg "write gate" documentation/` shows new usage guide references.

## 4) [HUMAN/ChatGPT] Review & GPU Tasks

1. **M3-R1 — Gate Review & Threshold Tuning**
   - Review implementation for design compliance and pointer-only guarantees.  
   - Run `scripts/run_m3_gate_calibration.sh --scenario A --n 512 --threshold-sweep` on GPU if needed; confirm τ achieves 1–5 writes/1k with high precision.

2. **M3-R2 — Scenario C Validation**
   - Execute `scripts/run_m3_gate_calibration.sh --scenario C --n 512 --pin-eval`.  
   - Verify pinned traces survive eviction and reward-positive items have high salience.

3. **M3-R3 — Privacy & Pointer Audit**
   - Inspect `reports/m3-write-gate/trace_samples.json` or equivalent to confirm pointer-only payloads and absence of raw text.

## 5) Deliverables & Artifacts
- `src/hei_nw/gate.py`, `src/hei_nw/eviction.py`, updated `store.py`, `pack.py`, telemetry modules.  
- Gate calibration scripts & plots under `scripts/` and `reports/m3-write-gate/`.  
- Scenario C enhancements plus unit fixtures.  
- Documentation: `documentation/write-gate.md`, README updates, quick-validate instructions.

## 6) Definition of Done (DoD) Checklist
1. Gate telemetry (PR-AUC, precision/recall, clutter rate, calibration curve) recorded for scenarios A & C.  
2. τ tuned to achieve 1–5 writes / 1k tokens, results logged in reports.  
3. Trace store persists pointer-only payloads; packer enforces redaction; manual privacy check complete.  
4. Eviction/decay removes expired traces while protecting `pin=True`.  
5. `git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."`  
6. `black .`, `ruff .`, `mypy .`, `pytest -q` pass; changed-line coverage ≥ 85%.  
7. `scripts/run_m3_gate_calibration.sh` produces telemetry artifacts in `reports/m3-write-gate/`.  
8. Documentation updated with gate usage and tuning guidance.

## 7) QA Gates & CI Commands
- `black .`  
- `ruff .`  
- `mypy .`  
- `pytest -q` (≥85% diff coverage)  
- `pytest tests/test_gate.py tests/test_eviction.py tests/test_trace_writer.py`  
- Optional slow run: `pytest -m slow`  
- `scripts/run_m3_gate_calibration.sh --scenario A --n 48 --seed 13`

## 8) Risks & Mitigations
1. **Ambiguous salience signals:** feature definitions mis-specified → Mitigate with deterministic unit tests & scenario fixtures verifying expected ranking.  
2. **Telemetry noise on small splits:** gate metrics may be unstable → Use deterministic seeds, smoothing bins, and document interpretation.  
3. **Privacy regressions:** accidental raw text storage → enforce pointer-only schema, add tests asserting absence of plaintext, include manual audit in DoD.

