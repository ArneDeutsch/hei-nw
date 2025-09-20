# M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional) — Milestone Review

### 1) Executive summary
The milestone set out to deliver a working B1 retrieval stack: DG keyer → ANN candidates → modern Hopfield completion, with full harness wiring, retrieval-health metrics, and the engineering acceptance gates (parity guard, oracle probe, retrieval-only, Hopfield lift, decoding sanity).

The repo now contains the planned modules (`keyer`, `store`, `recall`, retrieval metrics), expanded harness/report wiring, CI scripts, and probe automation. Unit tests exist for the keyer, ANN wrapper, Hopfield readout, store, recall, and scripts. Reports under `reports/m2-*` show the flows were exercised with the tiny model.

However, critical requirements fail. The ANN wrapper is a flat index (not HNSW), B0 prompts still include the episode text so B0 achieves 100% EM (headroom never opens), the “retrieval-only” probe drifts far from P@1, Hopfield re-ranking hurts top-1 accuracy (negative completion lift), and the acceptance script exits early instead of running the mandated memory-dependent baseline when headroom is blocked. The decoding guard also enforces only ≥0.9 non-empty rate without the alphabetic-first-token check.

**Overall verdict: Fail.** Retrieval diagnostics do not satisfy the milestone’s engineering gates, the ANN/store implementation diverges from design, and the headroom-aware acceptance flow is incomplete.

### 2) Evidence snapshot: repo & planning anchors
- **Repo tree (short):** `src/hei_nw/{keyer.py,store.py,recall.py,eval/harness.py,metrics/retrieval.py}`; `tests/{test_keyer.py,test_store_ann.py,test_store_ep.py,test_hopfield_readout.py,test_recall.py,metrics/test_retrieval.py,utils/test_scripts.py}`; `scripts/run_m2_retrieval.sh,compare_b0_b1_m2.sh,m2_isolation_probes.sh,run_m2_acceptance.sh`; reports under `reports/m2-*`.
- **Planning anchors used:** `planning/project-plan.md:61-85` (M2 scope/DoD); `planning/design.md:240-247` (ANN = HNSW defaults); `planning/validation-plan.md:20-42` (headroom gate & Scenario A expectations); `planning/milestone-2-plan.md` tasks throughout.
- **Assumptions/limits:** Validation checked with the tiny GPT-2 fixture; no evidence of Qwen acceptance run; GPU-only requirements not executed.

### 3) DoD / Acceptance verification table
| Item | Evidence | Status | Notes |
| --- | --- | --- | --- |
| Parity guard (B1 empty ≈ B0) | `reports/m2-parity-guard/A_B0_report.md:5` & `A_B1_report.md:5` | Pass | Both EM=1.000 on the parity guard run. |
| Oracle probe EM ≥ 0.8 | `reports/m2-acceptance/probes/E1/A_B1_report.md:5` | Pass | Oracle run reaches EM=1.000. |
| Retrieval-only ≈ P@1 (±5 pts) | `reports/m2-probes/E2/A_B1_report.md:5-42` | Fail | EM=0.250 vs P@1=0.469 (gap 22 pts). |
| Hopfield completion lift ≥ 0 | `reports/m2-retrieval-stack/A_B1_report.md:37-42` | Fail | Completion lift = −0.292. |
| Decoding sanity (first token/100% non-empty) | `scripts/gate_non_empty_predictions.py:20-60` | Fail | Script only enforces ≥0.9 non-empty and bans a few prefixes; alphabetic first-token and 100% requirement unmet. |
| Headroom gate recorded | `reports/m2-acceptance/summary.md:1-13` | Pass | Summary logs “Headroom Gate: BLOCKED (EM_B0=1.000)”. |
| Statistical uplift or fallback | `scripts/run_m2_acceptance.sh:109-154` | Fail | When headroom blocked the script exits without running the memory-dependent baseline mandated in the plan. |
| Retrieval health logged | `reports/m2-retrieval-stack/A_B1_report.md:37-42` | Pass | JSON/Markdown include P@1, MRR, near-miss, collision, completion lift. |
| Hopfield read-only test | `tests/test_hopfield_readout.py:1-18` | Pass | Test asserts params unchanged and `requires_grad=False`. |
| End-to-end B1 recall wiring | `src/hei_nw/eval/harness.py:773-933` | Pass | Harness builds `RecallService`, feeds packed mem tokens into adapter. |
| Hopfield ablation artifact (lift ≥ 0) | `reports/m2-retrieval-stack/completion_ablation.png` & `A_B1_report.md:37-42` | Fail | Plot exists but lift stays ≤0; requirement explicitly calls for ≥0. |
| Zero stubs | `scripts/grep_no_stubs.sh:1-11` | Pass | Stub sweep script present; repo grep clean. |
| Docs & API exports | `src/hei_nw/__init__.py:1-9` | Pass | Public classes exported with docstrings in modules. |
| CI gates configured | `.github/workflows/ci.yml:1-40` | Pass | Workflow runs black, ruff, mypy, pytest, diff-cover, parity guard, retrieval smoke, no-stubs. |

### 4) Task-by-task review (plan order)

#### M2-T1 [CODEX] DG Keyer (k-WTA)
- **Intent:** Implement sparse DG keyer with L1-normalised k-WTA output.
- **Findings:** `src/hei_nw/keyer.py:21-94` defines `DGKeyer` + `to_dense`; tests in `tests/test_keyer.py:1-44` cover sparsity, scaling, round-trip.
- **Gaps / Risks:** None noted.
- **Status:** **Pass**

#### M2-T2 [CODEX] ANN Index wrapper (HNSW/IP)
- **Intent:** Provide FAISS HNSW wrapper with configurable `M`/`efSearch`.
- **Findings:** `src/hei_nw/store.py:19-68` creates `faiss.IndexFlatIP`, ignores HNSW parameters; tests only cover basic nearest neighbour.
- **Gaps / Risks:** Spec calls for `IndexHNSWFlat` with tunable breadth (`design.md:240-247`); current Flat IP index misses the design’s recall/latency characteristics.
- **Status:** **Fail**

#### M2-T3 [CODEX] Modern Hopfield Readout (inference-only)
- **Intent:** Add read-only Hopfield module with fixed updates.
- **Findings:** `src/hei_nw/store.py:105-170` implements `HopfieldReadout`; tests `tests/test_hopfield_readout.py:1-18` confirm no parameter mutation.
- **Gaps / Risks:** None at unit level.
- **Status:** **Pass**

#### M2-T4 [CODEX] Associative Store (ANN + Hopfield + metrics hooks)
- **Intent:** Compose keyer, ANN, Hopfield, diagnostics into `EpisodicStore`.
- **Findings:** `src/hei_nw/store.py:225-405` builds store; tests `tests/test_store_ep.py:1-44` exercise top-1 retrieval and diagnostics.
- **Gaps / Risks:** Store synthesises “hidden states” via hashed bag-of-words (`_hash_embed`, lines 225-238) instead of actual model activations, leaving a conceptual gap to the design. Retrieval quality issues cascade downstream.
- **Status:** **Partial**

#### M2-T5 [CODEX] Retrieval metrics (P@k/MRR/etc.)
- **Intent:** Implement retrieval-health metrics functions and tests.
- **Findings:** `src/hei_nw/metrics/retrieval.py:1-99`; tests `tests/metrics/test_retrieval.py:1-43`.
- **Gaps / Risks:** Functions behave as specified; misuse occurs later.
- **Status:** **Pass**

#### M2-T6 [CODEX] Recall API returning memory tokens
- **Intent:** Wrap store + packer into `RecallService`.
- **Findings:** `src/hei_nw/recall.py:1-68`; test `tests/test_recall.py:1-18` checks token cap.
- **Gaps / Risks:** None noted.
- **Status:** **Pass**

#### M2-T7 [CODEX] Harness integration (B1 uses retrieval)
- **Intent:** Integrate RecallService, log retrieval metrics, support dev probes.
- **Findings:** `src/hei_nw/eval/harness.py:288-933` builds recall path, logs metrics; tests `tests/test_harness_b1.py:1-35` ensure reports exist.
- **Gaps / Risks:** Prompts still inject the full episode for B0 (`_build_prompt`, lines 297-347), so headroom never opens; retrieval metrics mix raw ANN rankings with Hopfield predictions, yielding negative lift and large EM vs P@1 gaps (`reports/m2-probes/E2/A_B1_report.md:5-42`). Engineering gates fail.
- **Status:** **Fail**

#### M2-T8 [CODEX] Hopfield ablation & plot
- **Intent:** Provide `--no-hopfield`, emit ablation plot.
- **Findings:** Plot writing in `src/hei_nw/eval/report.py:96-142`; CLI flag handled in harness; probes script `scripts/m2_isolation_probes.sh:62-129`; plot file present.
- **Gaps / Risks:** Hopfield-on vs off shows zero/negative lift, failing the intended diagnostic.
- **Status:** **Fail**

#### M2-T9 [CODEX] CLI script & wiring
- **Intent:** Scripts for acceptance flow, headroom gate, probes.
- **Findings:** `scripts/run_m2_retrieval.sh:1-92`, `scripts/compare_b0_b1_m2.sh:1-140`, `scripts/m2_isolation_probes.sh:1-160`; test `tests/utils/test_scripts.py:24-58` checks presence/flags.
- **Gaps / Risks:** Acceptance script exits when headroom blocked without running the required memory-dependent baseline (`scripts/run_m2_acceptance.sh:109-154`). Headroom fallback flow is incomplete.
- **Status:** **Partial**

#### M2-T10 [CODEX] Public API & docs polish
- **Intent:** Export key classes and ensure docstrings.
- **Findings:** `src/hei_nw/__init__.py:1-9`; modules carry docstrings.
- **Gaps / Risks:** None.
- **Status:** **Pass**

#### M2-T11 [CODEX] No-stubs sweep
- **Intent:** Enforce stub-free repo.
- **Findings:** `scripts/grep_no_stubs.sh:1-11`; CI mirrors it.
- **Gaps / Risks:** None.
- **Status:** **Pass**

#### HUMAN/ChatGPT Task 1 — Code review pass
- **Intent:** Human review of key files.
- **Findings:** No evidence of review results or sign-off in repo.
- **Gaps / Risks:** Task outcome undocumented.
- **Status:** **Fail**

#### HUMAN/ChatGPT Task 2 — Sanity run (tiny model)
- **Intent:** Run B0/B1 on tiny model and check retrieval metrics.
- **Findings:** `reports/m2-retrieval-stack/A_B1_report.md:1-48` present with retrieval block.
- **Gaps / Risks:** Metrics exist but show regression.
- **Status:** **Pass**

#### HUMAN/ChatGPT Task 3 — Ablation check
- **Intent:** Hopfield on/off ablation with positive lift.
- **Findings:** `reports/m2-retrieval-stack/A_B1_report.md:37-42` shows negative lift; plot exists but no improvement.
- **Gaps / Risks:** Diagnostic fails.
- **Status:** **Fail**

#### HUMAN/ChatGPT Task 4 — Acceptance delta (if headroom passes)
- **Intent:** Run headroom-aware uplift or fallback baseline.
- **Findings:** Headroom blocked; the mandated memory-dependent baseline was not executed (`scripts/run_m2_acceptance.sh:109-154`).
- **Gaps / Risks:** Acceptance evidence missing.
- **Status:** **Fail**

### 5) Design & validation alignment
- **Design mapping:** DGKeyer/RecallService exist but the ANN index is not the HNSW specified in `planning/design.md:240-247`, and the store feeds on hashed text (`src/hei_nw/store.py:225-238`) rather than real LLM hidden states, undermining the intended pattern separation pipeline. Hopfield re-ranking currently degrades accuracy (negative lift in `reports/m2-retrieval-stack/A_B1_report.md:37-42`), violating the expectation in `design.md:200-207`.
- **Validation mapping:** Scenario A runs keep the episode text in B0 prompts (`src/hei_nw/eval/harness.py:297-347`), so B0 already answers perfectly, contradicting the `validation-plan.md:30-42` requirement that B1≫B0 on partial cues and the headroom gate logic at `validation-plan.md:20-24`. Retrieval metrics are logged, but the diagnostic goals (positive lift, P@1 agreement) are unmet.

### 6) Quality & CI assessment
- Tooling is configured (`pyproject.toml:1-38` for black, ruff, mypy, pytest; diff-cover enforced in `.github/workflows/ci.yml:1-40`).
- CI additionally runs parity guard, retrieval smoke, and the stub sweep.
- Tests cover keyer/store/recall/scripts but lack assertions for retrieval-only vs P@1, Hopfield lift, or headroom fallback. No automated check ensures B0 prompts omit episodes or that the acceptance script runs the memory-dependent baseline.

### 7) Gaps and Follow-up tasks

```
### M2-F1 [CODEX] Stabilise retrieval diagnostics and Hopfield lift

* **Goal:** Make retrieval-only EM track P@1 and ensure Hopfield re-ranking no longer suppresses top-1 accuracy.
* **Key changes:**
  1) Refactor `src/hei_nw/eval/harness.py` to reuse a single ANN query, compute metrics against the same ranking used for predictions, and log baseline vs Hopfield P@1 explicitly.
  2) Adjust Hopfield scoring (e.g., temperature, candidate set, normalisation) or gating so that completion lift is non-negative on Scenario A probes.
* **Tests:**
  - Extend `tests/test_harness_b1.py` with a synthetic regression that asserts retrieval-only EM equals P@1 within 0.05.
  - Add a probe-mode test (e.g., fixture under `tests/eval/`) verifying Hopfield-on top-1 ≥ Hopfield-off on the supplied fixture.
* **Quality gates:** `black .` · `ruff check .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  PYTHONPATH=src scripts/m2_isolation_probes.sh --probe ALL
  jq '.retrieval.completion_lift' reports/m2-probes/E3/A_B1_metrics.json
  ```

### M2-F2 [CODEX] Implement HNSW ANN index with tunable search

* **Goal:** Align `ANNIndex` with the design’s HNSW configuration and expose search knobs.
* **Key changes:**
  1) Replace `faiss.IndexFlatIP` with `faiss.IndexHNSWFlat` in `src/hei_nw/store.py`, wiring `m`, `efConstruction`, and `efSearch`, plus a setter.
  2) Update metadata handling and add defensive checks for ef/k bounds.
* **Tests:**
  - Extend `tests/test_store_ann.py` to assert the underlying FAISS type is HNSW and that adjusting `ef_search` changes recall on a crafted dataset.
* **Quality gates:** `black .` · `ruff check .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  PYTHONPATH=src pytest tests/test_store_ann.py
  ```

### M2-F3 [CODEX] Enforce memory-dependent prompting and headroom fallback

* **Goal:** Ensure B0 runs without episodic text by default and always generate the memory-dependent baseline when headroom blocks uplift.
* **Key changes:**
  1) Modify `_build_prompt` / mode dispatch in `src/hei_nw/eval/harness.py` so B0 omits episodes unless an explicit override is given; keep B1 prompts identical to facilitate fair comparison.
  2) Introduce a `--qa.memory_dependent_baseline` toggle (per plan) and update `scripts/run_m2_acceptance.sh` to call it automatically when headroom is blocked, producing baseline metrics in `summary.md`.
* **Tests:**
  - Add a unit test for prompt construction (e.g., `tests/test_harness_prompts.py`) asserting B0 drops the episode while B1 retains it.
  - Extend `tests/utils/test_scripts.py` to cover the new acceptance-summary behaviour.
* **Quality gates:** `black .` · `ruff check .` · `mypy .` · `pytest -q`
* **Acceptance check:**
  ```bash
  bash scripts/run_m2_acceptance.sh
  grep "Memory-dependent baseline" reports/m2-acceptance/summary.md
  ```

### M2-F4 [CODEX] Tighten decoding sanity guard

* **Goal:** Match the DoD by requiring 100% non-empty predictions and alphabetic leading tokens.
* **Key changes:**
  1) Update `scripts/gate_non_empty_predictions.py` to default to threshold=1.0 and enforce `str.isalpha()` (after stripping) on the first token, with clear error reporting.
  2) Add unit tests in `tests/utils/test_scripts.py` covering pass/fail cases for both checks.
* **Quality gates:** `black .` · `ruff check .` · `pytest -q`
* **Acceptance check:**
  ```bash
  python scripts/gate_non_empty_predictions.py reports/m2-retrieval-stack/A_B1_metrics.json
  ```

### M2-F4H [HUMAN/ChatGPT] Re-run M2 acceptance with headroom

* **Goal:** Produce acceptance evidence (engineering gates + headroom-aware uplift or documented fallback) on the target Qwen model once code fixes land.
* **Steps:** Run `scripts/run_m2_acceptance.sh`, capture `reports/m2-acceptance/summary.md`, hopfield ablation plot, and uplift logs; rerun `scripts/run_m2_retrieval.sh --hard-subset <file>` if headroom passes.
* **Acceptance check:** Attach updated reports (JSON + Markdown + plots) under `reports/m2-acceptance/` showing positive completion lift and either uplift CI or memory-dependent baseline metrics.
```

### 8) Final verdict
**Fail** — Retrieval diagnostics regress (Hopfield reduces accuracy, retrieval-only ≠ P@1), the ANN store misses the HNSW design, and the headroom-aware acceptance flow halts without the required fallback baseline.

**Required follow-ups:** M2-F1, M2-F2, M2-F3, M2-F4, M2-F4H.
