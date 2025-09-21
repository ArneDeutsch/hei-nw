# M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional) — Milestone Review

### 1) Executive summary
Milestone scope was to stand up the B1 retrieval stack—DG keyer, ANN candidate search, modern Hopfield completion, recall API, harness metrics, and headroom-aware acceptance gates—as defined in planning/project-plan.md:61 and expanded in planning/milestone-2-plan.md:31.
The repository now ships those components with documentation and regression tests: `DGKeyer`, `ANNIndex`, `HopfieldReadout`, `EpisodicStore`, and `RecallService` (`src/hei_nw/keyer.py:21`, `src/hei_nw/store.py:19`, `src/hei_nw/recall.py:15`) plus harness/report updates and automation scripts (`src/hei_nw/eval/harness.py:805`, `src/hei_nw/eval/report.py:65`, `scripts/run_m2_retrieval.sh:1`). Acceptance assets under `reports/m2-retrieval-stack/` and `reports/m2-acceptance/` demonstrate the required gates.
All DoD items pass: parity and oracle probes, retrieval-only vs P@1, headroom tracking, bootstrap uplift, Hopfield ablation, and stub sweeps (e.g., `reports/m2-parity-guard/A_B1_metrics.json`, `reports/m2-retrieval-stack/uplift_compare.txt`, `scripts/grep_no_stubs.sh:1`). Hopfield currently delivers a neutral lift (completion_lift = 0.0) per `reports/m2-acceptance/probes_summary.txt`, which meets the ≥0 requirement but is worth monitoring.

### 2) Evidence snapshot: repo & planning anchors
- **Repo tree (short)**: src/hei_nw/, tests/, scripts/, reports/m2-retrieval-stack/, reports/m2-acceptance/
- **Planning anchors used**: planning/project-plan.md:61; planning/milestone-2-plan.md:31; planning/design.md:75-89; planning/validation-plan.md:65-102
- **Assumptions/limits**: Relied on committed reports for Qwen acceptance runs; no new GPU reruns were executed during this review.

### 3) DoD / Acceptance verification table
| Item | Evidence (files/funcs/CLI) | Status | Notes |
| --- | --- | --- | --- |
| Engineering acceptance (parity, oracle, retrieval-only, Hopfield, decode) | reports/m2-parity-guard/A_B0_metrics.json; reports/m2-parity-guard/A_B1_metrics.json; reports/m2-retrieval-stack/probes/E1/A_B1_metrics.json; reports/m2-retrieval-stack/probes/E2/A_B1_metrics.json; reports/m2-retrieval-stack/A_B1_metrics.json; scripts/gate_non_empty_predictions.py:14 | Pass | B1 empty matches B0 (EM 0); oracle EM=1.0; retrieval-only EM equals P@1; Hopfield lift = 0.0; gating enforces non-empty alphabetic first tokens. |
| Headroom gate recorded | reports/m2-acceptance/summary.md; scripts/run_m2_retrieval.sh:45 | Pass | Summary logs “Headroom Gate: PASS (EM_B0 = 0.000)” with script echoing status. |
| Statistical uplift (hard subset, headroom pass) | reports/m2-retrieval-stack/uplift_compare.txt | Pass | Hard-subset lift = 1.000 with 95% CI [1.000, 1.000]; threshold exceeded. |
| Retrieval health logged | reports/m2-retrieval-stack/A_B1_metrics.json; src/hei_nw/eval/report.py:165 | Pass | JSON records p_at_1/mrr/near_miss/collision/completion_lift; Markdown report prints the retrieval block. |
| Hopfield inference read-only | tests/test_hopfield_readout.py:6 | Pass | Test asserts params stay frozen and unchanged after forward. |
| End-to-end B1 uses RecallService for memory tokens | src/hei_nw/eval/harness.py:805; src/hei_nw/recall.py:15 | Pass | Harness builds `RecallService` once per run and reuses tokenizer/max token settings while querying the store for packed traces. |
| Hopfield ablation artifacts | reports/m2-retrieval-stack/completion_ablation.png; tests/test_harness_ablation.py:11 | Pass | CLI writes with/without-Hopfield metrics and exports the plot (≥0 lift). |
| Zero stubs guard | scripts/grep_no_stubs.sh:1; tests/utils/test_scripts.py:13 | Pass | Guard script fails on stub markers; unit test validates regex. |
| Docs & API polish | src/hei_nw/__init__.py:1; src/hei_nw/keyer.py:21 | Pass | Public classes exported via `__all__` with docstrings on key components. |
| CI green path | .github/workflows/ci.yml:11; pyproject.toml:14 | Pass | CI runs black/ruff/mypy/pytest/diff-cover plus parity guard and M2 smoke; tooling config checked in. |

### 4) Task-by-task review (mirror plan order)
### M2-T1 [CODEX] DG Keyer (k-WTA)
- **Intent (from plan):** Create a sparse, L1-normalised DG keyer with tests.
- **Findings (evidence):** Implementation and helper in src/hei_nw/keyer.py:21; invariants covered by tests/test_keyer.py:8.
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T2 [CODEX] ANN Index wrapper (HNSW/IP)
- **Intent:** Provide FAISS HNSW wrapper with metadata alignment tests.
- **Findings:** ANNIndex lives in src/hei_nw/store.py:19; behaviour verified in tests/test_store_ann.py:8.
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T3 [CODEX] Modern Hopfield Readout (inference-only)
- **Intent:** Add read-only Hopfield module with refinement tests.
- **Findings:** src/hei_nw/store.py:155 defines HopfieldReadout; tests/test_hopfield_readout.py:6 validates immutability and refinement.
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T4 [CODEX] Associative Store (ANN + Hopfield + metrics hooks)
- **Intent:** Compose keyer/index/Hopfield into EpisodicStore with diagnostics.
- **Findings:** src/hei_nw/store.py:224 builds the store and query path; tests/test_store_ep.py:12 exercises top-1, near-miss, collision counters.
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T5 [CODEX] Retrieval metrics (P@k/MRR/near-miss/collision/completion lift)
- **Intent:** Implement and test retrieval-health metrics.
- **Findings:** Metrics implemented in src/hei_nw/metrics/retrieval.py:9; behaviour checks in tests/test_metrics_retrieval.py:16.
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T6 [CODEX] Recall API returning memory tokens
- **Intent:** Provide RecallService.build/recall wrappers over EpisodicStore.
- **Findings:** src/hei_nw/recall.py:15 implements build/recall and truncation; tests/test_recall.py:11 ensures token caps.
- **Gaps / Risks:** Harness repacks tokens directly from `service.store.query`, duplicating `RecallService.recall`; future drift is possible if the packing template changes.
- **Status:** Pass

### M2-T7 [CODEX] Harness integration (B1 uses retrieval)
- **Intent:** Feed adapter with retrieved tokens and log retrieval metrics/reports.
- **Findings:** src/hei_nw/eval/harness.py:805 builds the service, aggregates per-record diagnostics, and writes retrieval stats; Markdown reporting extended in src/hei_nw/eval/report.py:65. Tests test_harness_b1.py:52 and tests/eval/test_report_details.py:13 cover the new fields.
- **Gaps / Risks:** None beyond the packing duplication noted above.
- **Status:** Pass

### M2-T8 [CODEX] Hopfield ablation & plot
- **Intent:** Support `--no-hopfield` and produce completion-lift plot.
- **Findings:** CLI flag handled in src/hei_nw/eval/harness.py:1000 and orchestrated by scripts/m2_isolation_probes.sh:63; plotting helper in src/hei_nw/eval/report.py:221; tests/test_harness_ablation.py:11 verifies the workflow.
- **Gaps / Risks:** Current runs show completion lift = 0.0 and slightly better MRR without Hopfield (reports/m2-acceptance/probes_summary.txt); worth monitoring as tuning continues.
- **Status:** Pass

### M2-T9 [CODEX] CLI script & wiring
- **Intent:** Automate acceptance flow (headroom, probes, compare).
- **Findings:** scripts/run_m2_retrieval.sh:1, scripts/compare_b0_b1_m2.sh:1, and scripts/m2_isolation_probes.sh:1 implement the flow; presence/executable bits enforced by tests/utils/test_scripts.py:24.
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T10 [CODEX] Public API & docs polish
- **Intent:** Expose retrieval components via package init with doc coverage.
- **Findings:** src/hei_nw/__init__.py:1 exports DGKeyer/EpisodicStore/RecallService; module docstrings present (e.g., src/hei_nw/keyer.py:1).
- **Gaps / Risks:** None.
- **Status:** Pass

### M2-T11 [CODEX] No-stubs sweep
- **Intent:** Enforce stub-free repo via script/test.
- **Findings:** scripts/grep_no_stubs.sh:1 and tests/utils/test_scripts.py:13 guard against TODO/FIXME/NotImplementedError.
- **Gaps / Risks:** None.
- **Status:** Pass

### 5) Design & validation alignment
- DG keyer, ANN index, and Hopfield completion mirror the architecture described in planning/design.md:75-89 and are implemented in src/hei_nw/keyer.py:21 and src/hei_nw/store.py:19/155/224.
- Scenario A metrics (P@k, MRR, near-miss, collision, completion lift) match planning/validation-plan.md:65-80 via src/hei_nw/metrics/retrieval.py:9 and harness logging at src/hei_nw/eval/harness.py:1000.
- Headroom-aware acceptance flow and probes conform to planning/milestone-2-plan.md:131 and planning/validation-plan.md:97-104, with automation in scripts/run_m2_retrieval.sh:45 and scripts/m2_isolation_probes.sh:63.

### 6) Quality & CI assessment
- Tooling is standardised through pyproject.toml:14 (black, isort, ruff, mypy, pytest, coverage settings).
- CI pipeline (.github/workflows/ci.yml:11) runs format/lint/type/tests, enforces diff coverage, parity guard, M2 smoke, QA prompting smoke, and stub sweeps.
- Test suite spans unit and integration coverage for key modules (e.g., tests/test_keyer.py:8, tests/test_store_ep.py:12, tests/test_harness_b1.py:52, tests/eval/test_report_details.py:13), plus script-level smoke checks.

### 7) Gaps and Follow-up tasks
- None.

### 8) Final verdict
- Pass – Milestone deliverables and acceptance evidence are in place; monitor Hopfield tuning (currently neutral lift) in upcoming milestones but no blocking gaps for M2.
