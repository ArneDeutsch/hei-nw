# M2b — Schema-Agnostic Retrieval Hardening (B1 cross-scenario)

## 1) Milestone Summary

* Replace the stop-gap hashed episodic keys with a **schema-agnostic feature pipeline** that works across scenarios A, B, and C without scenario-specific hacks.
* Upgrade the DG/HNSW/Hopfield stack so that retrieval health meets design targets: **Scenario A P@1 ≥ 0.6 (mini set)** and **Scenarios B/C P@1 ≥ 0.3 on n ≥ 200**.
* Strengthen the acceptance harness so under-powered runs fail fast, telemetry calls out collisions, and large-sample checks become routine for Milestone 2 sign-off.

---

## 2) Dependencies / Inputs

* Planning/design specs: `planning/design.md` (§5.3 DG keyer, §5.4 associative store, §8 defaults) and `planning/validation-plan.md` (retrieval metrics, scenario expectations).
* Updated constraints from `planning/project-plan.md` (Milestone 2 acceptance notes) and `documentation/hei-nw.md` (2025-09 collision warning).
* Root-cause backlog from `reviews/pre-milestone-4-review.md` (RC-M2 issues, RC-M2-T4+ outstanding tasks).
* Existing code: `src/hei_nw/keyer.py`, `src/hei_nw/store.py`, `src/hei_nw/recall.py`, `src/hei_nw/datasets/{scenario_a,scenario_b,scenario_c}.py`, `src/hei_nw/eval/harness.py`, `scripts/run_m2_acceptance.sh`.

---

## 3) [CODEX] Implementation Tasks

### M2b-T1 — Schema-agnostic episode feature extractor
*Goal.* Emit canonical, schema-aware feature bundles (textual spans + structured slots) for every scenario without bespoke hacks.

*Key changes.*
1. Add `src/hei_nw/features/episode_features.py` with `EpisodeFeatureExtractor` producing `FeatureBundle(text_tokens, slot_tokens, numeric_channels, provenance)` from raw scenario records.
2. Update scenario generators (`scenario_a/b/c`) to output standardized metadata (e.g., `entity_slots`, `config_dims`, timestamps) consumed by the extractor.
3. Wire extractor into the recall/write path (packer + store build) so downstream modules only see feature bundles.

*Tests.*
* `tests/features/test_episode_features.py::test_bundle_shapes_and_types`
* `tests/features/test_episode_features.py::test_scenarios_share_schema`

*Acceptance check.*
```bash
PYTHONPATH=src python -m hei_nw.features.dump_features --scenario B -n 5 --out /tmp/m2b_features.jsonl
# file shows text+slot channels populated, no scenario-specific logic warnings
```

---

### M2b-T2 — Multi-channel DG keyer & sparsity controls
*Goal.* Generate low-collision sparse keys from feature bundles, with tunable sparsity and reproducibility.

*Key changes.*
1. Extend `DGKeyer` to accept `FeatureBundle`s (multi-channel projections, learned or PCA-initialized) and support per-channel k-WTA before concatenation.
2. Add configurable projection stack (`Linear` + LayerNorm) initialized from design defaults; expose `--dg.k_total`, `--dg.channel_weights` flags in harness/scripts.
3. Provide deterministic hashing fallback only for ablations; default path uses learned projections stored in `resources/dg_keyer/` checkpoints.

*Tests.*
* `tests/retrieval/test_dg_keyer.py::test_multi_channel_k_sparsity`
* `tests/retrieval/test_dg_keyer.py::test_collision_rate_drops_vs_hashed` (synthetic overlap < 5%)

*Acceptance check.*
```bash
PYTHONPATH=src python -m hei_nw.eval.key_diagnostics --scenario C --n 256 --stats collision_rate_by_schema
# reports collision_rate_by_schema.{B,C} <= 0.15 on synthetic mini-set
```

---

### M2b-T3 — ANN/Hopfield refresh + collision telemetry
*Goal.* Align the associative store with new keys, add collision/near-miss telemetry, and retune retrieval knobs.

*Key changes.*
1. Rebuild `EpisodicStore` to index channel-aware dense views (L2-normalized) and log per-scenario `collision_rate`, `near_miss_rate`, `hopfield_lift`.
2. Update HNSW defaults (`M`, `ef_construction`, `ef_search`) and expose per-scenario overrides; ensure Hopfield completion operates on channel-weighted similarities.
3. Extend `src/hei_nw/metrics/retrieval.py` and reports to surface the new telemetry.

*Tests.*
* `tests/retrieval/test_store_multi_channel.py::test_ann_round_trip`
* `tests/retrieval/test_store_multi_channel.py::test_metrics_include_collision_rates`

*Acceptance check.*
```bash
scripts/run_m2_retrieval_ci.sh --scenario B --ann.ef_search 128 --dg.k_total 96
# Retrieval health section lists collision_rate <= 0.2 and hopfield_lift > 0
```

---

### M2b-T4 — Scenario B/C harness integration & large-sample slices
*Goal.* Ensure scenarios B and C use the shared retrieval path and have acceptance-grade evaluation slices (n ≥ 200).

*Key changes.*
1. Extend `hei_nw.eval.harness` to accept multiple scenarios per run (A/B/C) and to materialize large-sample splits via `--hard-subset` manifests or generator parameters.
2. Update `scripts/run_m2_acceptance.sh` (and helper probes) to iterate over `{A,B,C}` when `M2B_SCENARIOS` is unset, recording per-scenario metrics and aggregated summary tables.
3. Produce reusable subsets (`data/m2b/hard_subset_{B,C}.jsonl`) with ≥ 200 items each, pulled via generator seeds and stored under `reports/m2b-eval/manifests/`.

*Tests.*
* `tests/eval/test_harness_multi_scenario.py::test_modes_share_feature_pipeline`
* `tests/eval/test_harness_multi_scenario.py::test_large_sample_manifest_count`

*Acceptance check.*
```bash
PYTHONPATH=src scripts/run_m2_acceptance.sh --hard-subset reports/m2b-eval/manifests/scenario_c_200.jsonl
# Summary includes Scenario C P@1 ≥ 0.30 (n=200) and notes manifest provenance
```

---

### M2b-T5 — Powered acceptance runner & CLI/doc sync (RC-M2-T4 + shared)
*Goal.* Block under-powered acceptance runs and make CLI/docs reflect the new multi-scenario flow.

*Key changes.*
1. Modify `scripts/run_m2_acceptance.sh` to enforce `N ≥ 200` unless `ALLOW_SMALL_SAMPLE=1` or `--dev-mode` is set; annotate `summary.md` with `UNDERPOWERED` when overrides are used.
2. Add `--out` flag parity (or update docs accordingly) and surface per-scenario sample sizes + P@1 in the summary table.
3. Update `documentation/quick-validate.md` and `prompts/HEI-NW_milestone_task_prompt_template.md` references so instructions match the runner behavior.

*Tests.*
* `tests/utils/test_scripts.py::test_m2_acceptance_blocks_small_samples`
* `tests/utils/test_scripts.py::test_m2_acceptance_help_matches_docs`

*Acceptance check.*
```bash
N=48 scripts/run_m2_acceptance.sh && echo "should fail" || echo "blocked as expected"
ALLOW_SMALL_SAMPLE=1 N=48 scripts/run_m2_acceptance.sh  # succeeds with UNDERPOWERED notice
```

---

### M2b-T6 — Reporting & documentation refresh
*Goal.* Reflect the new retrieval pipeline in docs and reports, closing the 2025-09 warning once metrics improve.

*Key changes.*
1. Update `documentation/hei-nw.md` and `documentation/quick-validate.md` to describe the feature pipeline, collision telemetry, and multi-scenario acceptance thresholds.
2. Extend `reports/m2-acceptance/summary.md` generation to embed per-scenario tables (P@k, MRR, collision rate, hopfield lift) and link to manifests.
3. Add `tests/docs/test_docs_links.py::test_m2b_docs_refer_to_feature_pipeline` ensuring docs mention schema-agnostic keys and P@1 goals.

*Acceptance check.*
```bash
rg "schema-agnostic" documentation/hei-nw.md && rg "P@1 ≥ 0.3" documentation/quick-validate.md
```

---

## 4) [HUMAN/ChatGPT] Review & GPU Tasks

1. **M2b-R1 — Feature pipeline audit.** Review feature bundles for leakage/pointer-only compliance; spot-check Scenario C reward/pin channels.
2. **M2b-R2 — Large-sample acceptance runs.** Execute `scripts/run_m2_acceptance.sh --hard-subset ...` on GPU with Qwen/Qwen2.5-1.5B-Instruct; confirm Scenario B/C P@1 ≥ 0.3, Scenario A P@1 ≥ 0.6.
3. **M2b-R3 — Collision telemetry sanity.** Inspect `reports/m2b-eval/*/retrieval_metrics.json` for collision_rate by scenario ≤ 0.2 and hopfield lift > 0.

---

## 5) Deliverables & Artifacts

* Code: `src/hei_nw/features/episode_features.py`, updated keyer/store/recall modules, enhanced metrics and harness logic.
* Tests: new feature, retrieval, harness, and script tests listed above.
* Scripts: refreshed `scripts/run_m2_acceptance.sh`, optional `hei_nw.features.dump_features` utility, documentation updates.
* Reports: `reports/m2b-eval/` with per-scenario metrics, manifests, collision telemetry, updated acceptance summary.

---

## 6) Definition of Done (DoD) Checklist

* ✅ Feature bundles emitted for scenarios A/B/C with shared schema; extractor unit tests green.
* ✅ Collision telemetry shows Scenario B/C collision_rate ≤ 0.2 on acceptance slices; hopfield lift ≥ 0.
* ✅ Scenario A mini-set (n≈48) P@1 ≥ 0.60; Scenarios B/C large-set (n≥200) P@1 ≥ 0.30 using Qwen/Qwen2.5-1.5B-Instruct.
* ✅ Acceptance runner blocks `N < 200` unless override flag used; summary annotates overrides.
* ✅ Reports and docs updated to reflect the schema-agnostic pipeline; 2025-09 warning replaced with current status + next steps.
* ✅ `black .`, `ruff .`, `mypy .`, `pytest -q` pass; diff coverage ≥ 85%; `scripts/grep_no_stubs.sh` clean.

---

## 7) QA Gates & CI Commands

* `black .`
* `ruff .`
* `mypy .`
* `pytest -q`
* `pytest tests/features/test_episode_features.py tests/retrieval/test_dg_keyer.py tests/retrieval/test_store_multi_channel.py`
* `pytest tests/eval/test_harness_multi_scenario.py tests/utils/test_scripts.py`
* `PYTHONPATH=src scripts/run_m2_retrieval_ci.sh --scenario B --n 32` (CI smoke, dummy model)

---

## 8) Risks & Mitigations

1. **Feature leakage or schema drift.** Mitigate with shared extractor tests and pointer-only checks before writes.
2. **Keyer training instability.** Store projection weights under versioned checkpoints; provide deterministic fallback + CI regression tests.
3. **Large-sample runtime.** Slice manifests to 200–256 items and document GPU wall-clock; allow CI to skip via dummy model while human runs full acceptance.
4. **Collision telemetry regressions.** Add alarm thresholds in reports and fail acceptance runner when collision_rate > 0.25 unless `ALLOW_COLLISIONS=1` (optional future guard).

