# T0 — Gate & Retrieval Preconditions: RESULT

## Decision

**Block M4 (T1+), open “M3/M2 Root-Cause.”**
M2 statistical acceptance is **not met**, and M3 gate calibration shows **threshold-insensitive behavior** (likely a wiring/labeling bug). We shouldn’t proceed to scheduler/replay until these are fixed.

---

## Evidence (with exact artifacts)

### M2 — Retrieval stack (accept/reject)

* **Headroom Gate (small set):** PASS — EM\_{B0}=0.000 < 0.7. (Console + `reports/m2-acceptance/uplift_compare.txt`)
* **Statistical uplift (hard criterion):** **FAIL** — ΔEM = **+0.104** with 95% CI **\[0.042, 0.188]**, but the plan requires **mean uplift ≥ +0.30** with CI excluding 0. (`reports/m2-acceptance/uplift_compare.txt`)
* **Oracle probe (E1):** **FAIL** — EM = **0.000** (expected ≥0.8). (`reports/m2-acceptance/probes/E1/A_B1_report.md`, summarized in `reports/m2-acceptance/probes_summary.txt`)
* **Hopfield ablation (E3):** No lift — EM **0.104** with and without Hopfield → memory tokens aren’t changing answers. (`reports/m2-acceptance/probes/E3/*`)
* **Retrieval health:** P\@1 **0.292**, MRR **0.493** — only moderate, likely too weak for a +0.30 EM lift. (`reports/m2-acceptance/uplift_compare.txt`)
* **Sample size note:** the acceptance track should use a **hard subset (n≥200–500)**. Your current run evaluated **48** items, which is below spec for final acceptance. (`reports/m2-acceptance/uplift_compare.txt`; see `planning/validation-plan.md` “Tracks” & “accept/reject” sections)

**Conclusion (M2):** **Rejected** — fails uplift threshold and oracle sanity; also underpowered (n too small).

---

### M3 — Gate calibration (accept/reject)

* **Threshold sweep τ ∈ {0.8, 1.0, 1.2, 1.4, 1.6}:** **No effect** on writes.

  * `writes_per_1k_tokens` ≈ **3.275** for **every** τ; `write_rate` fixed at **0.5**.
    See `reports/m3-accept/A_sweep_summary.json` and per-τ telemetry `reports/m3-accept/tau_*/A_gate_telemetry.json`.
* **Telemetry looks unrealistically “perfect”:** PR-AUC **1.0**, precision **1.0**, recall **1.0** **for all τ** (same files as above). That usually indicates labeling or scoring is degenerate, or τ is ignored in the decision path.
* **Calibration plots** were rendered, but since writes don’t move with τ, the gate is not actually **calibratable** as required.

**Conclusion (M3):** **Rejected (calibration)** — τ is not influencing writes; telemetry suggests a bug in label generation or thresholding.

> Minor plan mismatch: the script doesn’t support `--out` for M2; it writes to `reports/m2-acceptance/` by default. You already adapted, which is fine.

---

## What I’d record in the ledger (single line)

“**T0 (2025-09-24): Block.** M2 uplift +0.104 \[0.042, 0.188] < +0.30; E1 oracle EM=0.000; E3 no Hopfield lift; M3 gate sweep τ∈\[0.8..1.6] yields fixed write\_rate=0.5, PR-AUC=1.0 across τ → threshold/label bug suspected. Open ‘M3/M2 Root-Cause’, halt T1+.”

---

## Root-Cause work I recommend opening (tight, code-aligned)

### M3 — Gate threshold & telemetry

1. **Wire τ into the decision.** In `src/hei_nw/write_gate.py`, audit the **score S** and **threshold τ** usage. Ensure `should_write = (S ≥ τ) ∨ pin` (plus any K-WTA is **optional**, not overriding τ).
   *Add a unit/integration test:* τ sweep must make writes **monotone non-increasing**.
2. **Fix labels & PR curve.** Ensure `should_remember_label` comes from the generator’s event labels (as spec’d), not from the model’s decision. Verify that the PR curve **varies with τ**. Guard against all-positive/all-negative label sets.
3. **Expose write-rate target:** Keep target band 1–5 / 1k tokens; after fix, pick τ to hit the target and re-emit `A_sweep_summary.json` with non-flat writes.

### M2 — Retrieval & prompting

4. **Oracle probe wiring.** In the M2 acceptance workflow, verify the **E1 “oracle” variant actually injects gold memory tokens** (or gold facts) into the prompt. A 0.000 EM strongly suggests they are missing/misplaced. Check the harness that assembles E1 (see `scripts/run_m2_acceptance.sh` call chain and the probe runner it invokes).
5. **Memory-token integration.** Inspect the B1 **prompt template** (e.g., `src/hei_nw/prompting.py` / templates used by the QA runner). Confirm memory tokens are present and clearly demarcated, and that the instruction tells the model to **prefer** them for answering. E3’s “no change” implies tokens are ignored.
6. **Retrieval strength.** Increase DG Keyer sparsity and HNSW search breadth (efSearch), and/or a small Hopfield steps/temperature retune. Target **P\@1 ≥ 0.6** on the probe mini-set before repeating uplift.
7. **Sample size for acceptance.** Prepare/run the **hard subset (n≥200–500)** for the statistical track once items 4–6 are green; the current 48-item run is not sufficient for final acceptance.

---

## Re-run commands after fixes (minimal)

```bash
# M3: sweep to verify τ actually moves writes
scripts/run_m3_gate_calibration.sh --scenario A --n 512 --threshold-sweep "0.8 1.0 1.2 1.4 1.6" --out reports/m3-accept

# M2: full acceptance (small-set + hard subset if ready)
scripts/run_m2_acceptance.sh                 # writes to reports/m2-acceptance/
# (if you have a curated hard subset:)
scripts/run_m2_acceptance.sh --hard-subset path/to/hard_subset.jsonl
```

---

### Bottom line

* **M2:** fails uplift threshold and oracle sanity → **not acceptable**.
* **M3:** gate sweep flat across τ with perfect PR metrics → **not acceptable**.
  → **Open “M3/M2 Root-Cause” and keep M4 blocked** until the above items are fixed and re-validated.

Here’s a **root-cause remediation task pack** for Codex, using the same structure and tone as the existing milestone plans.

---

# \[CODEX] Root-Cause Remediation Tasks — M3/M2 Blockers

## RC-M3-T1 — Gate decision must be τ-sensitive (kill label fallback in calibration)

**Goal.** Ensure the write gate’s **threshold τ actually controls writes** during calibration; eliminate “label fallback” that forces writes when `gate_features` are missing.

**Key changes.**

1. `src/hei_nw/eval/harness.py`

   * Change `_apply_gate(... )` signature to include `allow_label_fallback: bool = True`.
   * Guard the fallback path:

     ```py
     if allow_label_fallback and (not has_gate_payload) and truth_label and not write:
         write, fallback = True, True
     ```
   * Plumb a new CLI flag:

     * `--gate.allow_label_fallback / --no-gate.allow_label_fallback` (default: **true**).
   * Thread the flag into `_apply_gate(...)` call sites.
2. `scripts/run_m3_gate_calibration.sh`

   * Pass `--no-gate.allow_label_fallback` for all runs.
   * Ensure `--gate.use_for_writes` remains **on** and `--gate.debug_keep_labels` **off** in calibration.

**Tests.**

* `tests/test_gate_calibration_integration.py::test_tau_moves_write_rate`

  * Run harness over a small synthetic set with/without fallback; assert `writes_per_1k_records(τ=low) > writes_per_1k_records(τ=high)`.

**Acceptance check (CLI).**

```bash
scripts/run_m3_gate_calibration.sh --scenario A --n 512 \
  --threshold-sweep "0.8 1.0 1.2 1.4 1.6"
# In reports/m3-accept/A_sweep_summary.json:
# assert write_rate and/or writes_per_1k_tokens vary monotonically with τ
```

---

## RC-M3-T2 — Always populate gate features in Scenario A

**Goal.** **No record** enters calibration without `gate_features` (`surprise`, `novelty`, `reward`, `pin`) populated.

**Key changes.**

1. `src/hei_nw/datasets/scenario_a.py`

   * Ensure every **positive and negative** record includes a non-empty `gate_features` dict:

     * `surprise`: use `surprise_from_prob` with model-agnostic heuristics (e.g., uniform prior → small surprise; or tie to question entropy proxy).
     * `novelty`: reuse `novelty_from_similarity(similarity)`; for hard negatives, set similarity high (low novelty).
     * `reward`: false by default; pins only for explicitly marked items.
     * `pin`: propagate any existing pin logic.
2. Add a lightweight helper: `src/hei_nw/gate.py` (or reuse existing) for generating sane default features.

**Tests.**

* `tests/test_scenario_a_gate_features.py::test_all_records_have_gate_features`
* `tests/test_scenario_a_gate_features.py::test_feature_ranges` (0–1, booleans, etc.)

**Acceptance check.**

* Run calibration (RC-M3-T1); verify **no** diagnostics have `has_gate_payload == false` in `A_gate_telemetry.json`.

---

## RC-M3-T3 — Gate telemetry sanity & monotonicity guards

**Goal.** Prevent degenerate PR-AUC=1.0 unless labels are actually all-positive; make monotonicity part of CI.

**Key changes.**

1. `src/hei_nw/telemetry/gate.py`

   * Add `label_distribution` to metrics: `{positives, negatives, positive_rate}`.
2. `src/hei_nw/eval/harness.py`

   * When producing the “subset” summaries, include `label_distribution`.
3. CI check: if `positives in {0,total}`, warn in report and tag run as **non-calibratable**.

**Tests.**

* `tests/test_gate_metrics.py::test_pr_auc_non_trivial_when_labels_mixed`
* `tests/test_gate_metrics.py::test_label_distribution_present`

**Acceptance check.**

* Re-run calibration; `A_sweep_summary.tsv` shows varying `write_rate` and **non-trivial** `pr_auc` with mixed labels.

---

## RC-M3-T4 — Calibration UX: target write-rate auto-τ and sweep summaries

**Goal.** One-shot τ selection to hit a target write-budget and clearer summaries.

**Key changes.**

1. `scripts/run_m3_gate_calibration.sh`

   * Add `--target-rate-per-1k {tokens|records} --target VALUE` to binary-search τ over a bounded interval.
   * Persist chosen τ in `A_auto_selected_tau.json` with metric used.
2. `src/hei_nw/eval/report.py`

   * Include `writes_per_1k_tokens`, `write_rate`, and `label_distribution` in the calibration plot footer.

**Tests.**

* `tests/utils/test_scripts.py::test_m3_auto_tau_selection_runs`

**Acceptance check.**

```bash
scripts/run_m3_gate_calibration.sh --scenario A --n 512 --target-rate-per-1k tokens --target 3
# Verify OUT/*/A_auto_selected_tau.json exists and τ differs from default when needed
```

---

## RC-M2-T1 — Oracle probe must uplift EM (wire end-to-end)

**Goal.** E1 “oracle trace” should **materially raise EM** (≥0.80 on Scenario A mini-set).

**Key changes.**

1. `src/hei_nw/eval/harness.py`

   * Double-check `dev.oracle_trace` path produces `selected=[{"answers":[...]}]` (already present) **and** that these pack into memory tokens (via `RecallService`).
   * Ensure QA build includes either:

     * adapter cross-attn with `mem_tokens`, **and/or**
     * `memory_prompt` textual fallback with the decoded slots (`who/what/where/when`).
   * For E1 runs, set `qa.memory_dependent_baseline=True` to also stuff a plain text memory prompt (belt-and-suspenders).
2. `src/hei_nw/models/base.py`

   * Confirm `generate(..., mem_tokens=..., adapter=...)` path is taken for B1; keep the current “Memory:” section as fallback when `memory_prompt` is set.

**Tests.**

* `tests/probes/test_e1_oracle.py::test_oracle_probe_has_high_em` (n=48, assert `em_relaxed >= 0.8`)
* `tests/probes/test_e1_oracle.py::test_selected_contains_answers`

**Acceptance check.**

```bash
scripts/run_m2_acceptance.sh   # E1 section should report EM≥0.8
```

---

## RC-M2-T2 — Memory-aware QA prompting (don’t ignore memory)

**Goal.** Ensure the model is instructed to **use** memory tokens.

**Key changes.**

1. `src/hei_nw/eval/harness.py`

   * For B1, set QA defaults: `prompt_style=chat`, `qa.answer_hint=True`, and pass a short system hint:

     > “Use the provided Memory snippets when answering. Prefer them over prior text.”
   * Make hint controllable: `--qa.answer_hint/--no-qa.answer_hint` already exists; default **True** for B1.
2. `src/hei_nw/models/base.py`

   * In `build_prompt`, for chat templates, prepend a **system** message when `memory_prompt` is non-empty.

**Tests.**

* `tests/test_prompting.py::test_memory_hint_in_chat_prompt`

**Acceptance check.**

* Rerun M2 uplift; **non-empty rate stays \~1.0**, EM lift increases vs. baseline.

---

## RC-M2-T3 — Retrieval strength knobs (DG/HNSW/Hopfield) + better defaults

**Goal.** Improve retrieval health to **P\@1 ≥ 0.6** on Scenario A mini-set.

**Key changes.**

1. `src/hei_nw/eval/harness.py`

   * Add flags: `--ann.m`, `--ann.ef_construction`, `--ann.ef_search`; plumb into `RecallService.build(...)`.
   * Add `--hopfield.steps` and `--hopfield.temperature` (already present; keep defaults but expose in scripts).
2. `src/hei_nw/recall.py` / `src/hei_nw/store.py`

   * Respect the new ANN flags (defaults: `m=32, ef_construction=200, ef_search=128`).
3. `scripts/run_m2_retrieval.sh`

   * Use stronger default `--ann.ef_search 128` and `--dg.k` tuned (e.g., 8–16 depending on hidden size).

**Tests.**

* `tests/retrieval/test_retrieval_health.py::test_p_at_1_above_0p6_on_mini`

**Acceptance check.**

```bash
scripts/run_m2_acceptance.sh
# In retrieval health: P@1 ≥ 0.6, MRR increases; E3 Hopfield improves top-1 vs. baseline
```

---

## RC-M2-T4 — Acceptance runner: enforce sample size or explicit dev override

**Goal.** Avoid “passing” underpowered runs.

**Key changes.**

1. `scripts/run_m2_acceptance.sh`

   * If `N < 200` and `--hard-subset` not provided, **print a red warning** and set a `UNDERPOWERED=true` flag in the summary.
   * Exit non-zero unless `ALLOW_SMALL_SAMPLE=1` environment var is set.
2. Persist `summary.md` with the `UNDERPOWERED` note.

**Tests.**

* `tests/utils/test_scripts.py::test_m2_acceptance_enforces_sample_size`

**Acceptance check.**

```bash
N=48 scripts/run_m2_acceptance.sh && echo "should fail" || echo "correctly blocked"
ALLOW_SMALL_SAMPLE=1 scripts/run_m2_acceptance.sh  # allowed for dev
```

---

## RC-SHARED-T1 — Report-level visibility for gate labels & writes

**Goal.** Make it obvious when calibration is non-informative.

**Key changes.**

1. `src/hei_nw/eval/report.py`

   * Add a “Gate Label Mix” line: `positives / total (rate)`.
   * If `positive_rate ∈ {0,1}`, annotate plots “degenerate label distribution”.

**Tests.**

* `tests/test_reports.py::test_gate_report_includes_label_mix`

**Acceptance check.**

* Re-run calibration; check `A_gate_calibration.png` footer shows label mix.

---

## RC-SHARED-T2 — CLI contract & docs sync (+ optional `--out` for M2)

**Goal.** Eliminate drift between script behavior and docs.

**Key changes.**

1. `scripts/run_m2_acceptance.sh`

   * (Option A) Implement `--out PATH` to override `OUT` envvar.
   * Update `usage()` accordingly.
2. `documentation/quick-validate.md` and any prompt templates referencing flags:

   * Clarify that M2 writes to `reports/m2-acceptance/` by default (if you pick Option B and **don’t** add `--out`, fix docs instead).

**Tests.**

* `tests/utils/test_scripts.py::test_m2_acceptance_has_out_flag_or_docs_match` (assert exactly one of: flag exists **or** docs disclaim it)

**Acceptance check.**

* Verify `--help` matches actual behavior; CI runs the help checks.

---

# How to run after merge (Human)

1. **Gate calibration (expect τ-sensitivity):**

   ```bash
   scripts/run_m3_gate_calibration.sh --scenario A --n 512 \
     --threshold-sweep "0.8 1.0 1.2 1.4 1.6"
   ```

   Verify sweep summary shows changing write rates; label mix is non-degenerate.

2. **M2 acceptance (expect E1 EM ≥ 0.8, improved retrieval P\@1):**

   ```bash
   ALLOW_SMALL_SAMPLE=1 scripts/run_m2_acceptance.sh
   # then with a hard subset when ready:
   scripts/run_m2_acceptance.sh --hard-subset path/to/hard_subset.jsonl
   ```

---

## Risks & Mitigations

* **R1: Label distribution still degenerate** → Add **adversarial negatives** in Scenario A; assert mixed labels in tests.
* **R2: Model prompt drift (chat templates)** → Keep `template_policy=auto` with system memory hint; fall back to plain prompt in tests.
* **R3: Retrieval knobs regress latency** → Track `adapter_latency_overhead_s` and ANN `efSearch` in reports; cap efSearch in CI mini runs.

---

## Appendix — Post-Fix Acceptance Snapshot (2025-09-25)

**Context.** Follow-up run after implementing the single-token normalization fix and defaulting acceptance scripts to label-driven writes (`USE_GATE_WRITES=0`). See `reports/m2-acceptance/retrieval/A_B1_report.md` (seed 7, n=48) for raw metrics.

**Command.** `PYTHONPATH=src ALLOW_SMALL_SAMPLE=1 scripts/run_m2_acceptance.sh`

**Outcomes.**

- `ΔEM = +0.542` with 95% CI `[0.396, 0.688]` (Scenario A small set).
- Retrieval health now clears the design bar: `P@1 = 0.604`, `MRR = 0.757` (`reports/m2-acceptance/retrieval/A_B1_metrics.json`).
- Oracle probe scores EM **1.000**; normalization fix prevents “Fay left …” style answers from being rejected.
- Hopfield still offers no improvement on the probe sweep; we fall back to heuristically re-ranked ANN results (RC-M2-T3 follow-up keeps Hopfield optional).

**Implementation notes.**

- `_normalize_prediction(..., single_token=True)` strips boilerplate tokens when the answer-hint is active (`src/hei_nw/eval/harness.py`).
- New unit tests cover both normalization modes (`tests/test_harness_dev_flags.py`).
- Acceptance scripts and isolation probes accept `USE_GATE_WRITES=1` to re-enable gate-driven writes once telemetry is trustworthy.

**Next steps.**

- Restore τ-sweeps and gate-driven writes after verifying calibration (RC-M3-T1/T3).
- Tune retrieval knobs (DG `k`, HNSW `ef_search`, cue embeddings) toward the design target `P@1 ≥ 0.6` before claiming final acceptance.
