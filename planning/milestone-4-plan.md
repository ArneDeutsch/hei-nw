# M4 — Replay Queue, CA2-Style Scheduler, and LoRA Consolidation (B2/B3)

## 1) Milestone Summary

* Build a **ReplayQueue** plus a **CA2-style scheduler** that prioritizes items by salience, recency, diversity, and overlap.
* Add **LoRA-based consolidation** to distill episodic traces into the base model, enabling **B2** (after replay, memory ON) and **B3** (after replay, memory OFF) evaluation modes.
* Prove consolidation: **B3 approaches B1** on episodic targets with **no base-task regression**; shuffled replay performs worse than scheduled.

---

## 2) Dependencies / Inputs

* Planning docs: `planning/design.md` (§5.8 Replay & Consolidation; defaults §11), `planning/validation-plan.md` (modes B0–B3; scenarios A–E; drift guard; CI), `planning/project-plan.md` (M4 scope, verification, DoD).
* Existing code/APIs to call:

  * Episodic plumbing: `src/hei_nw/{gate.py,pack.py,store.py,recall.py,keyer.py,eviction.py}`
  * Evaluation harness (to extend): `src/hei_nw/eval/harness.py`
  * Datasets A–E: `src/hei_nw/datasets/`
  * Model loader: `src/hei_nw/models/base.py`
  * Metrics & scripts: `src/hei_nw/metrics/*`, `scripts/compute_lift_ci.py`, `scripts/run_parity_guard.sh`
* Tooling: `codex-env/requirements.txt` (transformers, peft, trl, accelerate, datasets, faiss, pytest, ruff, mypy, black).

---

## 3) \[CODEX] Implementation Tasks

### M4-T1 — \[CODEX] Implement `ReplayQueue` and item schema

* **Goal:** Provide a priority queue for episodic traces with the fields required by the scheduler.
* **Key changes:**

  1. New module `src/hei_nw/replay.py`:

     * `@dataclass QueueItem { trace_id: str; score_S: float; created_at: float; diversity_hash: int; group_id: int | None; key_hint: list[int] | None }`
     * `ReplayQueue.push(items: Iterable[dict]) -> None` (accepts TraceWriter payloads; derives fields)
     * `ReplayQueue.pop_batch(k:int, policy:"ca2|shuffled", **kwargs) -> list[QueueItem]`
     * `simhash_for_slots(entity_slots: Mapping) -> int` (diversity buckets)
     * `key_hint_from_slots(entity_slots) -> list[int]` (lightweight DG-key proxy using `DGKeyer` on cue constructed from slots)
  2. Update `TraceWriter` (if needed) to expose `records` with `group_id` present (already captured via `trace_id` suffix and `entity_slots`).
* **Tests:**

  * `tests/test_replay_queue.py::test_push_and_pop_order_ca2`
  * `tests/test_replay_queue.py::test_diversity_bucket_reduces_duplicates`
  * `tests/test_replay_queue.py::test_shuffled_policy_uniformity`
  * Coverage ≥ 90% for this module.
* **Quality gates:** `black . && ruff . && mypy . && pytest -q tests/test_replay_queue.py`
* **Acceptance check:**
  `python - <<'PY'\nfrom hei_nw.replay import ReplayQueue; q=ReplayQueue(); ...; print('ok')\nPY`

---

### M4-T2 — \[CODEX] CA2-style scheduler scoring

* **Goal:** Implement scheduler scoring that prefers high-S, inverse recency, diversity bonus, and **minimizes overlap** via key similarity.
* **Key changes:**

  * In `src/hei_nw/replay.py` add:

    * `CA2Scheduler(score_w=(1.0, 1.0, 0.1, 0.5))` combining: `S_norm`, `1/recency`, `diversity_bonus`, `-overlap_penalty`.
    * Overlap proxy: cosine similarity between `key_hint` vectors of current batch and candidates (K-WTA mask from `DGKeyer`).
* **Tests:**

  * `tests/test_scheduler.py::test_scheduler_prefers_high_S_recent_diverse`
  * `tests/test_scheduler.py::test_overlap_penalty_spreads_batches`
* **Quality gates:** same as above.
* **Acceptance check:**
  `python -m pytest -q tests/test_scheduler.py`

---

### M4-T3 — \[CODEX] Row builders for replay (LM and Cue→Answer)

* **Goal:** Build supervised training rows from pointer-only traces deterministically.
* **Key changes:**

  1. New module `src/hei_nw/training/rows.py`:

     * `LMReplayRowBuilder(scenario:str, n:int, seed:int)` regenerates records via dataset generator; extracts `episode_text` slice referenced by `tokens_span_ref` (we set `start=0,end=len(episode)` in harness), yields `(input_ids, labels)` for next-token loss.
     * `CueAnswerRowBuilder(...)` synthesizes `(cue, answer)` from `entity_slots` (`who/what/where/when`) into HF dataset rows.
     * Both ensure **no raw text** is re-persisted into the trace store; text is used transiently for training batches only.
  2. Helper: `parse_trace_id(trace_id) -> (pointer_doc, record_index)` (already structured by harness).
* **Tests:**

  * `tests/test_rows.py::test_lm_replay_row_reconstructs_from_index`
  * `tests/test_rows.py::test_cue_answer_row_uses_entity_slots_only`
  * `tests/test_rows.py::test_no_raw_text_written_to_store` (guard)
* **Quality gates:** same as above.
* **Acceptance check:**
  `python - <<'PY'\nfrom hei_nw.training.rows import LMReplayRowBuilder; print('ok')\nPY`

---

### M4-T4 — \[CODEX] LoRA training & consolidation worker

* **Goal:** Train LoRA on replay batches; export adapter; optionally merge for B3.
* **Key changes:**

  1. New module `src/hei_nw/training/lora.py`:

     * `LoRAConfig(r=32, alpha=16, lr=1e-4, wd=0.01, steps=500, batch_size=4, mix=(0.5,0.3,0.2))`
     * `ConsolidationWorker(model_id:str, outdir:Path, device:str)` with:

       * `prepare_adamw()`, `attach_lora(peft)`, `train_on(replay_ds, cue_ds)`, `save_adapter()`, `merge_and_save_base()` (for B3)
     * Accelerate support (single-GPU default).
  2. Save artifacts under `artifacts/m4/lora/` and record training args JSON.
* **Tests:**

  * `tests/test_lora.py::test_attach_and_save_adapter_smoke` (uses dummy model; skip if transformers unavailable on CI)
  * `tests/test_lora.py::test_merge_noop_on_dummy_model`
* **Quality gates:** same as above plus `pytest -m "not slow"`
* **Acceptance check:**
  `python -c "from hei_nw.training.lora import LoRAConfig; print('ok')"`

---

### M4-T5 — \[CODEX] Extend harness with B2/B3 modes

* **Goal:** Enable evaluation after replay with memory ON (B2) and corticalized (B3).
* **Key changes:**

  * Modify `src/hei_nw/eval/harness.py`:

    * Add CLI flags: `--adapter.path`, `--adapter.merge_for_b3`.
    * Implement mode handlers:

      * **B2**: load base + **attach LoRA adapter**, memory ON (same retrieval path as B1).
      * **B3**: load base + **merge LoRA** (or load merged weights), **memory OFF** (retrieval disabled).
    * Keep outputs JSON/MD compatible with B0/B1.
* **Tests:**

  * `tests/test_harness_b2_b3.py::test_modes_registered`
  * `tests/test_harness_b2_b3.py::test_b3_disables_memory_path`
* **Quality gates:** same as above.
* **Acceptance check:**
  `python -m hei_nw.eval.harness --mode B3 --scenario A --n 8 --seed 13 --adapter.path artifacts/m4/lora --adapter.merge_for_b3 --outdir reports/m4-smoke`

---

### M4-T6 — \[CODEX] Replay runner CLI & scripts

* **Goal:** End-to-end script: write traces (B1), build queue, replay→train LoRA, then run B2/B3 evals.
* **Key changes:**

  1. New CLI `src/hei_nw/eval/replay_runner.py`:

     * Args: `--scenario {A,B,C}`, `--n`, `--seed`, `--cycles`, `--policy {ca2,shuffled}`, `--steps-per-cycle`, `--mix`, `--lora.r`, `--outdir`
     * Flow:

       1. Run harness **B1** with `--gate.use_for_writes` to collect `TraceWriter.records` (in-process).
       2. Push to `ReplayQueue`; for each cycle sample batches via scheduler.
       3. Build LM & Cue→Answer datasets and train LoRA.
       4. Evaluate **B2** and **B3** with adapter path; save `*_{mode}_metrics.json` and `*_{mode}_report.md`.
  2. Shell wrapper `scripts/run_m4_replay.sh` with sane defaults (3 cycles; A/B/C).
* **Tests:**

  * `tests/utils/test_scripts.py::test_run_m4_replay_cli_parses` (extend existing script-parser tests)
* **Quality gates:** same as above.
* **Acceptance check:**
  `bash scripts/run_m4_replay.sh --scenario A --n 64 --seed 7 --outdir reports/m4-run1`

---

### M4-T7 — \[CODEX] Interference experiment (scheduler vs shuffled)

* **Goal:** Demonstrate retention worsens with shuffled replay.
* **Key changes:**

  * Extend `replay_runner.py` to accept `--policy shuffled` and run comparison.
  * New plotting script `scripts/plot_replay_retention.py` → saves `reports/interference/{scenario}/retention.png` and JSON summary.
* **Tests:**

  * `tests/test_interference.py::test_shuffled_worse_than_ca2_on_proxy_metric` (synthetic proxy: fewer unique groups recalled).
* **Quality gates:** same as above.
* **Acceptance check:**
  `python scripts/plot_replay_retention.py --runs reports/m4-run1 --outdir reports/interference/A`

---

### M4-T8 — \[CODEX] Reporting & docs

* **Goal:** Produce review-ready artifacts and update docs.
* **Key changes:**

  * Add `reports/m4/README.md` describing runs, configs, and acceptance checks.
  * Update `README.md` usage; add a short “M4 quick start”.
  * Add `documentation/m4-method-notes.md` (1–2 pages).
* **Tests:** `tests/test_report_md.py` add section presence checks.
* **Quality gates:** same as above.
* **Acceptance check:**
  `bash scripts/make_report.sh reports/m4-run1`

---

### M4-T9 — \[CODEX] Stub eradication and CI wiring

* **Goal:** Ensure no stubs/mocks remain; wire CI for new tests.
* **Key changes:**

  * Add `scripts/grep_no_stubs.sh` to CI stage for M4 paths.
  * Ensure all new public APIs have docstrings and are imported at least once in an integration test.
* **Tests:** none beyond CI.
* **Quality gates:** run repo-wide grep (below).
* **Acceptance check:**
  `bash scripts/grep_no_stubs.sh`

---

## 4) \[HUMAN/ChatGPT] Review & GPU Tasks

1. **Design sanity pass (10–15 min):**

   1. Skim `src/hei_nw/replay.py`, `training/rows.py`, `training/lora.py` against `planning/design.md §5.8`.
   2. Confirm scheduler terms and replay mix (50/30/20) defaults exist.
2. **GPU run — A/B/C small (engineering track):**

   1. `bash scripts/run_m4_replay.sh --scenario A --n 128 --seed 13 --cycles 3 --outdir reports/m4-run1`
   2. Repeat for **B** and **C**.
   3. **Success signal:** in each `*_B3_report.md`, B3 EM rises vs B0 and approaches B1; see retention curve.
3. **B3 uplift CI (statistical track):**

   1. `python scripts/compute_lift_ci.py --a reports/m4-run1/A_B0_metrics.json --b reports/m4-run1/A_B3_metrics.json --metric em --n_boot 2000`
   2. **Success signal:** 95% CI for (B3−B0) excludes 0.
4. **Drift guard on base tasks:**

   1. `bash scripts/run_parity_guard.sh` (pre/post replay B0 core tasks).
   2. **Success signal:** ΔEM within ±1.
5. **Interference check (policy ablation):**

   1. `bash scripts/run_m4_replay.sh --scenario A --policy shuffled --outdir reports/m4-run1-shuffled`
   2. `python scripts/plot_replay_retention.py --runs reports/m4-run1 reports/m4-run1-shuffled --outdir reports/interference/A`
   3. **Success signal:** CA2 curve dominates shuffled; JSON notes “shuffled worse”.

---

## 5) Deliverables & Artifacts

* Code: `src/hei_nw/replay.py`, `src/hei_nw/training/{rows.py,lora.py}`, `src/hei_nw/eval/replay_runner.py`, harness updates.
* Scripts: `scripts/run_m4_replay.sh`, `scripts/plot_replay_retention.py`.
* Adapter & weights: `artifacts/m4/lora/{adapter,merged}/`.
* Reports:

  * `reports/m4-run*/{A,B,C}_{B1,B2,B3}_{metrics.json,report.md}`
  * `reports/interference/{scenario}/retention.{png,json}`
  * `reports/m4/README.md`

---

## 6) Definition of Done (DoD) Checklist

* **Replay & Scheduler implemented:** `ReplayQueue` and **CA2 scheduler** with diversity & overlap terms.
* **LoRA consolidation:** Worker trains on **50/30/20** mix; adapter saved; option to merge for B3.
* **Modes:** Harness supports **B2** (adapter attached, memory ON) and **B3** (adapter merged, memory OFF).
* **Verification runs:** A/B/C show **B3↑ toward B1**; **shuffled < CA2**.
* **Acceptance metrics:** `B3` retains **≥80–90% of B1** on targets; `(B3−B0)` 95% CI excludes 0; **no base-task regression** (±1 EM).
* **Artifacts produced** as listed; reports include charts and JSON.
* **No stubs/mocks:**
  `git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."`
* **Docs updated** (README quick start; M4 notes).
* **All quality gates green** (below).

---

## 7) QA Gates & CI Commands

* **Formatting:** `black .`
* **Lint:** `ruff .`
* **Types:** `mypy .`
* **Unit/integ tests:**

  * CI: `pytest -q -m "not slow"`
  * Local slow (if any): `pytest -q -m slow`
* **Coverage:** ≥85% changed-lines (diff-cover acceptable).
* **Repo grep:** `bash scripts/grep_no_stubs.sh`
* **Smoke E2E:** `bash scripts/run_m4_replay.sh --scenario A --n 32 --cycles 1 --outdir reports/m4-smoke`

---

## 8) Risks & Mitigations

1. **Base-task drift during replay**
   *Mitigation:* small LR, cosine decay, early-stop on rising base loss; enforce parity guard (±1 EM) in the run script.
2. **Row reconstruction mismatch (pointer ↔ record)**
   *Mitigation:* deterministically regenerate datasets (`seed`, `n`, `scenario`); unit test that `trace_id` → `record_index` round-trips; assert hashed `pointer_doc` matches recomputed digest.
3. **LoRA adapter incompatibility / HF API churn**
   *Mitigation:* isolate adapter attach/merge in `training/lora.py` with version-tolerant checks; provide dummy-model smoke tests; fall back to “attach only” path if merge fails and note in report.

---

### Branching & PR hygiene

* Branch: `feat/m4-replay-lora`
* Single PR title: `M4 — Replay Queue, CA2-Style Scheduler, and LoRA Consolidation (B2/B3)`
* PR checklist mirrors DoD, requires one **\[HUMAN]** review before merge.

---

### Notes tied to HEI-NW docs

* Scheduler terms and replay mix match `planning/design.md §5.8`.
* Modes **B2/B3** and acceptance match `planning/validation-plan.md` and `planning/project-plan.md` (consolidation; uplift CI; drift guard).
* Ensure **no raw episode text** is stored in traces; training uses transient regeneration only.
