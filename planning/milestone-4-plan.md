## Milestone summary for **M4 — Replay Queue, CA2-Style Scheduler, and LoRA Consolidation (B2/B3)**

**Goal.** Add a **replay/consolidation pipeline**: mine replay rows from pointer-only traces, schedule them with a **CA2-style** policy (recency × salience × similarity/spacing), and **consolidate** into the model via **LoRA** so that **B3 (memory OFF)** approaches **B1 (memory ON)** while **B2 (post-replay, memory ON)** remains healthy.
**Key acceptance.** On scenarios A/B/C with 1–N short replay cycles, **B3 retains ≥ 80–90% of B1** on targets; **(B3−B0)** bootstrap CI **excludes 0**; **base-task drift guard** holds (≤ ±1 EM). Show **scheduler effect**: shuffled ordering **worsens** retention (figure + JSON).

---

## 1) Traceability Matrix (DoD → Evidence)

| DoD Item                                                 | Implementation Anchors                                                                                               | Tests                                                                                             | CLI/Script                                          | Artifact Path(s)                                                                                           | Pass Signal                                                                     |
| -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Replay mining** from traces (A/B/C) to cue→answer rows | `src/hei_nw/replay.py` (`ReplayQueue`, `mine_replay_rows`, `CA2Scheduler`)                                           | `tests/test_replay_queue.py` (row count, deterministic ordering, pointer mapping)                 | `scripts/run_m4_replay_consolidation.sh --dry-run`  | `reports/m4-replay/*/replay_queue.jsonl`, `.../scheduler_history.json`                                     | `replay_queue.jsonl` validates schema; ≥ 1 row per write; no PII fields present |
| **CA2-style scheduler** improves retention vs shuffled   | same as above + `src/hei_nw/replay.py:ca2_priority`                                                                  | `tests/test_scheduler_effect.py` (synthetic loss proxy monotonic wrt priority; ablation harness)  | `scripts/m4_scheduler_ablation.sh`                  | `reports/m4-ablation/scheduler_ablation.json`, `.../scheduler_ablation.png`                                | `scheduler_ablation.json.effect_size > 0` and figure emitted                    |
| **LoRA consolidation** trainer                           | `src/hei_nw/consolidate.py` (`train_lora`, `apply_lora`, CPU-friendly fallback)                                      | `tests/test_consolidate_lora.py` (tiny run converges on dummy model; weights saved/loaded)        | `scripts/run_m4_replay_consolidation.sh --epochs 1` | `reports/m4-replay/*/training_history.json`, `.../lora_config.json`, `.../weights/`                        | Loss ↓ ≥ 10% on toy set; artifacts exist & validate                             |
| **Extend harness to B2/B3**                              | `src/hei_nw/eval/harness.py` (`_evaluate_mode_b2/_b3`, `MODE_HANDLERS`), new helpers in `src/hei_nw/utils/ledger.py` | `tests/eval/test_harness_b2_b3.py` (CLI runs w/ dummy model; metrics json present)                | `python -m hei_nw.eval.harness --mode B2/B3 ...`    | `reports/m4-replay/*/{A,B,C}_B2_metrics.json`, `.../{A,B,C}_B3_metrics.json`, `.../retention_summary.json` | `retention_summary.json.b3_over_b1 ≥ 0.80` and CI of (B3−B0) excludes 0         |
| **Base-task drift guard**                                | `src/hei_nw/consolidate.py` (periodic eval on base prompts)                                                          | `tests/test_drift_guard.py` (synthetic invariant set; ΔEM within ±1)                              | via `run_m4_replay_consolidation.sh` (auto)         | `.../drift_guard.json`                                                                                     | `abs(delta_em) ≤ 1.0`                                                           |
| **Ledger** of all runs                                   | `src/hei_nw/utils/ledger.py` + script hooks                                                                          | `tests/test_ledger_index.py`                                                                      | (auto from all above)                               | `reports/index.jsonl`                                                                                      | New line appended; schema-valid                                                 |
| **Anti-stub & schema guard**                             | `scripts/grep_no_stubs.sh` (tightened), `schemas/*.json`, `src/hei_nw/schemas.py`                                    | `tests/test_no_stubs.sh`, `tests/test_artifact_schemas.py`, `tests/scripts/test_cli_contracts.py` | `scripts/grep_no_stubs.sh`                          | —                                                                                                          | CI fails on stubs; all artifacts validate                                       |

---

## 2) Artifact Contract

> Ship machine-readable schemas at `schemas/` (generated from `src/hei_nw/schemas.py` Pydantic models and also stored as JSON Schema). Tests validate artifacts against Pydantic models.

### Canonical files

* **Replay queue**

  * `reports/m4-replay/<run_slug>/replay_queue.jsonl`
  * **Schema (per line)** `ReplayRow`:

    * `trace_id: str` (e.g., `"group-12-9ab3:0"`)
    * `group_id: int`
    * `scenario: "A"|"B"|"C"`
    * `cue: str` *(no PII; derived from generator)*
    * `answer: str`
    * `salience: float` *(S from M3 gate; unitless)*
    * `recency: int` *(lag tokens, ≥0)*
    * `similarity: float` *(0–1 cosine proxy; unitless)*
    * `weight: float` *(scheduler-computed replay weight)*
    * `pointer: object` *(opaque pointer summary, **no** raw text; e.g., `{doc:"group-12-…", start:int, end:int}`)*
  * **Example line**:

    ```json
    {"trace_id":"group-0-a8ba2c19:0","group_id":0,"scenario":"A","cue":"Where did ...?","answer":"in Paris","salience":1.87,"recency":42,"similarity":0.74,"weight":0.92,"pointer":{"doc":"group-0-a8ba2c19","start":0,"end":43}}
    ```

* **Scheduler history**

  * `reports/m4-replay/<run_slug>/scheduler_history.json`
  * **Schema** `SchedulerHistory`:

    * `policy: "ca2"|"shuffled"`
    * `count: int`
    * `weights: {"alpha":float,"beta":float,"gamma":float,"lambda_recency":float,"lambda_similarity":float}`
    * `order_sample: [trace_id, ...]` *(first 16 ids for audit)*
    * `stats: {"gini":float,"mean_weight":float}`

* **Training**

  * `.../training_history.json` — `TrainingHistory`:

    * `epochs:int`, `steps:int`, `loss:[float]`, `eval_em:[float]`
    * `best_step:int`, `best_eval_em:float`, `early_stopped:bool`
  * `.../lora_config.json` — `LoRAConfigDump`:

    * `rank:int`, `alpha:float`, `lr:float`, `target_modules:[str]`, `total_params:int`, `trainable_params:int`
  * `.../weights/` — saved adapter or LoRA weights (framework-neutral torch state dict)

* **Metrics**

  * `.../{A,B,C}_B2_metrics.json`, `.../{A,B,C}_B3_metrics.json` — **same shape** as existing `*_B1_metrics.json` (see `reports/m3-run1/tau_1.2/A_B1_metrics.json`) with added field:

    * `aggregate.retention_ref: "B1"|"B0"` *(identity of baseline used for retention)*
  * `.../retention_summary.json` — `RetentionSummary`:

    * `scenario:"A"|"B"|"C"`, `b1_em:float`, `b3_em:float`, `b0_em:float`, `b3_over_b1:float`, `b3_minus_b0:float`, `ci_low:float`, `ci_high:float`, `n:int`

* **Ablation**

  * `reports/m4-ablation/scheduler_ablation.json`:

    * `scenario`, `cycles:int`, `ca2_retention:float`, `shuffled_retention:float`, `effect_size:float`

* **Ledger**

  * Append to `reports/index.jsonl` — `RunLedgerLine`:

    * `ts:str(ISO8601)`, `milestone:"M4"`, `scenario`, `seed:int`, `model:str`,
      `mode:"B2"|"B3"|...`, `params:object` *(tau, lora, cycles, scheduler, n, …)*,
      `artifacts:[str]` *(relative paths)*, `notes:str?`

> **Units/compatibility:** all EM/F1 metrics remain **fractions \[0,1]** (unchanged), latencies in **seconds**, write rates per **1k tokens** (unchanged). No conversion needed.

---

## 3) CLI Contract

### A) `scripts/run_m4_replay_consolidation.sh`

**Purpose:** Mine replay rows → schedule (CA2) → train LoRA → evaluate B2/B3 → log ledger.

**Flags**

| Flag                             | Type / Default                       | Meaning                                     |       |
| -------------------------------- | ------------------------------------ | ------------------------------------------- | ----- |
| `--scenario {A,B,C}`             | str / **A**                          | Dataset generator scenario                  |       |
| `--n N`                          | int / **48**                         | Records per scenario for mining & eval      |       |
| `--seed S`                       | int / **13**                         | Global seed                                 |       |
| `--model ID`                     | str / **Qwen/Qwen2.5-1.5B-Instruct** | Base model id (supports dummy path)         |       |
| `--tau T`                        | float / **1.5**                      | Gate threshold to select writes (reuse M3)  |       |
| `--cycles K`                     | int / **1**                          | Replay cycles                               |       |
| `--queue-size Q`                 | int / **512**                        | Max items in queue per cycle                |       |
| `--scheduler {ca2,shuffled}`     | str / **ca2**                        | Scheduling policy                           |       |
| `--alpha/--beta/--gamma/--delta` | float / **from M3 default**          | Salience weights used for `S` in scheduling |       |
| `--lambda-recency R`             | float / **1.0**                      | Recency weight                              |       |
| `--lambda-sim L`                 | float / **1.0**                      | Similarity weight                           |       |
| `--lora-rank r`                  | int / **8**                          | LoRA rank                                   |       |
| `--lora-alpha a`                 | float / **16.0**                     | LoRA alpha                                  |       |
| `--lora-lr lr`                   | float / **1e-4**                     | Learning rate                               |       |
| `--epochs E`                     | int / **1**                          | Epochs                                      |       |
| `--batch-size B`                 | int / **8**                          | Batch size                                  |       |
| `--out DIR`                      | str / **reports/m4-replay**          | Output root                                 |       |
| `--resume`                       | flag                                 | Resume if weights exist                     |       |
| `--dry-run`                      | flag                                 | Build queue + history only (no training)    |       |
| \`--help                         | -h\`                                 | —                                           | Usage |

**Examples**

```bash
scripts/run_m4_replay_consolidation.sh --scenario A --n 64 --cycles 2 --tau 1.6 \
  --scheduler ca2 --lora-rank 8 --epochs 1 --out reports/m4-runA
```

**Outputs**

* `reports/m4-runA/replay_queue.jsonl`, `scheduler_history.json`, `training_history.json`,
  `{A}_B2_metrics.json`, `{A}_B3_metrics.json`, `retention_summary.json`
* Appends one line to `reports/index.jsonl`

---

### B) `scripts/m4_scheduler_ablation.sh`

Runs the same pipeline twice (CA2 vs shuffled) and summarizes the **retention delta**.

**Flags**

| Flag                                  | Type / Default |
| ------------------------------------- | -------------- |
| `--scenario {A,B,C}` / **A**          |                |
| `--n N` / **64**                      |                |
| `--seed S` / **13**                   |                |
| `--cycles K` / **1**                  |                |
| `--tau T` / **1.5**                   |                |
| `--out DIR` / **reports/m4-ablation** |                |
| \`--help                              | -h\`           |

**Outputs**

* `reports/m4-ablation/scheduler_ablation.json`, `scheduler_ablation.png`
* Ledger line

---

### C) Harness (existing) extended

`python -m hei_nw.eval.harness`

* `--mode` now **supports**: `B0`, `B1`, **`B2`**, **`B3`**
* For `B2/B3`: inherits existing flags (`--scenario`, `-n`, `--seed`, `--model`, Hopfield/QA flags) and adds:

  * `--replay-root DIR` *(defaults to sibling dir produced by run\_m4 script)*
  * `--baseline-ref {"B1","B0"}` *(for retention computation in report)*

**CLI existence tests** will assert these flags appear in `--help`.

---

## 4) Observability & Chronology

* **Single ledger:** `reports/index.jsonl`. Every script appends:

```json
{
  "ts":"2025-09-24T12:34:56Z",
  "milestone":"M4",
  "scenario":"A",
  "seed":13,
  "model":"Qwen/Qwen2.5-1.5B-Instruct",
  "mode":"B3",
  "params":{"tau":1.6,"cycles":2,"scheduler":"ca2","lora":{"rank":8,"alpha":16,"lr":1e-4}},
  "artifacts":[
    "reports/m4-runA/A_B2_metrics.json",
    "reports/m4-runA/A_B3_metrics.json",
    "reports/m4-runA/retention_summary.json"
  ]
}
```

* **Index maintenance:** `src/hei_nw/utils/ledger.py` exposes `append_ledger(outdir, line)`; script wrappers call it. Test ensures append-only behavior and schema compliance.

---

## 5) Plan Tasks (strict format)

### **T0 — Gate & Retrieval Preconditions (Owner: Human)**

* **Rationale:** Stop-the-line verification before M4. M3 gate must be calibrated; M2 retrieval acceptance must hold.
* **Changes:** None (uses existing scripts).
* **Tests:** N/A (manual acceptance capture in ledger).
* **CLI:**

  * `scripts/run_m3_gate_calibration.sh --scenario A --n 512 --threshold-sweep "0.8 1.0 1.2 1.4 1.6" --out reports/m3-accept`
  * `scripts/run_m2_acceptance.sh --out reports/m2-accept`
  * **Success signals:** `reports/m3-accept/A_sweep_summary.json` exists and shows **writes\_per\_1k\_tokens ∈ \[1,5]** at some τ; `reports/m2-accept/compare_b0_b1.json` shows **headroom gate pass** (or switch to alt reporting per plan).
* **Artifacts:** `reports/m3-accept/*`, `reports/m2-accept/*`; ledger entries added.
* **Quality gates:** N/A.
* **Done means:** Human signs off “M2 & M3 acceptance met” in commit message and links artifact paths. If unmet, open “M3/M2 Root-Cause” and **block T1+**.

---

### **T1 — Implement Replay Miner & CA2 Scheduler (Owner: Codex)**

* **Rationale:** DoD rows “Replay mining” and “Scheduler effect”.
* **Changes:**

  * Add `src/hei_nw/replay.py`:

    * `ReplayQueue(capacity:int)`, `mine_replay_rows(records, gate, store_diag)`, `CA2Scheduler(weights)` and `ca2_priority(row)` implementing: `priority = w_s*S + λ_r*recency_norm + λ_sim*similarity`.
    * Deterministic ordering, tie-break on `trace_id`.
  * Wire `mine_replay_rows` to existing A/B/C generators via `group_id` + pointer doc (see `reports/*/A_trace_samples.json` pattern) ensuring **no raw episode\_text** is written.
* **Tests:**

  * `tests/test_replay_queue.py`: builds 12 rows with controlled S/recency/similarity; asserts queue length, order, stability.
  * `tests/test_scheduler_effect.py`: synthetic mini-loop where higher priority correlates with faster toy loss drop.
  * **CLI existence test:** `tests/scripts/test_cli_contracts.py` checks `scripts/run_m4_replay_consolidation.sh --help` lists flags (see contract).
* **CLI:**

  * `scripts/run_m4_replay_consolidation.sh --scenario A --n 48 --dry-run --out reports/m4-replay-smoke`
  * **Success signals:** `replay_queue.jsonl` has ≥ 24 rows; validates schema; `scheduler_history.json.policy=="ca2"`.
* **Artifacts:** As per Artifact Contract.
* **Quality gates:** `black`, `ruff`, `mypy`, `pytest -q`.
* **Done means:** `replay_queue.jsonl` schema-valid; PII checker passes (no `episode_text` field); deterministic order across runs with same seed.

---

### **T2 — LoRA Trainer & Weight I/O (Owner: Codex)**

* **Rationale:** Consolidation mechanism for B2/B3.
* **Changes:**

  * Add `src/hei_nw/consolidate.py`:

    * `train_lora(model, tok, rows, config) → (history, weights_dir)`. Implement **real** training with Torch:

      * If HF model present, wrap a safe subset of `nn.Linear` in a minimal LoRA module (no external deps); else fall back to training the existing `EpisodicAdapter` parameters on dummy pathway. Both paths save weights.
    * `apply_lora(model, weights_dir)`; `evaluate_on_prompts(...)` for drift guard.
  * Add `src/hei_nw/schemas.py` (Pydantic models) + emit JSON Schemas to `schemas/*.json` in a small codegen step in `consolidate.py` (first run).
* **Tests:**

  * `tests/test_consolidate_lora.py`: synthetic (CPU) training for 30–50 steps reduces loss ≥ 10%, weights saved/loaded reproduce eval.
  * `tests/test_artifact_schemas.py`: validates produced JSON via Pydantic.
* **CLI:** via T4 wrapper.
* **Artifacts:** `training_history.json`, `lora_config.json`, `weights/`.
* **Quality gates:** `black`, `ruff`, `mypy`, `pytest -q`.
* **Done means:** Training on dummy path converges; artifacts exist & validate; `apply_lora` changes model outputs on replay probes.

---

### **T3 — Extend Harness to B2/B3 (Owner: Codex)**

* **Rationale:** Evaluate after consolidation (memory ON/OFF).
* **Changes:**

  * `src/hei_nw/eval/harness.py`:

    * Implement `_evaluate_mode_b2` and `_evaluate_mode_b3`.
    * Add to `MODE_HANDLERS`.
    * New args: `--replay-root`, `--baseline-ref`.
    * For B2: ensure memory tokens path active; for B3: **memory disabled** and **LoRA applied**.
  * `src/hei_nw/utils/ledger.py`: `append_ledger`.
* **Tests:**

  * `tests/eval/test_harness_b2_b3.py`: run dummy model for A with tiny `replay_queue.jsonl` fixture; assert `*_B2_metrics.json` and `*_B3_metrics.json` produced and contain `aggregate` block.
  * CLI help presence asserted in `tests/scripts/test_cli_contracts.py`.
* **CLI:**

  * `python -m hei_nw.eval.harness --mode B2 --scenario A -n 8 --seed 7 --model hei-nw/dummy-model --replay-root reports/m4-replay-smoke`
  * `python -m hei_nw.eval.harness --mode B3 ...`
  * **Success signals:** metrics JSON files exist; schema-valid; ledger appended.
* **Artifacts:** `.../{A,B,C}_B2_metrics.json`, `.../{A,B,C}_B3_metrics.json`.
* **Quality gates:** as above.
* **Done means:** End-to-end pipeline runs on CPU dummy path; metrics JSON shape matches B1 with `retention_ref` added.

---

### **T4 — Wrap End-to-End Orchestration Scripts (Owner: Codex)**

* **Rationale:** Reproducible, review-friendly entry points.
* **Changes:**

  * Add `scripts/run_m4_replay_consolidation.sh` (bash; mirrors M2/M3 script style).
  * Add `scripts/m4_scheduler_ablation.sh`.
  * Both call `append_ledger`.
* **Tests:** `tests/scripts/test_run_m4_replay_consolidation.py` (smoke on dummy model, asserts files exist), update `tests/scripts/test_cli_contracts.py`.
* **CLI:** see Contract.
* **Artifacts:** as per Contract.
* **Quality gates:** as above.
* **Done means:** Smoke test passes; `scheduler_ablation.json.effect_size > 0` on synthetic.

---

### **T5 — Retention & CI Stats (Owner: Codex)**

* **Rationale:** Formalize acceptance computation + CI visibility.
* **Changes:**

  * `src/hei_nw/eval/report.py`: add `compute_retention(b1_path,b3_path,b0_path) → RetentionSummary` with bootstrap CI.
  * `scripts/make_report.sh`: include M4 retention lines if present.
* **Tests:** `tests/test_retention_summary.py` (bootstrap CI sane; monotonic examples).
* **CLI:** auto-invoked by T4 script.
* **Artifacts:** `retention_summary.json`.
* **Quality gates:** as above.
* **Done means:** `retention_summary.json.b3_over_b1 ≥ 0.80` on synthetic; function used by script.

---

### **T6 — Drift Guard (Owner: Codex)**

* **Rationale:** DoD requires no base regression.
* **Changes:** `src/hei_nw/consolidate.py:evaluate_on_prompts` with a small **invariant prompt set** (added in `src/hei_nw/testing/invariants.py`).
* **Tests:** `tests/test_drift_guard.py` verifies ΔEM ≤ 1 on invariant set after training.
* **CLI:** part of T4 run; emits `drift_guard.json`.
* **Artifacts:** `.../drift_guard.json`.
* **Quality gates:** as above.
* **Done means:** Guard passes on synthetic.

---

### **T7 — Ledger & Schema Guards (Owner: Codex)**

* **Rationale:** Anti-stub + schema validation (cross-cutting).
* **Changes:**

  * Tighten `scripts/grep_no_stubs.sh` to also flag `pass  # TODO`.
  * Add `tests/test_no_stubs.sh` to run it.
  * Add `tests/test_artifact_schemas.py` validating all JSON against Pydantic models (`src/hei_nw/schemas.py`).
* **Tests:** as above.
* **CLI:** `bash scripts/grep_no_stubs.sh`.
* **Artifacts:** `schemas/*.json`.
* **Quality gates:** as above.
* **Done means:** CI fails if stubs present; all artifacts validate.

---

### **T8 — Documentation & READMEs (Owner: Codex)**

* **Rationale:** Reviewer efficiency and reproducibility.
* **Changes:** Update `documentation/hei-nw.md` and add `documentation/consolidation.md` (how to run M4, interpret retention, ablation).
* **Tests:** `tests/test_report_md.py` updated to include new files (existence/headers).
* **CLI:** N/A.
* **Artifacts:** Markdown docs.
* **Quality gates:** as above.
* **Done means:** Docs render; links valid.

---

## 6) Risk Register & Fallbacks

1. **LoRA path on real HF models may be environment-blocked (no GPU/PEFT).**
   *Mitigation:* Implement **PEFT-free minimal LoRA** on `nn.Linear` and always support a **CPU dummy path** that truly trains (no stubs). Separate weights dirs per model id.

2. **Replay mining could accidentally leak raw episode text.**
   *Mitigation:* Miner reconstructs cue/answer **only** via scenario generator IDs (`group_id`, pointer doc prefix `group-<id>-...`). Add a PII linter: assert `episode_text` never serialized in replay artifacts.

3. **Scheduler brittleness to weight choices.**
   *Mitigation:* Expose weights as CLI flags; log `scheduler_history.json`; run **ablation** with shuffled ordering; acceptance requires a measurable drop vs CA2.

---

## Repository-wide Anti-Stub Gates (enforced here)

* **CI stub scan:** keep `scripts/grep_no_stubs.sh` and **tighten** to catch `pass  # TODO` and `raise NotImplementedError`. Add test that executes it.
* **Schema validation test:** `tests/test_artifact_schemas.py` loads every produced JSON via Pydantic models in `src/hei_nw/schemas.py`.
* **CLI flag parity tests:** `tests/scripts/test_cli_contracts.py` asserts help outputs match this plan for:

  * `scripts/run_m4_replay_consolidation.sh`
  * `scripts/m4_scheduler_ablation.sh`
  * `python -m hei_nw.eval.harness` (B2/B3 flags)

---

### Common sweeps table (ready-to-paste)

| Sweep                                    | Example                                                                                                    | Expected Outputs per value                                        |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `--tau-sweep` (via multiple invocations) | `for t in 1.2 1.4 1.6; do scripts/run_m4_replay_consolidation.sh --tau $t --out reports/m4-tau_${t}; done` | Each run emits `replay_queue.jsonl`, B2/B3 metrics, a ledger line |
| `--cycles`                               | `... --cycles 1/2/3`                                                                                       | `retention_summary.json` trend: `b3_over_b1` non-decreasing       |
| Scheduler                                | `--scheduler ca2` vs `--scheduler shuffled`                                                                | `scheduler_ablation.json.effect_size > 0`                         |

---

**That’s the full, executable plan.** It is directly actionable for Codex (T1–T8) with crisp PASS signals and guards to keep us out of stubland.
