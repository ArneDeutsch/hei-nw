# M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional) — Milestone Review

### 1) Executive summary (2–4 short paragraphs)

**Intent.** M2 was to deliver the end-to-end **retrieval path** for HEI-NW: DG k-WTA keyer → dense view for ANN → candidate set → **Modern Hopfield** completion → **memory tokens** fed to the episodic adapter in B1. The DoD includes: retrieval health metrics in JSON; an ablation (`--no-hopfield`) and plot; read-only Hopfield; B1 wired through the harness; **quality lift** on Scenario A with a real model (Qwen2.5-1.5B) showing **B1 − B0 ≥ +0.30 EM**; “zero stubs”; docs/API; CI green.

**What exists.** The repo contains concrete implementations for the keyer (`src/hei_nw/keyer.py`), ANN wrapper + Hopfield + store (`src/hei_nw/store.py`), recall service producing memory tokens (`src/hei_nw/recall.py`), harness integration & metrics/reporting (`src/hei_nw/eval/*`, `src/hei_nw/metrics/retrieval.py`), runnable scripts, and M2 reports under `reports/m2-retrieval-stack/` (including the ablation PNG). Unit and integration tests are present for each component and for the harness wiring (e.g., `tests/test_harness_b1.py`, `tests/test_hopfield_readout.py`, `tests/test_keyer.py`).

**Verdict.** **Partial.** The retrieval stack is implemented and wired; diagnostics and ablation exist and are exercised in tests. However, two DoD items aren’t met: (1) **no evidence** of the required **+0.30 EM** lift with the target model; the committed reports use the tiny model and show **0.00 EM**. (2) No CI workflow is present, so “CI green” cannot be satisfied.

---

### 2) Evidence snapshot: repo & planning anchors

* **Repo tree (short)**

  * `src/hei_nw/`: `keyer.py`, `store.py`, `recall.py`, `adapter.py`, `eval/{harness.py,report.py}`, `metrics/{retrieval.py,text.py,timing.py}`, `baselines/{long_context.py,rag.py}`, `models/{base.py}`
  * `tests/`: component and harness tests incl. `test_keyer.py`, `test_store_{ann,ep}.py`, `test_hopfield_readout.py`, `test_harness_{b1,cli,ablation}.py`, `metrics/test_retrieval.py`, `utils/test_scripts.py`
  * `scripts/`: `run_m2_retrieval.sh`, `run_m2_retrieval_ci.sh`, `compare_b0_b1_m2.sh`, `compare_b0_b1.py`, `grep_no_stubs.sh`
  * `reports/m2-retrieval-stack/`: `A_B0_metrics.json`, `A_B1_metrics.json`, `A_B1_no-hopfield_metrics.json`, `*_report.md`, `completion_ablation.png`
  * `planning/`: `milestone-2-plan.md`, `design.md`, `validation-plan.md`, `project-plan.md`

* **Planning anchors used**

  * `planning/milestone-2-plan.md` — “# M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional)”
  * `planning/design.md` — §5.3 DG Keyer; §5.4 Associative Store (Hopfield); §8 defaults
  * `planning/validation-plan.md` — Scenario A; modes B0–B3; retrieval metrics
  * `planning/project-plan.md` — milestone scopes/DoD

* **Assumptions/limits**

  * I reviewed by reading files (no GPU runs). Committed reports use the tiny model; hence the Qwen-based EM lift DoD is **unverified**.
  * Several files contain mid-line ellipses in this ZIP (e.g., within docstrings/lines) but the overall structure, tests, and reports indicate real implementations. Please confirm the raw repo has the non-elided sources.

---

### 3) DoD / Acceptance verification table

| Item                                                       | Evidence (files/funcs/CLI)                                                                                                                                                                                                             | Status                | Notes                                                                         |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | ----------------------------------------------------------------------------- |
| **Quality lift on A (B1−B0 ≥ +0.30 EM with Qwen2.5-1.5B)** | Scripts default model to Qwen `run_m2_retrieval.sh` (line contains `MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"`); committed reports show **EM 0.00** for tiny model (`reports/m2-retrieval-stack/A_B0_report.md`, `A_B1_report.md`). | **Fail (unverified)** | Needs a real run on Qwen2.5-1.5B and committing the metrics proving +0.30 EM. |
| **Retrieval health logged**                                | `harness.py` builds `retrieval = { "p_at_1", "mrr", "near_miss_rate", "collision_rate", "completion_lift" }`; `A_B1_metrics.json` contains `retrieval`.                                                                                | **Pass**              | `tests/test_harness_b1.py` asserts finite numbers.                            |
| **Hopfield inference is read-only**                        | `HopfieldReadout` stores patterns with `requires_grad=False`; `tests/test_hopfield_readout.py::test_inference_is_read_only`.                                                                                                           | **Pass**              | Also checks forward doesn’t mutate state.                                     |
| **End-to-end B1 (adapter receives memory tokens)**         | `RecallService.recall()` → `pack_trace()`; `models/base.py` injects `mem_tokens` via adapter path; `harness` B1 uses `RecallService`.                                                                                                  | **Pass**              | `tests/test_harness_b1.py` runs B1 and writes reports.                        |
| **Ablation & plot**                                        | `--no-hopfield` flag in `harness.py`; `save_completion_ablation_plot()` writes `completion_ablation.png`; `tests/eval/test_report_details.py::test_ablation_plot_written`.                                                             | **Pass**              | Also `tests/test_harness_ablation.py` ensures PNG exists and lift ≥ 0.        |
| **Zero stubs**                                             | `scripts/grep_no_stubs.sh`; `tests/utils/test_scripts.py::test_no_stubs_regex`.                                                                                                                                                        | **Pass**              | No `TODO`/`FIXME`/`NotImplementedError` hits.                                 |
| **Docs & API**                                             | Docstrings present; `__all__` in `src/hei_nw/__init__.py`; `tests/test_root_imports.py`.                                                                                                                                               | **Pass**              | Public classes importable.                                                    |
| **CI green**                                               | No `.github/workflows` present.                                                                                                                                                                                                        | **Fail**              | Add GH Actions (format/lint/type/test + tiny run).                            |

---

### 4) Task-by-task review

#### M2-T1 \[CODEX] DG Keyer (k-WTA)

* **Intent.** Pool hidden state → project → k-WTA with signed top-k → L1 normalize; `to_dense` helper.
* **Findings.** `src/hei_nw/keyer.py` defines `DGKeyer` and `to_dense`; tests assert **exact k non-zeros** and **L1=1** (`tests/test_keyer.py`: “exactly k non-zero entries per sample”; “L1 norm of retained values is 1”).
* **Gaps/Risks.** N/A for functionality; confirm source is not elided in your clone.
* **Status.** **Pass**.

#### M2-T2 \[CODEX] ANN Index wrapper (HNSW/IP)

* **Intent.** Dense keys → ANN top-K; keep aligned metadata.
* **Findings.** `store.ANNIndex.add/search` with FAISS; `tests/test_store_ann.py` validates neighbor correctness and meta alignment.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T3 \[CODEX] Modern Hopfield Readout (inference-only)

* **Intent.** Fixed-step refinement over candidate patterns; read-only.
* **Findings.** `store.HopfieldReadout` registers `patterns` with `requires_grad=False`; tests verify invariance of `state_dict()` and that refinement **changes** query (`tests/test_hopfield_readout.py`).
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T4 \[CODEX] Associative Store (ANN + Hopfield + metrics hooks)

* **Intent.** Build from records; query with cue→key→ANN→(optional Hopfield)→selected traces + diagnostics.
* **Findings.** `store.EpisodicStore.from_records` pipelines keyer/ANN; `query(..., use_hopfield=True/False)` returns `selected`, `candidates`, `diagnostics` (near\_miss/collision). `tests/test_store_ep.py` checks top-1 self, near-miss, collision.
* **Gaps/Risks.** Uses hashed BoW embedding (`_hash_embed`)—simple but adequate for Scenario A.
* **Status.** **Pass**.

#### M2-T5 \[CODEX] Retrieval metrics (P\@k, MRR, near-miss, collision, completion lift)

* **Intent.** Implement metrics and surface them.
* **Findings.** `metrics/retrieval.py` implements `precision_at_k`, `mrr`, `near_miss_rate`, `collision_rate`, `completion_lift`; unit tests cover each.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T6 \[CODEX] Recall API returning memory tokens

* **Intent.** Wrap store to produce packed memory token IDs.
* **Findings.** `recall.RecallService.build/recall`; `tests/test_recall.py` ensures non-empty tokens and 128 cap.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T7 \[CODEX] Harness integration (B1 uses retrieval)

* **Intent.** B1 path feeds `mem_tokens` into adapter; compute b0 vs b1 latency delta; log retrieval.
* **Findings.** `harness._evaluate_mode_b1` constructs `RecallService`, computes retrieval, writes `adapter_latency_overhead_s`; `tests/test_harness_b1.py` validates report content.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T8 \[CODEX] Hopfield ablation & plot

* **Intent.** `--no-hopfield` run + PNG plot comparing completion lift.
* **Findings.** Flag present; `report.save_completion_ablation_plot()` writes bar chart; tests for both code path and file emission.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T9 \[CODEX] CLI script & wiring

* **Intent.** One-shot scripts for M2 acceptance and comparison.
* **Findings.** `scripts/run_m2_retrieval.sh` (defaults Qwen2.5-1.5B); `run_m2_retrieval_ci.sh` (pins tiny model); `compare_b0_b1_m2.sh` asserts **EM lift ≥ 0.30** against produced JSON; `tests/utils/test_scripts.py` asserts presence and executable bits.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T10 \[CODEX] Public API & docs polish

* **Intent.** Docstrings and stable exports.
* **Findings.** Module docstrings; `__all__ = ["DGKeyer","EpisodicStore","RecallService"]`; `tests/test_root_imports.py` ensures import.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

#### M2-T11 \[CODEX] No-stubs sweep

* **Intent.** Enforce absence of stub markers.
* **Findings.** `grep_no_stubs.sh` and test verifying regex over files.
* **Gaps/Risks.** None.
* **Status.** **Pass**.

---

### 5) Design & validation alignment

* **Design mapping.** Implements §5.3 **DG Keyer** (`keyer.DGKeyer` with defaults `d=2048,k=64`), §5.4 **Associative Store** (FAISS ANN + `HopfieldReadout`), and adapter memory injection per §4 pipeline (`models/base.generate` combines `mem_tokens` with the adapter). Storage/query flow in `EpisodicStore` matches the intended “cue → candidates → completion → selected traces”.
* **Validation mapping.** Scenario **A** (partial cue) is exercised in B0/B1, with retrieval diagnostics written into JSON and a **Hopfield ablation** plot. Modes B2/B3 stay out-of-scope for M2 (per plan). Metrics match `validation-plan.md` (EM/F1/latency + retrieval P\@k/MRR + diagnostics).

---

### 6) Quality & CI assessment

* **Tooling.** `pyproject.toml` contains config for **black**, **ruff**, **mypy**, **pytest** (and coverage). Requirements include **faiss-cpu** and plotting libs.
* **Testing depth.** Good balance of unit (metrics, keyer, Hopfield), component (ANN/store/recall), and harness E2E with the tiny model. Determinism handled via `set_global_seed` in tests/harness.
* **CI.** **Missing.** There is no `.github/workflows/` file. Scripts exist to reproduce M2 locally (including a CI-friendly tiny run), but pipeline automation is absent.

---

### 7) Gaps and **Follow-up tasks**

#### M2-F1 \[CODEX] Wire a minimal CI workflow (typing, lint, tests, tiny M2 run)

* **Goal:** Make “CI green” measurable on PRs.
* **Key changes:**

  1. Add `.github/workflows/ci.yml` running on `push`/`pull_request`: `pip install -r codex-env/requirements.txt`; run `ruff .`, `black --check .`, `mypy .`, `pytest -q`.
  2. Add a job step to run `scripts/run_m2_retrieval_ci.sh` (uses tiny model) and upload `reports/m2-retrieval-stack/*` as artifacts.
* **Tests:**

  * Reuse existing test suite; ensure CI completes under 15 minutes using tiny model.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  # In PR on GitHub
  # CI shows all jobs green; artifacts contain *_metrics.json and completion_ablation.png
  ```

#### M2-F2 \[HUMAN/ChatGPT] Run acceptance on Qwen2.5-1.5B and commit metrics

* **Goal:** Satisfy “B1 − B0 ≥ +0.30 EM” on Scenario A (small set) with the target model.
* **Steps:**

  1. GPU box with \~8–16GB VRAM (Qwen2.5-1.5B-Instruct, 4-bit OK).
  2. `bash scripts/run_m2_retrieval.sh` (defaults already target Qwen).
  3. `bash scripts/compare_b0_b1_m2.sh` to check EM lift.
  4. Commit `reports/m2-retrieval-stack/*` produced by the run.
* **Acceptance check:** The script prints `EM lift ≥ 0.30`, and the committed `A_B1_metrics.json` vs `A_B0_metrics.json` shows `aggregate.em` satisfying the threshold.

#### M2-F3 \[CODEX] Harden retrieval JSON & docs in the report

* **Goal:** Make retrieval diagnostics self-describing in Markdown.
* **Key changes:**

  1. In `eval/report.py::build_markdown_report`, add a “## Retrieval” section when `summary["retrieval"]` exists with key-value lines for `P@1`, `MRR`, `Near-miss rate`, `Collision rate`, `Completion lift`.
  2. Include a short gloss of near-miss/collision definitions at the end of the report.
* **Tests:**

  * Extend `tests/eval/test_report_details.py` to assert the new lines are present for B1 runs.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 4 \
    --seed 0 --model tests/models/tiny-gpt2 --outdir /tmp/m2
  grep -q "Completion lift:" /tmp/m2/A_B1_report.md
  ```

#### M2-F4 \[CODEX] Add a smoke test for `run_m2_retrieval.sh`

* **Goal:** Ensure the default Qwen-oriented script at least parses on CPU-only environments.
* **Key changes:**

  1. Add `tests/utils/test_m2_script_parse.py` that shells `bash -n scripts/run_m2_retrieval.sh` and asserts exit 0.
* **Tests:** New test file as above.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  pytest -q tests/utils/test_m2_script_parse.py
  ```

---

### 8) Final verdict

**Partial.** The retrieval stack (DG keyer → ANN → Hopfield → memory tokens) is implemented, tested, and integrated; retrieval metrics and ablation are present with artifacts. The milestone falls short on **two DoD items**: **(i)** no committed proof of **+0.30 EM** with Qwen2.5-1.5B, and **(ii)** no CI workflow.
**Minimum follow-ups to meet DoD:** **M2-F1**, **M2-F2**. (Optional polish: **M2-F3**, **M2-F4**.)
