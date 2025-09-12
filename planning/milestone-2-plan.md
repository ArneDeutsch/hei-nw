# M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional)

## 1) Milestone Summary

* Implement the **retrieval path** for HEI-NW: DG keying (k-WTA sparse keys) → **ANN** candidate search → **Modern Hopfield** completion → **memory tokens** for the adapter.
* Wire retrieval into the **B1** harness and log **retrieval health** metrics (P\@k/MRR, near-miss/collision, completion lift).
* Validate on Scenario **A** (partial-cue) with an **ablation** (Hopfield off) and show **B1 − B0 ≥ +30 EM** on the small set.

---

## 2) Dependencies / Inputs

* Planning docs (already in repo):

  * `planning/project-plan.md` — M2 scope/acceptance.
  * `planning/design.md` — §5.3 DG Keyer, §5.4 Associative Store (Hopfield), §8 Configuration defaults.
  * `planning/validation-plan.md` — Scenario A, retrieval metrics.
* Existing code (present in repo and used by M2):

  * Adapter & packer: `src/hei_nw/adapter.py`, `src/hei_nw/pack.py`
  * Datasets: `src/hei_nw/datasets/scenario_a.py`
  * Harness & reports: `src/hei_nw/eval/harness.py`, `src/hei_nw/eval/report.py`
  * Metrics scaffolding: `src/hei_nw/metrics/*`
  * Baselines (reference only): `src/hei_nw/baselines/*`
* Tooling: `pytest`, `ruff`, `black`, `mypy`; FAISS in env; tiny model under `tests/models/tiny-gpt2`.

---

## 3) \[CODEX] Implementation Tasks

### M2-T1 — \[CODEX] DG Keyer (k-WTA)

* **Goal:** Project context → sparse key with k-WTA and L1-normalized values.
* **Key changes:**

  * New: `src/hei_nw/keyer.py`

    * `class DGKeyer(nn.Module)`: `forward(H_t: Tensor) -> dict(indices: Tensor, values: Tensor, dim: int)`

      * `q = W_q · pool(H_t)`; `topk` by magnitude; keep signs; L1-normalize retained values.
    * `to_dense(key, dim) -> Tensor` helper (dense float32 view for ANN).
    * Defaults from design §8: `d=2048`, `k=64`.
* **Tests:**

  * `tests/test_keyer.py::test_k_wta_sparsity_invariants` (exact k non-zeros; L1=1; stable under scaling).
  * `tests/test_keyer.py::test_to_dense_roundtrip` (indices/values ↔ dense).
  * Coverage target ≥90% in this file.
* **Quality gates:** `black .` · `ruff .` · `mypy .` · `pytest -q`
* **Acceptance check:**

  ```bash
  python - <<'PY'
  from hei_nw.keyer import DGKeyer
  import torch; k=DGKeyer(d=2048,k=64); H=torch.randn(2,8,768)
  out=k(H); assert out["values"].abs().sum(dim=-1).allclose(torch.ones(2)), "L1 norm != 1"
  PY
  ```

### M2-T2 — \[CODEX] ANN Index wrapper (HNSW/IP)

* **Goal:** Fast top-K over dense views of DG keys.
* **Key changes:**

  * New: `src/hei_nw/store.py`

    * `class ANNIndex`: HNSW-Flat (FAISS) with inner-product (cosine after L2 norm).

      * `add(vectors: np.ndarray, meta: list[dict])` (stores `meta` with group\_id/should\_remember/trace payload).
      * `search(query: np.ndarray, k:int) -> list[dict]` (returns `meta` + scores).
* **Tests:**

  * `tests/test_store_ann.py::test_ann_returns_expected_neighbor` (toy vectors).
  * `tests/test_store_ann.py::test_ann_meta_alignment` (vectors/docs length match).
* **Quality gates:** standard.
* **Acceptance check:**

  ```bash
  python - <<'PY'
  import numpy as np; from hei_nw.store import ANNIndex
  x=np.eye(4).astype('float32'); idx=ANNIndex(dim=4); idx.add(x,[{"id":i} for i in range(4)])
  r=idx.search(x[0:1],k=1); assert r[0]["id"]==0
  PY
  ```

### M2-T3 — \[CODEX] Modern Hopfield Readout (inference-only)

* **Goal:** Complete partial cues from candidates (1–3 fixed updates).
* **Key changes:**

  * New in `src/hei_nw/store.py`:

    * `class HopfieldReadout(nn.Module)`: parameters (pattern matrix `M`) stored in module; `requires_grad=False` by default; `forward(cue, candidates) -> refined_query or scores`. Fixed steps, softmax attention with temperature; no training path in M2.
* **Tests:**

  * `tests/test_hopfield_readout.py::test_inference_is_read_only` (state\_dict identical pre/post forward; all params `requires_grad=False`).
  * `tests/test_hopfield_readout.py::test_refinement_changes_query` (refined ≠ input on synthetic).
* **Quality gates:** standard.
* **Acceptance check:** run the two tests above.

### M2-T4 — \[CODEX] Associative Store (ANN + Hopfield + metrics hooks)

* **Goal:** Compose DGKeyer, ANNIndex, HopfieldReadout into a retrieval store.
* **Key changes:**

  * Extend `src/hei_nw/store.py`:

    * `class EpisodicStore`:

      * `build_from_records(records, tokenizer, max_mem_tokens)` → indexes only `should_remember=True` items; computes & stores: dense key, `group_id`, `answers`, and `trace` payload (who/what/where/when) derivable from `episode_text`/fields.
      * `query(cue_text:str, top_k_candidates:int=64, return_m:int=4, use_hopfield:bool=True)` → returns selected traces and retrieval diagnostics.
* **Tests:**

  * `tests/test_store_ep.py::test_build_and_query_top1_self` (query with exact cue retrieves its episode).
  * `tests/test_store_ep.py::test_near_miss_and_collision_counters` (with confounders).
* **Quality gates:** standard.
* **Acceptance check:** quick synthetic:

  ```bash
  PYTHONPATH=src python - <<'PY'
  from hei_nw.datasets import scenario_a
  from transformers import AutoTokenizer
  from hei_nw.store import EpisodicStore
  tok=AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
  recs=scenario_a.generate(n=8, seed=0)
  st=EpisodicStore.from_records(recs, tok, max_mem_tokens=64)
  q=recs[0]["cues"][0]; out=st.query(q)
  assert out["selected"], "no selection"
  PY
  ```

### M2-T5 — \[CODEX] Retrieval metrics (P\@k, MRR, near-miss, collision, completion lift)

* **Goal:** Compute retrieval-health metrics with clear definitions.
* **Key changes:**

  * New: `src/hei_nw/metrics/retrieval.py`

    * `precision_at_k`, `mrr`,
    * `near_miss_rate`: retrieved item has same `group_id` but `should_remember=False`,
    * `collision_rate`: top-1 `group_id` ≠ query’s `group_id` while a matching `should_remember=True` exists in store,
    * `completion_lift`: Δ(top-1 correctness) when `use_hopfield=True` vs `False` on the same queries.
* **Tests:** `tests/metrics/test_retrieval.py::{test_p_at_k,test_mrr,test_near_miss_and_collision,test_completion_lift}`.
* **Quality gates:** standard.
* **Acceptance check:** run tests.

### M2-T6 — \[CODEX] Recall API returning memory tokens

* **Goal:** One call that goes from cue → memory tokens for the adapter.
* **Key changes:**

  * New: `src/hei_nw/recall.py`

    * `class RecallService`: wraps `EpisodicStore` and `pack.pack_trace`

      * `build(records, tokenizer, max_mem_tokens)`
      * `recall(cue_text) -> list[int]` (concat up to `return_m` packed traces; ≤128 tokens per §8).
* **Tests:** `tests/test_recall.py::test_recall_returns_token_ids_length_bound`.
* **Quality gates:** standard.
* **Acceptance check:** short script calling `RecallService` on Scenario A.

### M2-T7 — \[CODEX] Harness integration (B1 uses retrieval)

* **Goal:** B1 path computes `mem_tokens` via RecallService and logs new metrics.
* **Key changes:**

  * Modify `src/hei_nw/eval/harness.py`:

    * Build a store once per run from generated **records** (index `should_remember=True` only).
    * For each record, use first `cues[0]` as query → `mem_tokens` → pass to adapter; fill `EvalItem.recall_at_k` with RAG baseline if present.
    * Collect retrieval-health metrics (from `metrics/retrieval.py`) and attach to summary (`summary["retrieval"] = {...}`).
  * Modify `src/hei_nw/eval/report.py`: append a **Retrieval** section (P\@k, MRR, near-miss/collision, completion-lift).
* **Tests:**

  * `tests/test_harness_b1.py` (extend): assert summary contains `retrieval` with finite numbers.
  * `tests/eval/test_report_details.py` (extend) to check new section exists.
* **Quality gates:** standard.
* **Acceptance check:**

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 12 --seed 7 \
    --model tests/models/tiny-gpt2 --outdir reports/m2-retrieval-stack
  jq '.retrieval' reports/m2-retrieval-stack/A_B1_metrics.json
  ```

### M2-T8 — \[CODEX] Hopfield ablation & plot

* **Goal:** Run B1 twice (with/without Hopfield) and plot completion-lift.
* **Key changes:**

  * Harness: add `--no-hopfield` flag to disable Hopfield in `EpisodicStore.query`.
  * `src/hei_nw/eval/report.py`: save `reports/m2-retrieval-stack/completion_ablation.png` (matplotlib).
* **Tests:** `tests/eval/test_report_details.py::test_ablation_plot_written`.
* **Quality gates:** standard.
* **Acceptance check:** two B1 runs (flag on/off) produce plot file.

### M2-T9 — \[CODEX] CLI script & wiring

* **Goal:** One-shot script to reproduce the M2 acceptance.
* **Key changes:**

  * New: `scripts/run_m2_retrieval.sh` (B0 vs B1, ablation on A).
  * New: `scripts/compare_b0_b1_m2.sh` invoking existing compare tool to assert `B1−B0 ≥ 0.30 EM`.
* **Tests:** `tests/utils/test_scripts.py` extend with presence/executable bit.
* **Quality gates:** shellcheck not enforced; keep POSIX-sh.

### M2-T10 — \[CODEX] Public API & docs polish

* **Goal:** Make modules discoverable and documented.
* **Key changes:**

  * `src/hei_nw/__init__.py` export `DGKeyer`, `EpisodicStore`, `RecallService`.
  * Docstrings on all public classes/methods; type hints; `__all__` updates.
* **Tests:** `pytest -q` ensures import path works (`tests/*/test_imports.py` extend if present).
* **Quality gates:** standard.
* **Acceptance check:** `python -c "import hei_nw, inspect; print(hei_nw.__all__)"`

### M2-T11 — \[CODEX] No-stubs sweep

* **Goal:** Enforce zero placeholders.
* **Key changes:** none (check only).
* **Tests/Acceptance check:**

  ```bash
  bash scripts/grep_no_stubs.sh
  ```

---

## 4) \[HUMAN/ChatGPT] Review & GPU Tasks

1. **Code review pass**

   * Check `keyer.py`, `store.py`, `recall.py`, harness/report diffs → verify docstrings, types, invariants (k-WTA, read-only Hopfield).
2. **Sanity run (CPU ok with tiny model)**

   ```bash
   export PYTHONPATH=src
   python -m hei_nw.eval.harness --mode B0 --scenario A -n 24 --seed 7 \
     --model tests/models/tiny-gpt2 --outdir reports/m2-retrieval-stack
   python -m hei_nw.eval.harness --mode B1 --scenario A -n 24 --seed 7 \
     --model tests/models/tiny-gpt2 --outdir reports/m2-retrieval-stack
   ```

   **Success signal:** `A_B1_metrics.json` contains `retrieval` with finite P\@k/MRR and `A_B1_report.md` shows the Retrieval section.
3. **Ablation check**

   ```bash
   python -m hei_nw.eval.harness --mode B1 --scenario A -n 24 --seed 7 \
     --model tests/models/tiny-gpt2 --outdir reports/m2-retrieval-stack --no-hopfield
   ```

   **Success signal:** `completion_ablation.png` written; completion-lift in JSON > 0 on average.
4. **Acceptance delta**

   ```bash
   python scripts/compare_b0_b1.py reports/m2-retrieval-stack/A_B0_metrics.json \
                                   reports/m2-retrieval-stack/A_B1_metrics.json
   ```

   **Success signal:** exit code `0` and printed lift `≥ +0.30 EM` on the small set.

---

## 5) Deliverables & Artifacts

* **Code:**

  * `src/hei_nw/keyer.py`, `src/hei_nw/store.py`, `src/hei_nw/recall.py`
  * `src/hei_nw/metrics/retrieval.py`
  * Harness/report changes as above.
* **Scripts:** `scripts/run_m2_retrieval.sh`, `scripts/compare_b0_b1_m2.sh`
* **Reports:** under `reports/m2-retrieval-stack/`

  * `A_B{0,1}_metrics.json` and `A_B{0,1}_report.md`
  * `completion_ablation.png`
* **Tests:** new/extended under `tests/` as listed per task.

---

## 6) Definition of Done (DoD) Checklist

* [ ] **Quality lift:** On Scenario A small set, `B1 − B0 ≥ +30 EM` (≥ +0.30 absolute).
* [ ] **Retrieval health logged:** JSON includes `retrieval: { p_at_k, mrr, near_miss_rate, collision_rate, completion_lift }` with finite values.
* [ ] **Hopfield inference is read-only:** unit/integration test proves parameters unchanged and `requires_grad=False`.
* [ ] **End-to-end B1:** Harness uses RecallService to feed **memory tokens** into the adapter.
* [ ] **Ablation:** `--no-hopfield` run produces `completion_ablation.png` and JSON with completion-lift ≥ 0.
* [ ] **Zero stubs:**

  ```bash
  git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."
  ```
* [ ] **Docs & API:** Public classes have docstrings; modules importable.
* [ ] **CI green:** formatting, linting, typing, tests, coverage gates pass.

---

## 7) QA Gates & CI Commands

* **Format:** `black .` (no diff)
* **Lint:** `ruff .` (no errors)
* **Types:** `mypy .` (no new issues)
* **Tests:**

  * `pytest -q`
  * Coverage on changed lines ≥ **85%** (CI config respects diff-coverage)
  * Optional split: `pytest -m "not slow"` in CI; `pytest -m slow` locally.
* **E2E smoke:**

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 12 --seed 7 \
    --model tests/models/tiny-gpt2 --outdir reports/m2-retrieval-stack
  ```

---

## 8) Risks & Mitigations

1. **Sparse→dense ANN fidelity** (cosine on dense zeros may blunt separation).
   *Mitigations:* L2-normalize dense views; assert k-WTA invariants; add unit test on separability; expose `efSearch` knob and set `efS=64` per §8.
2. **No EM lift with tiny model** (signal too weak).
   *Mitigations:* Increase `n` slightly for the small set (e.g., 24–48); ensure store indexes only positives; verify cue templates match trace packing; confirm adapter actually receives non-empty tokens.
3. **Metric definitions ambiguous (near-miss/collision).**
   *Mitigations:* Encode definitions in `metrics/retrieval.py` tied to Scenario A fields (`group_id`, `should_remember`), document in docstrings, and assert with fixture tests using controlled confounders.

---

### Branching & PR hygiene

* Branch: `feat/m2-retrieval-stack`
* Single PR: **“M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional)”** with a checklist mirroring the DoD.
* Require one **\[HUMAN]** review before merge.

> This plan is grounded in `planning/project-plan.md` (M2 scope/acceptance), `planning/design.md` (§5.3, §5.4, §8), and `planning/validation-plan.md` (Scenario A, retrieval metrics).
