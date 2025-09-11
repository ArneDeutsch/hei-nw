# M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton) — **Executable Task Plan**

## 1) Milestone Summary

* Add a **no-op-equivalent Episodic Adapter** (single cross-attention block) and **Memory Token Packer** API, wire them into the generation path, and enable `--mode B1` in the harness.
* Verify **functional equivalence to B0** when memory is empty (`M_t=None`) and **record adapter latency overhead**.
* Produce CI-passing unit/integration tests, reports, and CLI scripts; **no stubs/mocks** left behind.

---

## 2) Dependencies / Inputs

* From repo (already present):

  * Harness & CLI: `src/hei_nw/eval/harness.py`, `src/hei_nw/utils/cli.py`
  * Baselines: `src/hei_nw/baselines/`
  * Datasets A–E: `src/hei_nw/datasets/`
  * Base model utilities: `src/hei_nw/models/base.py`
  * Metrics: `src/hei_nw/metrics/`
  * Tests scaffolding: `tests/test_harness_cli.py`, `tests/test_e2e.py`, scenarios tests
* Planning constraints:

  * **Project plan M1** scope/doD: Add **Episodic Adapter** API and **Memory Token Packer**; **no memory store yet**; **B1(with empty memory) ≈ B0**; log latency budget.
  * **Design**: Adapter is a slim cross-attention that will later attend over **packed memory tokens**; API surface `EpisodicAdapter.forward(H_t, M_t)`.
  * **Validation plan**: Modes `B0..B3` must exist; for M1 we only **exercise B1 with empty memory** and confirm **metric equivalence** and **latency recording**.

---

## 3) \[CODEX] Implementation Tasks

### M1-T1 — \[CODEX] Create `EpisodicAdapter` (read-only, no-op when `M_t` is empty)

* **Goal:** Provide a real cross-attention module with a path that returns `H_t` unchanged when `M_t` is `None` or length 0.
* **Key changes:**

  1. Add `src/hei_nw/adapter.py`:

     * `class EpisodicAdapter(nn.Module)` with `__init__(hidden_size:int, n_heads:int, dropout:float=0.0)` and `forward(H_t:Tensor, M_t:Tensor|None, attn_mask:Tensor|None=None) -> Tensor`.
     * Use `nn.MultiheadAttention` (batch-first) + residual & LayerNorm (`PreNorm`), ensure shape `[batch, seq, hidden]`.
     * If `M_t is None or M_t.shape[1]==0`: **return `H_t` byte-for-byte** (no LayerNorm/Dropout in this branch).
  2. Add docstrings specifying **read-only** semantics (no writes, no state).
* **Tests:**

  * `tests/test_adapter.py::test_noop_when_memory_empty` — asserts `torch.allclose(out, H_t)` and identity of `.detach().cpu().numpy()`.
  * `tests/test_adapter.py::test_shapes_with_memory_tokens` — random `H_t` `[2,7,hidden]`, `M_t` `[2,k,hidden]` runs, preserves shape.
  * Coverage target for this file ≥ **95%**.
* **Quality gates:**
  `black . && ruff . && mypy . && pytest -q tests/test_adapter.py`
* **Acceptance check:**
  `python3 -c "from hei_nw.adapter import EpisodicAdapter; import torch; h=32; m=EpisodicAdapter(h,4); H=torch.randn(2,5,h); assert torch.equal(m(H,None), H); print('OK')"`

---

### M1-T2 — \[CODEX] Implement **Memory Token Packer** (deterministic, capped)

* **Goal:** Deterministically pack a (future) episode trace into **token IDs** with a fixed template and cap length by `max_mem_tokens`.
* **Key changes:**

  1. Add `src/hei_nw/pack.py`:

     * `def pack_trace(trace:dict, tokenizer, max_mem_tokens:int) -> list[int]`
     * Deterministic template (stable field order):
       `"<episodic>\nwho:{who}\nwhat:{what}\nwhere:{where}\nwhen:{when}\n</episodic>"`
       Missing fields render as empty strings; strip; encode with `tokenizer(...)["input_ids"]`; **truncate** to `max_mem_tokens`.
     * Pure function, no global state.
* **Tests:**

  * `tests/test_pack.py::test_pack_is_deterministic_and_capped`
  * `tests/test_pack.py::test_pack_handles_missing_fields`
* **Quality gates:** same as above.
* **Acceptance check:**
  `python - <<'PY'\nfrom transformers import AutoTokenizer\nfrom hei_nw.pack import pack_trace\nT=AutoTokenizer.from_pretrained('sshleifer/tiny-gpt2')\nids=pack_trace({'who':'Dana','what':'backpack','where':'Café Lumen','when':'2025-09-10'}, T, 16)\nassert len(ids)<=16\nprint('OK',len(ids))\nPY`

---

### M1-T3 — \[CODEX] Extend base generation API with optional memory tokens (plumbing only)

* **Goal:** Add a **stable API surface** for memory tokens without changing outputs when `mem_tokens=None or []`.
* **Key changes:**

  1. In `src/hei_nw/models/base.py`:

     * Update `generate(..., mem_tokens: list[int] | None = None, adapter: 'EpisodicAdapter|None' = None, **kwargs)`; **do not** change generation logic when `mem_tokens` falsy.
     * If both `adapter` and `mem_tokens` are provided **and non-empty**, for now **bypass** (documented) and log a `UserWarning` that read path is not active in M1; return normal output. (This prevents silent changes yet keeps the signature.)
  2. Expose factory: `def build_default_adapter(model) -> EpisodicAdapter` (read geometry from `model.config`).
* **Tests:**

  * `tests/test_models_base.py::test_generate_signature_accepts_mem_tokens` — call with `mem_tokens=None` and empty list; outputs equal.
* **Quality gates:** as standard.
* **Acceptance check:**
  `python - <<'PY'\nfrom hei_nw.models.base import load_base, generate\n_,_,_=load_base(model_id='sshleifer/tiny-gpt2', quant_4bit=False)\na=generate('Hello', max_new_tokens=4)\nb=generate('Hello', max_new_tokens=4, mem_tokens=[])\nassert a['text']==b['text']\nprint('OK')\nPY`

---

### M1-T4 — \[CODEX] Wire `--mode B1` in harness (empty memory path) and record latency

* **Goal:** Enable `B1` mode end-to-end; when memory is empty, metrics and outputs must match `B0` while we **record adapter overhead**.
* **Key changes:**

  1. Modify `src/hei_nw/eval/harness.py`:

     * Remove the M0 guard that rejects `args.mode != "B0"`.
     * For `mode=="B1"`: call generation exactly like `B0` but:

       * Construct adapter via `build_default_adapter(model)`.
       * Call `generate(..., adapter=adapter, mem_tokens=None)`.
       * Measure latency per item; add `"adapter_latency_overhead_s"` field equal to `(B1_latency - B0_latency)` in summary if both runs are requested, else just record B1 latency.
  2. Save reports under `reports/m1-episodic-adapter/` when `--mode B1`.
* **Tests:**

  * `tests/test_harness_b1.py::test_b1_runs_and_writes_reports(tmp_path)` — run `--mode B1 -n 0` and `-n 2` with tiny model, assert JSON and MD exist.
* **Quality gates:** as standard.
* **Acceptance check:**
  `python -m hei_nw.eval.harness --mode B1 --scenario A -n 0 --outdir reports/m1-episodic-adapter --model sshleifer/tiny-gpt2`

---

### M1-T5 — \[CODEX] Equivalence test: `B1(empty)` ≈ `B0`

* **Goal:** Demonstrate metric equivalence within ±0.1 EM/F1 across A–E on tiny splits.
* **Key changes:**

  1. Add `tests/test_b1_equivalence.py`:

     * For scenarios `A..E`, `n=4`, seed fixed, run `B0` and `B1` to collect item-level predictions. Assert **exact string equality** of predictions (stronger than ±0.1).
     * If exact equality flakes on some models, fallback to asserting EM/F1 diffs ≤ 0.1 with seed fixed.
* **Quality gates:** as standard.
* **Acceptance check:**
  `pytest -q tests/test_b1_equivalence.py`

---

### M1-T6 — \[CODEX] CLI scripts + report template

* **Goal:** Convenience scripts and MD summary for reviewers.
* **Key changes:**

  1. Add `scripts/run_b1_empty.sh` — runs A–E with `-n 8` using tiny model into `reports/m1-episodic-adapter/`.
  2. Add `scripts/compare_b0_b1.py` — loads metrics JSON pairs and prints a one-line diff table; exits non-zero if any EM/F1 delta > 0.1.
  3. Add `documentation/m1_adapter_notes.md` — brief design notes & how to extend to real reads in M2.
* **Tests:** smoke test for the comparator: `tests/test_compare_b0_b1.py`.
* **Quality gates:** as standard.
* **Acceptance check:**
  `python scripts/compare_b0_b1.py reports/baseline/A_B0_metrics.json reports/m1-episodic-adapter/A_B1_metrics.json`

---

### M1-T7 — \[CODEX] CI & quality plumbing updates

* **Goal:** Ensure CI runs B1 tests and coverage meets bar.
* **Key changes:**

  1. Update `.github/workflows/ci.yml`: run `pytest -m "not slow"` which now includes new B1 tests.
  2. Enforce **≥85%** changed-lines coverage; ensure `coverage.xml` is uploaded.
* **Tests:** CI green on PR.
* **Quality gates:** as standard.
* **Acceptance check:** CI badge/status green for PR `M1 — Episodic Adapter ...`.

---

### M1-T8 — \[CODEX] Stub-removal guardrail

* **Goal:** Enforce **no stubs/mocks** remain.
* **Key changes:**

  1. Add `scripts/grep_no_stubs.sh` and call it from CI:

     ```
     git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."
     ```
* **Tests:** none (script).
* **Quality gates:** as above.
* **Acceptance check:**
  `bash scripts/grep_no_stubs.sh`

---

## 4) \[HUMAN/ChatGPT] Review & GPU Tasks

1. **M1-H1 — Design/API review (quick):**

   1. Skim `adapter.py` & `pack.py` docstrings; 2) Confirm `generate(...)` new params & no-op behavior; 3) Approve if shapes and no-op branch are correct.
      **Success signal:** clear approval comments; no design changes required.

2. **M1-H2 — Equivalence & latency sanity run (CPU/tiny OK):**

   1. Run B0 then B1 on tiny splits:
      `python -m hei_nw.eval.harness --mode B0 --scenario A -n 8 --outdir reports/baseline --model sshleifer/tiny-gpt2`
      `python -m hei_nw.eval.harness --mode B1 --scenario A -n 8 --outdir reports/m1-episodic-adapter --model sshleifer/tiny-gpt2`
   2. Compare: `python scripts/compare_b0_b1.py ...`
   3. Open MD reports and confirm latency field present.
      **Success signal:** comparator exit code 0; report shows latency.

3. **M1-H3 — PR review & CI gate:**

   1. Verify PR title/branch; 2) Confirm CI green; 3) Run `bash scripts/grep_no_stubs.sh`.
      **Success signal:** CI green, grep outputs “No stubs.”

---

## 5) Deliverables & Artifacts

* Code:

  * `src/hei_nw/adapter.py` — **EpisodicAdapter**.
  * `src/hei_nw/pack.py` — **Memory Token Packer**.
  * `src/hei_nw/models/base.py` — extended `generate(...)`, `build_default_adapter(...)`.
  * `src/hei_nw/eval/harness.py` — `--mode B1` enabled; latency recorded.
* Tests:

  * `tests/test_adapter.py`, `tests/test_pack.py`, `tests/test_b1_equivalence.py`, `tests/test_harness_b1.py`, `tests/test_compare_b0_b1.py`.
* Scripts & docs:

  * `scripts/run_b1_empty.sh`, `scripts/compare_b0_b1.py`, `scripts/grep_no_stubs.sh`.
  * `documentation/m1_adapter_notes.md`.
* Reports:

  * `reports/m1-episodic-adapter/*_metrics.json` and `*_report.md` for A–E tiny runs.

---

## 6) Definition of Done (DoD) Checklist

* [ ] **Episodic Adapter API** exists and is **no-op identical** when `M_t` is empty (`tests/test_adapter.py::test_noop_when_memory_empty`).
* [ ] **Memory Token Packer** deterministic & capped (`tests/test_pack.py`).
* [ ] Harness **`--mode B1` runs**, writes metrics and report under `reports/m1-episodic-adapter/`.
* [ ] **Equivalence:** For A–E tiny splits, **B1(empty) predictions equal B0** (or EM/F1 deltas ≤ 0.1) — comparator passes.
* [ ] **Latency recorded** in B1 summaries (field present and non-null).
* [ ] **Docs & scripts** included (notes + run/compare scripts).
* [ ] **No stubs/mocks:**
  `git grep -nE "(TODO|FIXME|pass  # stub|raise NotImplementedError)" || echo "No stubs."`
* [ ] **Quality gates:** `black`, `ruff`, `mypy` clean; `pytest -q` green; coverage ≥85% changed lines.
* [ ] Single PR on branch `feat/m1-episodic-adapter` with this checklist; **1+ HUMAN review** before merge.

---

## 7) QA Gates & CI Commands

* **Format:** `black .`
* **Lint:** `ruff .`
* **Types:** `mypy .`
* **Tests (CI):** `pytest -q -m "not slow"`
* **Coverage:** changed lines ≥ **85%** (upload `coverage.xml` if configured)
* **Slow (local optional):** `pytest -q -m slow`
* **No-stubs check:** `bash scripts/grep_no_stubs.sh`

---

## 8) Risks & Mitigations

1. **HF generation hook complexity** (can’t easily inject cross-attn mid-stack in M1).
   *Mitigation:* Keep M1 **read-only plumbing only** (API + no-op path). Defer actual read integration to M2/M3 where we either wrap the model forward or implement a lightweight per-step decode loop.

2. **Non-determinism causing false diffs between B0 and B1.**
   *Mitigation:* Fix seeds, disable sampling in tests, and demand **exact prediction equality** on tiny model. Fall back to EM/F1 delta ≤ 0.1 only if exact equality is flaky on certain tokenizers.

3. **Latency measurements noisy on small hardware.**
   *Mitigation:* Record **per-item** elapsed times and aggregate in report; do not enforce numeric budget in M1 — only presence + reproducible computation. Enforce numerical budget in later milestones.

---

### Branching & PR hygiene

* Branch: `feat/m1-episodic-adapter`
* PR: **“M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)”**
* PR checklist mirrors **DoD** above.
