# Critical Second-Opinion Review of M2 (design → impl → validation → reports)

## 1) Executive summary

* **Verdict: M2 acceptance *not met*.** B1 delivers **empty generations** under the chat+newline stop setting, yielding **EM=0.000** vs **B0=1.000** (lift −1.000).
* **Most likely root cause:** stop handling bug in `models/base.generate()` **when adapter is enabled**: decoded text is truncated at the **first `"\n"`**, producing empty predictions. See `src/hei_nw/models/base.py` — “`if stop: stop_idx = text.find(stop)`” and the adapter branch “`generated_ids = output_ids[0]`” (no prompt slice).
* **Alt hypotheses (top 3):**

  1. **Chat + stop mismatch:** newline stop is safe in B0 (post-slice) but unsafe when using `inputs_embeds` (pre-slice).
  2. **Prompt echo in adapter path:** without slicing, decoded text may include leading newline/echo; `find('\n')` at index 0 → empty.
  3. **Over-aggressive memory packing (128 tokens)** may perturb decoding; however retrieval health is solid, pointing primarily to stop/truncation.
* **Smallest next action to falsify top hypothesis:** rerun **B1** with **`--qa.stop ""`** (becomes `None`) **and** `--qa.max_new_tokens 16`. Expect non-empty answers and EM>0 if bug is the cause.

## 2) Acceptance criteria check

* From `planning/milestone-2-plan.md`: *“Validate on Scenario **A** … with an **ablation** (Hopfield off) and show **B1 − B0 ≥ +30 EM** …”*
* From `planning/validation-plan.md`: Scenario A expects **B1≫B0 immediately**; metrics include EM/F1, recall\@k, near-miss/collision, latency.

**Observed (reports):**

* `reports/m2-retrieval-stack/A_B0_metrics.json` → EM=**1.000**, F1=**1.000**; `A_B0_report.md` agrees.
* `reports/m2-retrieval-stack/A_B1_metrics.json` → EM=**0.000**, F1=**0.000**; `A_B1_report.md` shows empty predictions.
* **Lift** B1−B0 = **−1.000** (target ≥ +0.30) → **FAIL**.

**Tiny table (Scenario A, N=48):**

| mode |  n |    EM |    F1 | latency (s) |  P\@1 |   MRR | near-miss | collision | completion\_lift |
| ---- | -: | ----: | ----: | ----------: | ----: | ----: | --------: | --------: | ---------------: |
| B0   | 48 | 1.000 | 1.000 |       0.070 |     – |     – |         – |         – |                – |
| B1   | 48 | 0.000 | 0.000 |       0.171 | 0.375 | 0.543 |     0.167 |     0.292 |           −0.292 |

**CI (95%, Agresti–Coull):** EM\_B0 ≈ \[0.909, 1.000], EM\_B1 ≈ \[0.000, 0.091], **lift** ≈ \[−0.997, −0.849] → strongly negative.

## 3) Run provenance & reproducibility

* Commands per `documentation/quick-validate.md` and `scripts/run_m2_retrieval.sh`:

  * `python -m hei_nw.eval.harness --mode B0 --scenario A … --qa.prompt_style chat --qa.max_new_tokens 8 --qa.stop $'\n' --qa.answer_hint`
  * `python -m hei_nw.eval.harness --mode B1 --scenario A … --qa.prompt_style chat --qa.max_new_tokens 8 --qa.stop $'\n' --qa.answer_hint`
  * Also runs **B1 --no-hopfield** for ablation.
* Model used matches acceptance: **`Qwen/Qwen2.5-1.5B-Instruct`** (`run_m2_retrieval.sh`). CI script warns it is **not** acceptance.
* Prompt template = **chat**; stop = literal newline; `max_new_tokens=8`; `seed=7`; Scenario A forced by the script. Mode flag **does** switch code paths (see `MODE_HANDLERS` and `_evaluate_mode_b1()` in `src/hei_nw/eval/harness.py`).

## 4) Validation pipeline audit (math & data)

* **EM/F1 math** (`src/hei_nw/metrics/text.py`):

  * strict EM: “`return 1.0 if prediction.strip() == truth.strip() else 0.0`”.
  * relaxed EM: “`strict_em(canonicalize(pred), canonicalize(truth))`” (lowercase, strip, drop punctuation).
  * token F1: whitespace tokens; zero if one side **empty**.
* **Why B1 predictions are empty (confirmed):**

  * In `src/hei_nw/models/base.py` `generate()`:

    * Adapter path: “`if adapter is not None and mem_tokens:` … `generated_ids = output_ids[0]`” (no prompt slicing).
    * Then: “`text = tokenizer.decode(generated_ids, skip_special_tokens=True)`”.
    * Stop handling: “`if stop: stop_idx = text.find(stop); … text = text[:stop_idx]`”.
  * With chat+newline stop, first token is often `"\n"`. `find("\n")==0` ⇒ **`text` becomes empty**. Reports show `"prediction": ""` in B1 JSON.
* **Sample size & variance:** N=48 (two cues × 24 episodes). With degenerate 1.0 vs 0.0, bootstrap CI of lift is −1.0; Agresti–Coull CI ≈ \[−0.997, −0.849].
* **Scenario A construction** (`src/hei_nw/datasets/scenario_a.py`):

  * Partial-cue QA with hard negatives; lags binned; fields who/what/where/when; no leakage detected from code.

## 5) Design compliance audit (DG → ANN → Hopfield → memory tokens → adapter)

* **DG keyer** (`src/hei_nw/keyer.py`): k-WTA with L1-norm; defaults **d=2048, k=64**; uses mean-pooled H; selects by `q.abs().topk(k)`; values L1-normalized.
* **ANN index** (`src/hei_nw/store.py`): `IndexFlatIP` with **L2 normalization** on add and search → cosine; exact (not HNSW).
* **Hopfield** (`src/hei_nw/store.py`): iterative refinement “`z = softmax((z @ patterns.T)/T) @ patterns`” for **steps** with **temperature**; ablation toggled by `--no-hopfield`.
* **Token packing** (`src/hei_nw/pack.py`): deterministic template; per-trace capped by `max_mem_tokens`; harness concatenates selected traces and **clips to 128** (`mem_tokens = tokens[:128]`).
* **Adapter injection**:

  * `src/hei_nw/models/base.py`: builds `EpisodicAdapter`; when `mem_tokens` present, computes embeddings and calls `adapter(prompt_embeds, mem_embeds)`; passes as `inputs_embeds` to generation: “`gen_input = {"inputs_embeds": adapted, "attention_mask": inputs["attention_mask"]}`”.
  * `src/hei_nw/adapter.py`: read-only cross-attention returning `H_t + dropout(attn(...))` — memory is **attended**, not ignored.

## 6) Implementation audit (“bug safari”)

* **Stop handling corner-case** (primary): adapter path decodes from **generated ids without prompt slice**, then **applies newline stop**, truncating to empty.
* **CLI & defaults:** `--qa.stop` parsed; `QAPromptSettings.stop_value()` maps empty string to `None`. Script passes **literal newline**, triggering the bug only in B1.
* **Tokenizer/model pairing:** Qwen Instruct uses chat template via `apply_chat_template`; fallback adds `ASSISTANT:`; both produce initial newline/space—unsafe with `find("\n")` at position 0.
* **DG/ANN/Hopfield plumbing:** Matches design; retrieval health is reasonable (P\@1≈0.375, MRR≈0.543). No `TODO|FIXME|NotImplementedError` in hot paths; dtype/device handled; pad/eos set.

## 7) Numbers, reconciled

* **Why retrieval didn’t translate to answers:** generation output was **blanked by stop**. Retrieval metrics healthy; completion\_lift ≤ 0 simply reflects that Hopfield re-rank didn’t matter once generations were empty (with Hopfield off: completion\_lift=0.0; with it on: −0.292).
* **Concrete cases:** `reports/m2-retrieval-stack/A_B1_metrics.json` shows `"prediction": ""` for many IDs; B0 path (same prompt) yields correct single-token names (EM=1.0).

## 8) Fault tree & minimal isolating experiments

**Goal:** isolate stop/truncation vs injection.

1. **Prompt sanity (no memory):** run B0 through the same path but with `adapter=None` — already OK (non-empty).

2. **Stop token probe:** run **B1** with **no stop** and more headroom:

   * Edit `scripts/run_m2_retrieval.sh` to pass `--qa.stop '' --qa.max_new_tokens 16`.
   * Expect non-empty predictions; EM>0 if bug is the cause.

3. **Adapter path echo check:** keep stop as newline but **left-strip decoded text** in `generate()`; or temporarily search `stop` from index **≥1**. Expect non-empty outputs.

4. **Injection sanity (dev modes):**

   * `--dev.retrieval_only` (should emit top retrieved answer directly; see parser help).
   * `--dev.oracle_trace` (inject ground-truth trace only) — EM should jump to \~1.0 if generation path is healthy.

## 9) Actionable fix list (ranked)

### Fast fixes (hours)

1. **Stop handling in adapter path** (`src/hei_nw/models/base.py`):

   * **Change**: before applying `stop`, **strip leading whitespace** from decoded text in the adapter branch:
     Replace: `if stop: stop_idx = text.find(stop)`
     With: `text = text.lstrip(); stop_idx = text.find(stop)`
   * **Alt**: search from index 1 → `stop_idx = text.find(stop, 1)`; or drop `stop` entirely for chat mode and rely on `max_new_tokens`.
   * **Expected delta**: B1 predictions become non-empty; EM should reflect retrieval quality; uplift likely positive on Scenario A.
   * **Guard test**: add unit test that decodes `"\nAlice\n"` with `stop="\n"` returns `"Alice"`, not empty.

2. **Script safeguard** (`scripts/run_m2_retrieval.sh`):

   * **Change**: pass empty stop for chat runs: `--qa.stop ''` (maps to `None`). Optionally set `--qa.max_new_tokens 12`.
   * **Risk**: minimal; bounded by `max_new_tokens`.

3. **Decode slice parity** (`src/hei_nw/models/base.py`):

   * **Change**: in adapter path, **mirror B0 slicing** by computing the number of newly generated tokens and decoding only those (e.g., compare `output_ids.shape[-1]` before/after).
   * **Expected**: stop can remain `"\n"` safely; less brittle to prompt echoes.

### Structural (if needed)

4. **Memory budget**: consider capping to **64 tokens** total (not per-trace) to reduce interference; sweep {32,64,96,128}.

5. **Hopfield tuning**: validate `steps∈{1,2,3}`, `T∈{0.3,0.5,1.0}`; ensure ablation parity still holds; watch completion\_lift.

6. **Keyer k**: sweep `k∈{32,64,96}`; verify near-miss/collision trade-offs.

## 10) Go / no-go

* **Go with fast fixes**: With stop handling fixed (and/or stop disabled) **this week**, M2 should meet the +0.30 EM gate on Scenario A. The retrieval stack is healthy; the failure is in **decode/stop plumbing**, not retrieval quality.

---

## TL;DR (one page for PM)

* M2 **fails acceptance** because B1 answers are **truncated to empty strings** by a **newline stop bug** in the adapter generation path.
* **Evidence:**

  * Reports: B0 EM=1.000 vs B1 EM=0.000; retrieval P\@1=0.375, MRR=0.543; predictions are empty in JSON.
  * Code: adapter path decodes then `find('\n')` → empty when first token is newline; B0 slices prompt before applying stop, avoiding this.
* **What to fix now:** strip leading whitespace or disable newline stop for chat; or slice generated tokens in adapter path before stop.
* **Why we’re confident:** degenerate metrics (1.0 vs 0.0) + healthy retrieval + direct code path showing truncation; CI bounds show lift ≪ +0.30.
* **Ablation:** Hopfield off has same EM=0 but completion\_lift 0.0 vs −0.292 → re-rank works but masked by empty generations.
* **Next run recipe:** `bash scripts/run_m2_retrieval.sh` after changing to `--qa.stop ''` and `--qa.max_new_tokens 16` — expect non-empty answers and positive EM.

---

### M2-F1 — \[CODEX] Fix adapter-path newline stop truncation in `generate()`

* **Goal:** Prevent **blank predictions** in B1 when `prompt_style=chat` and `stop="\n"` by making stop-handling robust in the **adapter path**.

* **Key changes:**

  * Edit `src/hei_nw/models/base.py` in `generate()`:

    * Current adapter branch decodes **all** returned ids:

      > `"if adapter is not None and mem_tokens: ... generated_ids = output_ids[0]"`
      > and then truncates by first newline:
      > `"stop_idx = text.find(stop)"`
    * **Change**: After decoding, **strip leading whitespace** *before* applying `stop`. E.g., insert right after decode:

      ```python
      text = text.lstrip()  # avoid empty on leading newline in chat
      ```
    * Keep existing prompt-slice for the non-adapter branch:

      > `"generated_ids = output_ids[0][prompt_len:]"` (unchanged).
    * Optional safety: restrict the first stop search to index ≥1:

      ```python
      stop_idx = text.find(stop, 1) if stop else -1
      ```

      (Use either `lstrip()` or `find(..., 1)`; prefer `lstrip()`.)
  * Add an inline comment explaining parity differences between `inputs_embeds` vs `input_ids` paths.

* **Tests:**

  * New: `tests/models/test_base_generate_newline.py`

    * `test_adapter_branch_does_not_truncate_to_empty_on_newline_stop()`:

      * Build `EpisodicAdapter()`, call
        `generate("Hello", max_new_tokens=2, stop="\n", adapter=adapter, mem_tokens=[2])`
      * Assert `out["text"].strip() != ""` and `out["generated_tokens"] >= 1`.
    * `test_plain_vs_adapter_stop_parity()`:

      * Compare `generate(..., stop="\n")` with and without adapter; both non-empty.
  * Keep existing `tests/models/test_base_generate.py` intact.

* **QA gates & CI commands:**

  ```bash
  pytest -k "test_base_generate_newline or test_base_generate"
  ```

* **Definition of Done:**

  * B1 predictions are **never empty** due to a leading newline.
  * All tests pass locally and in CI.

---

### M2-F2 — \[CODEX] Adjust M2 run script defaults for chat mode (no explicit newline stop)

* **Goal:** Make acceptance runs **robust** by removing brittle newline stop and providing **more headroom** for short answers.

* **Key changes:**

  * Edit `scripts/run_m2_retrieval.sh` (all three invocations):

    * Replace `--qa.stop $'\n'` with `--qa.stop ''` (maps to **None** via `QAPromptSettings.stop_value()`).
    * Bump `--qa.max_new_tokens 8` → `--qa.max_new_tokens 16`.
  * Update `documentation/quick-validate.md` lines describing newline stop to reflect **no‐stop / max\_new\_tokens=16**.

* **Tests:**

  * Update `tests/utils/test_scripts.py::test_run_m2_retrieval_flags`:

    * Expect `--qa.max_new_tokens 16`.
    * Expect `--qa.stop ''` instead of newline.

* **QA gates & CI commands:**

  ```bash
  bash scripts/run_m2_retrieval_ci.sh   # unchanged smoke
  pytest -k test_run_m2_retrieval_flags
  ```

* **Definition of Done:**

  * Script changes present; doc updated.
  * Related test updated and passing.

---

### M2-F3 — \[CODEX] Add non-empty-prediction gate for B1

* **Goal:** Fail fast when B1 predictions are mostly blank, **before** uplift checks.

* **Key changes:**

  * New: `scripts/gate_non_empty_predictions.py`

    * Input: a metrics JSON path (e.g., `reports/m2-retrieval-stack/A_B1_metrics.json`).
    * Logic: read `records[*].prediction`; compute `non_empty_rate = mean(pred.strip() != "")`.
    * Exit non-zero if `non_empty_rate < 0.9`. Print the rate.
  * New: `scripts/gate_non_empty_predictions.sh` wrapper (calls Python script on A\_B1 file).

* **Tests:**

  * New: `tests/test_gate_non_empty_predictions.py`

    * Write a temp JSON with `records` containing some `prediction=""`.
    * Assert exit non-zero below threshold and zero above.

* **QA gates & CI commands:**

  ```bash
  python scripts/gate_non_empty_predictions.py reports/m2-retrieval-stack/A_B1_metrics.json
  ```

* **Definition of Done:**

  * Gate script exists, documented in quick-validate notes (optional).
  * Unit tests pass.

---

### M2-F4 — \[CODEX] Report surface for non-empty rate & decode diagnostics

* **Goal:** Improve **observability**: expose `non_empty_rate` and minimal decode diagnostics in reports.

* **Key changes:**

  * Edit `src/hei_nw/eval/harness.py` near summary construction (≈ lines 880–900):

    * Compute `non_empty_rate = sum(bool(it.prediction.strip()) for it in items)/len(items) if items else 0.0`.
    * Add to `summary["aggregate"]`.
  * Edit `src/hei_nw/eval/report.py::build_markdown_report()` to render it under “## Aggregate Metrics”.
  * (Keep the “## Debug” section minimal; no per-item dumps.)

* **Tests:**

  * Update `tests/eval/test_report_details.py` to assert the markdown includes “Non-empty rate”.
  * Add unit test that JSON contains `"non_empty_rate"` in `aggregate`.

* **QA gates & CI commands:**

  ```bash
  pytest -k test_report_details
  ```

* **Definition of Done:**

  * New field present in JSON and markdown.
  * Tests updated and green.

---

### M2-F5 — \[CODEX] Minimal isolation probes script for M2

* **Goal:** Add a **tiny** driver to falsify the top hypothesis quickly (stop handling vs. retrieval stack).

* **Key changes:**

  * New: `scripts/m2_isolation_probes.sh`

    * Probes (N=16):

      * **No stop**: `--qa.stop '' --qa.max_new_tokens 16` (B1).
      * **Stop on**: `--qa.stop $'\n'` (B1) — to contrast.
      * **Retrieval-only**: `--dev.retrieval_only` (B1).
      * **Oracle trace**: `--dev.oracle_trace` (B1).
    * Save outputs under `reports/m2-probes/` with suffixes.
  * Document in `documentation/quick-validate.md` (Optional section “Isolation Probes”).

* **Tests:**

  * `tests/utils/test_scripts.py`:

    * Assert script exists and is executable.

* **QA gates & CI commands:**

  ```bash
  bash scripts/m2_isolation_probes.sh
  ```

* **Definition of Done:**

  * Script runs on the tiny model (CI smoke) and Qwen locally.
  * Artifacts appear in `reports/m2-probes/`.

---

### M2-F6 — \[CODEX] Unit tests for chat prompt ↔ stop semantics

* **Goal:** Lock in correct **chat** prompting and stop handling through the harness layer.

* **Key changes:**

  * New: `tests/eval/test_chat_stop_semantics.py`

    * Build one record (Scenario-A like).
    * Use `_evaluate_records(...)` with `QAPromptSettings(prompt_style="chat", stop="")`.
    * Monkeypatch `hei_nw.models.base.generate` to capture `stop` and ensure it is **None** (normalized by `stop_value()` when `""`).
    * Variant where `stop="\n"` confirms pass-through of literal newline.

* **QA gates & CI commands:**

  ```bash
  pytest -k test_chat_stop_semantics
  ```

* **Definition of Done:**

  * Tests enforce that `'' -> None` and literal `"\n"` is honored.

---

### M2-F7 — \[CODEX] Acceptance gate script refactor: keep EM-lift check, add helpful printouts

* **Goal:** Keep the **existing** acceptance gate but improve the operator readout.

* **Key changes:**

  * Edit `scripts/compare_b0_b1_m2.sh`:

    * After `compare_b0_b1.py`, run a short inline Python that:

      * Prints B1 `non_empty_rate` (from JSON) if present else computes it from `records`.
      * Prints retrieval health (P\@1/MRR) if present.
    * No change to the **EM≥+0.30** requirement.

* **Tests:**

  * Update `tests/test_compare_b0_b1.py` (optional): ensure script remains runnable (`--help` already covered in `test_utils`).

* **QA gates & CI commands:**

  ```bash
  bash scripts/compare_b0_b1_m2.sh
  ```

* **Definition of Done:**

  * Gate output includes **non-empty rate** and retrieval hints.
  * Exit code logic unchanged.

---

### M2-F8 — \[CODEX] Hopfield ablation parity micro-test (deterministic toy)

* **Goal:** Prove ablation wiring is correct even when generations are OK.

* **Key changes:**

  * New: `tests/test_hopfield_ablation_parity.py`

    * Build a tiny store with known “patterns” and a query so that Hopfield re-scoring **changes** ranks.
    * Assert:

      * With `--no-hopfield`, selection equals baseline ANN top-K.
      * With Hopfield, top-1 **differs** as expected (using the same candidates).

* **QA gates & CI commands:**

  ```bash
  pytest -k hopfield_ablation_parity
  ```

* **Definition of Done:**

  * Test demonstrates **wiring parity** and effect.

---

### M2-F9 — \[CODEX] Optional: memory token cap sweep hook (64/96/128)

* **Goal:** Add a simple **config hook** to cap total memory tokens and enable quick sweeps.

* **Key changes:**

  * Edit `src/hei_nw/pack.py` (or the call site in harness B1 path):

    * Introduce `--mem.max_tokens` (default 128) enforced after concatenation:

      > `"mem_tokens = tokens[:max_mem_tokens]"`
  * Add CLI parse in `src/hei_nw/eval/harness.py` (B1 only).

* **Tests:**

  * `tests/test_pack.py` add `test_total_memory_token_cap_enforced()`.

* **QA gates & CI commands:**

  ```bash
  pytest -k memory_token_cap
  ```

* **Definition of Done:**

  * Cap enforced; default preserves current behavior (128).

---

### M2-F10 — \[CODEX] Docs: update `quick-validate.md` to match fixes and gates

* **Goal:** Keep operator instructions in lock-step with code and scripts.

* **Key changes:**

  * Update **Section 4** to reflect `--qa.stop ''` and `--qa.max_new_tokens 16`.
  * Mention the new **non-empty gate** and optional **isolation probes**.

* **Tests:** None (docs only).

* **QA gates & CI commands:** N/A

* **Definition of Done:**

  * Doc reflects current scripts and acceptance flow.
  * PR passes CI (lint) and links to new scripts.

---

## Quick acceptance checklist for this task bundle

* [ ] `generate()` no longer blanks on leading newline with adapter path.
* [ ] `run_m2_retrieval.sh` uses **no stop** and **16 tokens**.
* [ ] New **non-empty gate** exists and passes for healthy runs.
* [ ] Reports/markdown include **Non-empty rate**.
* [ ] Isolation probes script present and executable.
* [ ] Hopfield ablation parity micro-test added.
* [ ] All tests (existing + new) pass in CI.

---

## Handy commands

```bash
# Unit tests
pytest -q

# M2 acceptance run (now robust)
bash scripts/run_m2_retrieval.sh
bash scripts/gate_non_empty_predictions.sh           # new fast-fail
bash scripts/compare_b0_b1_m2.sh                    # uplift gate

# Isolation probes (diagnostics)
bash scripts/m2_isolation_probes.sh
```
