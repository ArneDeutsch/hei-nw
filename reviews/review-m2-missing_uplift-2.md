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
