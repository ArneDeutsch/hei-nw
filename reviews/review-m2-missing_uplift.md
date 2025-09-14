# review-m2-missing\_uplift.md

## 1) Executive summary

* **Problem.** Scenario A shows **no EM improvement**: `B0.em=0.00`, `B1.em=0.00` in committed M2 artifacts, so **B1 − B0 = 0.00 < +0.30** required uplift (see `reports/m2-retrieval-stack/A_B0_metrics.json`, `A_B1_metrics.json`).
* **Most-likely root causes (ranked):**

  1. **Prompting / model-format mismatch.** The harness sends a raw continuation prompt (no chat template, no answer constraint) to an **instruction-tuned** model, causing long, off-format completions (often non-English) and near-zero EM. Evidence: `_build_prompt()` concatenates `episode + cue` without any instruction; `generate()` uses the generic **text-generation** pipeline with no chat template and no `stop` (see `src/hei_nw/eval/harness.py::_build_prompt`, `src/hei_nw/models/base.py::load_base`, `::generate`).
  2. **EM metric brittleness.** `exact_match()` is strict string equality (only `.strip()`), so any punctuation/casing/extra words yield EM=0 even for semantically correct answers (see `src/hei_nw/metrics/text.py::exact_match`).
  3. **Hopfield provides no rank lift.** `retrieval.completion_lift == 0.0` and `p@1=0.375` indicate Hopfield isn’t improving candidate ordering; default `steps=1` and `temperature=1.0` may be too weak (see `src/hei_nw/store.py::HopfieldReadout` and B1 JSON).
  4. **ANN metric mismatch.** Index uses **IP** with **L2-normalized queries only**; stored vectors are **not normalized**. This biases scores by database vector norms and can degrade recall quality (see `src/hei_nw/store.py::ANNIndex.search`, `EpisodicStore.from_records`).
  5. **Answer packing → memory tokens OK but under-used by the model.** Tokens are built (`pack_trace`) and injected through the adapter, but the base prompt never asks to *use* episodic details nor to answer tersely; the adapter’s effect is swamped by generation format. Injection is proven to affect logits (unit test), but not aligned to the task (see `src/hei_nw/pack.py`, `src/hei_nw/recall.py`, `tests/models/test_base_generate.py::test_mem_tokens_affect_generation`).

> Prior M2 review already flagged the **missing +0.30 EM**; that review’s evidence references 0.00 EM and absent CI. In this repo snapshot, **CI exists** (`.github/workflows/ci.yml`), but the **EM gap remains**.&#x20;

---

## 2) Evidence ledger

### Repo map (short)

* **Core path:** `src/hei_nw/keyer.py` (DG k-WTA), `src/hei_nw/store.py` (ANN + Hopfield + store), `src/hei_nw/recall.py` (Recall → mem tokens), `src/hei_nw/pack.py` (trace→tokens), `src/hei_nw/adapter.py` (cross-attn), `src/hei_nw/models/base.py` (load/generate, adapter wiring), `src/hei_nw/eval/harness.py` (B0/B1 eval), `src/hei_nw/metrics/*` (text & retrieval), `src/hei_nw/datasets/scenario_a.py`.
* **Scripts & artifacts:** `scripts/run_m2_retrieval.sh`, `scripts/run_m2_retrieval_ci.sh`, `scripts/compare_b0_b1_m2.sh`; reports under `reports/m2-retrieval-stack/`.
* **CI:** `.github/workflows/ci.yml`.

### Planning anchors checked

* `planning/milestone-2-plan.md` (M2 scope & acceptance).
* `planning/validation-plan.md` (Scenario A, metrics, modes).
* `planning/design.md` (§ DG Keyer, store/Hopfield, defaults).

### Run provenance (committed)

* **Commands.** `scripts/run_m2_retrieval.sh` runs `B0`, `B1`, and `B1 --no-hopfield` on **Scenario A** with `MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"`, `N=24`, `SEED=7`.
* **Harness defaults.** If `--model` omitted, `DEFAULT_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"` (see `src/hei_nw/eval/harness.py`, near “DEFAULT\_MODEL\_ID”).
* **Reports.** This repo’s `reports/m2-retrieval-stack/*.json` show `aggregate.em=0.0` for both B0 and B1; `retrieval` block exists in B1.
* **Quick-validate.** `documentation/quick-validate.md` instructs running baseline and M2 stack; the steps match the scripts present (file has ellipses but references M2 artifacts and acceptance report paths).

---

## 3) Validation audit (challenge everything)

**Dataset & labels.**

* **Construction.** `src/hei_nw/datasets/scenario_a.py::generate` builds `(episode_text, cues[0], answers[who,what,where,when])` with **hard negatives** and `lag` bins; each record has `group_id` and `should_remember`.
* **Size/balance.** Default `N=24` primaries and `confounders_ratio=1.0` → **48 records** (confirmed by `A_B0_metrics.json`/`A_B1_metrics.json` record counts).
* **Leakage.** No cross-group reuse of episodes; negatives come from same or similar templates but with edits to single fields (name/item/place/day), which is intended “near-miss” pressure, not leakage.
* **Label noise.** Synthetic—deterministic ground truth; no noise expected.

**Metrics math & brittleness.**

* **EM.** `exact_match(pred, truth)` is strict equality after `.strip()` (no case or punctuation normalization) → extremely brittle for instruction models that return sentences (`src/hei_nw/metrics/text.py`).
* **F1.** Token-level F1 uses whitespace tokenization only; still OK for quick checks but sensitive to punctuation.
* **Recall\@k.** Implemented for text baseline in `baselines/rag.py` and used for reporting; retrieval P\@k/MRR implemented separately (`src/hei_nw/metrics/retrieval.py`).
* **Off-by-one / averaging.** Aggregation summarizes per-record metrics via `_aggregate_metrics` in the harness; no micro/macro issues surfaced in code; bins are explicit in `eval/report.py`.

**Sample size & variance.**

* With **24 primary prompts**, the minimum step in EM is **\~0.042**. Hitting **+0.30** requires ≥8 exact matches where baseline had 0.
* Observed **P\@1≈0.375** in B1 retrieval (top-1 correct group \~9/24). Under an **oracle** generation mapping, **max EM ≈ 0.375**—*barely above* the +0.30 target and statistically fragile (SE ≈ 0.099; 95% CI \~ 0.375±0.19). Thus even perfect generation on top-1 may not robustly clear +0.30 at N=24; increasing N improves power.

**Prompting & truncation.**

* **Template.** `_build_prompt()` returns `f"{episode}\n{cue}\n"`, no system/user roles, no “answer with one word,” no English constraint (`src/hei_nw/eval/harness.py`).
* **Generation.** `generate()` uses greedy decoding (`do_sample=False`, `temperature=1.0`, `top_p=1.0`, `top_k=0`), `max_new_tokens=32`, and **no `stop`** (`src/hei_nw/models/base.py`).
* **Observed pathology.** B1 records show multi-token outputs including non-English boilerplate (e.g., Chinese preambles), ensuring EM=0 while F1 also collapses (see `reports/m2-retrieval-stack/A_B1_metrics.json` “prediction” fields).

**Mode semantics & flagging.**

* **B0 vs B1 difference.** B1 constructs `RecallService`, queries store, **packs traces → mem\_tokens**, builds `EpisodicAdapter`, and calls `_evaluate_records(..., adapter, mem_tokens)`; B0 calls `_evaluate_records` without adapter/tokens.
* **Hopfield switch.** `--no-hopfield` flips `use_hopfield=False` in the B1 flow; the ablation PNG is written by `eval/report.py::save_completion_ablation_plot`.
* **Adapter timing.** B1 logs `adapter_latency_overhead_s` (B1 latency minus B0) to prove adapter path executed.

---

## 4) Design audit (DG Keyer → ANN → Hopfield → tokens → adapter)

**DG Keyer.**

* **Correctness.** `DGKeyer.__init__(d=2048,k=64)`; forward: mean-pool → linear → **k-WTA by |q|** → **L1-normalize retained values** (docstring); tests assert **exact k non-zeros** and **L1=1** (`tests/test_keyer.py`).
* **Ablation-ready.** `k` is a ctor arg; harness doesn’t expose `--dg.k`, but small change can enable sweep.

**ANN.**

* **Index.** `faiss.IndexFlatIP` (**inner product**). Before search, **query is L2-normalized**; **database vectors are not** (see `ANNIndex.search`). This effectively ranks by `cosine(query, db) * ||db||`—a known pitfall.
* **K.** `top_k_candidates=64` default; `return_m=4`.
* **IDs & meta.** `EpisodicStore.from_records` aligns `meta[i]` with `vectors[i]`; the `trace` payload carries `answers` & `episode_text`.

**Hopfield.**

* **Parameters.** Defaults `steps=1`, `temperature=1.0`. Update: `z←softmax((z·Mᵀ)/T)·M` with per-step L2 norm; `return_scores=True` returns attention over candidates.
* **Observed.** `completion_lift=0.0` in B1 JSON: Hopfield **does not improve** top-1 vs ANN under current settings.

**Token packing.**

* `pack_trace()` serializes **who/what/where/when** in a fixed template delimited by `<episodic> ... </episodic>`; capped to `max_mem_tokens` per trace and **128 total** (`src/hei_nw/pack.py`, `src/hei_nw/recall.py`).
* Ordering: top candidates first; truncation after 128 tokens.

**Adapter wiring.**

* **Injection site.** `generate()` embeds `mem_tokens`, computes adapter cross-attn (`EpisodicAdapter`) with prompt embeddings, and calls model with **inputs\_embeds** (not token IDs).
* **Effect exists.** Unit test shows outputs differ when `mem_tokens` are present (`tests/models/test_base_generate.py::test_mem_tokens_affect_generation`), so the path isn’t a no-op.
* **But** the base prompt never *asks to use memory* or to produce a short canonical answer, so usefulness doesn’t translate to EM.

---

## 5) Implementation audit (bug safari)

* **Stubs / TODOs.** None by `scripts/grep_no_stubs.sh`; tests enforce this.
* **CLI flags honored.** `--no-hopfield` switches code path; `--model` overrides default; `--baseline` handled for LC/RAG comparisons.
* **Device/dtype.** `build_default_adapter()` infers and moves adapter to model device/dtype; defensive fallbacks included.
* **Quant defaults.** Harness calls `load_base(..., quant_4bit=False)` to avoid bitsandbytes fragility; OK for robustness, costs VRAM.
* **Seeds.** `set_global_seed(seed)` sets `random`, `numpy`, `torch`; cudnn/cuBLAS nondeterminism not fully guarded (acceptable for eval).
* **Tokenizer/model pair.** **Critical:** instruction model (`Qwen2.5-Instruct`) is used with **plain text continuation**, **no chat template** and **no stop/format constraint** → catastrophic for EM.
* **ANN metric mismatch.** IP + normalize query only is a subtle bug that can distort nearest neighbors.
* **Top-K off-by-one.** None observed; `top_k_candidates` and slice logic are straightforward.
* **Masking / position IDs.** Adapter operates in embedding space pre-model; `attention_mask` passed from prompt; memory isn’t part of the mask (cross-attn happens inside adapter only). No obvious zero-masking bug.

---

## 6) Isolation experiments (fast to run)

> All commands assume: `export PYTHONPATH=src`

**E0 — Sanity (prove mem\_tokens in path)**

* **Cmds:**

  ```bash
  # 2 samples, dump B1 report
  python -m hei_nw.eval.harness --scenario A --mode B1 -n 2 --seed 0 \
    --model Qwen/Qwen2.5-1.5B-Instruct --outdir /tmp/m2/E0
  ```
* **Artifacts:** `/tmp/m2/E0/A_B1_metrics.json` with `adapter_latency_overhead_s` present; check any “prediction” ≠ B0 (seed-matched) on same records.
* **Pass/Fail:** If predictions identical to B0 across both items (despite adapter), suspect tokens not constructed; otherwise path is active.

**E1 — Oracle retrieval upper bound**

* **Idea:** Bypass ANN/Hopfield; feed the **ground-truth trace** as the *only* memory.
* **Cmd:** (add a small harness flag `--oracle-trace` or temporary patch: in B1 loop set `res_h["selected"]=[truth_trace]` when `group_id` matches).
* **Artifacts:** `/tmp/m2/E1/A_B1_oracle.json` summary.
* **Interpretation:** EM≈P\@1 upper bound. If EM jumps ≫0 with oracle memory, **injection is fine** and the issue is **retrieval or prompting**.

**E2 — Retrieval-only head (no LLM)**

* **Idea:** Return `answers[0]` from retrieved top-1 trace directly.
* **Cmd:** add `--retrieval-only` to harness (15-line change) that writes predictions from `selected[0]["answers"][0]`.
* **Artifacts:** `/tmp/m2/E2/A_B1_retonly.json`.
* **Interpretation:** If EM≈P\@1 here but EM≈0 with LLM, **generation formatting** is to blame.

**E3 — Hopfield ablation/sweep**

* **Cmds:**

  ```bash
  python -m hei_nw.eval.harness --scenario A --mode B1 -n 64 --seed 0 --outdir /tmp/m2/H_no --no-hopfield
  python -m hei_nw.eval.harness --scenario A --mode B1 -n 64 --seed 0 --outdir /tmp/m2/H_yes
  ```

  Optional param sweep (tiny edit or env) for `steps∈{1,2,4}`, `T∈{1.0,0.5,0.25}`.
* **Artifacts:** `completion_ablation.png`, metrics JSONs.
* **Interpretation:** If `completion_lift` stays ≈0 and EM unchanged, Hopfield is neutral; focus on prompting/EM metric.

**E4 — k sweep (DG keyer sparsity)**

* **Cmd (expose `--dg.k` in harness or via env; quick loop):**

  ```bash
  for k in 16 32 64 128; do
    python -m hei_nw.eval.harness --scenario A --mode B1 -n 64 --seed 0 \
      --model Qwen/Qwen2.5-1.5B-Instruct --outdir /tmp/m2/k$k  # with dg.k wired to $k
  done
  ```
* **Artifacts:** Metrics per `k`.
* **Interpretation:** Rising P\@1/MRR with moderate `k` suggests better keys; if flat, ANN/Hopfield dominates.

**E5 — Prompt template swap**

* **Cmds:** after implementing **chat template + concise answer directive** (see §9), run:

  ```bash
  python -m hei_nw.eval.harness --scenario A --mode B0 -n 32 --seed 7 --outdir /tmp/m2/P_B0
  python -m hei_nw.eval.harness --scenario A --mode B1 -n 32 --seed 7 --outdir /tmp/m2/P_B1
  ```
* **Artifacts:** `A_B0_metrics.json`, `A_B1_metrics.json`.
* **Interpretation:** If EM lift jumps (even with same retrieval), **root cause confirmed**: prompting/metric mismatch.

---

## 7) Findings (Hypothesis → Evidence → Verdict)

1. **H:** Instruction-tuned model mis-prompted (no chat format / no “short answer”).
   **E:** `_build_prompt()` returns only `episode + cue`; `generate()` uses **text-generation** without chat template/stop; predictions are long/non-English; EM=0.
   **V:** **Supports (strong)**.

2. **H:** EM metric too strict; near-correct answers counted wrong.
   **E:** `exact_match()` strict equality; no normalization; F1 also collapses due to extra tokens.
   **V:** **Supports**—explains zero EM even with plausible content.

3. **H:** Memory tokens not injected.
   **E:** `generate()` passes `inputs_embeds=adapter(prompt_embeds, mem_embeds)`; test shows mem changes output.
   **V:** **Refutes** (injection path exists).

4. **H:** Adapter ignored by model due to masking/positions.
   **E:** Adapter operates on embeddings; `attention_mask` applied to model only; unit test shows effect.
   **V:** **Refutes**.

5. **H:** ANN metric mismatch lowers candidate quality.
   **E:** IP index with **query-only normalization**; database vectors unnormalized.
   **V:** **Supports (moderate)**.

6. **H:** Hopfield too weak to reorder candidates.
   **E:** `steps=1`, `T=1.0`; `completion_lift=0.0`.
   **V:** **Supports**.

7. **H:** k-WTA sparsity too tight/loose.
   **E:** Defaults `k=64` not swept; unknown sensitivity.
   **V:** **Inconclusive** (needs E4).

8. **H:** Dataset construction wrong / label leakage.
   **E:** Synthetic generator with explicit `group_id` and `should_remember`; hard negatives but no leakage.
   **V:** **Refutes**.

9. **H:** Seed/nondeterminism hides gains.
   **E:** Seeds fixed; cuBLAS nondet possible but unlikely to zero out EM systematically.
   **V:** **Refutes** (weak).

10. **H:** Wrong model actually used.
    **E:** Scripts default to `Qwen/Qwen2.5-1.5B-Instruct`; harness default matches; reports show non-English text consistent with Qwen defaults under raw continuation.
    **V:** **Refutes** wrt identity; **supports** mis-prompt explanation.

11. **H:** Token packing bug (empty or wrong fields).
    **E:** `pack_trace()` packs ordered fields; `RecallService` slices answers correctly; 128-token global cap.
    **V:** **Refutes**.

12. **H:** CI missing allowed regressions.
    **E:** CI exists (`.github/workflows/ci.yml`). Prior review’s “no CI” is out-of-date.&#x20;
    **V:** **Refutes**.

---

## 8) Root-cause ranking & decision

1. **Primary cause:** **Prompting / format mismatch + brittle EM.**

* The model is **instruction-tuned** but receives raw continuation input; outputs are verbose and often non-English, making **`exact_match` virtually impossible**.
  **Minimum fix:** Apply the model’s **chat template** (via tokenizer) and add a **concise answer instruction** + **`stop`** to isolate the first token(s). Normalize EM.

2. **Secondary contributors:**

* **Hopfield neutral** under current params; try `steps>1` and lower `T`.
* **ANN normalization mismatch**; use true cosine or normalize database vectors on add.

**Decision:** Implement the **prompting fix + EM normalization** first; this alone should unlock **≥ +0.30** on N≈32 given observed P\@1. Then tune Hopfield/ANN if needed.

---

## 9) Actionable plan (≤10 steps)

1. **Make B0/B1 prompts instruction-friendly.** In `src/hei_nw/eval/harness.py::_build_prompt`, switch to **chat template**:

   * Use `tok.apply_chat_template([...], add_generation_prompt=True)` (via `models/base.generate`), or add a simple instruction header:
     `"[SYSTEM] Answer the question using the episode. Reply with ONLY the single correct word/name.\n[EPISODE] ...\n[QUESTION] ...\n[ANSWER]"`.
2. **Constrain outputs.** In `models/base.generate`, pass `stop="\n"` and `max_new_tokens=4` when a QA prompt is detected (B0/B1); keep greedy decoding.
3. **Normalize EM.** Replace `exact_match()` with canonicalization: lowercase; strip punctuation; collapse whitespace. Keep current exact compare for a “strict EM” secondary metric.
4. **Add a debug toggle.** In B1 path, log `len(mem_tokens)` and the **first 8 decoded mem tokens** to the JSON summary (e.g., `summary["debug"]["mem_len"]`, `["mem_preview"]`).
5. **Fix ANN scoring.** Either: (a) **normalize database vectors on add** (`faiss.normalize_L2(vec_array)`) to make IP=cosine; or (b) switch to `IndexFlatL2` and L2-normalize both sides.
6. **Strengthen Hopfield.** Expose CLI for `--hopfield.steps` and `--hopfield.T`; set defaults to `steps=2`, `T=0.5`.
7. **Expose DG `k`.** Add `--dg.k` to harness to enable quick sweeps.
8. **Small N acceptance run.** Re-run Scenario A with `n=32`, `seed=7` to check uplift.
9. **Lock baseline.** Keep current B0 prompt **identical** except for chat template (same instruction), to ensure the only difference is memory injection.
10. **One-shot acceptance command:**

    ```bash
    export PYTHONPATH=src
    bash scripts/run_m2_retrieval.sh         # now uses improved prompt & metrics
    bash scripts/compare_b0_b1_m2.sh         # prints: EM lift >= 0.30
    ```

---

## 10) Appendices

### A. Commands (repro & checks)

```bash
# Inspect current artifacts
jq '.aggregate.em' reports/m2-retrieval-stack/A_B0_metrics.json
jq '.aggregate.em,.retrieval' reports/m2-retrieval-stack/A_B1_metrics.json

# Tiny sanity (CPU ok)
python -m hei_nw.eval.harness --mode B0 --scenario A -n 4 --seed 0 \
  --model tests/models/tiny-gpt2 --outdir /tmp/m2/T_B0
python -m hei_nw.eval.harness --mode B1 --scenario A -n 4 --seed 0 \
  --model tests/models/tiny-gpt2 --outdir /tmp/m2/T_B1

# Qwen run (GPU)
python -m hei_nw.eval.harness --mode B0 --scenario A -n 32 --seed 7 \
  --model Qwen/Qwen2.5-1.5B-Instruct --outdir /tmp/m2/Q_B0
python -m hei_nw.eval.harness --mode B1 --scenario A -n 32 --seed 7 \
  --model Qwen/Qwen2.5-1.5B-Instruct --outdir /tmp/m2/Q_B1
python -m hei_nw.eval.harness --mode B1 --scenario A -n 32 --seed 7 \
  --model Qwen/Qwen2.5-1.5B-Instruct --outdir /tmp/m2/Q_B1_noH --no-hopfield

# Compare (after implementing fixes)
bash scripts/compare_b0_b1_m2.sh
```

### B. Artifacts

* `reports/m2-retrieval-stack/A_B0_metrics.json` — baseline.
* `reports/m2-retrieval-stack/A_B1_metrics.json` — with memory.
* `reports/m2-retrieval-stack/A_B1_no-hopfield_metrics.json` — ablation.
* `reports/m2-retrieval-stack/completion_ablation.png` — Hopfield plot.
* Temporary: `/tmp/m2/*` — per-experiment outputs.

### C. Checklists

**Validation checklist**

* [ ] Prompt uses chat template or explicit instruction.
* [ ] `stop` configured; `max_new_tokens` small for QA.
* [ ] EM canonicalization applied; strict-EM reported separately.
* [ ] Seeds set; N≥32 for Scenario A acceptance.
* [ ] Reports include retrieval block and (new) `debug.mem_len`.

**Design checklist**

* [ ] DG `k` exposed and swept.
* [ ] ANN cosine or normalized IP; candidate K adequate.
* [ ] Hopfield `steps,T` tuned; lift > 0.
* [ ] Token packing order/limits verified; no collisions.
* [ ] Adapter path active; latency overhead reported.

**Implementation checklist**

* [ ] No silent flag ignores; CLI validated.
* [ ] No stubs; tests cover harness/report.
* [ ] Model/tokenizer pair correct; chat prompt path tested.
* [ ] CI runs unit tests and tiny M2 smoke; artifacts uploaded.

---

### Notes against prior review

* The earlier milestone review correctly called out **missing +0.30 EM**. CI **is present** in this repo snapshot (`.github/workflows/ci.yml`), so that part of the earlier report is out-of-date; the **uplift gap remains** and is traced above to **prompting/metric** issues.&#x20;

**End of report.**
