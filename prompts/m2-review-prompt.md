# 🔍 Critical Second-Opinion Review of M2 (design → impl → validation → reports)

You are a **principal ML systems auditor**. Your job is to **challenge every assumption** in HEI-NW’s Milestone 2 and produce a concrete, evidence-backed review that explains **why the expected EM uplift is still missing**, what the numbers *actually* say, and what to fix **now**.

## Scope & ground rules

* You have a repository ZIP named **`hei-nw-main.zip`** in this chat. **Unzip** it and work locally.
* Treat prior outputs—including `reviews/review-m2-missing_uplift.md` and the quick-validate run—as **claims to be verified**, not facts.
* Focus on M2 scope per:

  * `planning/milestone-2-plan.md`, `planning/design.md`, `planning/project-plan.md`, `planning/validation-plan.md`
  * Acceptance target: **Scenario A**, **B1 − B0 ≥ +0.30 EM** with a Hopfield ablation included.
* Evidence over opinion. Prefer **file paths, function names, CLI flags, config keys, and ≤25-word code excerpts**. Avoid long pastes—summarize and point to lines.
* Don’t edit files under `research/`, `planning/`, `prompts/`, `reviews/`, `models/` (see `AGENTS.md`). You may quote them.

## What to read and cross-check first

1. **Planning anchors**

   * `planning/milestone-2-plan.md`: required components, acceptance check.
   * `planning/validation-plan.md`: metrics definitions, mode matrix (B0/B1/B2/B3), Scenario A construction.
   * `planning/design.md`: DG keyer, ANN, Modern Hopfield, memory-token packing, adapter injection, defaults (§ refs).
   * `planning/project-plan.md`: success criteria & dependencies.

2. **Implementation surfaces (likely hot paths)**

   * `src/hei_nw/`: `store.py` (EpisodicStore, ANN index, Hopfield readout), `pack.py` (memory tokens), `adapters/` (injection site), `metrics/text.py` (exact\_match/f1), `models/base.py` (loader/generate), `harness/` (mode wiring & prompts).
   * `scripts/`: `run_b0_small.sh`, `run_m2_retrieval.sh`, `run_m2_retrieval_ci.sh`, `compare_b0_b1_m2.sh`.

3. **Evidence from reports**

   * `reports/baseline/` → `A_B0_metrics.json`, `A_B0_report.md` (and combined report).
   * `reports/m2-retrieval-stack/` → `A_B0_*`, `A_B1_*`, `A_B1_no-hopfield_*`, `combined_report.md`, `completion_ablation.png`.

4. **Run recipe used**

   * `documentation/quick-validate.md` (commands, model choice, prompts, stop sequences, `max_new_tokens`, seeds).

> Claim currently visible in the repo outputs (verify, don’t trust):
>
> * **Scenario A**: `A_B0_report.md` shows **EM=1.000**; `A_B1_report.md` and `A_B1_no-hopfield_report.md` show **EM=0.000** (F1=0, empty predictions), with retrieval diagnostics like **P\@1≈0.375, MRR≈0.543**, **near-miss≈0.167**, **collision≈0.292**, and **completion\_lift ≤ 0**.
> * This implies **negative uplift** (B1 worse than B0). Your job is to confirm the numbers and explain *why*.

## Deliverable (single Markdown file)

Write **`reviews/review-m2-critical-second-opinion.md`** with these sections:

1. **Executive summary (≤ 10 bullets)**

   * Your verdict: “M2 acceptance met / not met”, with a one-sentence *why*.
   * The single most likely root cause and the top 3 alternative hypotheses.
   * The smallest next action to falsify the top hypothesis.

2. **Acceptance criteria check**

   * Quote the acceptance from `milestone-2-plan.md` and `validation-plan.md`.
   * Extract **B0 vs B1 EM** (and F1) from the reports (file paths). Compute **B1−B0** and say if **≥ +0.30**.
   * Include a tiny table: mode, n, EM, F1, latency, and (for B1) P\@1/MRR/near-miss/collision/completion\_lift.

3. **Run provenance & reproducibility**

   * Show the exact **commands/scripts/configs** used (from `quick-validate.md` and shell scripts).
   * Confirm the **model** used matches acceptance (**`Qwen/Qwen2.5-1.5B-Instruct`** for M2; CI script is *not* acceptance).
   * Verify **prompt template**, **stop sequence** (`"\n"`), **`max_new_tokens=8`**, seeds, Scenario A selection, and mode flag actually flip code paths.

4. **Validation pipeline audit (math & data)**

   * **Metrics math**: derive how EM/F1 is computed (`metrics/text.py`). Check casing/whitespace/normalization, micro vs macro, off-by-ones, masking of blanks.
   * **Empty/blank outputs**: explain why predictions are empty in B1 (if confirmed). Is it the stop token, truncation, template, or adapter?
   * **Sample size & variance**: compute/approximate a **95% CI** for EM lift; determine whether +0.30 was statistically reachable.
   * **Scenario A construction**: confirm partial-cue generation, confounders, and leakage checks as per plan.

5. **Design compliance audit (DG → ANN → Hopfield → memory tokens → adapter)**

   * **DG keyer**: k-WTA behavior (`k`, sparsity, norms).
   * **ANN index**: metric (cosine vs dot/IP), vector normalization, recall\@K upper bound, candidate count.
   * **Hopfield**: β/temperature, iterations, convergence/energy delta; **pre/post rank change** on candidates; ablation parity.
   * **Token packing**: memory token cap (looks like 128), ordering, collisions, serialization correctness.
   * **Adapter injection**: prove `mem_tokens` are **actually** fed to the model and **attended** (not ignored).

6. **Implementation audit (“bug safari”)**

   * Search for `TODO|FIXME|pass|NotImplementedError` in hot paths.
   * Check config defaults & CLI parsing: wrong fallbacks, ignored flags, device/dtype flips (fp16↔fp32), quantization or chat-template mismatch.
   * Verify tokenizer/model pair and exact chat template for Qwen Instruct; confirm stop conditions and BOS/EOS handling.
   * Check top-K boundaries, shape/dtype invariants, off-by-one in candidate filtering.

7. **Numbers, reconciled**

   * Reproduce the **B0=1.000 vs B1=0.000** discrepancy from the JSONs; explain why retrieval health (P\@1≈0.375, MRR≈0.543) did **not** translate to answers.
   * Inspect a few **concrete cases** (IDs, prompts, predictions). Are B1 generations empty due to `max_new_tokens`, early stop, or wrong role formatting?
   * Interpret **completion\_lift ≤ 0** with/without Hopfield; what does that say about the DG/ANN/Hopfield stack?

8. **Fault tree & minimal isolating experiments**

   * Provide a ranked fault tree with **yes/no branches** to isolate the defect in ≤3 tiny runs (e.g., N=16).
   * Example probes (adapt to findings):

     * **Prompt sanity**: same question to the base model via `generate()` path used in B1 (no memory) → Is output non-empty?
     * **Stop token**: set `stop=None`, `max_new_tokens=16` → Do answers appear?
     * **Injection sanity**: feed **only** memory tokens + direct cue → Does output degrade?
     * **Template swap**: use plain prompt vs chat template → Does B1 recover?

9. **Actionable fix list (ranked)**

   * For each fix: what to change (file/line/flag), expected delta, risk, and a tiny test to guard it.
   * Separate **“fast fixes”** (prompt/stop/length/wiring) from **“structural”** (index metric, Hopfield β/iters, adapter attention).

10. **Go/no-go**

* State whether M2 can pass **this week** with fast fixes. If not, what structural change is mandatory?

## Style & output constraints

* Keep the review **focused, evidenced, and adversarial**; nothing is assumed correct without proof.
* Use **short quotes (≤25 words)** with file paths/line refs and **bullet-first writing**.
* End with a **one-page TL;DR** a PM could read.

## Optional helper checklists (use if helpful)

* **Mode semantics**: prove B0 vs B1 differ *only* by memory injection.
* **Prompting**: confirm chat roles/fields for Qwen Instruct; verify system/user turns, stop sequences, truncation, and `max_new_tokens`.
* **Retrieval→generation handshake**: show the path from `query()` to `mem_tokens` to `generate()`, and verify attention actually incorporates the tokens (not silently dropped).
* **Metrics**: guard against “empty string counts as wrong” confounds; compute EM on normalized answers and compare to strict EM.
