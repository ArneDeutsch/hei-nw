Here’s a single, copy-paste **meta-prompt** you can give to ChatGPT to drive a deep, adversarial root-cause investigation into the missing **+0.30 EM** uplift. It pushes for rigorous reasoning across **design, implementation, and validation** and culminates in a concrete review document + a stepwise path to isolate the fault.

---

# 🔎 Meta-Prompt: Adversarial Root-Cause Review of Missing +0.30 EM (HEI-NW — M2)

You are a **principal ML systems reviewer**. Treat this as a **red-team audit**. Assume **nothing** is correct: the **design**, the **implementation**, and the **validation** pipeline may all be flawed. Your mandate is to either (a) **prove** the precise reason(s) we did **not** achieve **B1 − B0 ≥ +0.30 EM** on Scenario A, **or** (b) produce a **step-by-step plan** that deterministically finds the root cause with minimal additional runs.

## Scope & constraints

* Repository ZIP is available in this chat: `hei-nw-main.zip`. Unzip and work with its contents.
* reviews/milestone-2-review.md is provided in addition with the review of the latest milestone that detects the missing uplift.
* The data for this check was produced by following `documentation/quick-validate.md`. **Do not trust** that this implies correctness—**verify** it.
* Prefer **evidence** (file paths, function names, CLI flags, short excerpts ≤25 words) over opinion.
* Avoid long code pastes; point to exact lines/defs and summarize.
* Output must be a single Markdown file called **`review-m2-missing-uplift.md`** with the structure below.

## Deliverable structure (write exactly this doc)

1. **Executive summary** (max 10 bullets)

   * One-line problem statement; observed EM results (B0, B1); whether required **+0.30 EM** was met.
   * Top 3–5 most likely root causes (ranked).
2. **Evidence ledger**

   * **Repo map (short)**: key modules you inspected (paths only).
   * **Planning anchors**: headings/sections you matched in `planning/*.md`.
   * **Run provenance**: commands, configs, model IDs, seeds, dataset slice; prove they match `quick-validate.md`.
3. **Validation audit** (challenge everything)

   * **Dataset & labels**: size, class balance, label noise checks, leakage checks; confirm Scenario A construction.
   * **Metrics math**: derive EM/F1 computation from code; check off-by-one, micro/macro averaging, masking, casing, normalization.
   * **Sample size & variance**: bootstrap CI for EM lift; show if the +0.30 target is statistically reachable with current n.
   * **Prompting & truncation**: verify template, stop sequences, max tokens, context length; detect silent truncation or BOS/EOS mishandling.
   * **Mode semantics**: prove B0 vs B1 pathways differ only by retrieval injection; confirm flags actually flip code paths.
4. **Design audit** (DG Keyer → ANN → Modern Hopfield → memory tokens → adapter)

   * **DG Keyer**: k-WTA correctness; actual `k`, sparsity, normalization; ablate with k∈{32,64,128}; check for gradient/dropout interaction during eval.
   * **ANN**: index type/metric; recall\@K upper bound; candidate set size; ID alignment with store records; check for dim mismatch/cosine vs dot.
   * **Hopfield**: iterations, β/temperature, convergence/energy monotonicity; verify read-only; does it **improve** candidate rank? show pre/post rank deltas.
   * **Token packing**: verify selected traces → memory tokens; max token cap; ordering; collisions; serialization bugs.
   * **Adapter wiring**: show where `mem_tokens` are injected; verify the model actually **conditions** on them (not ignored).
5. **Implementation audit** (bug safari)

   * Search for stub patterns: `TODO|FIXME|NotImplementedError|pass` in critical paths.
   * Config fallbacks/miswiring: wrong defaults, ignored CLI flags, device/dtype fallbacks (e.g., fp16→fp32), quantization mismatches.
   * Seed control; nondeterminism (torch, numpy, cuBLAS, FAISS).
   * I/O shape/type checks; off-by-one in top-K; wrong tokenizer/model pair; instruction vs base model mix-up.
6. **Isolation experiments (fast to run)**

   * **E0 (sanity)**: B0 vs B1 on **1–2 samples** with verbose tracing; prove `mem_tokens` present in model input.
   * **E1 (oracle retrieval upper bound)**: If ground-truth trace is provided to the adapter (bypassing ANN/Hopfield), what EM do we get?
   * **E2 (retrieval only)**: Replace model generation with a trivial head using retrieved text; does EM rise? If yes, injection path is suspect.
   * **E3 (Hopfield ablation)**: With/without Hopfield; compare `completion_lift`, pre/post ranks.
   * **E4 (k sweep)**: k∈{16,32,64,128}; plot P\@1/MRR vs EM lift.
   * **E5 (prompt template swap)**: minimal changes (system/user separation, few-shot off/on) to detect prompting brittleness.
   * For each, specify exact command(s), expected artifacts, and pass/fail interpretation.
7. **Findings**

   * Table of **Hypothesis → Evidence → Verdict (supports/refutes)** for at least **12** distinct hypotheses across **validation**, **design**, and **implementation**.
8. **Root-cause ranking & decision**

   * Ranked list with confidence; name the **single most probable** cause and the **minimum fix**.
9. **Actionable plan (≤10 steps)**

   * Concrete code/config changes; tests to add; one-shot acceptance command producing artifacts that prove the fix.
10. **Appendices**

* **A. Commands**: exact shell invocations to reproduce everything.
* **B. Artifacts**: required output paths (JSON/MD/PNGs).
* **C. Checklists**:

  * Validation checklist (dataset, metrics, seeds, prompt).
  * Design checklist (DG/ANN/Hopfield/adapter).
  * Implementation checklist (wiring, flags, CI).

## Method (how to proceed)

1. **Unpack & map the repo**

   * Unzip `hei-nw-main.zip` to `./hei-nw-main/`. Build a short tree. Identify modules likely implementing: **DG keyer**, **ANN**, **Hopfield**, **recall/memory tokens**, **adapter**, **harness/eval**, **metrics**, **scripts**.
   * Locate `documentation/quick-validate.md`. Extract the **exact commands** and **model identifiers** (e.g., Qwen 2.5 1.5B Instruct vs tiny). Check for optional flags controlling mode (`B0/B1`), Hopfield, k, context length, and seeds.
2. **Prove the run actually used the intended model & mode**

   * Pull model name from logs/config; verify tokenizer aligns with model; check any quantization flags (4-bit/8-bit) and their impact on max tokens.
   * Show **B1** code path was engaged: find where `mem_tokens` are created and added to the model input; confirm the path executed (log lines or saved JSON).
3. **Instrument retrieval quality vs generation quality**

   * Extract **retrieval diagnostics**: P\@1, MRR, near-miss, collision, candidate K, Hopfield completion lift. If missing, add lightweight logging and rerun a **tiny subset**.
   * Show **pre→post** Hopfield rank changes and whether the selected trace matches ground truth.
4. **Audit metrics & acceptance math**

   * Derive the EM calculation from code: exact normalization, casing/whitespace rules, answer extraction. Recompute EM manually on 5–10 samples to catch parsing errors.
   * Bootstrap EM lift (≥1000 resamples) to estimate CI; determine if **+0.30** is even statistically plausible for the dataset size used by quick-validate.
5. **Form hypotheses & test cheaply**

   * Create at least **12** hypotheses spanning validation/design/implementation (e.g., “adapter ignores memory tokens”, “Hopfield temperature too high”, “k too small → low recall”, “prompt truncation”, “metric bug”).
   * For each, map **one command** and **one artifact** that would confirm/refute it—favor **E0–E5** experiments above.
6. **Conclude with a minimal fix**

   * If a root cause is found, propose the **smallest diff** (config or code) plus a **single command** that yields artifacts proving **B1 − B0 ≥ +0.30 EM** on Scenario A, using the model specified by `quick-validate.md`.

## Review heuristics & tripwires (explicitly check)

* **Silent fallbacks**: wrong CLI flag names; defaults overriding supplied params; CPU/FP32 fallback; tokenizer mismatch; instruction vs base model confusion.
* **k-WTA correctness**: exactly **k** non-zeros; L1 normalization; stability under different seeds.
* **ANN**: metric mismatch (cosine vs L2/IP); index trained vs not; ID alignment; dimensionality checks.
* **Hopfield**: β/temperature/iterations too low (no refinement) or too high (collapse); energy non-monotonic; pattern matrix shape bugs.
* **Memory tokens**: ordering reversed; truncation by max prompt length; exceeding token cap; wrong special tokens.
* **Adapter wiring**: tokens appended but not attended (e.g., attention mask zeros); position IDs wrong; caching interfering.
* **Validation**: dataset too small; label canonicalization mismatch; case/punctuation stripping inconsistency.
* **Repro**: non-fixed seeds; mixed precision instability; FAISS nondeterminism; padding masks affecting logits.

## Output requirements

* Provide **`review-m2-missing-uplift.md`** as your only output.
* Every claim must cite a **file path** and, when helpful, a **line range** or **function/class name**.
* Include **commands** and **artifact paths** that a human can run to reproduce your findings on a dev machine (GPU where needed).

---

### (Optional) One-shot command skeletons the review may use

* B0 vs B1 quick check (tiny slice):

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --scenario A --mode B0 -n 32 --seed 0 --outdir /tmp/m2/B0
  PYTHONPATH=src python -m hei_nw.eval.harness --scenario A --mode B1 -n 32 --seed 0 --outdir /tmp/m2/B1
  ```
* Hopfield ablation:

  ```bash
  PYTHONPATH=src python -m hei_nw.eval.harness --scenario A --mode B1 --no-hopfield -n 64 --outdir /tmp/m2/B1_noH
  ```
* k sweep:

  ```bash
  for k in 16 32 64 128; do
    PYTHONPATH=src python -m hei_nw.eval.harness --scenario A --mode B1 --dg.k $k -n 64 --outdir /tmp/m2/k$k
  done
  ```

---

**End of meta-prompt.**

