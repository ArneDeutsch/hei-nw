Here’s a numbers-first, principal-level stress-test based only on the repo + reports you provided.

# A) Artifact inventory & consistency

**Scenarios:** A–E exist under `reports/baseline/…`. M2 adds Scenario **A** runs for B0, **B1**, and **B1(no-hopfield)** under `reports/m2-retrieval-stack/…`.

**Variants present**

* **B0** (baseline) for A–E.
* **B1** (adapter + retrieval) and **B1(no-hopfield)** only for **Scenario A**.

**Sample sizes & seeds**

* Baseline A: **N=128** (`reports/baseline/A_B0_metrics.json`).
* M2 A runs: **N=48** each for B0/B1/B1(no-hopfield) (`reports/m2-retrieval-stack/*A_*_metrics.json`).
* Script `scripts/run_m2_retrieval.sh` shows defaults `N=24`, `SEED=7`; two B1 runs are invoked (with/without hopfield), so per-file N=24 would be expected. The actual **N=48** per file indicates either doubled generation or merged records—flagging a minor **reporting consistency issue**.

**Config knobs actually used (M2):**

* `--qa.prompt_style chat`, `--qa.max_new_tokens 16`, `--qa.stop ''` (empty stop), `--qa.answer_hint` on.
* Hopfield: `--hopfield.steps 2`, `--hopfield.temperature 0.5`.

**Gating rules:** A non-empty prediction gate at **≥0.90** exists (see `scripts/gate_non_empty_predictions.py` + tests). All M2 runs show **non\_empty\_rate = 1.000**, so they pass the gate.

# B) Metric sanity & definitions

**Relaxed vs strict EM:** tests confirm relaxed EM ignores punctuation/case and collapses whitespace; strict EM is exact string match (see `tests/metrics/test_em_relaxed.py` and `src/hei_nw/metrics/text.py`). F1 is token-level F1; `recall@k` is defined but n/a for Scenario A decoding. Retrieval metrics include **P\@1, MRR, near-miss, collision, completion lift** (see `src/hei_nw/metrics/retrieval.py` and tests).

**Important interpretation:** any “EM = −1” in a compare script is **uplift** (B1 − B0). Here, we compute deltas directly from per-run JSON.

# C) The numbers (Scenario A)

From:

* `reports/m2-retrieval-stack/A_B0_metrics.json`
* `reports/m2-retrieval-stack/A_B1_metrics.json`
* `reports/m2-retrieval-stack/A_B1_no-hopfield_metrics.json`

| Variant         |  N | EM\_relaxed | EM\_strict |    F1 | Non-empty | Latency(s) |  P\@1 |   MRR | Near-miss | Collision | Completion lift | ΔEM\_relaxed | ΔEM\_strict |        ΔF1 |
| --------------- | -: | ----------: | ---------: | ----: | --------: | ---------: | ----: | ----: | --------: | --------: | --------------: | -----------: | ----------: | ---------: |
| B0              | 48 |       1.000 |      1.000 | 1.000 |     1.000 |      0.070 |       |       |           |           |                 |        0.000 |       0.000 |      0.000 |
| B1              | 48 |       0.000 |      0.000 | 0.000 |     1.000 |      0.331 | 0.375 | 0.543 |     0.167 |     0.292 |      **−0.292** |   **−1.000** |  **−1.000** | **−1.000** |
| B1(no-hopfield) | 48 |       0.000 |      0.000 | 0.000 |     1.000 |      0.328 | 0.375 | 0.543 |     0.167 |     0.292 |           0.000 |   **−1.000** |  **−1.000** | **−1.000** |

Additional M2 debug (from reports):

* **Adapter latency overhead:** \~**+0.26s** (B1: +0.263s; B1-no-hopfield: +0.260s).
* **Memory token length:** fixed **128 tokens** per query (debug shows `mem_len =[128,…]`).
* **Memory preview:** tokens look like `"<episodic>\nwho:Ivan"`, i.e., packing seems active.
* **B1 predictions** are **nonsense for the task** (e.g., `"•\n\n## 1. Introduction\nIn this section the author"`) while **B0** predicts the exact name (e.g., “Ivan”). Non-empty=1.0, so they are **not empty** but **format-mismatched**.

Baseline A–E snapshot (B0 only):

* A: EM/F1 **1.000**, latency \~0.065s.
* B–E: EM/F1 **0.000** each, non-empty=1.000, latencies \~0.63–0.67s (baseline tasks are harder/different).

# D) Fast triage using metric signatures

**Signature observed:** **B0 EM ≈ 1.0**, **B1 EM ≈ 0.0**, **non-empty ≈ 1.0**, retrieval **P\@1 \~ 0.375**.

* This strongly points to a **decode/prompt/eval mismatch** rather than retrieval or memory failure.
* Corroborating evidence:

  * B1 predictions contain a **markdowny template** unrelated to the QA answer. This is consistent with a **prompt/rendering/stop or slicing bug** in the **B1 generation path**.
  * **B1 ≈ B1(no-hopfield)** (both zero EM/F1), implying the **Hopfield component is either inactive or its effect is masked** by decode issues.
  * Retrieval health is **not terrible** (P\@1=0.375, MRR=0.543; near-miss=0.167, collision=0.292), so **contexts are at least partly relevant**, yet the decoder doesn’t leverage them.
  * **Completion lift** for B1 is **negative (−0.292)** vs **0.000** with no-hopfield. That suggests **Hopfield readout hurts** the top-1 correctness relative to the baseline retrieval—but given decoding is broken, treat this signal as **secondary**.

# E) Cross-check with design & acceptance

* **Claimed mechanism of lift (M2):** DG keyer → ANN → **modern Hopfield completion** → packed memory tokens improve **short-answer adherence** and correctness.
  **Contradiction:** We see **correct answers in B0**, **irrelevant long-form markdown in B1**. That’s a **decode/formatting fault**, not an algorithmic shortfall per se.
* **Acceptance target:** **B1 − B0 ≥ +0.30 EM** on Scenario A with an ablation.
  **Observed:** **−1.00 EM**. **Fail.**
* **Config drift vs validation defaults:** tests prefer **stop="\n"** and **max\_new\_tokens=8** for short answers; M2 script uses **stop=''** and **max\_new\_tokens=16**. B0 survives this; B1 does not—again implicating **B1-specific generation**.

# F) Likely root cause(s) with evidence

1. **Implementation – B1 generation/slicing bug (most likely).**

   * Pattern: perfect B0 → zero B1 with non-empty outputs and moderate retrieval P\@1.
   * Content: B1 outputs a boilerplate “report” header instead of a single token/name—typical of **chat template/stop/slice mix-ups**.
   * Known risk: when using adapters/`inputs_embeds`, it’s easy to **fail to slice off the prompt tokens** and/or **ignore stop**. The typical tell is retrieving `output_ids[0]` rather than `output_ids[0, input_len:]` in the adapter branch, or mis-computing `input_len` when `inputs_embeds` is used.
   * **Probability: \~0.65**

2. **Evaluation / format – minor.**

   * Relaxed EM + token F1 both **0.0**; so this isn’t about punctuation/case/aliasing.
   * **Probability: \~0.10**

3. **Algorithm – retrieval/Hopfield quality – secondary at this stage.**

   * P\@1=0.375 isn’t stellar but should yield **some** decoding wins if the prompt/stop path were correct.
   * **Completion lift < 0** suggests Hopfield readout as currently wired might be detrimental, but we can’t trust it until B1 decode is fixed.
   * **Probability: \~0.25**

# G) Targeted experiments to flip the sign (minimal changes)

**Goal:** get **any** positive EM lift by making B1 decode produce the same short-answer format as B0.

1. **Fix B1 token slicing.**
   In `models/base.generate(...)`, **always** compute `input_len` from the rendered prompt ids and slice the generated sequence:
   `generated_ids = output_ids[0, input_len:]` (or equivalent) **even when using adapters/inputs\_embeds**.
   *Expected if sound:* B1 predictions collapse to single names; EM\_relaxed > 0; **ΔEM > 0** vs B0 on at least some items.
   *Failure implies:* the bug is elsewhere in prompting/stop.

2. **Restore short-answer stop semantics for Scenario A.**
   Run M2 with **`--qa.stop "\n"`** and **`--qa.max_new_tokens 8`** (as in tests), keeping `--qa.prompt_style chat`.
   *Expected if sound:* short, newline-terminated answers; **EM/F1 jump from 0.0**; latency drops.
   *If no change:* slicing still wrong or wrong prompt rendered.

3. **Cut memory tokens to a sane length.**
   Debug shows **128** memory tokens. Try **`max_mem_tokens ∈ {8,16,32}`**.
   *Expected if sound:* lower latency (+ less prompt pollution), **higher EM** once (1)–(2) are fixed.
   *If no change:* memory positioning/format may be at fault.

4. **Prompt format isolation.**
   Toggle prompt style to **`--qa.prompt_style plain`** (no chat system Preamble), with explicit instruction: “Respond with only a single word.”
   *Expected if sound:* B1 aligns to single tokens; **EM increases**.
   *If worse:* chat template was fine; look back at (1).

5. **Component ablation once decode works.**
   **B1 vs B1(no-hopfield)** with (1)–(3) applied. Then sweep **`--hopfield.steps ∈ {1,2,4}`** and **`--hopfield.temperature ∈ {0.3, 0.5, 0.7}`**.
   *Expected if sound:* **P\@1** and **completion lift** improve with Hopfield on; EM improves.
   *If completion lift stays ≤0 and EM is flat:*\* Hopfield wiring is ineffective/mis-scored → implementation issue in the readout or an algorithmic miss.

# H) Evaluation audit (quick)

* **Normalization:** already relaxed; F1=0 confirms this isn’t about case/punct/aliases.
* **Sample sizes:** M2 N=48 per run → OK for smoke; baseline A N=128. But **M2 “B0 KV cache bytes” disagree across files** (e.g., 23.27MB in baseline A vs 26.90MB in M2 reports). That suggests **prompt length differences** across runs; acceptable but note it when comparing latency/compute.
* **Seeds:** M2 script sets `SEED=7`; reports don’t record the seed explicitly—minor traceability gap.

# I) Verdict & plan

**1) Executive summary (≤10 lines)**

* B0 (Scenario A) is perfect (EM=1.000). B1 and B1(no-hopfield) are **both 0.000 EM/F1 with non-empty=1.000** and \~**+0.26s** latency overhead.
* B1 output text is **format-mismatched** (“report-like” markdown), not short answers.
* Retrieval is **moderate** (P\@1=0.375, MRR=0.543), so context isn’t the blocker.
* **Most probable fault:** **B1 generation path** (prompt rendering/stop/slicing) → **implementation issue**.
* Hopfield readout shows **negative completion lift**, but decoding must be fixed before judging algorithmic value.
* Acceptance (“B1−B0 ≥ +0.30 EM”) **not met**.

**2) Root-cause scoreboard (probabilities)**

* **Implementation: 65%** — B1 decode/slicing bug; evidence: B1 emits markdown boilerplate; B0 ok, non-empty=1.0; P\@1 moderate.
* **Algorithm: 25%** — Hopfield completion lift negative; effect masked by decode failure.
* **Evaluation: 10%** — Relaxed EM & F1 both 0 → not a normalization issue.

**3) Top 5 next actions (ordered, concrete)**

1. **Fix prompt slicing in B1 generate** (use `input_len` to slice generated ids in the adapter branch).
2. **Re-run Scenario A with** `--qa.stop "\n"`, `--qa.max_new_tokens 8` (leave other flags as is).
3. **Reduce memory token budget** to 16 (and 8/32 for sensitivity).
4. **Try plain prompt style** (`--qa.prompt_style plain`) with explicit “return a single word/name” line.
5. **Once decode OK, sweep Hopfield parameters** (`steps={1,2,4}`, `temperature={0.3,0.5,0.7}`) and compare **completion lift** and **EM uplift** vs **B1(no-hopfield)**.

**4) Risks & unknowns**

* If B1 decode fix does **not** restore short answers, there may be **prompt-template collisions** between memory packing and the chat pattern → try plain style and stricter stop.
* **Completion lift metric wiring** (exact baseline definition) is not visible; if after fixes Hopfield still hurts P\@1, inspect the **readout target and loss** used to select the completed key.
* **N inconsistencies** (M2 per-file N=48 vs script N=24) muddle reproducibility; lock N and seed in the saved JSON.

**5) What extra artifact would decide it fast**

* A short snippet (\~20 lines) from `models/base.py` covering **prompt rendering → model call → output slicing** for the **adapter path**, to validate the slicing/stop handling directly.

---

## Appendix — Scenario A table & checks

(Computed from the three M2 JSONs; see table above.)

* **Uplift:** `ΔEM_relaxed = −1.000` and `ΔF1 = −1.000` for both B1 variants.
* **Latency:** B1 adds \~**+0.26s** per query.
* **Retrieval health:** `P@1=0.375`, `MRR=0.543`, `near-miss=0.167`, `collision=0.292`.
* **Completion lift:** −0.292 (Hopfield on) vs 0.000 (Hopfield off).
* **Debug:** memory tokens fixed at **128**; preview looks reasonable (`"<episodic>\nwho:Ivan"`).

**Interpretation:** Retrieval is passable; decode is the bottleneck. Fix B1 generation (slice/stop), then right-size memory tokens and evaluate Hopfield ablation.
