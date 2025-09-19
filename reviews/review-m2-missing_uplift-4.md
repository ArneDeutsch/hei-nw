# M2 Critical Second-Opinion Review — HEI‑NW
*(design → impl → validation → reports)*

> Role: principal ML systems auditor. Scope: Scenario A, **B1 − B0 ≥ +0.30 EM**, with Hopfield ablation. Evidence > opinions.

---

## 1) Executive summary

- **Verdict: _M2 acceptance not met_.** B1 yields **EM=0.000** vs B0 **EM=1.000** on Scenario A (**lift = -1.000**, 95% CI -1.000…-0.852).
- **Most likely root cause:** the **adapter+memory injection corrupts decoding**: B1 produces a **fixed Markdown blob** (e.g., `•\n\n## 1`) instead of an answer.
- **Alt. hypotheses (ranked):**
  1) **Stop/length/template interaction** in **chat + `\n` stop + 8 tokens** path silently bypasses truncation.  
  2) **Memory packing** starts with **identical header tokens** (e.g., `<episodic>\nwho:…`), which the adapter **overweights**, biasing the first token.  
  3) **Hopfield readout** perturbs candidate ordering without improving content (**completion_lift −0.292**, ablation 0.000), worsening bias.
- **Smallest next action to falsify #1:** rerun B1 on N=16 with **`--qa.stop '' --qa.max_new_tokens 16`**; if EM pops back to ≈B0, the stop/length path is the proximate fault.

## 2) Acceptance criteria check

- **Plan quotes:**  
  - *Milestone*: “**Validate on Scenario A … and show B1 − B0 ≥ +30 EM**” (`planning/milestone-2-plan.md`).  
  - *Validation*: “**Pass if: (i) B1 − B0 ≥ +X (e.g., +30 EM) immediately**” (`planning/validation-plan.md`).

- **Numbers from reports:**  
  Sources:
  - `reports/m2-retrieval-stack/A_B0_metrics.json` and `A_B0_report.md`  
  - `reports/m2-retrieval-stack/A_B1_metrics.json` and `A_B1_report.md`  
  - `reports/m2-retrieval-stack/A_B1_no-hopfield_metrics.json`

| Mode | n | EM | F1 | Latency (s) | P@1 | MRR | Near-miss | Collision | Completion lift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 48 | 1.000 | 1.000 | 0.070 | – | – | – | – | – |
| B1 | 48 | 0.000 | 0.000 | 0.167 | 0.375 | 0.543 | 0.167 | 0.292 | -0.292 |
| B1 (no Hopfield) | 48 | 0.000 | 0.000 | 0.168 | 0.375 | 0.543 | 0.167 | 0.292 | 0.000 |

- **Acceptance result:** **B1−B0 = -1.00 EM** ⇒ **FAIL** (target ≥ +0.30).

## 3) Run provenance & reproducibility

- **Commands used** (`documentation/quick-validate.md` & scripts):  
  - Baseline: `bash scripts/run_b0_small.sh` (B0 across A–E, seed=7).  
  - M2 stack: `bash scripts/run_m2_retrieval.sh` → runs **B0 → B1 → B1(no-hopfield)** on Scenario A.
- **Model selection:** `scripts/run_m2_retrieval.sh` defaults to **`MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"`** → matches acceptance (**Qwen/Qwen2.5‑1.5B‑Instruct**). CI uses a tiny local model only for tests.
- **Prompting/decoding defaults (Scenario A):** `src/hei_nw/eval/harness.py` sets  
  `QAPromptSettings(prompt_style="chat", max_new_tokens=8, stop="\n", answer_hint=True)` (see None).
- **Run config recorded in JSON:** `qa: { prompt_style: "chat", max_new_tokens: 8, stop: "\n", answer_hint: true }`, `seed: 7`.
- **Mode wiring flips code paths:** B1 invokes **`RecallService`** + **adapter**; B0 omits both (see `eval/harness.py`: generate called with/without `adapter`/`mem_tokens`).

## 4) Validation pipeline audit (math & data)

- **EM/F1 math (`src/hei_nw/metrics/text.py`):**  
  - **Strict EM:** `return 1.0 if prediction.strip() == truth.strip() else 0.0`.  
  - **Relaxed EM:** canonicalize (lowercase, strip punctuation, collapse spaces) then strict.  
  - **F1:** whitespace tokens with exact-count overlap. **Micro-average** across records in `_aggregate_metrics`.
- **Stop/blank handling (`src/hei_nw/models/base.py`):** after decode, if adapter is used, **leading whitespace is stripped** before truncation; then `_truncate_at_stop(text, stop)` applies. Tests cover substring stops and leading-newline edge-cases.
- **Observed predictions (B1):** examples from `A_B1_metrics.json/records`:  
- pred="•\n\n## 1" vs truth="Fay"
- pred="•\n\n## 1" vs truth="Bob"
- pred="•\n\n## 1" vs truth="Ivan"
  → **Non-empty rate = 1.0**; failures are **wrong content**, not silence.
- **Sample size & variance:** n=48. **Newcombe 95% CI for lift** = -1.000…-0.852. Target +0.30 was statistically reachable; observed lift is decisively **negative**.
- **Scenario A generation:** synthetic partial cues with hard negatives; no leakage observed in dataset construction (`src/hei_nw/datasets/scenario_a.py`).

## 5) Design compliance audit (DG → ANN → Hopfield → memory tokens → adapter)

- **DG keyer (`src/hei_nw/keyer.py`):** k‑WTA on linear projection; retains top‑k by magnitude and **L1‑normalizes** values: “`values = values / l1.clamp_min(...)`” (lines ~66–71).
- **ANN index (`src/hei_nw/store.py`):** FAISS **inner product** with **L2 normalization** → **cosine similarity**: “`IndexFlatIP` … `normalize_L2(vectors)` / `normalize_L2(query)`” (lines ~39, 65, 92).
- **Hopfield readout:** normalized patterns + softmax attention with **temperature T** and **steps** (lines ~158–166); exposed as `--hopfield.steps`, `--hopfield.temperature` (scripts use **2** and **0.5**).
- **Token packing (`src/hei_nw/pack.py` & `recall.py`):** fields ordered **who/what/where/when**; per‑trace cap **64**; total cap **128** via `truncate_memory_tokens`. Debug shows prefix tokens **`<episodic>\nwho:…`** (`A_B1_metrics.json.debug.mem_preview`).
- **Adapter injection:** `generate()` embeds `mem_tokens`, applies **`EpisodicAdapter`** cross‑attention, then decodes (`src/hei_nw/models/base.py`: “`adapted = adapter(prompt_embeds, mem_embeds)`” → pass as `inputs_embeds`).

**Conclusion:** The stack matches the design on paper. The failure is **not** missing components; it’s **interaction** at generation.

## 6) Implementation audit (“bug safari”)

- **Flags & defaults are honored:** CLI plumbs `--qa.*`, `--mem.max_tokens`, `--no-hopfield` into the harness and JSON run_config.
- **Tokenizer/template:** for chat, code tries **`tokenizer.apply_chat_template(messages, add_generation_prompt=True)`** and falls back to a simple `SYSTEM/USER/ASSISTANT:` format if absent.
- **Stop/newline path:** `_truncate_at_stop(text, stop)` searches for the **substring** and truncates; with `stop="\n"` **any** newline should cut. Leading whitespace is stripped when adapter is used to avoid empty outputs.
- **DG/ANN invariants:** checked (no obvious off‑by‑ones; division by zero guarded).  
- **Hot-path TODO/FIXME:** none material. Only a defensive `pass` in device/dtype detection.
- **Potential sharp edges (likely relevant):**
  - **Residual strength:** `adapter` returns `H_t + attn_out` **without a learnable scale** → can **overpower** prompt tokens, causing **mode collapse** at the first token.
  - **Constant memory prefix:** all memories begin with identical tokens (`<episodic>…`) → **shared keys** dominate early attention.
  - **Tight decode budget:** **8 tokens** + `stop="\n"` makes any **formatting token** fatal; the observed `•\n` pattern fits this.
  - **Hopfield effect:** **completion_lift −0.292** vs **0.000** (ablation) shows readout **hurts** ordering/content on this setup.

## 7) Numbers, reconciled

- **B0=1.000 vs B1=0.000** reproduced from JSON. Latency roughly doubles (0.070→0.167 s).
- **Retrieval is “healthy-ish”:** `P@1=0.375`, `MRR≈0.543`, `near_miss≈0.167`, `collision≈0.292` — **not perfect**, but **shouldn’t** zero out EM given the episode is still in the prompt.
- **Why no answer then?** In B1 the **generation path** is altered: with adapter+mem, the first token becomes a **bullet/heading**; with `stop="\n"` and 8 tokens, output **never reaches** the answer token. Ablation (no Hopfield) **does not help** EM (still 0.000) and removes the (negative) completion lift — pointing to **generation bias**, not candidate recall.

## 8) Fault tree & minimal isolating experiments (≤3 tiny runs)

**Goal:** isolate whether the proximate fault is **decode policy** vs **adapter scaling** vs **memory content**.

1) **Prompt/stop sanity** (decode policy) — *expected: recover EM if policy is to blame*  
   ```bash
   python -m hei_nw.eval.harness --mode B1 --scenario A -n 16 --seed 7      --qa.prompt_style chat --qa.max_new_tokens 16 --qa.stop ''      --outdir reports/m2-debug1
   ```
   - If EM ≈ 1.0, the **`\n` stop + 8 tokens** combo is the trigger.

2) **Adapter scaling off** (injection strength) — *expected: recover EM when residual is reduced*  
   - Quick hack: set **`dropout=0.0`** (already) **and** add a scale: in `src/hei_nw/adapter.py` return `H_t + 0.2*attn_out`.  
   - Or emulate via **zero memories**: `--mem.max_tokens 0` (if allowed) or set `--dev.retrieval_only` (no generate) to verify retrieval isn’t the culprit. If you can’t pass 0, try **`--mem.max_tokens 32`**.

3) **Template swap** (chat vs plain) — *expected: recover if chat template interacts badly with inputs_embeds*  
   ```bash
   python -m hei_nw.eval.harness --mode B1 --scenario A -n 16 --seed 7      --qa.prompt_style plain --qa.max_new_tokens 16 --qa.stop ''      --outdir reports/m2-debug2
   ```

**Telltales:**  
- **If (1) fixes it:** prioritize **stop/length** policy change.  
- **If (2) fixes it, but (1) doesn’t:** **adapter scale/packing** dominates.  
- **If only (3) fixes it:** the **chat template + inputs_embeds** path is misaligned for Qwen.

## 9) Actionable fix list (ranked)

**Fast fixes (hours):**

1) **Loosen decode:** in scripts/harness, change Scenario A defaults to **`max_new_tokens=16`, `stop=None`** (file: `src/hei_nw/eval/harness.py`, Scenario‑A defaults). Tiny test: `tests/models/test_base_generate_newline.py` already guards substring stops — add a case for `stop=None`.
2) **Reduce adapter impact:** in `src/hei_nw/adapter.py` scale residual: `return H_t + 0.2*self.dropout(attn_out)` (make **0.2** a config). Tiny test: assert non‑collapse on a fixed seed.
3) **De‑prefix memories:** in `src/hei_nw/pack.py` remove the constant `<episodic>` header; start with the **answer fields**. Tiny test: `debug.mem_preview` no longer shows `<episodic>` first.
4) **Lower memory cap:** run with `--mem.max_tokens 64` (already per‑trace 64; also cap total=64). Tiny test: `debug.mem_len` ≤ 64.

**Structural fixes (days):**

5) **Add gating / layer scale** to adapter (learnable α with init 0.1).  
6) **Rework Hopfield settings**: tune `steps∈(1, 2, 3)`, `T∈[0.5,1.5]`; log **pre/post rank** deltas per query.  
7) **Regularize memory packing** with **field tags** (`who:`, `what:`) and **no decorative tokens**; add **positional enc** for memories.

## 10) Go / no-go

- **This week?** Yes, with **fast fixes (1–3)** you can likely **clear +0.30 EM** on the small set.  
- **If still blocked:** implement **adapter gating** (#5). Hopfield tuning is secondary given completion_lift≤0.

---

## Optional checklists (abridged)

- **Mode semantics:** B0 vs B1 differ **only** by `RecallService` + adapter injection; prompts identical (`eval/harness.py` prompt path).  
- **Prompting:** Qwen chat uses `apply_chat_template(..., add_generation_prompt=True)`; stop is a raw `\n` substring.  
- **Retrieval→generation handshake:** `query()`→`pack_trace()`→`mem_tokens`→`generate(adapter=…, mem_tokens=…)`; adapter is applied via `inputs_embeds` so attention **must** incorporate memories.  
- **Metrics:** EM computed on normalized text; **non‑empty** predictions counted. **Relaxed EM==EM** in aggregates; strict EM also logged.

---

## TL;DR (one page for PM)

- **Result:** M2 **fails** acceptance. **B1 EM=0.000**, **B0 EM=1.000** (lift −1.00). Retrievers look okayish; **generation collapses** under B1.  
- **Smoking guns:** fixed **`•\n\n## 1`** outputs; **8‑token** budget; `stop="\n"`; constant memory prefix; **adapter lacks gating**.  
- **Do now:** rerun **B1** with **`--qa.stop '' --qa.max_new_tokens 16`**; if fixed, **update defaults**. If not, **scale adapter residual (0.2)** and **remove memory header**.  
- **Ablation:** Hopfield currently **hurts** (completion_lift −0.292 → 0.000 when off). Don’t rely on it for uplift until tuned.  
- **Confidence:** high—numbers and file‑paths corroborate across reports and code.

---

## M2-FIX-01 — \[CODEX] Relax QA decode policy for Scenario A (defaults)

**Why**
The combo **chat + stop=`"\n"` + max\_new\_tokens=8** produces truncated, non-answers in B1 (e.g., `"•\n\n## 1"`). Loosen defaults per review.&#x20;

**Edits (surgical)**

* `src/hei_nw/eval/harness.py`: Scenario-A defaults → `QAPromptSettings(prompt_style="chat", max_new_tokens=16, stop=None, answer_hint=True)` (replace the A-specific default).
  *Existing default line shows `max_new_tokens=8, stop="\n"` for A; update to the above.*

**CLI & config**

* No new flags; keep `--qa.max_new_tokens` / `--qa.stop` override semantics.

**Tests**

* Extend `tests/models/test_base_generate_newline.py`: add case with `stop=None` to ensure no premature truncation.
* New tiny run test: B1 on A with `-n 8` returns non-empty, non-markdown tokens (smoke).

**Acceptance / Done**

* On Scenario A (N=16, seed=7): **B1 EM ≥ 0.50** and **B1−B0 ≥ +0.30** (small set).
* No degradation: B0 EM stays \~1.0 on A (±0.0–0.02).&#x20;

---

## M2-FIX-02 — \[CODEX] Add residual gate to EpisodicAdapter

**Why**
Adapter currently returns `H_t + attn_out` with **no scale**, causing first-token mode collapse. Add a learnable/configurable α.&#x20;

**Edits (surgical)**

* `src/hei_nw/adapter.py`:

  * `__init__(..., scale: float = 0.2)`; register `self.alpha = nn.Parameter(torch.tensor(scale))`.
  * Return: `H_t + self.alpha * self.dropout(attn_out)` (≤25 words).
* `src/hei_nw/models/base.py`: `build_default_adapter(model)` accepts `scale` and forwards it.

**CLI & config**

* `src/hei_nw/eval/harness.py`: add `--adapter.scale` (float, default 0.2). Pass to `build_default_adapter`.

**Tests**

* New: `tests/models/test_build_default_adapter.py` — verifies `alpha` exists and is on model device/dtype.
* New: seed-fixed generate: with α=0.0 vs α=0.5, first token distribution differs but doesn’t collapse.

**Acceptance / Done**

* With α∈\[0.1,0.3], B1 on A (N=16) **recovers EM ≥ 0.50** under chat/stop=None.&#x20;

---

## M2-FIX-03 — \[CODEX] De-prefix memory packing (remove constant header)

**Why**
All memories start with identical header tokens (e.g., “`<episodic>\nwho:`”), over-biasing the adapter’s earliest attention.&#x20;

**Edits (surgical)**

* `src/hei_nw/pack.py`:

  * Remove decorative/global header tokens; begin with field tags and values directly: `who: {name} what: {item} where: {place} when: {date}`.
* `src/hei_nw/recall.py`: ensure `pack_trace(...)` calls unchanged, just gets the new format.

**CLI & config**

* None.

**Tests**

* New: `tests/metrics/test_mem_preview.py` — `debug.mem_preview` (first 8 decoded tokens) **does not** contain `<episodic>` or Markdown bullets.
* Existing B1 debug path still logs `mem_len` and preview.

**Acceptance / Done**

* With FIX-01 defaults + this change, B1 EM improves by ≥+0.10 absolute over pre-change (N=16).&#x20;

---

## M2-FIX-04 — \[CODEX] Wire `--mem.max_tokens` end-to-end (remove hardcoded 64)

**Why**
B1 handler builds `RecallService` with **hard-coded 64**; CLI provides `--mem.max_tokens`. Align both pack and service. (Prevents over-long memories interacting with decode.)

**Edits (surgical)**

* `src/hei_nw/eval/harness.py`:

  * In B1 path, pass `max_mem_tokens=args.mem_max_tokens` when constructing `RecallService.build(...)` (replace `64`).
  * Ensure truncation call uses the **same** `args.mem_max_tokens`.

**CLI & config**

* Keep `--mem.max_tokens` default 128.

**Tests**

* New: `tests/test_harness_b1_debug.py` — for `--mem.max_tokens 32`, **all** `debug.mem_len` ≤ 32.

**Acceptance / Done**

* Sanity: mem length distribution respects the CLI; no regressions in B0.

---

## M2-FIX-05 — \[CODEX] Chat-template resilience & fallback

**Why**
Qwen chat template + `inputs_embeds` can be brittle; ensure robust fallback to plain prompts.&#x20;

**Edits (surgical)**

* `src/hei_nw/models/base.py` (`build_prompt`):

  * Add `template_policy ∈ {"auto","plain"}`; if `"plain"` or `apply_chat_template` fails, build the **plain** prompt.
* `src/hei_nw/eval/harness.py`: add `--qa.template_policy` (default `"auto"`).

**Tests**

* Extend `tests/test_harness_prompting.py`: chat vs plain produce non-empty answers; plain doesn’t include “ASSISTANT:” in final decode.

**Acceptance / Done**

* With `"plain"` + FIX-01 defaults, B1 EM on A (N=16) ≥ 0.50 if chat path still collapses.

---

## M2-FIX-06 — \[CODEX] Stop-handling robustness (trim & token-aware option)

**Why**
`stop="\n"` cut any newline; leading trim exists, but add a token-aware option and defend against markdown bullets.&#x20;

**Edits (surgical)**

* `src/hei_nw/models/base.py`:

  * New kwarg `stop_mode ∈ {"substring","none"}` (default `"none"` for Scenario A).
  * Keep current substring truncation for compatibility when requested.

**CLI & config**

* `--qa.stop_mode` (default `"none"`). Scenario A defaults → `"none"`.

**Tests**

* Extend newline tests to assert no truncation occurs with `"none"` when output begins with `•`.

**Acceptance / Done**

* No substring truncation used by default in acceptance runs; B0 untouched.

---

## M2-FIX-07 — \[CODEX] Hopfield diagnostics (pre/post rank deltas)

**Why**
Completion lift ≤ 0; need visibility into Hopfield’s net effect per query.&#x20;

**Edits (surgical)**

* `src/hei_nw/store.py` (EpisodicStore.query): include `pre_top1_group`, `post_top1_group`, and `rank_delta` in diagnostics; aggregate into metrics.

**CLI & config**

* None.

**Tests**

* `tests/metrics/test_retrieval.py`: given a synthetic case, turning on Hopfield doesn’t **worsen** top-1 when the gold key is already top-k.

**Acceptance / Done**

* Reports include per-mode `hopfield_rank_improved_rate`. No perf regressions.

---

## M2-FIX-08 — \[CODEX] Scripts: acceptance runnable with safe defaults

**Why**
Make uplift check 1-command reproducible with safer decode defaults.&#x20;

**Edits (surgical)**

* `scripts/run_m2_retrieval.sh`: set default `--qa.max_new_tokens 16 --qa.stop ''`.
* New: `scripts/run_m2_acceptance.sh`: runs B0,A then B1,A (with and without Hopfield) using the safer defaults.

**Tests**

* `tests/utils/test_scripts.py`: checks presence of new flags and model default remains `Qwen/Qwen2.5-1.5B-Instruct`.

**Acceptance / Done**

* Script emits combined report with **B1−B0 ≥ +0.30 EM** on A (small set).&#x20;

---

## M2-FIX-09 — \[CODEX] Telemetry: memory & first-token probes

**Why**
We need proof B1 no longer collapses to markdown bullets and that memories look reasonable.&#x20;

**Edits (surgical)**

* `src/hei_nw/eval/harness.py`:

  * `debug.mem_preview_str` (decode first N mem tokens).
  * `debug.first_token` (first generated token when adapter is active).

**Tests**

* Unit: ensure fields exist and are strings/tokens for B1 runs.

**Acceptance / Done**

* Debug JSON shows human-readable preview without `<episodic>`; first token is alphanumeric ≥80% of time (A, N=16).

---

## M2-FIX-10 — \[CODEX] Tiny acceptance guard (CI, N=16)

**Why**
Prevent regressions; enforce uplift in CI on a tiny set before full runs.&#x20;

**Edits (surgical)**

* New test `tests/acceptance/test_m2_uplift_tiny.py`:

  * Runs A with FIX-01 defaults (chat/stop=None, 16 tokens), computes EM.
  * **Assert B1−B0 ≥ +0.30** (seed-fixed; skip if no HF models available).

**CLI & config**

* Use `tests/models/tiny-gpt2` fallback guarded by markers when HF unavailable.

**Acceptance / Done**

* CI shows green tiny uplift before publishing reports.

---

## M2-FIX-11 — \[CODEX] Memory cap preset for A (balanced length)

**Why**
128 total tokens can still crowd the prompt; use a **lower** cap for A to reduce interference.

**Edits (surgical)**

* `src/hei_nw/eval/harness.py`: for Scenario A defaults, set `--mem.max_tokens 64` (still user-overridable).

**Tests**

* Verify `mem_len` median ≤ 64; EM not worse than 128 cap on N=16.

**Acceptance / Done**

* With this preset, B1 EM not lower than with 128; latency not higher.

---

## M2-FIX-12 — \[CODEX] Prompt literal: “no Markdown” hint

**Why**
Reinforce single-token answer behavior; avoid markdown bullets/headers in completions.&#x20;

**Edits (surgical)**

* `src/hei_nw/eval/harness.py` (Scenario A chat prompt builder):
  Append: *“Respond with only the single word (no punctuation, no Markdown).”*

**Tests**

* New prompt unit test: ensure the user message contains the added clause in chat mode when `answer_hint=True`.

**Acceptance / Done**

* First token non-punctuation rate ≥90% on A (N=16).

---

### Notes & guardrails

* **Out of scope (for now):** changing ANN metric/type, wholesale Hopfield redesign. We only instrument/tune; the negative completion-lift is tracked, not “fixed” here.&#x20;
* **Regression fences:** Always re-compute B0 on A to confirm **base-task drift ∈ \[−1, +1] EM** (from validation plan).&#x20;
