## What’s wrong in the plans (with receipts)

1. **Acceptance hinges on “B1 − B0 ≥ +0.30 EM” without guaranteeing headroom.**

   * Milestone 2 summary demands it: “**show B1 − B0 ≥ +30 EM** on the small set.” `planning/milestone-2-plan.md` (line \~6).
   * Repeated in DoD: “**B1−B0 ≥ +30 EM** on scenario A small set.” (DoD §6) `planning/milestone-2-plan.md` (line \~295).
   * Validation plan: “**Pass if (i) B1 − B0 ≥ +X (e.g., +30 EM) immediately**.” `planning/validation-plan.md` §9.
   * Design doc also bakes this in under “**Acceptance**: immediate B1−B0 ≥ +30 EM …”. `planning/design.md`.
     **Why this is wrong now:** E0 showed **B0=1.00 EM** on Scenario A small slice, so uplift ≥+0.30 is **mathematically impossible** even if B1 is perfect.

2. **Plans don’t guard against the “B1 generation path breaks decoding” failure mode.**

   * M1 DoD has a parity guard (“**B1(with empty memory) ≈ B0**”). `planning/project-plan.md` M1 DoD.
   * **M2 DoD drops this guard.** There’s no explicit acceptance that **B1 with empty memory ≈ B0**. When the adapter path switches to `inputs_embeds`, chat-template misalignment can tank EM (as E0 showed).

3. **Small-set acceptance conflates *retrieval* quality with *generation/injection* correctness.**

   * M2 mentions “retrieval health metrics” and a Hopfield ablation, but **no oracle test** (ground-truth trace injection) and **no retrieval-only probe** to decouple components. `planning/milestone-2-plan.md` §3/§4.

4. **CI/Stats intent is vague for the “small set”, while other docs call for tight CIs.**

   * Validation plan asks for “tight CIs” but does not specify **n** for each acceptance track; M2 “small set” presently runs with **N≈24**, which cannot support “tight CI” **and** uplift confidence. `planning/validation-plan.md` §§0,9.

5. **Risk register misses the concrete “chat template × inputs\_embeds” hazard.**

   * Modern Hopfield risks are listed; ANN risks are listed; **no explicit item** about generation APIs when switching from `input_ids`→`inputs_embeds` (position IDs, prompt slicing). `planning/milestone-2-plan.md` “Risks & mitigations”.

---

## What to correct so we can actually make progress

### A) Fix the acceptance math and sequencing

* **Introduce a Headroom Gate** before any uplift check:
  “If EM\_{B0} ≥ 0.7 on the target slice/config, **do not** use uplift as acceptance. Switch to a harder baseline config or a ‘memory-dependent’ prompt.”
  (Add to `planning/validation-plan.md` §9 and `planning/milestone-2-plan.md` DoD.)

* **Split acceptance into two tiers**

  1. **Engineering acceptance (small set):** prove wiring works.

     * **B1(empty memory) ≈ B0** (±0.1 EM) — parity guard.
     * **Oracle upper bound** (inject ground-truth trace) ≥ *X* EM (e.g., ≥0.8).
     * **Retrieval-only EM** tracks P\@1 (±5 pts).
     * **Hopfield on/off**: completion-lift ≥ 0 on average.
  2. **Statistical acceptance (larger set):** measure uplift **only on a hard subset** where B0 fails but retrieval finds the right trace. Report bootstrap CI.

* **Define a “memory-dependent baseline” for uplift**
  For uplift tests, **remove the episode text / hint from B0** (or use a smaller base model), while B1 gets **the same prompt + memory tokens**. This guarantees headroom and isolates memory value.

### B) Add explicit guards against the B1 decoding trap

* **Keep the M1 parity guard in M2 DoD**: B1 with empty memory must match B0 on Scenario A.
* **Add decoding sanity checks** to small-set acceptance:

  * ≥90% predictions begin with an alphabetic token;
  * No leading `<`, `•`, or “Human:” tokens;
  * Non-empty rate = 1.00.

### C) Specify sample sizes and CIs

* **Small-set (engineering)**: N≈24–64 just to trip the wiring checks. No uplift target here.
* **Statistical set**: N≥200–500 items for Scenario A, with paired bootstrap **95% CI** on EM lift. Put the exact n in `planning/validation-plan.md` §0 and §9.

### D) Make retrieval vs generation disentanglement mandatory

* **Add E0–E5 probes to acceptance** (at least E0, E1, E2, E3).
* **Document expected artifacts** (JSON fields, plots) and pass/fail interpretations in the plans.

---

## Proposed **\[CODEX]** tasks to adapt the plans (and a few tiny code helpers)

> These are lightweight edits that align the plan with reality **and** give us deterministic signals that HEI-NW’s current slice works.

1. **\[CODEX] Update M2 Acceptance to Headroom-Aware**

   * Edit: `planning/milestone-2-plan.md` (§1 summary, §6 DoD).
   * Replace “**B1 − B0 ≥ +0.30 EM on small set**” with:

     * **Engineering acceptance (small set):** (i) B1(empty)≈B0; (ii) Oracle EM ≥ 0.8; (iii) Retrieval-only EM ≈ P\@1; (iv) Hopfield completion-lift ≥ 0.
     * **Statistical acceptance (hard subset):** If **Headroom Gate** passes (EM\_{B0}<0.7), require **B1−B0 ≥ +0.30 EM** with 95% CI excluding 0.
   * Add a **Headroom Gate** box: if failed, instruct to switch to memory-dependent baseline for uplift.

2. **\[CODEX] Add Headroom Gate to Validation Plan**

   * Edit: `planning/validation-plan.md` §9 “Interpreting outcomes”.
   * Add a decision table: **Headroom?** → *yes*: run uplift; *no*: fall back to memory-dependent baseline or evaluate absolute EM\_{B1} + oracle.

3. **\[CODEX] Restore Parity Guard in M2 DoD**

   * Edit: `planning/milestone-2-plan.md` §6.
   * Add line: “**B1 with empty memory (no mem\_tokens) must match B0 within ±0.1 EM** on Scenario A.”

4. **\[CODEX] Add E0–E3 probes to plans + scripts**

   * Edit: `planning/milestone-2-plan.md` §3 tasks; `planning/project-plan.md` M2 section.
   * Add a table listing **E0/E1/E2/E3** with command, artifact path, and pass/fail.
   * Code: add `scripts/m2_isolation_probes.sh` invoking harness flags (`--dev.oracle_trace`, `--dev.retrieval_only`, `--no-hopfield`).
   * Minimal tests: sanity “files exist” test in `tests/utils/test_scripts.py`.

5. **\[CODEX] Define Memory-Dependent Baseline for Uplift**

   * Edit: `planning/validation-plan.md` §1/§9 and `planning/milestone-2-plan.md` §5 config defaults.
   * Specify **B0 prompt** without episode/hint (or use a smaller base model), and **B1 prompt** identical to B0 + memory tokens.
   * Code helper (optional): add `--qa.memory_dependent_baseline` switch in harness to auto-toggle prompt ingredients.

6. **\[CODEX] Specify Sample Sizes & Bootstrap CI**

   * Edit: `planning/validation-plan.md` §0, §9; `planning/design.md` “Acceptance”.
   * Record **n\_small = 24–64**, **n\_stat ≥ 200–500**; mandate bootstrap with 1000 resamples for EM lift.
   * Code helper: add `--metrics.bootstrap 1000` to harness and write `*_lift_ci` fields.

7. **\[CODEX] Add Decoding Sanity to Acceptance**

   * Edit: `planning/milestone-2-plan.md` DoD.
   * Add checks: first-token charset, no HTML bullets/headers, non-empty rate.
   * Code helper: emit `debug.first_token`, `debug.prefix_strip_applied` booleans in metrics JSON.

8. **\[CODEX] Add a Risk & Mitigation item for `inputs_embeds`**

   * Edit: `planning/milestone-2-plan.md` “Risks”.
   * Risk: **chat-template × inputs\_embeds misalignment** (prompt slicing, position IDs).
   * Mitigations: (i) M1 parity guard reused in M2; (ii) **plain** template fallback; (iii) explicit `position_ids`; (iv) unconditional prefix slicing by prompt length.

9. **\[CODEX] Clarify “tight CIs” vs small-set objectives**

   * Edit: `planning/validation-plan.md` §0.
   * State explicitly: **small-set** = engineering checks (no CI claim), **stat-set** = CI targets.

10. **\[CODEX] Update the one-shot M2 script & compare tool**

    * Edit: `scripts/run_m2_retrieval.sh`, `scripts/compare_b0_b1_m2.sh`.
    * Script prints **Headroom Gate status**, runs E0/E1/E2/E3, and only runs uplift compare if headroom passes.
    * Compare script: accepts `--hard-subset` file of item IDs (B0 wrong, retrieval correct) for fair uplift.

---

## Are M2 Definitions of Done reasonable?

**As written: flawed.** They’re not achievable on the current Scenario A “small set” because:

* They require **uplift on a slice where B0 saturates** (E0 showed B0=1.00 EM).
* They **omit the parity guard** (B1 empty ≈ B0), letting a decoding bug zero out EM while all retrieval metrics look healthy.
* They **don’t separate component correctness** (oracle, retrieval-only) from end-to-end uplift; result: you can’t localize failure quickly.

**Revised M2 DoD (concise):**

1. **Parity**: B1 with empty memory ≈ B0 (±0.1 EM).
2. **Retrieval health**: P\@1/MRR finite; near-miss/collision logged; completion-lift ≥ 0 (Hopfield on vs off).
3. **Oracle upper bound**: B1 with ground-truth trace EM ≥ 0.8 on small set.
4. **Retrieval-only probe**: EM tracks P\@1 within ±5 pts.
5. **Decoding sanity**: first-token checks pass; non-empty rate = 1.00; prefix strip applied on ≥95% generations.
6. **Uplift (only if headroom)**: On **hard subset**, **B1−B0 ≥ +0.30 EM** with 95% CI excluding 0.

This keeps M2 laser-focused on proving the **retrieval stack + memory injection** actually helps (and doesn’t break the model), and it gives us deterministic, low-run-count levers to isolate issues.

---

### TL;DR

* The plans hard-require uplift where **uplift is impossible** (baseline saturates).
* M2 DoD lacks a critical **parity guard** and **component-level probes**.
* Apply the \[CODEX] edits above to make acceptance **headroom-aware**, add **oracle/retrieval-only** checks, specify **sample sizes/CI**, and include **decoding sanity**.
  That will let us *actually* test whether HEI-NW works—and debug it fast when it doesn’t.

---

## \[CODEX-1] Fix B1 decoding (root cause)

**Goal.** Make B1’s `inputs_embeds` path decode cleanly (no chat boilerplate / HTML), returning just the answer.

**Edits.**

* `src/hei_nw/models/base.py :: generate`

  * **Unconditional prompt slicing when using `inputs_embeds`:** after `output_ids = _model.generate(...)`, replace the equality-guarded strip with:
    *“If adapter+mem\_tokens and len(generated\_ids) > prompt\_len, slice `generated_ids = generated_ids[prompt_len:]`.”*
  * **Add explicit `position_ids` when using `inputs_embeds`:**

    ```python
    if adapter is not None and mem_tokens:
        attn = inputs["attention_mask"]
        position_ids = attn.cumsum(dim=1) - 1
        position_ids.masked_fill_(attn.eq(0), 0)
        gen_input = {
            "inputs_embeds": adapted,
            "attention_mask": attn,
            "position_ids": position_ids,
        }
    ```
  * **Tiny debug:** return `{"prefix_stripped": True|False}` for visibility (doesn’t affect harness).

**Acceptance.**

```bash
PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 4 --seed 7 \
  --model Qwen/Qwen2.5-1.5B-Instruct --outdir /tmp/m2/decoding_fix \
  --qa.prompt_style chat --qa.answer_hint --qa.max_new_tokens 16 --qa.stop ''
jq '.debug.first_token' /tmp/m2/decoding_fix/A_B1_metrics.json | head
```

**Pass if:** first tokens are alphabetic names (no leading `<`, `•`, “Human:”); `aggregate.non_empty_rate == 1.0`.

---

## \[CODEX-2] Add “plain template” escape hatch for B1

**Goal.** Provide a one-flag way to bypass chat-template quirks during B1 while we lock in the decoding fix.

**Edits.**

* No code change needed (already supported): `--qa.template_policy plain`.
* Update script default for M2 runs:

  * `scripts/run_m2_retrieval.sh` → add `--qa.template_policy plain` on **B1** invocations only.

**Acceptance.**

```bash
bash scripts/run_m2_retrieval.sh
jq '.aggregate.non_empty_rate' reports/m2-retrieval-stack/A_B1_metrics.json
```

**Pass if:** `non_empty_rate == 1.0` and predictions no longer contain boilerplate.

---

## \[CODEX-3] Headroom gate + memory-dependent baseline

**Goal.** Ensure uplift is only checked when there is headroom; otherwise use a baseline that needs memory.

**Edits.**

* `src/hei_nw/eval/harness.py`

  * **QAPromptSettings**: add `omit_episode: bool = False`.
  * **CLI**: add `--qa.omit_episode` (BooleanOptionalAction).
  * **\_build\_prompt(...)**: if `omit_episode`, pass an empty episode (it already renders as “(none)”).
* `scripts/compare_b0_b1_m2.sh`

  * Implement **Headroom Gate**: read B0 EM; if `EM_B0 >= 0.70`, **skip uplift failure** and print “Headroom gate triggered”.
* New script `scripts/run_m2_uplift_headroom.sh`

  * B0: `--qa.omit_episode` (and keep answer hint).
  * B1: same prompt + memory + `--qa.template_policy plain`.

**Acceptance.**

```bash
# Headroom-aware path
bash scripts/run_m2_uplift_headroom.sh
bash scripts/compare_b0_b1_m2.sh
```

**Pass if:** script prints either (a) “Headroom gate triggered” **or** (b) uplift computed on the memory-dependent baseline (no error exit).

---

## \[CODEX-4] Isolation probes (E0–E3) as a one-shot

**Goal.** Bake the fast probes into a single dev entry point to localize failures in minutes.

**Edits.**

* New `scripts/m2_isolation_probes.sh`:

  * **E0**: tiny B0/B1 (n=4).
  * **E1**: `--dev.oracle_trace`.
  * **E2**: `--dev.retrieval_only`.
  * **E3**: `--no-hopfield`.
  * Print a compact table and exit non-zero only on wiring failures (e.g., empty outputs).
* CI: add a job “M2 isolation probes (tiny model)” calling this script with `MODEL=tests/models/tiny-gpt2`.

**Acceptance.**

```bash
bash scripts/m2_isolation_probes.sh
```

**Pass if:** the script prints 4 sections with metrics; **no non-empty gate failure**.

---

## \[CODEX-5] Bootstrap CI for EM lift (hard subset)

**Goal.** Quantify uplift with uncertainty when headroom exists.

**Edits.**

* New `scripts/compute_lift_ci.py`:

  * Input: B0/B1 metrics JSON.
  * Compute per-item EM and paired bootstrap (≥1000) for **lift**.
  * Optional `--hard-subset` filter file (item IDs where B0 wrong & retrieval top1 correct).
* CI step: run this script only when Headroom Gate passes.

**Acceptance.**

```bash
python scripts/compute_lift_ci.py reports/m2-retrieval-stack/A_B0_metrics.json \
  reports/m2-retrieval-stack/A_B1_metrics.json --resamples 1000
```

**Pass if:** prints `lift_mean`, `ci_low`, `ci_high` without error.

---

## \[CODEX-6] Parity guard (B1 empty ≈ B0)

**Goal.** Ensure adapter path doesn’t degrade generation when memory is empty.

**Edits.**

* New script `scripts/run_parity_guard.sh`:

  * B0: default Scenario A.
  * B1: `--mem.max_tokens 0` (forces empty memory).
  * Use `scripts/compare_b0_b1.py --threshold 0.1`.
* CI: insert before M2 smoke.

**Acceptance.**

```bash
bash scripts/run_parity_guard.sh
```

**Pass if:** compare script exits **0** (EM/F1 deltas ≤ 0.1).

---

## \[CODEX-7] Unit test for prefix stripping in `inputs_embeds`

**Goal.** Lock the decoding fix with a deterministic test that doesn’t need a big model.

**Edits.**

* `tests/models/test_base_generate_strip.py` (new):

  * Monkeypatch `_model.generate` to return `torch.cat([input_ids[0], new_ids])` when `inputs_embeds` present.
  * Assert: with adapter+mem, `generate(...).text` decodes **only** `new_ids` (no prompt echo).
  * Assert: `prefix_stripped` flag is **True**.

**Acceptance.**

```bash
pytest -q tests/models/test_base_generate_strip.py
```

**Pass if:** green.

---

## \[CODEX-8] Guardrails: first-token & non-empty gates

**Goal.** Fail fast when decoding regresses.

**Edits.**

* Extend `scripts/gate_non_empty_predictions.py` to also gate on **invalid first tokens** (e.g., `<`, `•`, “Human:”).
* Add to `scripts/run_m2_retrieval.sh` after B1 runs.

**Acceptance.**

```bash
bash scripts/run_m2_retrieval.sh
bash scripts/gate_non_empty_predictions.sh reports/m2-retrieval-stack/A_B1_metrics.json
```

**Pass if:** gate passes.

---

## \[CODEX-9] One-button M2 acceptance runner

**Goal.** Provide a single command that produces all artifacts and enforces the updated DoD.

**Edits.**

* New `scripts/m2_acceptance.sh`:

  1. **Parity guard** (CODEX-6).
  2. **Isolation probes** (CODEX-4).
  3. **Headroom check** on regular config; if no headroom → print notice & exit **0**.
  4. **Memory-dependent uplift run** (CODEX-3) + **bootstrap CI** (CODEX-5).
  5. Emit summary (`reports/m2-acceptance/summary.md`) with: B0/B1 EM, uplift, CI, retrieval health, Hopfield ablation.

**Acceptance.**

```bash
bash scripts/m2_acceptance.sh
```

**Pass if:** summary generated and script exits **0** (or non-zero only when an explicit DoD guard fails).

---

## \[CODEX-10] (Optional) k-sweep harness for retrieval diagnostics

**Goal.** Make it trivial to see whether DG k impacts retrieval vs EM (to avoid chasing the wrong fix).

**Edits.**

* New `scripts/m2_k_sweep.sh`:

  ```bash
  for k in 16 32 64 128; do
    PYTHONPATH=src python -m hei_nw.eval.harness --mode B1 --scenario A -n 64 \
      --seed 7 --outdir "reports/m2-k/$k" --dg.k $k --qa.template_policy plain
  done
  ```
* Plot MRR vs EM in a tiny Python snippet (optional).

**Acceptance.**

```bash
bash scripts/m2_k_sweep.sh
```

**Pass if:** runs complete; reports contain retrieval metrics per-k.

---

# What “done” looks like for Milestone 2 (post-implementation)

* ✅ **Decoding fixed**: B1 no longer emits chat boilerplate; non-empty = 1.00; first-token gate passes.
* ✅ **Parity guard holds**: B1 (empty memory) ≈ B0 (≤0.1 deltas).
* ✅ **Isolation probes pass**: E0–E3 run without gates failing; E1 (oracle) shows **high EM**; E2 tracks P\@1.
* ✅ **Headroom respected**: uplift only enforced when B0 < 0.70 (or on the memory-dependent baseline).
* ✅ **Uplift demonstrated (when eligible)**: on the hard subset or memory-dependent baseline, **B1 − B0 ≥ +0.30 EM** with 95% CI excluding 0.
* ✅ **Artifacts**: `reports/m2-acceptance/summary.md` + per-run metrics JSONs are produced and checked into CI.

These tasks, in this order, will (1) remove the B1 decoding blocker, (2) make uplift measurable and fair, and (3) give you one-shot, reproducible acceptance for Milestone 2.

