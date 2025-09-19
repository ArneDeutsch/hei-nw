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
