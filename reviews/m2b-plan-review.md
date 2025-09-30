# M2b — Plan Review: “Schema-Agnostic Retrieval Hardening”

### 1) Executive summary

**Verdict (short): *Proceed with M2b, with a few guardrails and one fix***. The plan addresses the real blockers from M2: weak/unstable keys, missing multi-channel features, under-powered acceptance, and insufficient telemetry. The proposed **schema-agnostic feature extractor (T1)** plus **multi-channel DG keyer with tunable sparsity (T2)** and **ANN/Hopfield refresh (T3)** are the right levers to lift P@1 where you need it (A≥0.6 mini; B/C≥0.3 @ n≥200). The acceptance runner/doc updates (T4–T6) close the “underpowered” loophole that hurt M2.

**On your “is this a workaround?” worry:** updating scenario generators to emit **standardized metadata** (e.g., `entity_slots`, `config_dims`, timestamps) is not a scoring hack; it’s establishing a **neutral, minimal contract** that prevents scenario-specific branching in the extractor. It’s aligned with the design (pointer-only traces; who/what/where/when slots; §5.10 telemetry) and with **AGENTS.md’s** “avoid dataset-specific workarounds.” The key is to strictly forbid using answers/labels or raw text in these metadata (already enforced in `pack.py`/`store.py`), which the plan implicitly respects.

If you ship M2b with the safeguards below, it is **likely** to produce the uplift and retrieval health needed to close M2 and to validate the design’s retrieval path honestly.

---

### 2) What’s strong (and why it maps to the root causes)

1. **T1 — Schema-agnostic feature bundles (text + slots + numeric channels)**

* Removes scenario-specific conditionals in the keyer path and concentrates task variance into the features themselves.
* Fits the design’s pointer-only discipline (`pack_trace`/`TraceWriter` ban raw text; normalized slots).
* Provides an explicit seam to add **timestamps/novelty counters** without leaking labels.

2. **T2 — Multi-channel DG keyer & k-WTA controls**

* Directly targets **pattern separation** and collision reduction—the main technical reason for poor P@1 in M2.
* Per-channel projections + per-channel k-WTA before concat is the right mechanism to combine **textual context features** with **structured slots** and **numeric signals** without hand-tuning for each scenario.
* Storing small learned projections as repo checkpoints replaces the stop-gap hashed embeddings with something stable and reproducible.

3. **T3 — ANN/Hopfield refresh + collision telemetry**

* Channel-aware dense views + explicit `collision_rate` / `near_miss_rate` and **Hopfield lift** make failures diagnosable (M2 lacked this).
* Retuning HNSW (`M`, `efConstruction`, `efSearch`) under the new key distribution is necessary; doing it with telemetry is good practice.

4. **T4–T6 — Large-sample manifests, powered acceptance, doc/report sync**

* Hard enforcement of **n ≥ 200** for B/C acceptance prevents the “underpowered acceptance” mistake flagged earlier.
* CI smoke + human large-sample runs is the right split: keeps PRs fast while still backing the statistical claims.

---

### 3) The “is this a workaround?” question about T1

> “Update scenario generators (A/B/C) to output standardized metadata… Isn’t that a workaround to make scenarios similar?”

**No—provided you enforce these constraints (which the repo largely already does):**

* **No label/answer leakage**: never include `expected`, `answers`, or any derived synonym in `entity_slots`/`extras`/numeric channels.
* **Pointer-only traces**: keep raw `episode_text` out of the trace payload; pass **spans** (`{doc,start,end}`) only. (Already enforced by `pack_trace`/`TraceWriter`.)
* **Stable, scenario-intrinsic counters only**: `novelty_counters`, `timestamps`, `config_dims` must be derived from generator state (e.g., index/time) **not** model output or ground-truth answers.
* **One extractor, zero `if scenario == …`**: the extractor consumes a **uniform schema**; any scenario branching inside it is disallowed.

With those in place, the metadata standardization is **algorithm-facing, not data-massaging**. It stops you from writing three one-off extractors (which would be the real workaround).

---

### 4) Gaps & risks (actionable)

1. **Scenario B gate features are still pseudo-random.**
   In `scenario_b.py` gate signals (`surprise`, `novelty`) are sampled from ranges. That’s fine for **gate calibration tests (M3)**, but it adds noise when you use gate-filtered writes in M2 experiments. **Recommendation:** for M2 acceptance runs, rely on **truth-label writes** (or T1 features → keyer) rather than gate decisions; keep gate randomness contained to M3. Make this explicit in the acceptance runner.

2. **Label fallback pathway must be disabled for acceptance.**
   The harness supports “label fallback” (index positives even if the gate blocks). That’s a useful dev mode, but acceptance for M2 should **fail** if keys/retrieval aren’t strong enough without fallback. **Recommendation:** acceptance script sets `ALLOW_LABEL_FALLBACK=0` and asserts it in the report header.

3. **Collision thresholds: pick them conservatively and alarm.**
   The plan mentions scenario collision ≤0.15/≤0.20 on mini-sets. Good. Add a hard **alarm** in the runner (mark `UNDERPOWERED`-style notices) when collision_rate exceeds the bound, to prevent “quiet regressions.”

4. **Document the keyer checkpoints and determinism.**
   Storing projection weights is correct; also document the **seed/initialization** so keys are reproducible across machines. Include a quick “checksum print” in the acceptance logs.

5. **Hopfield lift must be measured against a plain KNN baseline** under the new key distribution.
   Keep a report row “P@1 (KNN)” vs “P@1 (+Hopfield)” and require **lift > 0**; otherwise you may be hiding a keyer issue behind completion.

6. **Scenario-specific anythings inside the extractor = fail.**
   Add a tiny test that walks the extractor code object and **asserts no string literal ‘scenario_’ appears**. It’s a blunt guard, but it catches the worst drift.

---

### 5) Will this likely get us over the M2 bar?

**Yes, if you execute T1–T3 cleanly and enforce T4–T6:**

* Moving from hashed toy embeddings to **learned, multi-channel sparse keys** should improve P@1 materially on A and meaningfully on B/C.
* Channel weighting + per-channel k-WTA reduces cross-scenario interference without making scenarios “look the same.”
* Powered acceptance with **n ≥ 200** plus collision telemetry stops you from “passing by noise.”

Given your prior P@1≈0.29 on the small set and near-zero oracle lift, these changes target the exact failure modes.

---

### 6) Anti-workaround guardrails to include in M2b

* **Extractor contract test:**

  * `test_scenarios_share_schema`: A/B/C records all map to the same `FeatureBundle` types/shapes.
  * `test_no_label_leakage`: extractor rejects records containing `expected`/`answers` in slots/extras.
* **Pointer-only write test:** a failing test if any trace includes `episode_text`/`raw_text` etc.
* **No scenario branching test:** CI fails if the extractor module contains `scenario_` string literals.
* **Acceptance invariants:**

  * `ALLOW_LABEL_FALLBACK` **off**.
  * Enforce `n ≥ 200` for B/C (block otherwise).
  * Report `collision_rate`, `near_miss_rate`, `hopfield_lift` and fail on regression thresholds.
* **Reproducibility:** print keyer checkpoint hash + seed in the acceptance summary.

---

### 7) Decision & next steps

* **Decision:** ✅ **Go** on M2b as written, with the guardrails above.
* **Must-dos before calling M2 closed:**

  1. Land T1–T3 together (they’re coupled).
  2. Run the powered acceptance (B/C @ n≥200) with label-fallback **disabled**.
  3. Require **A mini P@1 ≥ 0.6** and **B/C P@1 ≥ 0.3**, **Hopfield lift > 0**, and **collision_rate ≤ bound**.
  4. Cut an artifacted report (`reports/m2b-eval/…`) and link manifests/seed/checkpoints.

---

### 8) Appendix — specific notes from repo inspection

* **Design alignment:** pointer-only traces + normalized `entity_slots` in `pack.py`/`store.py` already enforce the right boundary; T1 is consistent with this.
* **Scenario A** uses `make_gate_features` derived from recall probability and similarity (good, deterministic).
* **Scenario B** uses random gate spans (ok for M3; keep out of M2 acceptance).
* **Scenario C** already emits `novelty_counters` and annotated reward/pin flags; these are the sort of **neutral numeric channels** the extractor should consume.
* **Hashed embeddings** in the harness are a **fallback**; M2b correctly de-prioritizes them in favor of learned DG keys.
* **Docs** (`documentation/hei-nw.md`) explicitly call out fixing keyer instead of scenario hacks—M2b is consistent with this stance.

---

## TL;DR

Your intuition is right to be wary of “making scenarios similar.” In this plan, the **standardization is on the interface**, not the content. With the guardrails above, **M2b is sound and non-hacky**, and it is **likely** to produce the P@1 and acceptance you need to close M2 and move on to consolidation.
