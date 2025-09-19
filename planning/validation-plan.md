# 0) What “success” means (at a glance)

You want to see ALL of these, each with tight CIs:

* **Tracks:**
  * **Engineering (small set):** `n_small ≈ 24–64` items. Focus on wiring (parity, oracle, retrieval-only, decoding sanity). Report point estimates; no CI promises here.
  * **Statistical (hard subset):** `n_stat ≥ 200–500` items where B0 fails but retrieval succeeds. Run only when the **Headroom Gate** (EM\_{B0} < 0.7) passes and report paired bootstrap 95% CIs (≥1000 resamples) on EM lift.

* **One-shot recall:** After a single exposure, the model answers correctly **immediately** *with memory enabled* (partial cues allowed).
  Metrics: partial-cue EM/F1, recall\@k, and latency.&#x20;
* **Consolidation:** After a replay cycle, the model answers **with memory OFF** (knowledge distilled into weights), and original skills don’t regress.
  Metrics: pre→post deltas, retention curves, “no-drift” on base tasks.&#x20;
* **Efficiency:** For the same quality, total attention FLOPs and KV-cache footprint drop vs long-context baselines.&#x20;

# 1) Test harness (four switches)

Run every scenario under these modes:

* **B0**: Base model only.
* **B1**: Base + episodic memory (adapter + store) — **no replay**.
* **B2**: Base + memory **after replay** (consolidation ON).
* **B3**: Base **after replay with memory OFF** (tests corticalization).

When the Headroom Gate blocks uplift, rerun B0 with the **memory-dependent baseline** prompt (`--qa.memory_dependent_baseline` toggle): strip the episodic hint (or use the smaller base model) while B1 keeps the identical prompt plus memory tokens.

Also include long-context and RAG baselines for fairness.&#x20;

# 2) Scenarios you need (and how to build them)

## A. Episodic one-shot with partial cues (core for HEI-NW)

Generate single-exposure mini-stories with who/what/where/when. Query with **partial cues** (“who at where?”, “what at when?”), possibly after distractors.
Measure EM/F1, recall\@k, robustness vs number of distractors, latency. Expect: B1≫B0 immediately; after replay, B3 ≈ B1.&#x20;

**Fixture template (synthetic):**

```
Episode: “On 2025-09-10, Dana left a red backpack at Café Lumen.”
Cues:    [“who left a backpack at Café Lumen?”, “what did Dana leave?”]
Distractors: 50 similar episodes w/ different names/items/places.
```

This targets pattern separation (DG) and completion (CA3).

## B. Factual “hot-patch” (override prior)

Inject a single update that contradicts prior parametric knowledge (“X now CEO=Y”). Test immediate B1 (should succeed), then B3 after replay (the weight-level answer updates) while unrelated facts remain stable. Track contradiction rate across sessions.&#x20;

## C. Preference / config memory (user-like)

One-shot user fact (e.g., “server *stargazer* runs Postgres on port 5433”). Query by varied cues (“port for stargazer?” “db port on my server?”). Measure B1↑ immediately; B3↑ post-replay. Also log tool-use success if relevant.&#x20;

## D. Stress-interference

Write many **near-collision** episodes (same schema, tiny changes). Track accuracy vs. store size and **interference slope**. Run ablations: remove DG sparsity; remove Hopfield completion; random writes; reorder replay (remove CA2-like scheduling). Each ablation should degrade the predicted dimension (precision, partial-cue robustness, or retention).&#x20;

## E. Long-context control

Create tasks solvable by either pasting a huge transcript or by HEI-NW recall. Compare quality vs. compute (FLOPs, KV-mem). Expect HEI-NW to reach parity cheaper.&#x20;

*(If you also implement the semantic/spatial variants, add the schema-fit and navigation tests, but they’re optional for bare HEI-NW.)*&#x20;

# 3) Metrics to log (precise definitions)

**Episodic quality**

* Partial-cue **EM/F1**, **recall\@k**, and **latency** per query; **robustness curves** vs distractors/time lag.&#x20;

**Consolidation**

* **Pre→post delta**: (B3 − B0) on the target facts/skills after replay; **retention curves** across days/cycles; **no-drift** on base tasks (B3 vs B0 on a held-out core suite).&#x20;

**Write-gate quality**

* Treat “should-remember” items as positives (rare, useful, pinned). Compute **precision/recall**, PR-AUC, **clutter rate** (writes per 1k tokens), and a **calibration curve** for the salience score S to select τ.&#x20;

**Retrieval/store health**

* **P\@k / MRR** of associative lookup; **collision rate** vs load; **near-miss rate**; **completion success** (does Hopfield step fix partial cues?).&#x20;

**Scheduler/interference**

* **Retention after replay** with vs without CA2-like ordering; measure drop if you shuffle replay (should worsen).&#x20;

**Efficiency**

* Attention **FLOPs** vs long-context baselines; **KV cache MB** with/without MQA/GQA; wall-clock per query.&#x20;

# 4) Experimental protocol (step-by-step)

1. **Baselines**: run B0 (and long-context/RAG baselines).&#x20;
2. **One-shot phase** (freeze base weights): write episodes via the **salience gate**; then evaluate B1 immediately on partial-cue sets. Expect B1 ≫ B0.&#x20;
3. **Replay cycles**: run prioritized replay (CA2-style ordering; mixed batches). Re-test after each “sleep” cycle (B2, then **turn memory OFF** → B3). Expect B3 to climb toward B1 without base-task regression.
4. **Ablations**: no DG, no Hopfield, random writes, no CA2 scheduling, no replay. Record which metric collapses; this validates each sub-mechanism.&#x20;
5. **Stress**: burst writes of similar episodes; confirm HEI-NW’s interference curve is flatter than ablations.&#x20;

# 5) How big should the test sets be?

* For episodic one-shot **engineering sweeps**: set `n_small ≈ 24–64` items (paired seeds) to exercise probes and decoding checks; treat these as deterministic gates without CI claims.
* For Scenario A **statistical acceptance**: choose `n_stat ≥ 200–500` items on the hard subset (B0 wrong, retrieval correct). This yields \~±3–4-point 95% CI on EM lift with ≥1000 paired bootstrap resamples; include **hard negatives** and distractor-heavy splits.
* For consolidation deltas: enough items so B3−B0 CI excludes 0 (paired bootstrap or McNemar on correctness).
* For interference: sweep store sizes (e.g., 1k → 100k episodes) to fit a **precision-vs-load** curve. *(The doc prescribes the metrics/protocols, not fixed counts; choose sizes for statistical power.)*

# 6) Building the datasets quickly

Use programmatic generators:

* **Slot stories:** sample names/objects/places/times from disjoint pools; auto-generate cues and distractors.
* **Counterfactual updates:** pick facts the base gets wrong or is outdated on; create a single corrective episode + queries.
* **Preferences/config:** generate per-“user” key-value tuples with synonyms for cues.

Each record should also include a **“should-remember”** label to score write-gate precision/recall and tune τ.&#x20;

# 7) Choosing replay & learning settings (validation-friendly defaults)

Use the doc’s guidance: **prioritized replay**, **interference-aware ordering**, and **small-LR LoRA or light FT**, mixing episodic rows with a little generic data (e.g., **50/30/20** episodic/semantic/fresh if you also run the semantic path). Evaluate after each short cycle rather than one long run; stop if base-task loss rises.

# 8) Minimal evaluation loop (pseudocode)

```python
modes = [B0, B1, B2, B3]  # see above
def eval_suite(model, memory, dataset):
    metrics = {}
    for q in dataset.queries:
        ans = run(model, memory, q)      # memory toggled per mode
        metrics.update(score(q, ans))    # EM/F1/latency/compute
    return summarize(metrics)

# One-shot phase
freeze_weights()
B0_scores = eval_suite(base, OFF, episodic_set)
write_episodes_via_gate(episodic_set)    # uses S=α·surprise+β·novelty+...
B1_scores = eval_suite(base, ON, episodic_set)

# Replay & consolidation
for cycle in range(C):
    prioritized_replay_step()            # CA2-like ordering
    B2_scores = eval_suite(base, ON, episodic_set)
    B3_scores = eval_suite(base, OFF, episodic_set)
    check_no_drift(core_tasks)           # guardrails
```

Gate and replay match the specified algorithms.

# 9) Interpreting outcomes (accept/reject)

**Headroom Gate (Scenario A, small set)**

| EM\_{B0} on evaluated slice | Action |
| --- | --- |
| `< 0.7` | Proceed to statistical uplift on the hard subset. Compute `B1 − B0` with ≥1000 paired bootstrap resamples; claim success only if the 95% CI excludes 0 and the mean uplift ≥ +0.30 EM. |
| `≥ 0.7` | Do **not** claim uplift. Switch to the **memory-dependent baseline** (strip episodic hint for B0, keep identical prompt + memory tokens for B1) and report absolute EM\_{B1}, oracle upper bound, and retrieval-only diagnostics instead. |

* **Pass** if:
  (i) Engineering gates hold: parity guard, oracle EM ≥ 0.8, retrieval-only ≈ P@1 (±5 pts), Hopfield lift ≥ 0, decoding sanity checks pass;
  (ii) When the Headroom Gate passes, statistical uplift on the hard subset meets the criterion above;
  (iii) B3 after replay retains ≥80–90% of B1 on target items;
  (iv) base-task change ∈ \[−1, +1] EM points (no drift);
  (v) compute is ≤ baseline long-context for matched quality.
* **Fail** if: noisy writes (low gate precision), partial-cue recall doesn’t beat RAG/long-context, consolidation regresses core tasks (fix scheduler/LR), or Headroom Gate repeatedly fails without a memory-dependent baseline rerun.&#x20;
