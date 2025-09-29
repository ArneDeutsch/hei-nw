# HEI-NW Project Plan — Milestones with Built-in Verification

Below is a *do-once* milestone plan that implements HEI-NW, integrates it with a small LLM (e.g., Qwen2.5-1.5B-Instruct), and validates usefulness with tight CIs. Each milestone ends with **Definition of Done** (DoD) checks, quantitative **Acceptance Criteria**, and concrete **Artifacts**. Modes **B0–B3** and the scenario/metric suite are taken directly from the validation plan and the design spec so we can verify every step before moving on. &#x20;

---

## M0 — Repo, Baselines, and Harness (B0)

**Scope**

* Stand up the evaluation harness with switchable modes `B0…B3` (B0 wired first) and long-context/RAG baselines for fairness.&#x20;
* Select the base model (e.g., Qwen2.5-1.5B-Instruct), pin tokenizer, generation settings, and seeds.
* Implement dataset *generators* for all scenarios (A–E) with CI-friendly sizes and bootstrap evaluation; include “should-remember” labels.&#x20;
* Log metrics: EM/F1, recall\@k, latency, FLOPs/KV cache stats scaffold (even if 0s for now).&#x20;

* Augment generators with **hard negatives** for Scenario A (confounders that are semantically close).
* Add scaffolding to log **robustness vs. time‑lag** curves (bin episodes by lag and compute EM/Recall@k per bin).
**Verification**

* Run **B0** on small splits of A–E to establish the frozen baseline.
* Output baseline compute footprints for long-context vs. no-memory.

**DoD / Acceptance**

* **Harness:** `--mode B0` works; produces metric JSON and a short HTML/Markdown report.
* **Data:** Generators for A–E emit ≥500 episodic items per condition when requested (not necessarily run in CI).&#x20;
* **Metrics:** EM/F1 and latency appear in reports; compute metrics fields exist (may be empty).&#x20;

**Artifacts**

* `eval/` harness with `--mode {B0,B1,B2,B3}` and `--scenario {A..E}`.
* `datasets/` generators for Slot-Stories, Hot-Patch, Pref/Config, Stress, Long-Context.&#x20;
* `reports/baseline/` with B0 numbers.

---

## M1 — Episodic Adapter (Read-only) + Memory Tokens (B1 skeleton)

**Scope**

* Insert the **Episodic Adapter** (1× cross-attention) at \~60% depth; cap memory tokens. No memory hooked yet (M\_t=None).&#x20;
* Implement **Memory Token Packer** (deterministic template), returning ≤`max_mem_tokens`.&#x20;
* Add API surface: `EpisodicAdapter.forward(H_t, M_t)` and plumbing in generation loop.&#x20;

**Verification**

* Sanity: with `M_t=None` model logits unchanged to within tolerance (adapter off == B0).
* Latency overhead of a no-op adapter recorded.

**DoD / Acceptance**

* **Equivalence:** `B1(with empty memory)` ≈ `B0` across A–E within ±0.1 EM/F1.
* **Budget:** Adapter-added latency ≤ the design’s target budget per step at small token counts (recorded; we enforce later).&#x20;

**Artifacts**

* `hei_nw/adapter.py`, `hei_nw/pack.py`, unit tests.

---

## M2 — Retrieval Stack: DG Keyer + ANN + Modern Hopfield (B1 functional)

**Scope**

* Implement **DG Keyer** (k-WTA sparse key), **ANN index** (HNSW), and **modern-Hopfield completion**; wire **recall()** to return memory tokens. Defaults per spec.  &#x20;
* Add retrieval metrics: P\@k/MRR, collision & near-miss rate, completion lift.&#x20;
* Build a **schema-agnostic feature pipeline** for episodic keys (text + structured slots) instead of dataset-specific heuristics; demonstrate it on scenarios A–C.&#x20;

**Verification**

* Scenario **A** (partial-cue): run **B0 vs. B1** immediately after a single exposure (manually write traces for now) and confirm large lift. Expect **B1 ≫ B0**.&#x20;
* Run **E0–E3 probes** (sanity, oracle, retrieval-only, Hopfield ablation) via helper scripts to isolate retrieval vs. generation.
* Ablate Hopfield completion to show partial-cue drop.

**DoD / Acceptance**

* **Engineering acceptance:** `B1` with empty memory ≈ `B0` (±0.1 EM); oracle EM ≥ 0.8; retrieval-only EM tracks P\@1 (±5 pts); Hopfield completion lift ≥ 0; decoding sanity checks pass.&#x20;
* **Headroom-aware uplift:** If the Headroom Gate passes (EM\_{B0} < 0.7 on the hard subset), require `B1−B0 ≥ +0.30 EM` with 95% paired bootstrap CI excluding 0. Otherwise, switch to the memory-dependent baseline and report absolute EM\_{B1}``/``oracle diagnostics instead of uplift.&#x20;
* **Retrieval Health:** P\@k and MRR logged; near-miss and collision rates finite (non-NaN) on stress mini-set.&#x20; Achieve `P@1 ≥ 0.6` on Scenario A small set *and* demonstrate ≥0.3 P@1 on Scenarios B/C (n ≥ 200) without scenario-specific hacks.

* Unit/integration test: **modern‑Hopfield parameters are read‑only at inference**; updates occur only during consolidation.
**Artifacts**

* `hei_nw/keyer.py`, `hei_nw/store.py` (ANN + Hopfield), `hei_nw/recall.py`.
* Plots: completion vs. ablated completion.

---

## M3 — Neuromodulated Write Gate + Trace Store

**Scope**

* Implement **write gate**: `S = α·surprise + β·novelty + γ·reward + δ·pin`; threshold τ; pointer-only traces schema. &#x20;
* Add telemetry for gate PR-AUC, precision/recall, clutter rate (writes/1k tokens) and calibration curves using label from generators.&#x20;
* Implement **eviction/decay/protection** (pin=true never evicted).&#x20;

**Verification**

* On scenario A/C streams, measure gate PR-AUC and set τ to hit target write rate (≈1–5 / 1k tokens) with high precision.&#x20;
* Privacy check: packer redacts PII; traces hold **pointers only**.&#x20;

**DoD / Acceptance**

* **Gate:** PR-AUC reported; precision ≥ target (team-set) at write-rate ≤ 5/1k; calibration plot generated.&#x20;
* **B1 Live:** End-to-end write→recall works on A/C without manual writes.

**Artifacts**

* `hei_nw/write_gate.py`, `hei_nw/traces/` (JSONL/Parquet layout).&#x20;
* `reports/gate/` with PR-AUC & calibration.

---

## M4 — Replay Queue, CA2-Style Scheduler, and LoRA Consolidation (B2/B3)

**Scope**

* Implement **ReplayQueue** + **CA2-style scheduler** (S/recency/diversity, gradient-overlap proxy), and **ConsolidationWorker.step()** producing LM and Cue→Answer rows; train with **LoRA** first.&#x20;
* Add **B2** (after replay, memory ON) and **B3** (turn memory OFF → test corticalization) runs.&#x20;

**Verification**

* Run 1–N short replay cycles on A/B/C; chart **B3** rising toward **B1** while **base-task drift guard** holds.&#x20;
* Compare scheduler vs. shuffled ordering; retention should worsen when shuffled.&#x20;

**DoD / Acceptance**

* **Consolidation:** `B3` retains **≥80–90% of B1** on targets; **B3−B0** CI excludes 0. No base-task regression (±1 EM).&#x20;
* **Scheduler effect:** measurable drop with shuffled ordering (report figure).&#x20;

**Artifacts**

* `hei_nw/replay.py`, `hei_nw/consolidate.py`, LoRA configs, drift guard monitors.

---

## M5 — Interference & Capacity Characterization + Ablations

**Scope**

* Scenario **D** (near-collisions): sweep store sizes (1k→100k) and measure **interference slope**; run ablations (no DG, no Hopfield, random writes, no CA2).&#x20;
* Fit precision-vs-load curves; report collision/near-miss trends.&#x20;

**Verification**

* Each ablation degrades the predicted dimension (precision, partial-cue robustness, or retention).&#x20;

**DoD / Acceptance**

* **Curves:** Published curves with CIs for interference; ablations show expected collapses (table + plots).&#x20;

**Artifacts**

* `reports/interference/` figures + JSON summaries.

---

## M6 — Efficiency Benchmarks vs Long-Context/RAG

**Scope**

* Scenario **E**: tasks solvable by long transcript vs HEI-NW recall. Compare **quality vs FLOPs/KV-MB/latency** for matched quality.&#x20;
* Implement KV budget and token-capping in adapter; record budgets on dashboards.&#x20;

* Expose **MQA/GQA** adapter toggle in config and report its impact on KV‑cache MB and throughput.
**Verification**

* For matched EM, HEI-NW uses **≤** compute of long-context baselines.&#x20;

**DoD / Acceptance**

* **Efficiency:** Report shows attention **FLOPs** and **KV MB** advantages at parity quality.&#x20;

* Latency SLO: **P95 recall latency ≤ 20 ms** at ≤ 128 memory tokens; include breakdown for ANN, packer, and adapter stages.
**Artifacts**

* `reports/efficiency/` with side-by-side plots.

---

## M7 — Productionization & HF Integration (Base + Augmented)

**Scope**

* Package **base LLM** and **HEI-NW-augmented LLM** as:

  * a Python library (`hei_nw`),
  * a Hugging Face–style wrapper (`PreTrainedModel` subclass or Pipeline) exposing `mode={B0..B3}` and memory controls,
  * a simple REST server (FastAPI) for demo.
* Public APIs as per design’s implementation-grade interfaces; stable storage layouts.&#x20;

**Verification**

* Drop-in HF usage: `from transformers import pipeline` or `AutoModelForCausalLM` with a memory wrapper runs **B1** on scenario A sample, and **B3** after `consolidate()`.

**DoD / Acceptance**

* **UX:** One-line toggle for modes; config for adapter depth, k-WTA `k`, Hopfield steps, τ.
* **Safety:** PII redaction in packer on by default; per-trace delete and provenance/rollback supported.&#x20;

* Traces **encrypted at rest** with key rotation; documentation for KMS integration.
* Per‑trace **signed provenance** (tamper‑evident) covering data source, gate decision, and consolidation lineage.
**Artifacts**

* `pip install hei_nw/` wheel, HF integration docs, minimal Dockerfile.

---

## M8 — Observability, Telemetry, and Stability Hardening

**Scope**

* Dashboards for: gate PR-AUC, retrieval P\@k, replay overlap, drift guard, token budgets; alerts on clutter spikes.&#x20;
* Add decay TTLs, diversity buckets, pin-protection (already implemented; wire knobs & charts).&#x20;

* Scale‑out: **shard** associative store by domain/time; add **async prefetch** on likely keys and a **warm‑key cache**.
* Load tests for shard fan‑out and cache hit‑rates; verify no regression on latency SLO.
**Verification**

* Chaos tests: bursty writes, mixed domains; verify no alert floods, eviction behaves, pins protected.&#x20;

**DoD / Acceptance**

* **Stability:** 1-hour soak on scenario mix with zero data loss, bounded clutter rate, and green drift guard.

**Artifacts**

* Grafana/Plot reports; `reports/soak/`.

---

## M9 — Full Validation Run + Paper Package

**Scope**

* Freeze configs; run full suites for A–E under **B0–B3** with ablations and efficiency comparisons.
* Aggregate results; prepare a research-style write-up (paper draft) with method, metrics, ablations, and limitations, mirroring the plan’s **Pass/Fail** logic.&#x20;

**Verification**

* Results meet **Pass** conditions:
  (i) `B1−B0 ≥ +30 EM` immediately on episodic set;
  (ii) `B3` retains ≥80–90% of `B1` after replay;
  (iii) base-task change within \[−1, +1] EM;
  (iv) compute ≤ long-context at matched quality.&#x20;

* Final report includes **accuracy vs. time‑lag** curves (Scenario A), alongside distractor‑robustness plots.
**DoD / Acceptance**

* **Repro Bundle:** script to reproduce all figures/tables; seed files; fixed commits and model IDs.
* **Paper Draft:** includes algorithms (with pseudo from spec), datasets, metrics, and results/ablations.&#x20;

**Artifacts**

* `reports/final/` with notebooks/plots/JSON, and `paper/hei_nw_results.md`.

---

## Cross-Cutting Standards (apply to every milestone)

* **Freeze & Seal:** At the end of each milestone, tag a release; downstream milestones cannot change sealed artifacts—only add new ones. If a change is unavoidable, open a *change PR* that carries its own micro-validation replicating the earlier acceptance tests.
* **Tests:** Unit + golden tests for data generators and APIs; deterministic small-split CI runs for B0/B1/B3 on scenario A; full runs guarded behind manual trigger.
* **Repro:** All metrics logged to JSON; plots regenerated from JSON; seeds stored in `seeds/`.
* **Security/Privacy:** Packer redaction on; traces are pointer-only; explicit pin for “remember this”; per-trace delete and provenance. &#x20;

---

## What Codex Automates vs. Human Oversight

* **Codex-friendly tasks:** implementing classes/APIs from the design’s *public interfaces*; wiring CLI; generating datasets; writing unit tests and plotting scripts; repetitive ablation runs.&#x20;
* **Human review (you + ChatGPT):** design adherence checks (adapter hook point; k-WTA ranges; Hopfield steps), telemetry sanity, replay scheduler heuristics, paper narrative quality.&#x20;

---

## Deliverables Summary

* **Models:**

  1. **Base LLM (B0)** packaged;
  2. **HEI-NW LLM** with `mode={B0,B1,B2,B3}` and HF-style API.&#x20;
* **Memory System:** DG keyer, ANN+Hopfield store, write gate, replay scheduler, consolidation worker, decay/eviction, telemetry.&#x20;
* **Validation Assets:** Scenarios A–E, metrics suite, ablations, efficiency comparisons, final report meeting the Pass criteria.&#x20;
