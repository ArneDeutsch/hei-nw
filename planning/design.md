# HEI-NW Design Spec v1.0

**Hippocampal Episodic Index — Neuromodulated Writes**

---

## 1) Purpose & Scope

HEI-NW augments a decoder-only LLM with a persistent, content‑addressable **episodic memory** that supports one‑trial writes, partial‑cue recall, offline replay/consolidation into base weights, and principled forgetting. This document defines the end‑to‑end design sufficient to implement, deploy, and validate HEI‑NW against a small base model (e.g., Qwen2.5‑1.5B‑Instruct) and scale upward.

---

## 2) Executive Summary


> Note: The **30% semantic facts** slice used during replay may, in future, be supplied by a schema store (e.g., SGC‑RSS). This is informational only and does not change the MVP.

* **Augmentation, not surgery:** keep the base Transformer intact; add a slim **Episodic Adapter** (cross‑attention) and a **Memory Service** (DG keyer, associative store, replay queue).
* **Core loop:**

  * *Read:* turn the current hidden state into a **sparse key** (k‑WTA). Query the associative store (ANN + modern Hopfield) to retrieve/complete a **trace**. Pack trace → memory tokens; the adapter cross‑attends to them.
  * *Write (gated):* compute salience $S$ from **surprise**, **novelty**, **reward/pin**. If $S>\tau$ → write `(key, trace)` and enqueue for replay.
  * *Consolidate:* an offline worker replays prioritized episodes (anti‑interference scheduling) and distills into the base model (LoRA → optional full FT).
  * *Forget:* decay/evict low‑value or stale entries; protect pinned items.
* **Why:** enables durable one‑shot learning with bounded runtime compute; reduces dependence on giant contexts/RAG for user‑specific or rapidly changing facts.

---

## 3) Definitions

* **DG key**: Sparse key produced by a k‑WTA encoder from the LLM’s hidden state (pattern separation).
* **Associative store**: Content‑addressable memory that returns/“completes” traces from partial cues (modern Hopfield layer sitting atop an ANN index).
* **Trace**: The value payload for an episode (pointer to original tokens, structured slots, tiny state sketch, salience/provenance tags).
* **Replay Queue**: Prioritized queue of episode IDs used by the consolidation worker; ordered to minimize interference.
* **Episodic Adapter**: A slim cross‑attention block that attends over retrieved memory tokens.

---

## 4) System Architecture

```
Text → Base LLM (decoder-only)
             │
        [Query vector]
             │
      DG k-WTA Keyer  ──►  Associative Store (ANN + modern Hopfield)
             │                               │
       Write Gate (S) ────────┐               ▼
             │                ├──► Episodic Store: {key → trace}
             ▼                │               │
      Replay Queue ◄──────────┘               ▼
             │                          Pack → Memory Tokens
   Consolidation Worker                     │
   (prioritized replay)                Episodic Adapter
             │                               │
     Distill to Base LLM  ◄──────────────────┘
```

---

## 5) Components

### 5.1 Base LLM Integration

* **Hook point:** After block *N* (configurable), insert the **Episodic Adapter** (cross‑attention over memory tokens). Keep self‑attention and FFN untouched.
* **Runtime budget:** cap memory tokens per step; adapter uses GQA/MQA; enable FlashAttention in core.

### 5.2 Episodic Adapter

* **Inputs:** current token hidden states (`H_t`), memory tokens (`M_t`).
* **Operation:** one cross‑attention block + residual; optional gating scalar to blend memory vs. base evidence.
* **Outputs:** enriched hidden states `H'_t`.
* **Placement:** typically after middle blocks (e.g., 60% depth) for strong semantic context but before late decoding.

### 5.3 DG Keyer (Pattern Separation)

* **Projection:** `q = W_q · h_context` (concat pooled token states or use BOS state).
* **Sparsify:** k‑WTA (top‑k by magnitude) → binary mask + retained values; L1‑normalized values.
* **Key format:** CSR‑like `(indices:int16/32, values:int8/16)`; dimension *d*, sparsity *k* (defaults in §11).
* **Distance:** cosine on sparse vectors; ANN (HNSW/PQ) over sparse embeddings.

### 5.4 Associative Store (Completion)

* **Indexing layer:** ANN over DG keys; retrieves top‑K candidate traces quickly.
* **Completion layer:** **Modern Hopfield** readout:

  * Maintain learnable memory matrix `M` (patterns) and energy function; initialize patterns from stored keys; allow small updates during consolidation.
  * Given cue `k`, perform 1–3 fixed‑point updates (or a single differentiable read) to obtain a completed pattern `ĉ` used to rank/select traces.
* **Return:** `top_k` traces with scores; pass best (or a small set) to the packer.

### 5.5 Trace Schema (Value Payload)

```ts
interface TraceValue {
  tokens_span_ref: { doc: string; start: number; end: number }; // pointer only
  entity_slots: { who?: string; what?: string; where?: string; when?: string; extras?: Record<string,string> };
  lm_state_sketch?: { layer_ids: number[]; low_rank_q: number[][] }; // tiny bootstrap
  action_trace?: Array<{ tool?: string; args?: Record<string, any>; result_hash?: string; ts?: string }>;
  salience_tags: { surprise: number; novelty: number; reward?: boolean; pin?: boolean; S: number };
  provenance?: { source: "chat"|"tool"|"file"; timestamp: string; confidence?: number };
}
```

**Example:** “Server *stargazer* runs Postgres on port 5433.” (see Appendix D).

### 5.6 Memory Token Packer

* **Pack:** deterministic template that renders `entity_slots` + a short snippet from `tokens_span_ref` into ≤ *m* tokens; add a tiny binary blob token for `lm_state_sketch` if present.
* **Unpack:** not needed at inference; used by consolidation to reconstruct training rows.

### 5.7 Neuromodulated Write Gate

* **Signals:**

  * `surprise = -log p(next_token)` from base logits;
  * `novelty = 1 - max_sim(q, existing_keys)`;
  * `reward ∈ {0,1}` (system/tool/user events);
  * `pin ∈ {0,1}` (explicit “remember this”).
* **Score:** `S = α·surprise + β·novelty + γ·reward + δ·pin`.
* **Policy:** if `S > τ`, write `(key, trace)` and push key to **Replay Queue**.
* **Calibration:** maintain PR‑AUC for the gate against “should‑remember” labels (validation harness).

### 5.8 Replay & Consolidation

* **Queue item:** `{key_id, S, recency, diversity_hash, grad_overlap_hint, trace_ref}`.
* **Scheduler (CA2‑style):**

  * Score pool by `S`, inverse recency, diversity bonus.
  * Choose next to **minimize gradient overlap** with the running batch (use key similarity as proxy or small Jacobian sketches).
* **Batch composition:** default **50% episodic replay / 30% semantic facts / 20% fresh** (semantic path optional).
* **Training rows from a trace:**

  1. **LM replay row:** reconstruct the original token slice → next‑token loss.
  2. **Cue→Answer row:** synthesize a cue from `entity_slots` and target from snippet; supervise cross‑entropy.
  3. **(Optional) Imitation/policy row:** if `action_trace` exists, emit macro‑policy imitation row(s) for a future policy head; **off by default**.
* **Optimizer:** AdamW; start with **LoRA** on attention/FFN (r=16–64); escalate to selective full FT if stable.
* **LR:** small (see §11); cosine decay per cycle; early‑stop if base‑task loss rises.

### 5.9 Eviction / Decay / Protection

* **Decay:** age‑ and use‑weighted score; stochastic down‑scaling of low‑S entries.
* **Evict:** when capacity exceeded, remove lowest composite score subject to **diversity guard**.
* **Protect:** never evict `pin=true`; raise thresholds for high‑S items.

### 5.10 Telemetry, Provenance, Safety

* Log per‑write: `S` breakdown, source, time, SHA of span; enable rollback by `trace_ref`.
* Track retrieval P\@k, collision rate, near‑miss rate; emit alerts on clutter spikes.
* PII: redaction policy at packer; optional encryption at rest for store.

---

## 6) Public Interfaces (implementation‑grade)

### 6.1 Pythonic APIs

```python
class EpisodicAdapter(nn.Module):
    def __init__(self, d_model:int, n_heads:int, max_mem_tokens:int): ...
    def forward(self, H_t: Tensor, M_t: Tensor|None) -> Tensor: ...

class DGKeyer:
    def __init__(self, d:int, k:int): ...
    def key(self, h_state: np.ndarray) -> SparseKey: ...

class AssocStore:
    def lookup(self, key: SparseKey, top_k:int=8) -> list[TraceHit]: ...
    def write(self, key: SparseKey, trace: TraceValue) -> str: ...
    def decay(self) -> None: ...

class WriteGate:
    def score(self, surprise:float, novelty:float, reward:bool, pin:bool) -> float: ...

class ReplayQueue:
    def push(self, key_id:str, meta:dict) -> None: ...
    def sample_window(self, n:int) -> list[str]: ...

class ConsolidationWorker:
    def step(self, budget_tokens:int) -> dict: ...  # returns stats

# High-level entrypoints
recall(query_state) -> list[MemoryTokens]
maybe_write(query_state, trace, reward=False, pin=False) -> Optional[str]
consolidate(token_budget:int) -> ConsolidationReport
```

### 6.2 Storage Layouts

* **Keys:** `keys.{shard}.faiss` (ANN) + `keys/manifest.parquet`.
* **Traces:** `traces/` as compact JSONL or Parquet; large doc bodies live elsewhere (we store pointers only).
* **Queue:** `replay/priority.db` (SQLite or RocksDB).

---

## 7) Algorithms (pseudo)

### 7.1 Recall

```python
def recall(h_state):
    q = proj(h_state)
    key = kWTA(q, k=K)
    cand = ann.topk(key, Kc)
    comp = hopfield.complete(key, cand)
    traces = fetch_traces(comp)
    mem_tokens = pack(traces[:M])
    return mem_tokens
```

### 7.2 Maybe‑Write

```python
def maybe_write(h_state, trace, reward=False, pin=False):
    surprise = nll_next_token()
    novelty  = 1 - ann.max_sim(proj(h_state))
    S = a*surprise + b*novelty + c*int(reward) + d*int(pin)
    if S > tau:
        key = kWTA(proj(h_state))
        id  = store.write(key, trace)
        queue.push(id, meta={"S":S, ...})
        return id
```

### 7.3 Consolidate

```python
def consolidate(token_budget):
    batch = scheduler.compose(token_budget, mix={"episodic":0.5,"semantic":0.3,"fresh":0.2})
    rows  = []
    for ep in batch:
        text = load_span(ep.trace.tokens_span_ref)
        rows.append(lm_row(text))
        rows.append(qa_row(cue_from_slots(ep.trace.entity_slots), answer_from(text)))
    train_lora(rows, lr=LR, cosine_decay=True, early_stop_on_drift=True)
    return stats()
```

---

## 8) Configuration (defaults & ranges)

* **Adapter depth:** insert after block **⌊0.6·L⌋**; 1× cross‑attn layer.
* **DG keyer:** `d=2048`, `k=64` (≈3% sparsity). Range: `k∈[32,128]`.
* **ANN:** HNSW `M=32, efC=200, efS=64`; PQ if memory constrained.
* **Hopfield:** 64–256 patterns per shard; 1–3 update steps.
* **Recall top‑k:** candidates `Kc=64`, return `M=4` traces → ≤128 memory tokens.
* **Gate:** `α=1.0, β=1.0, γ=0.5, δ=0.8`; threshold `τ` tuned to PR‑AUC target; aim for \~1–5 writes / 1k tokens.
* **Replay mix:** 50/30/20 episodic/semantic/fresh.
* **LoRA:** r=32, α=16; LR `1e-4` (LoRA) or `1e-5` (full FT); cosine decay; weight decay `0.01`.
* **Eviction:** TTL 30 days default; min‑use guard; diversity bucketization by SimHash.

---

## 9) Validation & Evaluation Plan (adopted)

**Modes:**

* **B0:** base model only.
* **B1:** base + memory (no replay).
* **B2:** base + memory after replay.
* **B3:** base after replay with memory **OFF** (tests corticalization).

**Scenarios:**

* **A. Episodic one‑shot w/ partial cues:** single‑exposure stories (who/what/where/when), heavy distractors.
* **B. Factual hot‑patch:** contradict prior parametric knowledge; check immediate B1, then B3 after replay, with no unrelated drift.
* **C. Preference/config memory:** user facts/configs (e.g., ports, IDs) with paraphrased cues.
* **D. Stress‑interference:** many near‑collision episodes; sweep store size; ablations (no DG, no Hopfield, random writes, shuffled replay).
* **E. Long‑context control:** tasks solvable via giant context vs. HEI‑NW; compare quality vs. compute.

**Metrics:**

* Episodic EM/F1, recall\@k, latency; robustness vs. distractors/time.
* Consolidation Δ (B3−B0), retention curves; base‑task **no‑drift**.
* Gate precision/recall, clutter rate (writes/1k toks), S calibration.
* Retrieval P\@k/MRR; collision & near‑miss rate; completion lifts.
* Scheduler effect: retention with CA2‑style ordering vs. shuffled.
* Efficiency: attention FLOPs, KV‑MB with/without MQA/GQA.

**Dataset sizes:** ≥500 episodes/condition for tight CIs; paired bootstrap or McNemar for deltas; sweep store sizes 1k→100k for interference curves.

**Acceptance:** Engineering gates (parity guard, oracle EM ≥ 0.8, retrieval-only ≈ P@1, Hopfield lift ≥ 0, decoding sanity) plus headroom-aware uplift: if EM\_{B0} < 0.7 on the hard subset, require `B1−B0 ≥ +0.30 EM` with 95% paired bootstrap CI excluding 0; otherwise rerun with the memory-dependent baseline and report absolute EM\_{B1}``/``oracle metrics. B3 retains ≥80–90% of B1 after replay; base‑task change within ±1 EM; compute ≤ long‑context baseline at matched quality.

---

## 10) Operational Considerations

* **Scaling:** sharded stores by semantic domain/time; async prefetch; warm‑key caches.
* **Latency:** budget ≤10–20 ms per recall (ANN 5–10 ms, packer 1–2 ms, adapter <5 ms at small token counts).
* **Capacity:** aim ≤10^7 traces per shard (with PQ); background compaction.
* **Observability:** dashboards for gate PR‑AUC, retrieval P\@k, replay overlap, drift guard, token budgets.
* **Rollback:** per‑trace delete; signed provenance; replay can unlearn via counterexamples if needed.

---

## 11) Risks & Mitigations

* **Noisy writes:** tune τ; add user pin; use S calibration; apply write‑rate caps.
* **Interference during replay:** CA2‑style ordering; low LR; interleaved batches; early‑stop.
* **Privacy/PII:** pointer‑only traces, encryption at rest, redaction in packer.
* **Compute creep:** strict token budgets; cap memory tokens; MQA/GQA in adapter.

---

## 12) Implementation Plan

1. **MVP (weeks 1–2):** DG keyer, ANN index, trace store, packer, adapter integration; simple write‑gate; B0/B1 eval.
2. **Replay (weeks 3–4):** queue + scheduler; LM/QA row builders; LoRA distillation; B2/B3 eval.
3. **Stability (weeks 5–6):** eviction/decay; provenance; dashboards; ablations; parameter sweeps.
4. **Scale‑out (weeks 7+):** sharding; Hopfield readout tuning; PQ; policy hardening.

---

## Appendix A) Modern Hopfield Readout (sketch)

* Patterns `M ∈ R^{P×d}`; energy `E(z) = -log ∑_p exp(β·⟨z, M_p⟩)`; fixed‑point step `z_{t+1} = softmax(β·M·z_t)·M` with small *T*.
* Initialize `M` with normalized keys of top items; allow tiny updates during consolidation; keep read‑only at inference.

## Appendix B) k‑WTA Sparse Keyer

* Choose *k* via load/interference trade‑off; implement with top‑k op + straight‑through for any trained variants; store indices/values only.

## Appendix C) Memory Tokens (packer template)

* Template strings for slots + short snippet; cap to `max_mem_tokens`; deterministic so training/inference match.

## Appendix D) Example Trace (stargazer)

```json
{
  "tokens_span_ref": { "doc": "chat/2025-09-10", "start": 1821, "end": 1862 },
  "entity_slots": {
    "who": "user",
    "what": "server config: Postgres port=5433",
    "where": "host=stargazer",
    "when": "2025-09-10T10:12:33+02:00",
    "extras": { "service": "postgresql" }
  },
  "lm_state_sketch": { "layer_ids": [19, 23], "low_rank_q": [[0.14, -0.02], [0.03, 0.11]] },
  "salience_tags": { "surprise": 0.41, "novelty": 0.83, "pin": true, "S": 0.77 },
  "provenance": { "source": "chat", "timestamp": "2025-09-10T10:12:33+02:00", "confidence": 0.98 }
}
```
