# HEI-NW (Hippocampal Episodic Index — Neuromodulated Writes)

**HEI-NW** is a *memory add-on* for an LLM. You keep the base Transformer mostly as-is, and bolt on a small adapter plus a persistent “episodic store.” The system decides *when* to write a fresh memory (based on surprise/novelty/reward), *how* to index it (sparse key for low interference), *how* to recall from partial cues (associative lookup/completion), and *when* to distill the best memories back into the model weights via offline replay. Think: **LLM + smart, content-addressable cache with a write policy and nightly consolidation.**&#x20;

---

# What it is (in software terms)

* **Architecture:** An external memory module + a thin cross-attention adapter inside the LLM stack. No surgery to the core attention/FFN blocks; you *augment* the LLM rather than re-architect it. (You *may* fine-tune during consolidation, but the day-to-day inference path is an add-on.)&#x20;
* **Goal:** True one-shot, durable “episodic” memory that you can recall from partial cues, without stuffing everything into the context window or running constant fine-tunes.&#x20;

---

# The moving parts

1. **DG-style sparse keyer (pattern separation).**
   Take the LLM’s current hidden state, project it, and apply **k-WTA** (keep top-k activations) to form a *sparse* key. This reduces collisions so similar but distinct episodes don’t overwrite each other. (Biology metaphor: dentate gyrus.)&#x20;

2. **CA3-style associative memory (completion).**
   Store `(sparse_key → trace)` pairs; on recall, do KNN / modern-Hopfield-style retrieval that can *complete* a memory from a partial cue, then feed the retrieved trace back to the LLM via a cross-attention adapter. (Biology metaphor: CA3 autoassociation.)&#x20;

3. **Neuromodulated write-gate.**
   Compute a salience score `S` from things you already have in the LLM loop: **predictive surprise** (–log p of the next token), **novelty** (low similarity to existing keys), plus any **reward/user pin** flag. Only if `S > τ` do you write a new episode. (Biology metaphor: dopamine/norepinephrine gate encoding.)&#x20;

4. **Prioritized replay (consolidation).**
   A background scheduler replays high-salience episodes (interleaved to avoid interference) to **distill** them into the base model’s semantic weights (CLS-style “hippocampus → cortex”). Over time, frequently useful stuff becomes “known” by the LLM even without hitting the episodic store.&#x20;

5. **Eviction/decay.**
   Low-salience or stale entries decay (age/use TTL + diversity regularizers) so the store doesn’t bloat and retrieval stays precise.&#x20;

---

# How a request flows

**Read (inference):**

1. LLM produces a query embedding from its current state.
2. Sparse keyer makes a k-WTA key.
3. Associative store does content-addressable lookup (+ optional Hopfield completion).
4. Retrieved “trace” (compressed token span + entity/time slots + small state sketch) is **cross-attended** by the LLM to answer better.&#x20;

**Write (only if worth it):**

1. Compute salience `S = α·surprise + β·novelty + γ·reward + δ·user_pin`.
2. If `S > τ`, persist `(sparse_key, trace)` and enqueue for replay.&#x20;

**Consolidate (offline):**

1. Replay scheduler samples a diverse, high-value mix of episodes.
2. Fine-tune/distill the base model (LoRA or full) so the best facts/skills move from episodic memory into weights.&#x20;

---

# Pseudocode sketch (conceptual)

```python
def recall(h_state):
    q = proj(h_state)                    # LLM hook
    key = kWTA(q, k=K)                   # DG-like sparsity
    hits = assoc_store.lookup(key, top_k=8)  # KNN/Hopfield
    trace = complete(hits)               # optional Hopfield step
    return cross_attend(trace)           # small adapter block

def maybe_write(h_state, trace, reward=False, pin=False):
    surprise = nll_next_token()          # from logits
    novelty = 1 - max_sim(proj(h_state), assoc_store.keys())
    S = α*surprise + β*novelty + γ*int(reward) + δ*int(pin)
    if S > τ:
        key = kWTA(proj(h_state))
        assoc_store.write(key, trace)
        replay_queue.add(key, priority=S)

def consolidate():
    for batch in replay_queue.sample_interleaved():
        finetune_base_llm_on(batch)      # CLS-style replay → cortex
```

(Structures: sparse keys with PQ/HNSW; traces hold token spans, who/what/where/when slots, small state sketch.)&#x20;

---

# How it differs from things you already know

* **vs. plain RAG:** RAG is “nearest passages from a vector DB.” HEI-NW stores *episodes* with **sparse indexing and associative completion** + a **write policy** + **replay into weights**. It can recall from *partial cues* and actually *learn overnight*.&#x20;
* **vs. longer context/KV cache:** KV cache is transient and grows linearly with tokens; HEI-NW is *persistent* and capacity-managed.&#x20;
* **vs. continual fine-tuning only:** HEI-NW decides *what to remember first*, then consolidates later. You don’t fine-tune on every new fact; you fine-tune on the *right* ones.&#x20;

---

# What you’d build (minimal viable stack)

* **Core:** decoder-only LLM (standard Transformer).
* **Adapter:** one slim cross-attention block that can attend over recalled traces.
* **Memory:** ANN index (e.g., PQ + HNSW) to store sparse keys → traces; optional modern-Hopfield readout for completion.
* **Signals:** compute surprise from the LLM’s logits; novelty from max similarity; accept a user “pin” flag.
* **Jobs:** a replay scheduler that mixes high-salience episodes and runs periodic distillation/LoRA.&#x20;

---

# Limits & trade-offs to expect

* You need a good **write threshold**; too low = noisy store, too high = missed memories.
* Associative completion helps recall but can return near-misses if separation is weak; k-WTA and diversity-aware eviction help.
* Consolidation must be **interference-aware** (order and mix matter) to avoid drifting the base model.&#x20;

## Scenario C gating signals

- Scenario C emits configuration updates (`event_type="config_update"`) and status probes (`event_type="status_probe"`).
- Each record includes `novelty_counters` so gate tests can track how novelty decays across repeated exposures.
- Critical services (currently `postgres`) set `gate_features["reward"] = True` and attach a `reward_annotation` explaining the boost.
- Servers in the SRE pin list (currently `alpha`) produce `gate_features["pin"] = True` on configuration updates plus a `pin_annotation` describing the pin.
- The fixture `tests/fixtures/scenario_c_gate.json` anchors these semantics and underpins the scenario acceptance check.

## Implementation notes (current repo)

* The FAISS `IndexHNSWFlat` backend returns **distances**. We convert them to cosine-like scores (negative distance) and sort **descending** before modern-Hopfield re-ranking (`ANNIndex.search`). If Hopfield ever tanks P@1, check that this ordering logic still runs.
* The memory-dependent acceptance baseline surfaces retrieved episodes as short natural-language snippets (`Memory:` block) so the base model can answer without the original episode prompt. Keep the decoding guard (`scripts/gate_non_empty_predictions.py`) happy by leaving the user-facing question’s “single word” instruction unchanged.
