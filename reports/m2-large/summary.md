# Large-sample Retrieval Check (τ = 1.0)

All runs use `--seed 21`, `n = 256`, chat QA prompts, and hashed recall (`use_raw_hash=True`).

| Scenario | Mode | EM | P@1 | Notes |
| -------- | ---- | --- | --- | ----- |
| A | B0 | 0.000 | — | Baseline (no memory) |
| A | B1 (gate writes) | 0.211 | 0.230 | `writes/1k tokens ≈ 1.38`; uplift smaller but >0 due to limited writes |
| B | B0 | 0.000 | — | Hot-patch baseline |
| B | B1 (label writes) | 0.000 | 0.023 | Memory answers rarely top-1; hashing collides heavily |
| C | B0 | 0.000 | — | Preference/config baseline |
| C | B1 (label writes) | 0.000 | 0.023 | Same collision issue as scenario B |

Positive-only P@1 (should_remember=true) — Scenario B: 0.047, Scenario C: 0.031 (computed via `RecallService.store.query`).

**Takeaways**
- Gate-locked τ=1.0 maintains Scenario A uplift with actual gate decisions, but P@1 drops to ≈0.23 when writes are limited to ~1.4/1k tokens.
- Hashed embeddings + heuristic rerank do *not* scale to Scenario B/C at n=256; we observe severe collisions. Further work needed (richer features or scenario-specific keying).
