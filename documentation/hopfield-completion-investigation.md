# Hopfield Completion Investigation (M2)

## Context
- Goal: recover the expected positive retrieval lift from the Hopfield readout described in planning/design.md §5.4 and planning/validation-plan.md §3.
- Baseline observations prior to this work: the Hopfield path either regressed or matched ANN-only rankings (completion\_lift ≤ 0), while B1 uplift with cue-only prompts plateaued at ≈0.146 EM.

## Changes implemented
- Normalise ANN and Hopfield scores before blending, then mix them via a tunable weight (`_hopfield_blend = 0.2`).
- Compute the Hopfield query refinement from the stored pattern matrix instead of reusing raw ANN scores.
- Add a conservative guard (`_hopfield_margin = 0.01`) that prevents the reranker from demoting the ANN top hit unless it clearly increases confidence.
- Keep the ANN ordering for the remaining candidates to avoid exposing low-quality traces to the adapter.
- Added `tests/test_store_ep.py::test_hopfield_refinement_promotes_better_candidate` to pin the improvement path on a synthetic example.

## Experimental results
All runs use the default Qwen/Qwen2.5-1.5B-Instruct model (`scripts/run_m2_retrieval.sh` / `scripts/run_m2_uplift_headroom.sh` with `N=24`, `seed=7`, cue-only prompts).

| Mode | EM (relaxed) | P@1 | MRR | Notes |
| ---- | ------------ | --- | --- | ----- |
| B0 | 0.000 | – | – | Cue-only baseline without memory.
| B1 (Hopfield ON) | 0.208 | 0.375 | 0.543 | uplift +0.208 vs B0.
| B1 (Hopfield OFF) | 0.208 | 0.375 | 0.543 | identical to Hopfield ON on current Scenario A slice.

- Completion lift remains `0.0`; Hopfield neither improves nor regresses precision, suggesting the blend/guard avoids harm but still fails to surface better candidates on the acceptance slice.
- Retrieval-only probe (E2) reaches EM 0.375, setting an upper bound for adapter-informed answers. The gap indicates the adapter/prompt path (not Hopfield) is the limiting factor for EM uplift.

## Conclusions & Follow-up
1. **Algorithm health**: The episodic pipeline provides +0.208 EM uplift on the cue-only track, but Hopfield currently delivers *parity*, not an improvement. The readout is stabilized (no negative lift) yet still underdelivering versus design goals.
2. **Likely causes**: Scenario A cues are already very clean (exact lexical overlap), so ANN P@1 is high (0.375). Hopfield may only show a lift on harder variants (partial cues, higher collision rate) that we have not generated yet.
3. **Next steps**:
   - Generate a stress set with partial cues / higher interference (planning/validation-plan.md §2D) and rerun the probes to expose Hopfield’s benefits or highlight further issues.
   - Sweep `_hopfield_blend`, `_hopfield_margin`, and `hopfield_temperature` on that harder set; collect rank deltas to learn where the refinement helps.
   - Investigate adapter sensitivity (why EM lags retrieval-only). Without improving the downstream consumption of retrieved traces, Hopfield gains may remain hidden.
   - Consider feature scaling for the hashed embeddings (§5.3) or L2-normalising ANN scores before blending to better align magnitudes.

4. **Recommendation**: Continue Hopfield research under a dedicated milestone task. The current implementation is safe to ship for M2 (no regressions), but we should not expect it to provide measurable lift until we add harder evaluation sets and tune the blend parameters.

## Artifacts
- Reports regenerated under `reports/m2-retrieval-stack/` and `reports/m2-uplift-headroom/`.
- Unit/regression test: `tests/test_store_ep.py::test_hopfield_refinement_promotes_better_candidate`.

