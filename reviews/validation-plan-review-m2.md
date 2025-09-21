# Validation Plan Review (M2)

## Scope
- planning/design.md (§5–§9)
- planning/validation-plan.md (all)
- planning/milestone-2-plan.md (§1–§6)
- reports under reports/m2-retrieval-stack/, reports/m2-acceptance/, reports/m2-uplift-headroom/
- harness implementation in src/hei_nw/eval/harness.py and supporting modules

Runs were re-issued with the default Qwen/Qwen2.5-1.5B-Instruct model before reviewing. The refreshed artefacts now under `reports/m2-uplift-headroom/` reflect that rerun.

## Findings

1. **B1 prompts leak the ground-truth episode text.**
   - In `reports/m2-retrieval-stack/A_B1_metrics.json`, every recorded prompt begins with the full episode (for example, Fay leaving the umbrella). Meanwhile `reports/m2-retrieval-stack/A_B0_metrics.json` omits the episode entirely and only asks the cue.
   - This violates planning/design.md §5 and validation-plan.md §2, where B1 is supposed to rely on retrieved memory tokens and the partial cue, not a verbatim copy of the episode.
   - Because the answer is present verbatim in the B1 prompt, EM/CI values in `reports/m2-retrieval-stack/` and `reports/m2-acceptance/summary.md` cannot be taken as evidence of retrieval-stack quality. The uplift is dominated by prompt leakage, not DG→ANN→Hopfield retrieval.

2. **Memory-dependent baseline reveals the real uplift is still modest.**
   - After rerunning `scripts/run_m2_uplift_headroom.sh` with Qwen, `reports/m2-uplift-headroom/A_B1_metrics.json` shows EM≈0.146 with the episode removed from the prompt. That is the actual configuration envisaged by validation-plan.md §1–§4.
   - This exposes a ~15-point lift from memory tokens—better than zero, but far below the +0.30 EM target spelled out in design §9 and validation-plan §9. We are therefore not yet meeting the headroom-aware acceptance bar once the leakage is removed.

3. **Hopfield completion currently delivers no lift.**
   - `reports/m2-retrieval-stack/A_B1_metrics.json` records `completion_lift = 0.0`, and the no-Hopfield run in the same directory also scores EM=1.0 because of prompt leakage. In the leakage-free run (`reports/m2-uplift-headroom/A_B1_metrics.json`), completion_lift is negative (−0.292).
   - Validation-plan.md §3 and milestone-2-plan §6 expect Hopfield to raise MRR/P@1 or at minimum be non-negative. The current metrics suggest the readout is either misconfigured or being applied outside its design envelope.

4. **B0 default prompts are already the “memory-dependent” variant.**
   - `_qa_settings_from_args` forces `omit_episode=True` for mode B0. That means even in engineering runs we never observe the base model’s behaviour when it *does* receive the episode text. This is inconsistent with validation-plan.md §1, which only prescribes stripping the episode when the headroom gate blocks uplift.
   - As long as B1 still sees the episode, this asymmetry makes it impossible to attribute uplift to HEI-NW.

5. **Scenario/metric coverage is still limited to B0/B1 on Scenario A.**
   - Validation-plan.md §2–§4 and design §9 call for Scenario B–E coverage, long-context/RAG baselines, and eventually B2/B3 replay validation. These pathways are not yet implemented in the harness (MODE_HANDLERS only covers B0/B1).
   - This is acceptable for M2, but it means the validation plan remains largely aspirational; we should mark the outstanding pieces clearly before entering M3.

6. **“Episode: (none)” in the historical JSON was a stale artifact.**
   - The older `reports/m2-uplift-headroom/A_B0_metrics.json` emitted “Episode: (none)” because the prompt builder inserts that sentinel when the record’s `episode_text` is empty and `omit_episode=False`.
   - Rerunning the script with the current harness populates `run_config.qa.memory_dependent_baseline = true` and omits the episode block entirely, eliminating the confusing placeholder.

## Recommendations

1. **Fix the prompt asymmetry immediately.**
   - Ensure both B0 and B1 use the same cue-only prompt for the statistical track; rely on `mem_tokens` (and optional `memory_prompt`) to inject retrieved content. This is required before claiming any uplift.

2. **Re-evaluate acceptance once prompts are symmetric.**
   - With the leakage removed, re-run the headroom flow and compute the uplift/CI. Use the result to decide whether HEI-NW is ready for M3; current numbers (ΔEM ≈ +0.146) fall short of the +0.30 target.

3. **Investigate Hopfield completion.**
   - Instrument `src/hei_nw/store.py::EpisodicStore.query` to dump pre/post ranks and analyse why `completion_lift` is ≤0.0. Consider tuning `hopfield_temperature`, number of steps, or normalising candidate vectors as in design §5.4.

4. **Clarify the validation road-map for M3.**
   - Document which portions of validation-plan.md remain TODO (B/C/D/E, B2/B3 modes, long-context/RAG baselines) so they can be scheduled explicitly in milestone 3 rather than assumed complete.

5. **Housekeeping.**
   - Drop or archive stale reports that still contain the `(none)` sentinel; keep only leakage-free artefacts to avoid future confusion.

Until these gaps are addressed, I recommend *not* marking validation-plan.md as satisfied nor starting Milestone 3.
