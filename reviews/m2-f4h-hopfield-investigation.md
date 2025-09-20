# M2-F4H Acceptance Re-run – Retrieval Stack Investigation

## Context
- Trigger: M2-F4H follow-up demanded headroom-aware acceptance evidence on Qwen/Qwen2.5-1.5B-Instruct after latest fixes.
- Observed failure: running `scripts/run_m2_acceptance.sh` produced a **headroom PASS but zero uplift**. Memory-dependent baseline (`reports/m2-acceptance/memory_dependent/*.json`) stayed at 0 EM and uplift pipeline aborted. However, `scripts/run_m2_retrieval.sh --hard-subset …` still showed +1.0 EM on the hard subset (with episodes injected). This mismatch warranted a deep dive.

## Symptoms & Reproduction
1. Acceptance run:
   - `bash scripts/run_m2_acceptance.sh`
   - Artifacts: `reports/m2-acceptance/summary.md` (Headroom PASS, completion lift −0.292, EM lift 0.000).
2. Memory-dependent baseline inspection:
   ```python
   python - <<'PY'
   import json
   from pathlib import Path
   data = json.loads(Path('reports/m2-acceptance/memory_dependent/A_B1_metrics.json').read_text())
   print('aggregate EM', data['aggregate']['em_relaxed'])  # => 0.0
   print(data['records'][0]['prediction'])                 # => 'The customer did.'
   PY
   ```
   Every prediction is a generic response (`'thief'`, `'John Doe'`, …) despite 100% non-empty rate.
3. Retrieval-only probe: `reports/m2-acceptance/probes/E2/A_B1_metrics.json` shows the retrieval gate falling back to filler strings (“Ivan (retrieval miss)”), EM ≈ P@1 ≈ 0.083, confirming the store rarely surfaces the correct trace.
4. Manual store interrogation (see snippets under `analysis/` history):
   ```python
   from transformers import AutoTokenizer
   from hei_nw.datasets import scenario_a
   from hei_nw.recall import RecallService

   records = scenario_a.generate(n=24, seed=7)
   tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
   service = RecallService.build(records, tokenizer, max_mem_tokens=64, hopfield_steps=2, hopfield_temperature=0.5)
   store = service.store

   for rec in records:
       hop = store.query(rec['cues'][0], return_m=1, use_hopfield=True, group_id=int(rec['group_id']), should_remember=bool(rec['should_remember']))
       no = store.query(rec['cues'][0], return_m=1, use_hopfield=False, ...)
       # Count matches between top retrieved group and truth
   ```
   Result: **Hopfield top-1 hits = 2 / 48**, **no-Hopfield hits = 13 / 48**. Hopfield actively destroys recall.

5. Inspecting `ANNIndex.search` output revealed the immediate culprit: the list returned by `EpisodicStore.query` is **ascending by score** (worst neighbour first). Example:
   ```python
   scores = [r['score'] for r in store.index.search(query, k=10)]
   # => [1.61, 1.69, …, 1.72]  # strictly increasing
   ```
   Despite that, `EpisodicStore.query` (src/hei_nw/store.py:384-409) assumes `results[0]` is the strongest candidate. All downstream logic—baseline top-1, Hopfield reordering, diagnostics—operates on this inverted ordering.

6. Because Scenario-A acceptance (with episodes in the prompt) still hands the answer explicitly in the user message, `B1` succeeds even with unusable memory tokens. But the memory-dependent baseline removes the episode text (`QA` prompt omits the `Episode:` block), so the language model relies on retrieved memory tokens only. Since the store hands it the **worst** trace, EM stays at zero and uplift collapses.

## Alignment with Design & Plans
- **Design spec (§5.4 & §9 Acceptance)** expects the ANN+Hopfield stack to *increase* partial-cue recall and mandates a Hopfield ablation with `completion_lift ≥ 0`. Current behaviour violates both the retrieval-only guard (P@1 is garbage) and the Hopfield lift requirement (negative completion lift recorded in `reports/m2-acceptance/probes_summary.txt`).
- **Validation plan (§1, §9)** explicitly relies on a functioning memory-dependent baseline when the headroom gate passes. The baseline currently fails even on a 24-item engineering slice, so statistical validation is impossible.
- The project plan (M2 DoD) treats positive Hopfield lift and healthy retrieval metrics as gating criteria before moving on. These gates are failing due to implementation, not inherent design limits.

**Conclusion:** The design remains sound—the ANN results simply need to be ordered correctly before Hopfield re-ranking. Once the base ordering is fixed, we can reassess whether Hopfield still hurts; if it does, we may have to revisit its initialization (e.g. energy scaling/temperature). For now, the evidence points to an *implementation bug* rather than a fundamental design flaw.

## Impact
- The memory-dependent fallback (critical for headroom-Blocked cases and for plausibility checks) is unusable: EM stays at 0.0 even when retrieval should help.
- Engineering gates in M2 cannot legitimately pass: Hopfield lift is negative, retrieval-only ≈ P@1 guard is technically met (both ~0.08) but only because both are abysmal.
- Downstream milestones (write gate, replay) assume a functioning retrieval core; proceeding without fixing this invalidates the design assumptions.

## Recommended Remediation (Executable Codex Tasks)
1. **Fix ANN ordering**
   - Ensure `ANNIndex.search` or `EpisodicStore.query` sorts neighbours by descending score before selecting top-k / computing diagnostics.
   - Add regression tests covering the ordering. (Current unit `tests/test_store_ann.py` can be extended.)

2. **Re-validate Hopfield lift**
   - After ordering fix, rerun `scripts/run_m2_acceptance.sh` and `scripts/run_m2_retrieval.sh --hard-subset …`.
   - Compare Hopfield vs. no-Hopfield P@1 / EM; update probes to ensure completion lift ≥ 0.

3. **Strengthen retrieval-only probes**
   - Enhance `tests/test_harness_dev_flags.py` to assert that retrieval-only mode returns the `answers[0]` string for rememberable items when ANN ordering is correct.
   - Augment `reports/m2-acceptance/probes` generation to flag when Hopfield reduces P@1 or when completion lift is negative.

4. **Re-run headroom acceptance w/ baseline**
   - Once retrieval is fixed, rerun `scripts/run_m2_acceptance.sh` and confirm that memory-dependent baseline produces non-zero EM and a valid uplift (or documented fallback if headroom re-blocks).

5. **Documentation note**
   - Update `documentation/hei-nw.md` (or README) summarising the retrieval fix to avoid regressions, citing the need for descending score order prior to Hopfield refinement.

## Suggested Verification Artifacts After Fix
- `reports/m2-acceptance/summary.md` showing positive completion lift and non-zero baseline EM.
- `reports/m2-acceptance/probes_summary.txt` with Hopfield ≥ no-Hopfield on Scenario A.
- Updated unit tests demonstrating ANN ordering and retrieval-only correctness.

Once these tasks are complete, rerun the acceptance flow to confirm the design meets its own DoD. EOF
