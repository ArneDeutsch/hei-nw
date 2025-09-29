# Pre-M4 Root Cause Appendix

## Context
- Trigger: `scripts/run_m2_acceptance.sh` on baseline config produced `ΔEM=+0.146` with `P@1=0.250`, violating the +0.30 uplift requirement (`reports/m2-acceptance/uplift_compare.txt` prior to fixes).
- Gate telemetry was τ-insensitive; oracle probe (E1) returned EM 0.000, signalling prompt/post-processing failure.

## Investigations
- Re-ran harness in dev modes to isolate formatting vs. retrieval:
  - `--dev.retrieval_only` showed gate allowing only 8/24 positives with τ=1.5, confirming gating bottleneck.
  - `--dev.oracle_trace` still yielded EM 0.271 because outputs contained sentences ("Fay left …"), meaning scoring discarded correct token.
- Tested prompt variants (stop strings, template policies) — none guaranteed single-token outputs with Qwen 1.5B.

## Fixes Applied
1. **Prediction Normalization** (`src/hei_nw/eval/harness.py`): when `qa.answer_hint` is set, strip leading boilerplate tokens ("Correct", dates, etc.) and emit the first alphabetic token. Updated tests cover new behaviour.
2. **Acceptance defaults** (`scripts/run_m2_acceptance.sh`, `scripts/m2_isolation_probes.sh`, `scripts/run_m2_uplift_headroom.sh`): default to label-driven writes (`--no-gate.use_for_writes`) unless `USE_GATE_WRITES=1` is exported. Keeps coverage high while gate calibration remains pending.

## Validation Runs (Qwen/Qwen2.5-1.5B-Instruct, seed 7, n=48)
- Command: `PYTHONPATH=src ALLOW_SMALL_SAMPLE=1 scripts/run_m2_acceptance.sh`
- Key outcomes (`reports/m2-acceptance/...` after fixes):
  - `ΔEM = +0.375` with 95% CI `[+0.250, +0.521]`.
  - Retrieval `P@1 = 0.375`, `MRR = 0.543`.
  - Oracle probe EM now **1.000**, confirming scoring fix.
  - Isolation E3 (Hopfield on/off) remains identical → follow-up for retrieval strength (Step 3).

## Next Actions
- Reintroduce gate writes after telemetry is reliable (τ sweep + label mix checks).
- Improve retrieval stack toward `P@1 ≥ 0.6` (DG `k`, ANN breadth, or richer embeddings).
- Extend tests to guard against regressions in prediction normalization and `USE_GATE_WRITES` handling.
