# M2 Acceptance (gate writes, τ = 1.0)

- **Scenario:** A, n = 48 (seed 7)
- **Mode:** B1 with gate-driven writes (`USE_GATE_WRITES=1`, `τ=1.0`)
- **Uplift:** `ΔEM = +0.417`, 95% CI `[0.271, 0.562]`
- **Retrieval:** `P@1 = 0.583`, `MRR = 0.722`
- **Oracle (E1):** EM = 1.000
- **Hopfield Ablation (E3):** identical to baseline (`P@1 = 0.583`)
- **Gate sweep reference:** see `/tmp/m3-calibration-sweep.tar.gz` (τ→writes/1k tokens)
- **Command:** `PYTHONPATH=src USE_GATE_WRITES=1 GATE_THRESHOLD=1.0 scripts/run_m2_acceptance.sh`
