# M2 Acceptance Snapshot (label writes)

- **Scenario:** A, n = 48 (seed 7)
- **Mode:** B1 with label-driven writes (`USE_GATE_WRITES=0`)
- **Uplift:** `ΔEM = +0.542`, 95% CI `[0.396, 0.688]`
- **Retrieval:** `P@1 = 0.604`, `MRR = 0.757`
- **Oracle (E1):** EM = 1.000
- **Hopfield Ablation (E3):** No change (`P@1 = 0.312`)
- **Command:** `PYTHONPATH=src ALLOW_SMALL_SAMPLE=1 scripts/run_m2_acceptance.sh`
