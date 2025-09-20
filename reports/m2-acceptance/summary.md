# M2 Acceptance Summary

**Headroom Gate:** PASS (EM_B0 = 0.000 < 0.7).\n
**Retrieval (chat prompt)**\n- EM_B0: 0.000
- Non-empty rate (B0): 1.000
- EM_B1: 1.000
- EM_B1 (no Hopfield): 1.000
- Hopfield lift: +0.000
- Non-empty rate (B1): 1.000
- Retrieval: P@1=0.375, MRR=0.526, Near-miss=0.167, Collision=0.292, CompletionΔ=+0.000

**Memory-dependent uplift**\n- EM_B0 (no episode prompt): 0.000
- Non-empty rate (B0): 1.000
- EM_B1 (with memory tokens): 0.146
- Non-empty rate (B1): 1.000
- EM lift: +0.146
- 95% bootstrap CI: [+0.062, +0.250]
- Records evaluated: 48

See probes summary at reports/m2-acceptance/probes_summary.txt and uplift details in reports/m2-acceptance/uplift_compare.txt.
