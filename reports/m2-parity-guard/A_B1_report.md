# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.239s
- Adapter latency overhead: -0.008s

## Run config
- Seed: 7
- Requested records: 24
- Actual records: 48
- QA prompt style: chat
- QA max new tokens: 16
- QA stop: (empty string)
- QA answer hint: True
- Memory cap: 0 tokens
- Adapter residual scale: 0.200
- Hopfield: on (steps=1, temperature=1.0)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 19873910784
B0 KV cache bytes: 20404224

## Retrieval
- P@1: 0.292
- MRR: 0.493
- Near-miss rate: 0.125
- Collision rate: 0.333
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 1.500
- Writes: 24/48 (rate 0.500; 7.2 writes/1k tokens)
  • Legacy normalization: 500.0 writes/1k records
- Pinned episodes: 2 | Reward flags: 3
- Precision: 1.000 | Recall: 1.000 | PR-AUC: 1.000
- Clutter rate: 0.500 (7.2 writes/1k tokens)
- Pins-only PR-AUC: 1.000 | Clutter: 1.000 (13.7 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: yes
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
- Memory token preview: []

## Dataset notes
Hard negatives/confounders included (ratio 1.00)