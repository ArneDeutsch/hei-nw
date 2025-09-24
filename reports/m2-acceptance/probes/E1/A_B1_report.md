# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.258
- Non-empty rate: 1.000
- Latency: 0.159s
- Adapter latency overhead: 0.016s

## Run config
- Seed: 7
- Requested records: 24
- Actual records: 48
- QA prompt style: chat
- QA max new tokens: 8
- QA stop: (empty string)
- QA answer hint: True
- Memory cap: 64 tokens
- Adapter residual scale: 0.200
- Hopfield: on (steps=2, temperature=0.5)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | 0.278 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | 0.283 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | 0.241 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.232 | n/a |

## Compute
B0 attention FLOPs: 35298213888
B0 KV cache bytes: 27267072

## Retrieval
- P@1: 0.292
- MRR: 0.493
- Near-miss rate: 0.125
- Collision rate: 0.333
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 1.500
- Writes: 24/48 (rate 0.500; 5.4 writes/1k tokens)
  • Legacy normalization: 500.0 writes/1k records
- Pinned episodes: 2 | Reward flags: 3
- Precision: 1.000 | Recall: 1.000 | PR-AUC: 1.000
- Clutter rate: 0.500 (5.4 writes/1k tokens)
- Pins-only PR-AUC: 1.000 | Clutter: 1.000 (10.9 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: yes
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [23, 24, 23, 24, 23, 23, 24, 23, 23, 23, 25, 23, 23, 23, 24, 23, 23, 23, 23, 23, 23, 23, 23, 24, 23, 24, 23, 24, 23, 23, 23, 23, 23, 23, 25, 23, 23, 23, 24, 23, 23, 23, 23, 23, 23, 23, 23, 24]
- Memory token preview: [who, :, ĠFay, Ġwhat, :, Ġumbrella, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)