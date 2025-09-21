# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 0.000
- Latency: 0.040s
- Adapter latency overhead: -0.027s

## Run config
- Seed: 13
- Requested records: 48
- Actual records: 96
- QA prompt style: chat
- QA max new tokens: 16
- QA stop: None
- QA answer hint: True
- Memory cap: 64 tokens
- Adapter residual scale: 0.200
- Hopfield: on (steps=1, temperature=1.0)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 24 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 24 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 24 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 24 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 191833571328
B0 KV cache bytes: 89899008

## Retrieval
- P@1: 0.229
- MRR: 0.405
- Near-miss rate: 0.104
- Collision rate: 0.375
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 1.500
- Writes: 48/96 (rate 0.500; 500.0 writes/1k)
- Pinned episodes: 3 | Reward flags: 5
- Precision: 1.000 | Recall: 1.000 | PR-AUC: 1.000
- Clutter rate: 0.500 (500.0 writes/1k)
- Calibration bins: 10
- Pointer-only payload: NO — 384/384 missing pointer; banned keys: episode_text
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠFay, Ġwhat, :, Ġjacket, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)