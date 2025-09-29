# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.375
- EM_strict: 0.375
- F1: 0.375
- Non-empty rate: 1.000
- Latency: 0.158s
- Adapter latency overhead: 0.017s

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
| 0-1 | 12 | 0.250 | 0.250 | 0.250 | n/a |
| 1-3 | 12 | 0.417 | 0.417 | 0.417 | n/a |
| 3-7 | 12 | 0.333 | 0.333 | 0.333 | n/a |
| 7-30 | 12 | 0.500 | 0.500 | 0.500 | n/a |

## Compute
B0 attention FLOPs: 116375003136
B0 KV cache bytes: 49508352

## Retrieval
- P@1: 0.583
- MRR: 0.722
- Near-miss rate: 0.208
- Collision rate: 0.083
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 1.000
- Writes: 22/48 (rate 0.458; 2.7 writes/1k tokens)
  • Legacy normalization: 458.3 writes/1k records
- Pinned episodes: 2 | Reward flags: 3
- Precision: 1.000 | Recall: 0.917 | PR-AUC: 1.000
- Clutter rate: 0.458 (2.7 writes/1k tokens)
- Label distribution: 24 positive / 24 negative (rate 0.500)
- Pins-only PR-AUC: 1.000 | Clutter: 1.000 (6.0 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: yes
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠFay, Ġwhat, :, Ġumbrella, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)