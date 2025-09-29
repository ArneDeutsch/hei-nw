# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.331s
- Adapter latency overhead: 0.201s

## Run config
- Seed: 21
- Requested records: 256
- Actual records: 256
- QA prompt style: chat
- QA max new tokens: 16
- QA stop: (empty string)
- QA answer hint: True
- Memory cap: 128 tokens
- Adapter residual scale: 0.200
- Hopfield: on (steps=2, temperature=0.5)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 256 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 0 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 420730060800
B0 KV cache bytes: 217300992

## Retrieval
- P@1: 0.023
- MRR: 0.085
- Near-miss rate: 0.000
- Collision rate: 0.477
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 1.500
- Writes: 128/256 (rate 0.500; 3.6 writes/1k tokens)
  • Legacy normalization: 500.0 writes/1k records
- Pinned episodes: 19 | Reward flags: 12
- Precision: 1.000 | Recall: 1.000 | PR-AUC: 1.000
- Clutter rate: 0.500 (3.6 writes/1k tokens)
- Label distribution: 128 positive / 128 negative (rate 0.500)
- Pins-only PR-AUC: 1.000 | Clutter: 1.000 (7.1 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: yes
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [44, 44, 48, 48, 48, 44, 44, 48, 44, 48, 44, 48, 44, 48, 48, 44, 44, 44, 44, 48, 48, 48, 48, 44, 48, 44, 44, 44, 44, 44, 44, 44, 44, 48, 44, 44, 44, 48, 48, 44, 44, 44, 48, 48, 48, 44, 48, 44, 48, 44, 48, 44, 48, 44, 48, 44, 44, 48, 48, 44, 44, 44, 44, 48, 44, 48, 48, 44, 48, 44, 48, 48, 44, 48, 48, 44, 44, 44, 48, 48, 44, 48, 44, 44, 44, 44, 44, 48, 48, 48, 48, 48, 44, 48, 44, 48, 48, 44, 44, 44, 48, 48, 44, 44, 44, 48, 48, 48, 48, 48, 44, 44, 48, 48, 48, 44, 48, 44, 44, 48, 44, 44, 44, 48, 44, 44, 44, 48, 44, 44, 44, 44, 48, 44, 44, 48, 44, 44, 48, 48, 48, 44, 48, 44, 44, 48, 48, 44, 48, 48, 48, 48, 44, 44, 48, 48, 48, 48, 48, 48, 44, 48, 44, 44, 44, 44, 44, 44, 44, 44, 44, 48, 44, 44, 44, 48, 48, 44, 44, 44, 48, 44, 48, 44, 44, 48, 44, 44, 48, 48, 48, 44, 48, 44, 48, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 48, 48, 44, 48, 48, 44, 48, 44, 44, 44, 44, 48, 44, 48, 44, 48, 48, 48, 48, 44, 44, 48, 48, 48, 48, 48, 44, 44, 44, 44, 44, 48, 44, 44, 48, 48, 44, 44, 48, 44, 44, 48, 44, 48, 44, 44, 48, 44, 44, 44]
- Memory token preview: [who, :, ĠIvan, Ġwhat, :, ĠGlob, ex, Ġwhere]

## Dataset notes
None