# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.354s
- Adapter latency overhead: 0.182s

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
B0 attention FLOPs: 440853417984
B0 KV cache bytes: 222541824

## Retrieval
- P@1: 0.023
- MRR: 0.079
- Near-miss rate: 0.000
- Collision rate: 0.477
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 1.500
- Writes: 128/256 (rate 0.500; 3.5 writes/1k tokens)
  • Legacy normalization: 500.0 writes/1k records
- Pinned episodes: 18 | Reward flags: 28
- Precision: 1.000 | Recall: 1.000 | PR-AUC: 1.000
- Clutter rate: 0.500 (3.5 writes/1k tokens)
- Label distribution: 128 positive / 128 negative (rate 0.500)
- Pins-only PR-AUC: 1.000 | Clutter: 1.000 (7.1 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: yes
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
- Memory token preview: [who, :, Ġ, 1, 4, 0, 9, Ġwhat]

## Dataset notes
None