# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.125
- EM_strict: 0.125
- F1: 0.125
- Non-empty rate: 1.000
- Latency: 0.000s
- Adapter latency overhead: -3.829s

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
| 0-1 | 12 | 0.083 | 0.083 | 0.083 | n/a |
| 1-3 | 12 | 0.083 | 0.083 | 0.083 | n/a |
| 3-7 | 12 | 0.333 | 0.333 | 0.333 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 0
B0 KV cache bytes: 0

## Retrieval
- P@1: 0.375
- MRR: 0.543
- Near-miss rate: 0.167
- Collision rate: 0.292
- Completion lift: -0.292
- Hopfield rank improved rate: 0.125

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠIvan, Ġwhat, :, Ġkey, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)