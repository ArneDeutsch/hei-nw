# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 1.000
- EM_strict: 1.000
- F1: 1.000
- Non-empty rate: 1.000
- Latency: 0.049s
- Adapter latency overhead: -0.003s

## Run config
- Seed: 7
- Requested records: 64
- Actual records: 128
- QA prompt style: chat
- QA max new tokens: 16
- QA stop: (empty string)
- QA answer hint: True
- Memory cap: 64 tokens
- Adapter residual scale: 0.200
- Hopfield: on (steps=2, temperature=0.5)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 32 | 1.000 | 1.000 | 1.000 | n/a |
| 1-3 | 32 | 1.000 | 1.000 | 1.000 | n/a |
| 3-7 | 32 | 1.000 | 1.000 | 1.000 | n/a |
| 7-30 | 32 | 1.000 | 1.000 | 1.000 | n/a |

## Compute
B0 attention FLOPs: 81569918976
B0 KV cache bytes: 67688448

## Retrieval
- P@1: 0.180
- MRR: 0.326
- Near-miss rate: 0.055
- Collision rate: 0.375
- Completion lift: -0.148
- Hopfield rank improved rate: 0.180

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠIvan, Ġwhat, :, Ġkey, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)