# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 1.000
- EM_strict: 0.938
- F1: 0.938
- Non-empty rate: 1.000
- Latency: 0.048s
- Adapter latency overhead: -0.007s

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
| 0-1 | 12 | 1.000 | 0.833 | 0.833 | n/a |
| 1-3 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 3-7 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 7-30 | 12 | 1.000 | 0.917 | 0.917 | n/a |

## Compute
B0 attention FLOPs: 30479081472
B0 KV cache bytes: 25337856

## Retrieval
- P@1: 0.083
- MRR: 0.190
- Near-miss rate: 0.042
- Collision rate: 0.458
- Completion lift: -0.292
- Hopfield rank improved rate: 0.125

## Debug
- Memory token counts: [23, 23, 24, 23, 23, 23, 24, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 23, 23, 23, 24, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 23, 23, 23, 23, 23, 23]
- Memory token preview: [who, :, ĠFay, Ġwhat, :, Ġumbrella, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)