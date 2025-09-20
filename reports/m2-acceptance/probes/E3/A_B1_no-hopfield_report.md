# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 1.000
- EM_strict: 1.000
- F1: 1.000
- Non-empty rate: 1.000
- Latency: 0.046s
- Adapter latency overhead: -0.008s

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
- Hopfield: off

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 1-3 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 3-7 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 7-30 | 12 | 1.000 | 1.000 | 1.000 | n/a |

## Compute
B0 attention FLOPs: 30479081472
B0 KV cache bytes: 25337856

## Retrieval
- P@1: 0.375
- MRR: 0.543
- Near-miss rate: 0.167
- Collision rate: 0.292
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠGus, Ġwhat, :, Ġbackpack, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)