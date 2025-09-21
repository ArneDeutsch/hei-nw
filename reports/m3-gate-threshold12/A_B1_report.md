# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.008
- Non-empty rate: 0.168
- Latency: 0.093s
- Adapter latency overhead: 0.034s

## Run config
- Seed: 3
- Requested records: 128
- Actual records: 256
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
| 0-1 | 64 | 0.000 | 0.000 | 0.006 | n/a |
| 1-3 | 64 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 64 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 64 | 0.000 | 0.000 | 0.025 | n/a |

## Compute
B0 attention FLOPs: 536450998272
B0 KV cache bytes: 245317632

## Retrieval
- P@1: 0.078
- MRR: 0.202
- Near-miss rate: 0.027
- Collision rate: 0.449
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠIvan, Ġwhat, :, Ġumbrella, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)