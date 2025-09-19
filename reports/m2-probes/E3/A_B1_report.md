# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.011s
- Adapter latency overhead: 0.001s

## Run config
- Seed: 7
- Requested records: 16
- Actual records: 32
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
| 0-1 | 8 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 8 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 8 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 8 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 2618032
B0 KV cache bytes: 51776

## Retrieval
- P@1: 0.469
- MRR: 0.622
- Near-miss rate: 0.188
- Collision rate: 0.219
- Completion lift: -0.344
- Hopfield rank improved rate: 0.188

## Debug
- Memory token counts: [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
- Memory token preview: [who, :, ĠJudy, Ġwhat, :, Ġbackpack, Ġwhere, :]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)