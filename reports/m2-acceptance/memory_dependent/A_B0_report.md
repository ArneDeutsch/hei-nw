# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.086s

## Run config
- Seed: 7
- Requested records: 24
- Actual records: 48
- QA prompt style: chat
- QA max new tokens: 16
- QA stop: (empty string)
- QA answer hint: True
- Memory cap: n/a
- Adapter residual scale: n/a
- Hopfield: on (steps=2, temperature=0.5)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 18016653312
B0 KV cache bytes: 19445760

## Retrieval
- None

## Debug
- None

## Dataset notes
Hard negatives/confounders included (ratio 1.00)