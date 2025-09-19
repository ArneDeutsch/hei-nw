# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 1.000
- EM_strict: 1.000
- F1: 1.000
- Non-empty rate: 1.000
- Latency: 0.067s

## Run config
- Seed: 7
- Requested records: 64
- Actual records: 128
- QA prompt style: chat
- QA max new tokens: 8
- QA stop: \n
- QA answer hint: True
- Memory cap: n/a
- Hopfield: on (steps=1, temperature=1.0)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 32 | 1.000 | 1.000 | 1.000 | n/a |
| 1-3 | 32 | 1.000 | 1.000 | 1.000 | n/a |
| 3-7 | 32 | 1.000 | 1.000 | 1.000 | n/a |
| 7-30 | 32 | 1.000 | 1.000 | 1.000 | n/a |

## Compute
B0 attention FLOPs: 68842475520
B0 KV cache bytes: 62183424

## Retrieval
- None

## Debug
- None

## Dataset notes
Hard negatives/confounders included (ratio 1.00)