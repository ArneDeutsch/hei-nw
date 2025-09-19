# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 1.000
- EM_strict: 1.000
- F1: 1.000
- Non-empty rate: 1.000
- Latency: 0.070s

## Run config
- Seed: 7
- Requested records: 24
- Actual records: 48
- QA prompt style: chat
- QA max new tokens: 8
- QA stop: \n
- QA answer hint: True
- Memory cap: n/a
- Hopfield: on (steps=2, temperature=0.5)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 1-3 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 3-7 | 12 | 1.000 | 1.000 | 1.000 | n/a |
| 7-30 | 12 | 1.000 | 1.000 | 1.000 | n/a |

## Compute
B0 attention FLOPs: 25715171328
B0 KV cache bytes: 23273472

## Retrieval
- None

## Debug
- None

## Dataset notes
Hard negatives/confounders included (ratio 1.00)