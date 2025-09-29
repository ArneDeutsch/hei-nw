# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.199
- EM_strict: 0.098
- F1: 0.098
- Non-empty rate: 1.000
- Latency: 0.160s

## Run config
- Seed: 21
- Requested records: 256
- Actual records: 256
- QA prompt style: chat
- QA max new tokens: 16
- QA stop: (empty string)
- QA answer hint: True
- Memory cap: n/a
- Adapter residual scale: n/a
- Hopfield: on (steps=1, temperature=1.0)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 256 | 0.199 | 0.098 | 0.098 | n/a |
| 1-3 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 0 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 92343250944
B0 KV cache bytes: 101431296

## Retrieval
- None

## Debug
- None

## Dataset notes
None