# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.221s

## Run config
- Seed: 21
- Requested records: 256
- Actual records: 512
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
| 0-1 | 128 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 128 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 128 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 128 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 204692275200
B0 KV cache bytes: 213700608

## Retrieval
- None

## Debug
- None

## Dataset notes
Hard negatives/confounders included (ratio 1.00)