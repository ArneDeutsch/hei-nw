# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.656s

## Run config
- Seed: 7
- Requested records: 64
- Actual records: 64
- QA prompt style: plain
- QA max new tokens: 32
- QA stop: None
- QA answer hint: True
- Memory cap: n/a
- Adapter residual scale: n/a
- Hopfield: on (steps=1, temperature=1.0)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 64 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 0 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 19818086400
B0 KV cache bytes: 23592960

Baseline attention FLOPs: 15988654080
Baseline KV cache bytes: 20791296

## Retrieval
- None

## Debug
- None

## Dataset notes
None