# Evaluation Report

## Aggregate Metrics

- EM: 0.000
- F1: 0.000
- Latency: 0.694s
- Adapter latency overhead: 0.022s

## Lag bins
| Lag bin | count | EM | F1 | Recall@k |
| ------- | ----- | --- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 15469805568
B0 KV cache bytes: 18051072

## Retrieval
- P@1: 0.375
- MRR: 0.543
- Near-miss rate: 0.167
- Collision rate: 0.292
- Completion lift: 0.000

## Dataset notes
Hard negatives/confounders included (ratio 1.00)