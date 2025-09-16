# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 1.000
- EM_strict: 1.000
- F1: 1.000
- Latency: 0.070s

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
Hard negatives/confounders included (ratio 1.00)# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Latency: 0.170s
- Adapter latency overhead: 0.100s

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 23980228608
B0 KV cache bytes: 22474752

## Retrieval
- P@1: 0.375
- MRR: 0.543
- Near-miss rate: 0.167
- Collision rate: 0.292
- Completion lift: 0.000

## Debug
- Memory token counts: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
- Memory token preview: [<, ep, is, odic, >Ċ, who, :, G]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Latency: 0.171s
- Adapter latency overhead: 0.101s

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 23980228608
B0 KV cache bytes: 22474752

## Retrieval
- P@1: 0.375
- MRR: 0.543
- Near-miss rate: 0.167
- Collision rate: 0.292
- Completion lift: -0.292

## Debug
- Memory token counts: [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
- Memory token preview: [<, ep, is, odic, >Ċ, who, :I, van]

## Dataset notes
Hard negatives/confounders included (ratio 1.00)