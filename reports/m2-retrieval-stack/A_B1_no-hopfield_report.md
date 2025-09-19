# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.000
- Non-empty rate: 1.000
- Latency: 0.168s
- Adapter latency overhead: 0.098s

## Run config
- Seed: 7
- Requested records: 24
- Actual records: 48
- QA prompt style: chat
- QA max new tokens: 8
- QA stop: \n
- QA answer hint: True
- Memory cap: 128 tokens
- Hopfield: off

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 1-3 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 12 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 12 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 25780027392
B0 KV cache bytes: 23285760

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
Hard negatives/confounders included (ratio 1.00)