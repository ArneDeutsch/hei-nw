# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.027
- EM_strict: 0.000
- F1: 0.003
- Non-empty rate: 1.000
- Latency: 0.422s
- Adapter latency overhead: 0.167s

## Run config
- Seed: 13
- Requested records: 256
- Actual records: 256
- QA prompt style: plain
- QA max new tokens: 32
- QA stop: None
- QA answer hint: True
- Memory cap: 128 tokens
- Adapter residual scale: 0.200
- Hopfield: on (steps=1, temperature=1.0)

## Lag bins
| Lag bin | count | EM (relaxed) | EM_strict | F1 | Recall@k |
| ------- | ----- | ------------- | --------- | --- | -------- |
| 0-1 | 256 | 0.027 | 0.000 | 0.003 | n/a |
| 1-3 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 0 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 184956506112
B0 KV cache bytes: 141821952

## Retrieval
- P@1: 0.016
- MRR: 0.063
- Near-miss rate: 0.004
- Collision rate: 0.488
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 0.500
- Writes: 246/256 (rate 0.961; 47.1 writes/1k tokens)
  • Legacy normalization: 960.9 writes/1k records
- Pinned episodes: 19 | Reward flags: 33
- Precision: 0.520 | Recall: 1.000 | PR-AUC: 1.000
- Clutter rate: 0.961 (47.1 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: NO — 1024/1024 missing pointer; banned keys: episode_text
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52, 80, 52]
- Memory token preview: [who, :, Ġ, 1, 6, 1, 0, Ġwhat]

## Dataset notes
None