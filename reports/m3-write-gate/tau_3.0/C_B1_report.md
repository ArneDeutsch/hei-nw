# Evaluation Report

## Aggregate Metrics

- EM (relaxed): 0.000
- EM_strict: 0.000
- F1: 0.008
- Non-empty rate: 1.000
- Latency: 0.597s
- Adapter latency overhead: 0.321s

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
| 0-1 | 256 | 0.000 | 0.000 | 0.008 | n/a |
| 1-3 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 3-7 | 0 | 0.000 | 0.000 | 0.000 | n/a |
| 7-30 | 0 | 0.000 | 0.000 | 0.000 | n/a |

## Compute
B0 attention FLOPs: 252978819072
B0 KV cache bytes: 167983104

## Retrieval
- P@1: 0.008
- MRR: 0.027
- Near-miss rate: 0.000
- Collision rate: 0.145
- Completion lift: 0.000
- Hopfield rank improved rate: 0.000

## Write gate
- Threshold τ: 3.000
- Writes: 39/256 (rate 0.152; 5.6 writes/1k tokens)
  • Legacy normalization: 152.3 writes/1k records
- Pinned episodes: 19 | Reward flags: 33
- Precision: 1.000 | Recall: 0.305 | PR-AUC: 1.000
- Clutter rate: 0.152 (5.6 writes/1k tokens)
- Calibration bins: 10
- Pointer-only payload: NO — 1024/1024 missing pointer; banned keys: episode_text
- Calibration plot: not generated
- Trace samples: 5 (see metrics JSON)

## Debug
- Memory token counts: [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80]
- Memory token preview: [who, :, Ġ, 6, 0, 4, 3, Ġwhat]

## Dataset notes
None