from hei_nw.eval.report import build_markdown_report


def test_markdown_content_matches_summary() -> None:
    summary = {
        "aggregate": {
            "em": 0.5,
            "em_relaxed": 0.5,
            "em_strict": 0.25,
            "f1": 0.5,
            "latency": 0.1,
        },
        "lag_bins": [
            {
                "lag_bin": "0-1",
                "count": 1,
                "em": 0.5,
                "em_relaxed": 0.5,
                "em_strict": 0.25,
                "f1": 0.5,
                "recall_at_k": None,
            }
        ],
        "compute": {
            "b0": {"attention_flops": 0, "kv_cache_bytes": 0},
            "baseline": {"attention_flops": 1, "kv_cache_bytes": 2},
        },
        "dataset": {"hard_negative_ratio": 1.0},
        "run_config": {
            "seed": 7,
            "requested_records": 24,
            "actual_records": 24,
            "qa": {
                "prompt_style": "chat",
                "max_new_tokens": 16,
                "stop": "",
                "answer_hint": True,
            },
            "memory": {"max_tokens": 128},
            "adapter": {"scale": 0.25},
            "hopfield": {"enabled": True, "steps": 2, "temperature": 0.5},
        },
        "debug": {"mem_len": [1, 2], "mem_preview": ["who", "Alice"]},
    }
    md = build_markdown_report(summary, scenario="A")
    assert "- EM (relaxed): 0.500" in md
    assert "- EM_strict: 0.250" in md
    assert "- F1: 0.500" in md
    assert "- Latency: 0.100s" in md
    assert "- Seed: 7" in md
    assert "- Requested records: 24" in md
    assert "- Actual records: 24" in md
    assert "- QA prompt style: chat" in md
    assert "- QA max new tokens: 16" in md
    assert "- QA stop: (empty string)" in md
    assert "- QA answer hint: True" in md
    assert "- Memory cap: 128 tokens" in md
    assert "- Adapter residual scale: 0.250" in md
    assert "- Hopfield: on (steps=2, temperature=0.5)" in md
    assert "| 0-1 | 1 | 0.500 | 0.250 | 0.500 | n/a |" in md
    assert "Baseline attention FLOPs: 1" in md
    assert "Baseline KV cache bytes: 2" in md
    assert "Hard negatives/confounders included (ratio 1.00)" in md
    assert "- Memory token counts: [1, 2]" in md
    assert "- Memory token preview: [who, Alice]" in md
