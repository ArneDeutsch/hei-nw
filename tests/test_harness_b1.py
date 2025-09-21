from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from hei_nw.eval.harness import (
    DevIsolationSettings,
    ModelGeometry,
    QAPromptSettings,
    _aggregate_metrics,
    _evaluate_mode_b1,
)

TINY_MODEL = Path(__file__).resolve().parent.parent / "models" / "tiny-gpt2"


def _run(tmp_path: Path, n: int) -> dict:
    outdir = tmp_path / f"out_{n}"
    cmd = [
        sys.executable,
        "-m",
        "hei_nw.eval.harness",
        "--mode",
        "B1",
        "--scenario",
        "A",
        "-n",
        str(n),
        "--seed",
        "0",
        "--outdir",
        str(outdir),
        "--model",
        str(TINY_MODEL),
    ]
    subprocess.run(cmd, check=True)  # noqa: S603
    json_files = list(outdir.glob("*_metrics.json"))
    md_files = list(outdir.glob("*_report.md"))
    assert json_files and md_files
    data = json.loads(json_files[0].read_text())
    assert "aggregate" in data and "records" in data
    return data


@pytest.mark.slow
def test_b1_runs_and_writes_reports(tmp_path: Path) -> None:
    data = _run(tmp_path, 2)
    retrieval = data.get("retrieval")
    assert retrieval is not None
    for v in retrieval.values():
        assert isinstance(v, int | float) and math.isfinite(float(v))
    _run(tmp_path, 0)


def test_retrieval_only_em_tracks_p_at_1(monkeypatch: pytest.MonkeyPatch) -> None:
    class SimpleTokenizer:
        def __init__(self) -> None:
            self._vocab: dict[str, int] = {}

        def tokenize(self, text: str) -> list[str]:
            return text.split()

        def __call__(self, text: str) -> dict[str, list[int]]:
            tokens = self.tokenize(text)
            ids = [self._vocab.setdefault(tok, len(self._vocab) + 1) for tok in tokens]
            return {"input_ids": ids}

        def convert_ids_to_tokens(self, token_ids: list[int]) -> list[str]:
            inv_vocab = {idx: tok for tok, idx in self._vocab.items()}
            return [inv_vocab.get(tid, f"tok{tid}") for tid in token_ids]

    records = [
        {
            "episode_text": "Alice met Bob at the park.",
            "cues": ["Who met Bob?"],
            "answers": ["Alice"],
            "group_id": 1,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 1.2,
                "novelty": 0.9,
                "reward": False,
                "pin": False,
            },
        },
        {
            "episode_text": "Carol saw Dave near the river.",
            "cues": ["Who saw Dave?"],
            "answers": ["Carol"],
            "group_id": 2,
            "should_remember": True,
            "lag": 0,
            "gate_features": {
                "surprise": 1.1,
                "novelty": 0.85,
                "reward": False,
                "pin": False,
            },
        },
    ]

    plan: dict[int, dict[str, Any]] = {}
    for idx, rec in enumerate(records):
        plan[int(rec["group_id"])] = {
            "truth": str(rec["answers"][0]),
            "decoy_answer": f"decoy-{idx}",
            "decoy_group": int(rec["group_id"]) + 100,
            "top_correct": idx == 0,
        }

    tokenizer = SimpleTokenizer()

    class RetrievalProbeService:
        def __init__(self, mapping: dict[int, dict[str, Any]], tok: SimpleTokenizer) -> None:
            self._mapping = mapping
            self.store = self
            self.tokenizer = tok
            self.return_m = 2
            self.max_mem_tokens = 32

        def query(
            self,
            _cue: str,
            *,
            return_m: int,
            use_hopfield: bool,
            group_id: int,
            should_remember: bool,
        ) -> dict[str, Any]:  # noqa: ARG002
            entry = self._mapping[group_id]
            truth = entry["truth"]
            decoy_answer = entry["decoy_answer"]
            decoy_group = entry["decoy_group"]
            truth_entry = {
                "group_id": group_id,
                "answers": [truth],
                "trace": {"group_id": group_id, "answers": [truth]},
            }
            decoy_entry = {
                "group_id": decoy_group,
                "answers": [decoy_answer],
                "trace": {"group_id": decoy_group, "answers": [decoy_answer]},
            }
            if entry["top_correct"]:
                candidates = [truth_entry, decoy_entry]
            else:
                candidates = [decoy_entry, truth_entry]
            diag = {"near_miss": False, "collision": False, "rank_delta": 0}
            selected = candidates[: max(1, min(return_m, len(candidates)))]
            return {
                "selected": selected,
                "candidates": candidates,
                "diagnostics": diag,
                "baseline_candidates": candidates,
                "baseline_diagnostics": diag,
            }

    def fake_generate(*_: Any, **__: Any) -> dict[str, Any]:
        return {"text": "model-output", "prompt_tokens": 2, "generated_tokens": 1}

    monkeypatch.setattr("hei_nw.models.base.generate", fake_generate)
    monkeypatch.setattr(
        "hei_nw.models.base.build_default_adapter", lambda _model, *, scale=0.2: object()
    )

    def fake_recall_service(
        recs: list[dict[str, Any]],
        tok: SimpleTokenizer,
        max_mem_tokens: int,
        **_: Any,
    ) -> RetrievalProbeService:
        assert max_mem_tokens == 128
        assert recs == records
        return RetrievalProbeService(plan, tok)

    monkeypatch.setattr("hei_nw.eval.harness.RecallService.build", fake_recall_service)

    geometry = ModelGeometry(layers=2, hidden=8, heads=1, dtype="float32")
    qa_settings = QAPromptSettings(
        prompt_style="plain", max_new_tokens=4, stop=None, answer_hint=True
    )

    items, _compute, _baseline, extra = _evaluate_mode_b1(
        records,
        baseline="none",
        model=object(),
        tok=tokenizer,
        geom=geometry,
        no_hopfield=False,
        dg_keyer=None,
        qa=qa_settings,
        hopfield=None,
        dev=DevIsolationSettings(retrieval_only=True),
    )

    metrics = _aggregate_metrics(items)
    em_strict = float(metrics["em_strict"])
    p_at_1 = float(extra["retrieval"]["p_at_1"])

    assert pytest.approx(em_strict, abs=0.05) == p_at_1
    gate_summary = extra["gate"]
    assert gate_summary["writes"] == len(records)
    assert gate_summary["weights"]["alpha"] == pytest.approx(1.0)
