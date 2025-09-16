#!/usr/bin/env bash
set -euo pipefail

# Ensure we run from the repository root so relative paths resolve correctly.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH=src

OUTDIR="$(mktemp -d)"
export OUTDIR
trap 'rm -rf "${OUTDIR}"' EXIT

MODEL="${MODEL:-tests/models/tiny-gpt2}"
export MODEL

python -m hei_nw.eval.harness --mode B0 --scenario A -n 4 --seed 0 \
  --model "${MODEL}" --outdir "${OUTDIR}/plain" \
  --qa.prompt_style plain --qa.max_new_tokens 8 --qa.stop $'\n'

python -m hei_nw.eval.harness --mode B0 --scenario A -n 4 --seed 0 \
  --model "${MODEL}" --outdir "${OUTDIR}/chat"

python <<'PY'
import json
import os
from pathlib import Path
from unittest import mock

from hei_nw.eval import harness
from hei_nw.models.base import load_base

base = Path(os.environ["OUTDIR"])
tok, _, _ = load_base(model_id=str(Path(os.environ.get("MODEL", "tests/models/tiny-gpt2"))), quant_4bit=False)
for mode in ("plain", "chat"):
    data = json.loads((base / mode / "A_B0_metrics.json").read_text(encoding="utf8"))
    for idx, record in enumerate(data.get("records", [])):
        prediction = record.get("prediction", "")
        tokenized = tok(prediction, add_special_tokens=False)
        length = len(tokenized.get("input_ids", []))
        if length > 8:
            raise SystemExit(f"{mode} record {idx} produced {length} tokens (> 8)")

records = [
    {
        "episode_text": "Dana met her friend at the cafe.",
        "cues": ["Who met a friend?"],
        "answers": ["Dana"],
    }
]
qa = harness.QAPromptSettings(prompt_style="plain", max_new_tokens=8, stop="\n", answer_hint=True)
geom = harness.ModelGeometry(layers=1, hidden=1, heads=1, dtype="float32")

def fake_generate(prompt, **kwargs):
    return {"text": "Dana.", "generated_tokens": 2, "prompt_tokens": 4}

with mock.patch("hei_nw.models.base.generate", side_effect=fake_generate):
    items, _ = harness._evaluate_records(records, geom, qa)

relaxed = items[0].em_relaxed
strict = items[0].em_strict
if not (relaxed == 1.0 and strict == 0.0):
    raise SystemExit(
        f"Expected relaxed EM 1.0 and strict EM 0.0, got {relaxed} / {strict}"
    )
PY
