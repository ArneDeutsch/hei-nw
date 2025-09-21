from __future__ import annotations

from collections.abc import Sequence
from statistics import mean
from typing import Any, cast

import pytest
from transformers import PreTrainedModel

from hei_nw.eval import harness
from hei_nw.metrics import exact_match, token_f1
from hei_nw.models.base import build_default_adapter, load_base
from hei_nw.testing import DUMMY_MODEL_ID
from hei_nw.utils.seed import set_global_seed

set_global_seed(0)
_TOK, MODEL, _PIPE = load_base(model_id=DUMMY_MODEL_ID, quant_4bit=False)
GEOM = harness._model_geometry(MODEL)
ADAPTER = build_default_adapter(cast(PreTrainedModel, MODEL))


def _predict(
    records: Sequence[dict[str, Any]],
    adapter: object | None = None,
) -> tuple[list[str], list[str]]:
    items, _ = harness._evaluate_records(
        records,
        GEOM,
        harness.QAPromptSettings(),
        adapter=adapter,
        mem_tokens=None,
    )
    preds = [it.prediction for it in items]
    truths = [it.truth for it in items]
    return preds, truths


@pytest.mark.parametrize("scenario", list("ABCDE"))
def test_b1_matches_b0_predictions(scenario: str) -> None:
    records = harness.SCENARIOS[scenario](n=4, seed=0)
    set_global_seed(0)
    b0_preds, truths = _predict(records)
    set_global_seed(0)
    b1_preds, _ = _predict(records, adapter=ADAPTER)
    if b0_preds != b1_preds:
        em0 = mean(exact_match(p, t) for p, t in zip(b0_preds, truths, strict=True))
        em1 = mean(exact_match(p, t) for p, t in zip(b1_preds, truths, strict=True))
        f10 = mean(token_f1(p, t) for p, t in zip(b0_preds, truths, strict=True))
        f11 = mean(token_f1(p, t) for p, t in zip(b1_preds, truths, strict=True))
        assert abs(em0 - em1) <= 0.1
        assert abs(f10 - f11) <= 0.1
    else:
        assert b0_preds == b1_preds
