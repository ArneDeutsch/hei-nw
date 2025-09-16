"""Tests for relaxed exact match metric."""

from hei_nw.metrics import canonicalize, relaxed_em, strict_em


def test_relaxed_em_matches_punctuation_and_case() -> None:
    """Relaxed EM should ignore punctuation and case differences."""

    assert strict_em("Dana.", "dana") == 0.0
    assert relaxed_em("Dana.", "dana") == 1.0


def test_relaxed_em_collapses_whitespace_and_punctuation() -> None:
    """Extra spaces and punctuation should not affect relaxed EM."""

    prediction = "  DANA!!  "
    truth = "dana"
    assert canonicalize(prediction) == "dana"
    assert canonicalize(truth) == "dana"
    assert strict_em(prediction, truth) == 0.0
    assert relaxed_em(prediction, truth) == 1.0
