"""Scenario A: episodic one-shot stories with hard negatives and lag bins."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

NAMES = [
    "Alice",
    "Bob",
    "Charlie",
    "Dana",
    "Eli",
    "Fay",
    "Gus",
    "Hana",
    "Ivan",
    "Judy",
]

ITEMS = [
    "backpack",
    "notebook",
    "umbrella",
    "jacket",
    "phone",
    "laptop",
    "key",
    "bottle",
    "wallet",
    "book",
]

PLACES = [
    "Café Lumen",
    "the library",
    "the market",
    "the museum",
    "the park",
    "the station",
    "the bakery",
    "the office",
    "the school",
    "the beach",
]

_BASE_DATE = datetime(2025, 1, 1)


def _cycle_lag(idx: int, rng: random.Random, bins: Sequence[int]) -> int:
    """Return a lag value cycling through *bins* intervals."""
    if len(bins) < 2:
        raise ValueError("lag_spec bins must have at least two entries")
    span_idx = idx % (len(bins) - 1)
    low, high = bins[span_idx], bins[span_idx + 1]
    if high <= low:
        return low
    return rng.randint(low, high - 1)


def _format_date(offset: int) -> str:
    return (_BASE_DATE + timedelta(days=offset)).strftime("%Y-%m-%d")


def _build_record(
    name: str,
    item: str,
    place: str,
    day: int,
    lag: int,
    should_remember: bool,
    group_id: int,
    gate_features: Mapping[str, object],
) -> dict[str, object]:
    date_str = _format_date(day)
    text = f"On {date_str}, {name} left a {item} at {place}."
    cues = [
        f"who left a {item} at {place}?",
        f"what did {name} leave at {place}?",
        f"where did {name} leave the {item}?",
        f"when did {name} leave the {item} at {place}?",
    ]
    answers = [name, item, place, date_str]
    gate_payload = dict(gate_features)
    return {
        "episode_text": text,
        "cues": cues,
        "answers": answers,
        "should_remember": should_remember,
        "lag": lag,
        "group_id": group_id,
        "gate_features": gate_payload,
    }


def generate(
    n: int,
    seed: int,
    confounders_ratio: float = 1.0,
    hard_negative: bool = True,
    lag_spec: dict[str, Sequence[int]] | None = None,
) -> list[dict[str, object]]:
    """Generate Scenario A records.

    Parameters
    ----------
    n:
        Number of primary episodes to generate.
    seed:
        Random seed for reproducibility.
    confounders_ratio:
        Ratio of hard negatives to primary episodes.
    hard_negative:
        Whether to include hard negative distractors.
    lag_spec:
        Dictionary with ``"bins"`` key specifying lag bin edges.
    """
    rng = random.Random(seed)  # noqa: S311
    bins = lag_spec.get("bins", [0, 1, 3, 7, 30]) if lag_spec else [0, 1, 3, 7, 30]

    records: list[dict[str, object]] = []
    base_slots: list[tuple[str, str, str, int, int]] = []  # (name,item,place,day,lag)

    for i in range(n):
        name = rng.choice(NAMES)
        item = rng.choice(ITEMS)
        place = rng.choice(PLACES)
        day = rng.randint(0, 27)
        lag = _cycle_lag(i, rng, bins)
        base_slots.append((name, item, place, day, lag))
        gate_feats = {
            "surprise": 0.6 + rng.random() * 0.7,
            "novelty": 0.55 + rng.random() * 0.35,
            "reward": bool((i % 11) == 0),
            "pin": bool((i % 17) == 0),
        }
        records.append(_build_record(name, item, place, day, lag, True, i, gate_feats))

    if hard_negative:
        n_conf = int(round(n * confounders_ratio))
        for j in range(n_conf):
            base_idx = j % n
            name, item, place, day, lag = base_slots[base_idx]
            neg_name, neg_item, neg_place, neg_day = name, item, place, day
            to_change = rng.choice(["name", "item", "place", "day"])
            if to_change == "name":
                neg_name = rng.choice([s for s in NAMES if s != name])
            elif to_change == "item":
                neg_item = rng.choice([s for s in ITEMS if s != item])
            elif to_change == "place":
                neg_place = rng.choice([s for s in PLACES if s != place])
            else:
                neg_day = rng.randint(0, 27)
            gate_feats = {
                "surprise": 0.1 + rng.random() * 0.5,
                "novelty": 0.05 + rng.random() * 0.35,
                "reward": False,
                "pin": False,
            }
            records.append(
                _build_record(
                    neg_name,
                    neg_item,
                    neg_place,
                    neg_day,
                    lag,
                    False,
                    base_idx,
                    gate_feats,
                )
            )
    return records
