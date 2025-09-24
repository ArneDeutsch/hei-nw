"""Scenario A: episodic one-shot stories with hard negatives and lag bins."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta

from hei_nw.gate import make_gate_features

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

BaseSlot = tuple[str, str, str, int, int]


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


def _normalise_lag(lag: int, max_lag: int) -> float:
    if max_lag <= 0:
        return 0.0
    clamped = min(max(lag, 0), max_lag)
    return clamped / max_lag


def _lerp(high: float, low: float, weight: float) -> float:
    bounded = min(max(weight, 0.0), 1.0)
    return high + (low - high) * bounded


def _day_similarity(day_a: int, day_b: int, max_span: int) -> float:
    if max_span <= 0:
        return 1.0 if day_a == day_b else 0.0
    delta = abs(day_a - day_b)
    if delta == 0:
        return 1.0
    return max(0.0, 1.0 - delta / max_span)


def _slot_similarity(slot: BaseSlot, other: BaseSlot, max_span: int) -> float:
    name_sim = 1.0 if slot[0] == other[0] else 0.0
    item_sim = 1.0 if slot[1] == other[1] else 0.0
    place_sim = 1.0 if slot[2] == other[2] else 0.0
    day_sim = _day_similarity(slot[3], other[3], max_span)
    return (name_sim + item_sim + place_sim + day_sim) / 4.0


def _max_similarity(slot: BaseSlot, history: Sequence[BaseSlot], max_span: int) -> float | None:
    if not history:
        return None
    return max(_slot_similarity(slot, candidate, max_span) for candidate in history)


def _positive_recall_probability(lag: int, max_lag: int) -> float:
    lag_norm = _normalise_lag(lag, max_lag)
    return _lerp(0.55, 0.25, lag_norm)


def _negative_recall_probability(lag: int, max_lag: int) -> float:
    lag_norm = _normalise_lag(lag, max_lag)
    return _lerp(0.85, 0.6, lag_norm)


def _apply_priority_boost(prob: float, *, reward: bool, pin: bool) -> float:
    adjusted = prob
    if reward:
        adjusted = max(adjusted - 0.05, 0.05)
    if pin:
        adjusted = max(adjusted - 0.1, 0.05)
    return adjusted


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
    base_slots: list[BaseSlot] = []
    lag_cap = max(bins) if bins else 1
    if lag_cap <= 0:
        lag_cap = 1

    for i in range(n):
        name = rng.choice(NAMES)
        item = rng.choice(ITEMS)
        place = rng.choice(PLACES)
        day = rng.randint(0, 27)
        lag = _cycle_lag(i, rng, bins)
        slot = (name, item, place, day, lag)
        similarity = _max_similarity(slot, base_slots, lag_cap)
        reward_flag = (i % 11) == 0
        pin_flag = (i % 17) == 0
        recall_prob = _apply_priority_boost(
            _positive_recall_probability(lag, lag_cap),
            reward=reward_flag,
            pin=pin_flag,
        )
        gate_feats = make_gate_features(
            recall_prob=recall_prob,
            similarity=similarity,
            reward=reward_flag,
            pin=pin_flag,
        )
        records.append(_build_record(name, item, place, day, lag, True, i, gate_feats))
        base_slots.append(slot)

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
            neg_slot: BaseSlot = (neg_name, neg_item, neg_place, neg_day, lag)
            similarity = _slot_similarity(neg_slot, base_slots[base_idx], lag_cap)
            recall_prob = _negative_recall_probability(lag, lag_cap)
            gate_feats = make_gate_features(
                recall_prob=recall_prob,
                similarity=similarity,
                reward=False,
                pin=False,
            )
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
