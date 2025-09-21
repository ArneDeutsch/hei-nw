"""Scenario B: factual hot-patch contradictions."""

from __future__ import annotations

import random

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

COMPANIES = [
    "Acme Corp",
    "Globex",
    "Initech",
    "Umbrella",
    "Soylent",
    "Cyberdyne",
]


def generate(n: int, seed: int) -> list[dict[str, object]]:
    """Generate Scenario B records.

    Parameters
    ----------
    n:
        Number of items to generate.
    seed:
        Random seed for determinism.
    """
    rng = random.Random(seed)  # noqa: S311
    records: list[dict[str, object]] = []
    for i in range(n):
        company = rng.choice(COMPANIES)
        name = rng.choice(NAMES)
        if i % 2 == 0:
            context = f"Update: The CEO of {company} is now {name}."
            query = f"Who is the CEO of {company}?"
            expected = name
            should_remember = True
            gate_features = {
                "surprise": 1.1 + rng.random() * 0.4,
                "novelty": 0.8 + rng.random() * 0.15,
                "reward": bool(company == "Acme Corp"),
                "pin": bool(i % 7 == 0),
            }
        else:
            context = f"{company} produces widgets."
            query = f"What does {company} produce?"
            expected = "widgets"
            should_remember = False
            gate_features = {
                "surprise": 0.2 + rng.random() * 0.2,
                "novelty": 0.1 + rng.random() * 0.1,
                "reward": False,
                "pin": False,
            }
        records.append(
            {
                "context": context,
                "query": query,
                "expected": expected,
                "should_remember": should_remember,
                "gate_features": gate_features,
            }
        )
    return records
