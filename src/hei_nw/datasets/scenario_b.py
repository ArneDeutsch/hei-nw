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
        else:
            context = f"{company} produces widgets."
            query = f"What does {company} produce?"
            expected = "widgets"
            should_remember = False
        records.append(
            {
                "context": context,
                "query": query,
                "expected": expected,
                "should_remember": should_remember,
            }
        )
    return records
