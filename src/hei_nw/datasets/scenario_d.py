"""Scenario D: stress-interference with near-collisions."""

from __future__ import annotations

import random

SERVICES = ["alpha", "beta", "gamma"]


def generate(n: int, seed: int) -> list[dict[str, object]]:
    """Generate Scenario D records.

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
        service = SERVICES[i % len(SERVICES)]
        code = str(rng.randint(1000, 9999))
        if i % 2 == 0:
            context = f"Access code for {service} is {code}."
            query = f"What is the access code for {service}?"
            expected = code
            should_remember = True
        else:
            context = f"Temporary code for {service} is {code}."
            query = f"What is the temporary code for {service}?"
            expected = code
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
