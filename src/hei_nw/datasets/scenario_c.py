"""Scenario C: preference and configuration memory."""

from __future__ import annotations

import random

SERVERS = ["alpha", "beta", "gamma", "delta", "epsilon"]
SERVICES = ["postgres", "redis", "nginx", "mongodb", "mysql"]
COLORS = ["red", "blue", "green"]


def generate(n: int, seed: int) -> list[dict[str, object]]:
    """Generate Scenario C records.

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
        server = rng.choice(SERVERS)
        if i % 2 == 0:
            port = rng.randint(1024, 9999)
            service = rng.choice(SERVICES)
            context = f"Server {server} runs {service} on port {port}."
            query = rng.choice(
                [
                    f"What port does {server} use?",
                    f"Which port is {server} running on?",
                ]
            )
            expected = str(port)
            should_remember = True
        else:
            color = rng.choice(COLORS)
            context = f"{server} has a {color} status light."
            query = f"What color is the status light on {server}?"
            expected = color
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
