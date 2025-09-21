"""Scenario E: long-context control tasks."""

from __future__ import annotations

import random

WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "phi",
    "chi",
    "psi",
    "omega",
]


def generate(n: int, seed: int, context_length: int = 20) -> list[dict[str, object]]:
    """Generate Scenario E records.

    Parameters
    ----------
    n:
        Number of items to generate.
    seed:
        Random seed for determinism.
    context_length:
        Number of tokens in the synthetic long context.
    """
    rng = random.Random(seed)  # noqa: S311
    records: list[dict[str, object]] = []
    for i in range(n):
        tokens = [rng.choice(WORDS) for _ in range(context_length)]
        target_index = rng.randrange(context_length)
        target = tokens[target_index]
        context = " ".join(tokens)
        if i % 2 == 0:
            query = f"Which token is at position {target_index + 1}?"
            expected = target
            should_remember = True
            gate_features = {
                "surprise": 0.95 + rng.random() * 0.4,
                "novelty": 0.65 + rng.random() * 0.25,
                "reward": bool(target.startswith("a")),
                "pin": bool(i % 13 == 0),
            }
        else:
            context = "This is filler context."
            query = "Is this a filler context?"
            expected = "yes"
            should_remember = False
            gate_features = {
                "surprise": 0.2 + rng.random() * 0.15,
                "novelty": 0.05 + rng.random() * 0.1,
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
