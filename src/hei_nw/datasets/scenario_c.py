"""Scenario C: preference and configuration memory.

Acceptance check::

    PYTHONPATH=src python - <<"PY"
    from hei_nw.datasets.scenario_c import generate

    records = generate(8, seed=7)
    configs = [r for r in records if r["should_remember"]]
    assert any(r["gate_features"]["pin"] for r in configs)
    assert any(r["gate_features"]["reward"] for r in configs)
    for record in configs[:3]:
        counters = record["novelty_counters"]
        assert counters["config_index"] >= 0
        assert counters["status_index"] >= 0
        assert 0.0 <= record["gate_features"]["novelty"] <= 1.0
        if record["gate_features"]["reward"]:
            assert record["reward_annotation"]
    PY
"""

from __future__ import annotations

import random

SERVERS = ["alpha", "beta", "gamma", "delta", "epsilon"]
SERVICES = ["postgres", "redis", "nginx", "mongodb", "mysql"]
CRITICAL_SERVICES = {"postgres"}
PINNED_SERVERS = {"alpha"}
COLORS = ["red", "blue", "green"]


def _config_novelty(index: int) -> float:
    """Return novelty for a configuration event based on *index*."""

    return max(0.25, 1.0 - 0.2 * index)


def _status_novelty(index: int) -> float:
    """Return novelty for a status probe event based on *index*."""

    return max(0.05, 0.2 / (index + 1))


def _config_surprise(port: int, index: int) -> float:
    """Return deterministic surprise for configuration updates."""

    return 1.0 + ((port % 17) / 20.0) + 0.05 * index


def _status_surprise(index: int) -> float:
    """Return deterministic surprise for status probes."""

    return 0.3 + 0.07 * min(index, 3)


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
    server_counters: dict[str, dict[str, int]] = {}
    for i in range(n):
        server = rng.choice(SERVERS)
        counters = server_counters.setdefault(server, {"config_seen": 0, "status_seen": 0})
        config_index = counters["config_seen"]
        status_index = counters["status_seen"]
        if i % 2 == 0:
            port = rng.randint(1024, 9999)
            service = rng.choice(SERVICES)
            surprise = _config_surprise(port, config_index)
            novelty = _config_novelty(config_index)
            reward = service in CRITICAL_SERVICES
            counters["config_seen"] += 1
            should_remember = True
            context = f"Server {server} runs {service} on port {port}."
            query = rng.choice(
                [
                    f"What port does {server} use?",
                    f"Which port is {server} running on?",
                ]
            )
            expected = str(port)
            reward_annotation: str | None
            if reward:
                reward_annotation = f"Service {service} is marked critical; reward applied."
            else:
                reward_annotation = None
            pin = server in PINNED_SERVERS
            pin_annotation = f"{server} belongs to the always-pin SRE list." if pin else None
            event_type = "config_update"
            novelty_counters = {
                "config_index": config_index,
                "status_index": status_index,
            }
            payload = {
                "service": service,
                "port": port,
            }
        else:
            color = rng.choice(COLORS)
            surprise = _status_surprise(status_index)
            novelty = _status_novelty(status_index)
            reward = False
            pin = False
            counters["status_seen"] += 1
            should_remember = False
            context = f"{server} has a {color} status light."
            query = f"What color is the status light on {server}?"
            expected = color
            reward_annotation = None
            pin_annotation = None
            event_type = "status_probe"
            novelty_counters = {
                "config_index": counters["config_seen"],
                "status_index": status_index,
            }
            payload = {
                "status_color": color,
            }
        gate_features = {
            "surprise": surprise,
            "novelty": novelty,
            "reward": reward,
            "pin": pin,
        }
        record = {
            "context": context,
            "query": query,
            "expected": expected,
            "should_remember": should_remember,
            "gate_features": gate_features,
            "server": server,
            "event_type": event_type,
            "novelty_counters": novelty_counters,
            "reward_annotation": reward_annotation,
            "pin_annotation": pin_annotation,
        }
        record.update(payload)
        records.append(record)
    return records
