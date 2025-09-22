import json
import subprocess
import sys
from pathlib import Path


def _run_audit(paths: list[Path]) -> subprocess.CompletedProcess[str]:
    argv = [sys.executable, str(Path("scripts/audit_pointer_payloads.py"))]
    argv.extend(str(path) for path in paths)
    return subprocess.run(argv, check=False, capture_output=True, text=True)


def test_detects_banned_keys(tmp_path: Path) -> None:
    metrics = {
        "gate": {
            "pointer_check": {"pointer_only": True, "banned_keys": [], "missing_pointer": 0},
            "trace_samples": [
                {
                    "trace_id": "trace-001",
                    "has_pointer": True,
                    "banned_keys": ["episode_text"],
                }
            ],
        }
    }
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf8")

    pointer_payload = {
        "store": {
            "records": [
                {
                    "trace_id": "trace-001",
                    "tokens_span_ref": {"doc": "doc-1", "start": 0, "end": 12},
                    "entity_slots": {
                        "who": "Ada",
                        "what": "demo",
                        "where": "",
                        "when": "",
                        "extras": {"snippet": "raw text leak"},
                    },
                    "salience_tags": {"S": 0.7},
                    "eviction": {
                        "ttl_seconds": 3600,
                        "created_at": "2024-01-01T00:00:00Z",
                        "last_access": "2024-01-01T00:00:00Z",
                        "expires_at": "2024-01-01T01:00:00Z",
                    },
                }
            ]
        }
    }
    pointer_path = tmp_path / "store.json"
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf8")

    result = _run_audit([metrics_path, pointer_path])
    assert result.returncode == 1
    stdout = result.stdout.lower()
    assert "pointer_only=false" in stdout
    assert "episode_text" in stdout
    assert "snippet" in stdout


def test_passes_clean_payloads(tmp_path: Path) -> None:
    metrics = {
        "gate": {
            "pointer_check": {"pointer_only": True, "banned_keys": [], "missing_pointer": 0},
            "trace_samples": [
                {
                    "trace_id": "trace-002",
                    "has_pointer": True,
                    "banned_keys": [],
                }
            ],
        }
    }
    metrics_path = tmp_path / "metrics_clean.json"
    metrics_path.write_text(json.dumps(metrics), encoding="utf8")

    pointer_payload = {
        "store": {
            "records": [
                {
                    "trace_id": "trace-002",
                    "tokens_span_ref": {"doc": "doc-2", "start": 5, "end": 20},
                    "entity_slots": {"who": "Ada", "what": "demo", "where": "", "when": ""},
                    "salience_tags": {"S": 0.5},
                    "eviction": {
                        "ttl_seconds": 3600,
                        "created_at": "2024-01-01T00:00:00Z",
                        "last_access": "2024-01-01T00:30:00Z",
                        "expires_at": "2024-01-01T01:00:00Z",
                    },
                }
            ]
        }
    }
    pointer_path = tmp_path / "store_clean.json"
    pointer_path.write_text(json.dumps(pointer_payload), encoding="utf8")

    result = _run_audit([metrics_path, pointer_path])
    assert result.returncode == 0
    stdout = result.stdout.lower()
    assert "pointer_only=true" in stdout
    assert "banned_keys={}" in stdout
