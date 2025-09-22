from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_pins_only_render_smoke(tmp_path: Path) -> None:
    telemetry = {
        "scenario": "C",
        "threshold": 1.5,
        "calibration": [
            {"lower": 0.0, "upper": 0.5, "count": 3, "fraction_positive": 0.3, "mean_score": 0.2},
            {"lower": 0.5, "upper": 1.0, "count": 2, "fraction_positive": 0.8, "mean_score": 0.7},
        ],
        "pins_only": {
            "total": 1,
            "writes": 1,
            "precision": 1.0,
            "recall": 1.0,
            "pr_auc": 1.0,
            "write_rate": 1.0,
            "calibration": [
                {
                    "lower": 0.5,
                    "upper": 1.0,
                    "count": 1,
                    "fraction_positive": 1.0,
                    "mean_score": 0.9,
                },
            ],
        },
        "non_pins": {
            "total": 2,
            "writes": 1,
            "precision": 0.5,
            "recall": 0.5,
            "pr_auc": 0.5,
            "write_rate": 0.5,
            "calibration": [
                {
                    "lower": 0.0,
                    "upper": 0.5,
                    "count": 2,
                    "fraction_positive": 0.2,
                    "mean_score": 0.25,
                },
                {
                    "lower": 0.5,
                    "upper": 1.0,
                    "count": 1,
                    "fraction_positive": 0.6,
                    "mean_score": 0.65,
                },
            ],
        },
    }
    telemetry_path = tmp_path / "telemetry.json"
    telemetry_path.write_text(json.dumps(telemetry), encoding="utf8")
    out_path = tmp_path / "calibration_pins.png"

    result = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/plot_gate_calibration.py")),
            str(telemetry_path),
            "--out",
            str(out_path),
            "--pins-only",
            "--overlay-nonpins",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert out_path.exists()
