"""Unit tests for scripts/calibrate_camera.py."""

from __future__ import annotations

import json
from pathlib import Path
from scripts.calibrate_camera import main


def test_calibrate_camera_cli(tmp_path: Path) -> None:
    pts_path = tmp_path / "pts.json"
    pts_path.write_text(
        json.dumps(
            {
                "epsg_utm": 32723,
                "points": [
                    {"u": 100.0, "v": 100.0, "x_utm": 330010.0, "y_utm": 7390040.0},
                    {"u": 500.0, "v": 100.0, "x_utm": 330050.0, "y_utm": 7390040.0},
                    {"u": 550.0, "v": 400.0, "x_utm": 330050.0, "y_utm": 7390000.0},
                    {"u": 50.0, "v": 400.0, "x_utm": 330010.0, "y_utm": 7390000.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    out_yaml = tmp_path / "cam.yaml"
    rc = main(["--points", str(pts_path), "--out", str(out_yaml), "--headless"])
    assert rc == 0
    assert out_yaml.is_file()
