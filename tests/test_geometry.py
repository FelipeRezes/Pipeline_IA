"""Unit tests for src/tracker/geometry.py."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from tracker.geometry import (
    CameraProjector,
    compute_homography_dlt,
    utm_to_wgs84,
    wgs84_to_utm,
)


def test_wgs84_utm_conversions() -> None:
    x, y, epsg = wgs84_to_utm(-23.55052, -46.633308)
    assert epsg == 32723
    lat, lon = utm_to_wgs84(x, y, epsg)
    assert lat == pytest.approx(-23.55052, abs=1e-6)
    assert lon == pytest.approx(-46.633308, abs=1e-6)


def test_dlt_homography_and_projector(tmp_path: Path) -> None:
    src = np.array([[100.0, 100.0], [500.0, 100.0], [550.0, 400.0], [50.0, 400.0]], dtype=np.float64)
    dst = np.array([[330010.0, 7390040.0], [330050.0, 7390040.0], [330050.0, 7390000.0], [330010.0, 7390000.0]], dtype=np.float64)
    h_mat, rmse = compute_homography_dlt(src, dst, use_ransac=False)
    assert h_mat.shape == (3, 3)
    assert rmse < 1e-4

    yaml_path = tmp_path / "cam.yaml"
    yaml_path.write_text(yaml.safe_dump({"cam_id": "c1", "epsg_utm": 32723, "homography": h_mat.tolist()}), encoding="utf-8")
    proj = CameraProjector.from_yaml(yaml_path)
    gx, gy = proj.pixel_to_ground(100.0, 100.0)
    assert math.hypot(gx - 330010.0, gy - 7390040.0) < 1e-3
    u, v = proj.ground_to_pixel(gx, gy)
    assert math.hypot(u - 100.0, v - 100.0) < 1e-3

    bx, by, r_k = proj.bbox_to_ground((200.0, 200.0, 260.0, 250.0))
    assert math.isfinite(bx) and math.isfinite(by)
    assert r_k.shape == (2, 2) and np.all(np.linalg.eigvalsh(r_k) > 0)
    w_m, l_m = proj.bbox_dimensions_ground((200.0, 200.0, 260.0, 250.0))
    assert w_m > 0.0 and l_m > 0.0
