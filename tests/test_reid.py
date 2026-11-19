"""Unit tests for src/tracker/reid.py."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from tracker.config import ReIDConfig
from tracker.reid import ReIDExtractor, cosine_distance


def test_reid_hsv_fallback_and_cosine_distance() -> None:
    cfg = ReIDConfig(onnx_path=Path("hsv_hog_test"), allow_hsv_fallback=True)
    ext = ReIDExtractor(cfg)
    crop_a = np.full((128, 64, 3), (20, 20, 220), dtype=np.uint8)
    crop_b = np.full((128, 64, 3), (220, 40, 20), dtype=np.uint8)
    ea = ext.extract(crop_a)
    eb = ext.extract(crop_b)
    assert ea.shape == (512,) and ea.dtype == np.float32
    assert float(np.linalg.norm(ea)) == pytest.approx(1.0, abs=1e-4)
    assert cosine_distance(ea, ea) == pytest.approx(0.0, abs=1e-5)
    assert cosine_distance(ea, eb) > 0.05
    batch = ext.extract_batch([crop_a, crop_b])
    assert batch.shape == (2, 512)
