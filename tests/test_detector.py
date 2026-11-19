"""Unit tests for src/tracker/detector.py."""

from __future__ import annotations

from pathlib import Path
import numpy as np
from tracker.config import DetectorConfig
from tracker.detector import LocalPerception


def test_local_perception_active_and_exit_zone(tmp_path: Path) -> None:
    step = {"i": 0}

    def det_fn(_f: np.ndarray) -> np.ndarray:
        if step["i"] == 0:
            return np.array([[580.0, 200.0, 630.0, 250.0, 0.9, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    cfg = DetectorConfig(model_path=tmp_path / "unused.pt", tracker_type="bytetrack")
    ez = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]
    perc = LocalPerception(cfg, ez, detector_fn=det_fn)
    frame = np.full((480, 640, 3), 50, dtype=np.uint8)

    t0 = perc.process_frame(frame)
    assert len(t0) == 1 and t0[0].status == "ACTIVE"
    step["i"] = 1
    t1 = perc.process_frame(frame)
    assert len(t1) == 1 and t1[0].status == "EXITED_TO_BLIND"
