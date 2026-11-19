"""Unit tests for src/tracker/association.py."""

from __future__ import annotations

import numpy as np
import pytest
from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
from tracker.config import AssociationConfig
from tracker.ekf import FilterState
from tracker.roadgraph import BlindHypothesis


def test_cross_camera_associator_scale_and_match() -> None:
    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid", min_scale_samples=3))
    for _ in range(3):
        assoc.update_scale_calibration("cam_1", (2.0, 4.0), "cam_2", (2.5, 5.0))
    cw, cl = assoc.get_scale_factors("cam_1", "cam_2")
    assert cw == pytest.approx(1.25) and cl == pytest.approx(1.25)

    p5 = np.eye(5)
    st = FilterState(np.array([0.0, 0.0, 10.0, 0.0, 0.0]), p5, 0.0, 0.0, 10.0, 0.0, p5[:2, :2])
    hyp = BlindHypothesis("h1", [(0, 1, 0)], (0, 1, 0), 50.0, 50.0, 0.0, 10.0, 0.0, np.eye(2) * 4.0, 1.0, 5.0)
    emb = np.ones(512, dtype=np.float32) / np.sqrt(512.0)

    rec = BlindTrackRecord(1, "cam_1", 0.0, st, (2.0, 4.0), emb, None, [hyp])
    app = NewAppearance("cam_2", 10, 5.0, 50.2, 0.1, np.eye(2) * 0.5, (2.5, 5.0), emb, None)
    matches = assoc.associate([rec], [app])
    assert len(matches) == 1 and matches[0].global_id == 1
