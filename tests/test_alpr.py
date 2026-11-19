"""Unit tests for src/tracker/alpr.py."""

from __future__ import annotations

import cv2
import numpy as np
from tracker.alpr import ALPREngine, normalize_plate
from tracker.config import ALPRConfig


def test_alpr_normalize_and_recognize() -> None:
    assert normalize_plate("abc-1234") == ("ABC1234", True)
    assert normalize_plate("bra-2e19") == ("BRA2E19", True)
    assert normalize_plate("invalid") == ("INVALID", False)

    plate = np.full((80, 240, 3), 245, dtype=np.uint8)
    cv2.rectangle(plate, (2, 2), (237, 77), (25, 25, 25), 2)
    cv2.rectangle(plate, (4, 4), (235, 22), (180, 60, 20), -1)
    cv2.putText(plate, "ABC1D23", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (15, 15, 15), 3, cv2.LINE_AA)

    eng = ALPREngine(ALPRConfig(enabled=True, min_conf=0.3))
    res = eng.recognize(plate)
    assert res is not None
    assert res.normalized_text == "ABC1D23"
    assert res.format_valid is True
