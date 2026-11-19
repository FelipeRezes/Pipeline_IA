"""Unit tests for src/tracker/kinematics.py."""

from __future__ import annotations

import math
import numpy as np
import pytest

from tracker.kinematics import CTRV, LinearCV, TwoPointBicycle, circular_mean, wrap_angle


def test_wrap_angle_and_circular_mean() -> None:
    assert wrap_angle(3.0 * math.pi) == pytest.approx(math.pi, abs=1e-6)
    m = circular_mean([math.pi - 0.05, -math.pi + 0.05])
    assert abs(abs(m) - math.pi) < 0.06


def test_bicycle_ctrv_and_linearcv_steps_and_jacobians() -> None:
    bic = TwoPointBicycle(wheelbase_m=2.5)
    x5 = np.array([0.0, 0.0, 10.0, 0.1, 0.15], dtype=np.float64)
    nx5 = bic.step(x5, 0.1)
    j5 = bic.jacobian(x5, 0.1)
    assert nx5.shape == (5,) and j5.shape == (5, 5)
    assert np.all(np.isfinite(nx5)) and np.all(np.isfinite(j5))

    ctrv = CTRV()
    nx_c = ctrv.step(np.array([0.0, 0.0, 10.0, 0.2, 0.0], dtype=np.float64), 0.1)
    jc = ctrv.jacobian(np.array([0.0, 0.0, 10.0, 0.2, 0.3], dtype=np.float64), 0.1)
    assert nx_c.shape == (5,) and jc.shape == (5, 5)

    lcv = LinearCV()
    nx_l = lcv.step(np.array([0.0, 0.0, 8.0, 6.0], dtype=np.float64), 0.5)
    assert nx_l[0] == pytest.approx(4.0) and nx_l[1] == pytest.approx(3.0)
