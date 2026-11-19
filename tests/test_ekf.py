"""Unit tests for src/tracker/ekf.py."""

from __future__ import annotations

import numpy as np
import pytest

from tracker.config import EKFConfig
from tracker.ekf import (
    AdaptiveEKF,
    CTRVFilter,
    CanonicalEKF,
    IMMFilter,
    LinearCVFilter,
    create_filter,
)


@pytest.mark.parametrize("model_type", ["adaptive_bicycle", "canonical_bicycle", "ctrv", "linear_cv", "imm"])
def test_ekf_factory_and_lifecycle(model_type: str) -> None:
    flt = create_filter(EKFConfig(model_type=model_type))
    st0 = flt.initialize(100.0, 200.0, v0=10.0, theta0=0.0)
    assert st0.x_utm == pytest.approx(100.0)
    st1 = flt.predict(0.1)
    assert st1.x_utm > 100.0
    st2 = flt.update(np.array([101.0, 200.0]), np.diag([0.25, 0.25]), timestamp=0.1)
    assert np.all(np.linalg.eigvalsh(st2.P_pos_2x2) > 0.0)
    st_r = flt.reset_position_covariance(np.array([102.0, 200.0]), np.diag([0.05, 0.05]))
    assert float(np.trace(st_r.P_pos_2x2)) <= 0.15
