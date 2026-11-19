"""Nonlinear vehicle kinematic motion models and analytical Jacobians (TwoPointBicycle, CTRV, LinearCV)."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap angle in radians into [-pi, pi]."""
    a = float(angle)
    wrapped = (a + math.pi) % (2.0 * math.pi) - math.pi
    if wrapped == -math.pi and a > 0.0:
        return math.pi
    return float(wrapped)


def circular_mean(
    angles: Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    """Compute circular mean of angles in [-pi, pi] using atan2 of mean sin/cos."""
    arr = np.asarray(angles, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("angles must be non-empty")
    if weights is None:
        w = np.ones_like(arr, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != arr.size:
            raise ValueError("weights and angles must have the same length")
    s = float(np.sum(w * np.sin(arr)))
    c = float(np.sum(w * np.cos(arr)))
    if abs(s) < 1e-15 and abs(c) < 1e-15:
        return wrap_angle(float(arr[-1]))
    return float(math.atan2(s, c))


class TwoPointBicycle:
    """5-state Two-Point Bicycle kinematic model: x = [X, Y, v, theta, delta]^T."""

    MAX_STEER_RAD: float = 0.85

    def __init__(self, wheelbase_m: float = 2.5) -> None:
        if not math.isfinite(wheelbase_m) or wheelbase_m <= 0.0:
            raise ValueError("wheelbase_m must be finite and > 0")
        self.wheelbase_m = float(wheelbase_m)

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.size != 5:
            raise ValueError("TwoPointBicycle state must have 5 elements [X, Y, v, theta, delta]")
        if dt_f == 0.0:
            out = x.copy()
            out[3] = wrap_angle(float(out[3]))
            return out

        px, py, v, theta, delta = (
            float(x[0]),
            float(x[1]),
            float(x[2]),
            float(x[3]),
            float(np.clip(x[4], -self.MAX_STEER_RAD, self.MAX_STEER_RAD)),
        )
        tan_d = math.tan(delta)
        omega = (v / self.wheelbase_m) * tan_d

        if abs(omega) < 1e-6:
            th_mid = theta + 0.5 * omega * dt_f
            nx = px + v * math.cos(th_mid) * dt_f
            ny = py + v * math.sin(th_mid) * dt_f
            nth = wrap_angle(theta + omega * dt_f)
        else:
            nth_raw = theta + omega * dt_f
            nx = px + (v / omega) * (math.sin(nth_raw) - math.sin(theta))
            ny = py + (v / omega) * (-math.cos(nth_raw) + math.cos(theta))
            nth = wrap_angle(nth_raw)

        # Gentle steering return damping so extreme U-turn steering stays bounded
        ndelta = float(np.clip(delta, -self.MAX_STEER_RAD, self.MAX_STEER_RAD))
        return np.array([nx, ny, v, nth, ndelta], dtype=np.float64)

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.size != 5:
            raise ValueError("TwoPointBicycle state must have 5 elements")
        f_mat = np.eye(5, dtype=np.float64)
        if dt_f == 0.0:
            return f_mat

        v = float(x[2])
        theta = float(x[3])
        delta = float(np.clip(x[4], -self.MAX_STEER_RAD, self.MAX_STEER_RAD))
        L = self.wheelbase_m
        tan_d = math.tan(delta)
        sec2_d = 1.0 / max(math.cos(delta) ** 2, 1e-6)
        omega = (v / L) * tan_d
        domega_dv = tan_d / L
        domega_ddelta = (v / L) * sec2_d

        if abs(omega) < 1e-5:
            c = math.cos(theta)
            s = math.sin(theta)
            f_mat[0, 2] = c * dt_f
            f_mat[0, 3] = -v * s * dt_f
            f_mat[0, 4] = -0.5 * v * s * (dt_f**2) * domega_ddelta
            f_mat[1, 2] = s * dt_f
            f_mat[1, 3] = v * c * dt_f
            f_mat[1, 4] = 0.5 * v * c * (dt_f**2) * domega_ddelta
            f_mat[3, 2] = domega_dv * dt_f
            f_mat[3, 4] = domega_ddelta * dt_f
        else:
            th2 = theta + omega * dt_f
            s1, c1 = math.sin(theta), math.cos(theta)
            s2, c2 = math.sin(th2), math.cos(th2)
            # Note: v / omega = L / tan_d (independent of v!)
            r_turn = L / tan_d
            dr_ddelta = -L * sec2_d / (tan_d * tan_d)

            f_mat[0, 2] = r_turn * c2 * domega_dv * dt_f
            f_mat[0, 3] = r_turn * (c2 - c1)
            f_mat[0, 4] = dr_ddelta * (s2 - s1) + r_turn * c2 * domega_ddelta * dt_f

            f_mat[1, 2] = r_turn * s2 * domega_dv * dt_f
            f_mat[1, 3] = r_turn * (s2 - s1)
            f_mat[1, 4] = dr_ddelta * (-c2 + c1) + r_turn * s2 * domega_ddelta * dt_f

            f_mat[3, 2] = domega_dv * dt_f
            f_mat[3, 4] = domega_ddelta * dt_f

        return f_mat


class CTRV:
    """5-state Coordinated Turn Rate and Velocity model: x = [X, Y, v, theta, omega]^T."""

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.size != 5:
            raise ValueError("CTRV state must have 5 elements [X, Y, v, theta, omega]")
        if dt_f == 0.0:
            out = x.copy()
            out[3] = wrap_angle(float(out[3]))
            return out

        px, py, v, theta, omega = float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4])
        if abs(omega) < 1e-6:
            th_mid = theta + 0.5 * omega * dt_f
            nx = px + v * math.cos(th_mid) * dt_f
            ny = py + v * math.sin(th_mid) * dt_f
            nth = wrap_angle(theta + omega * dt_f)
        else:
            th2 = theta + omega * dt_f
            nx = px + (v / omega) * (math.sin(th2) - math.sin(theta))
            ny = py + (v / omega) * (-math.cos(th2) + math.cos(theta))
            nth = wrap_angle(th2)

        return np.array([nx, ny, v, nth, omega], dtype=np.float64)

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.size != 5:
            raise ValueError("CTRV state must have 5 elements")
        f_mat = np.eye(5, dtype=np.float64)
        if dt_f == 0.0:
            return f_mat

        v, theta, omega = float(x[2]), float(x[3]), float(x[4])
        if abs(omega) < 1e-5:
            c = math.cos(theta)
            s = math.sin(theta)
            f_mat[0, 2] = c * dt_f
            f_mat[0, 3] = -v * s * dt_f
            f_mat[0, 4] = -0.5 * v * s * (dt_f**2)
            f_mat[1, 2] = s * dt_f
            f_mat[1, 3] = v * c * dt_f
            f_mat[1, 4] = 0.5 * v * c * (dt_f**2)
            f_mat[3, 4] = dt_f
        else:
            th2 = theta + omega * dt_f
            s1, c1 = math.sin(theta), math.cos(theta)
            s2, c2 = math.sin(th2), math.cos(th2)
            inv_w = 1.0 / omega
            inv_w2 = inv_w * inv_w

            f_mat[0, 2] = inv_w * (s2 - s1)
            f_mat[0, 3] = v * inv_w * (c2 - c1)
            f_mat[0, 4] = v * dt_f * inv_w * c2 - v * inv_w2 * (s2 - s1)

            f_mat[1, 2] = inv_w * (-c2 + c1)
            f_mat[1, 3] = v * inv_w * (s2 - s1)
            f_mat[1, 4] = v * dt_f * inv_w * s2 - v * inv_w2 * (-c2 + c1)

            f_mat[3, 4] = dt_f

        return f_mat


class LinearCV:
    """4-state Constant Velocity linear model: x = [X, Y, vx, vy]^T."""

    def transition_matrix(self, dt: float) -> np.ndarray:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        return np.array(
            [
                [1.0, 0.0, dt_f, 0.0],
                [0.0, 1.0, 0.0, dt_f],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def step(self, state: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).ravel()
        if x.size != 4:
            raise ValueError("LinearCV state must have 4 elements [X, Y, vx, vy]")
        f_mat = self.transition_matrix(dt)
        return f_mat @ x

    def jacobian(self, state: np.ndarray, dt: float) -> np.ndarray:
        _ = state
        return self.transition_matrix(dt)
