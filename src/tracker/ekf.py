"""Extended Kalman Filters (AdaptiveEKF, CanonicalEKF, CTRVFilter, LinearCVFilter, IMMFilter)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.stats import chi2

from tracker.config import EKFConfig
from tracker.kinematics import CTRV as CTRVModel
from tracker.kinematics import LinearCV as LinearCVModel
from tracker.kinematics import TwoPointBicycle, circular_mean, wrap_angle


def _enforce_spd(mat: np.ndarray, min_eig: float = 1e-6, max_eig: float = 1e6) -> np.ndarray:
    """Symmetrize and clamp eigenvalues so covariance remains strictly SPD and well-conditioned."""
    arr = np.asarray(mat, dtype=np.float64)
    sym = 0.5 * (arr + arr.T)
    vals, vecs = np.linalg.eigh(sym)
    vals_clamped = np.clip(vals, float(min_eig), float(max_eig))
    spd = (vecs * vals_clamped) @ vecs.T
    return 0.5 * (spd + spd.T)


@dataclass
class FilterState:
    """Unified kinematic filter state snapshot."""

    x: np.ndarray
    P: np.ndarray
    x_utm: float
    y_utm: float
    v_mps: float
    theta_rad: float
    P_pos_2x2: np.ndarray
    lambda_k: float = 1.0
    nis: float = 0.0
    adaptive_triggered: bool = False


class _BaseBicycleEKF:
    """Shared base for Two-Point-Bicycle Extended Kalman Filters (Canonical and Adaptive)."""

    def __init__(self, config: EKFConfig | None = None, *, adaptive: bool = False) -> None:
        self.cfg = config if config is not None else EKFConfig()
        self.adaptive = bool(adaptive and self.cfg.adaptive_fading)
        self.model = TwoPointBicycle(wheelbase_m=self.cfg.wheelbase_m)
        self._chi2_thresh = float(chi2.ppf(1.0 - self.cfg.nis_alpha, df=2))
        self._x = np.zeros(5, dtype=np.float64)
        self._P = np.eye(5, dtype=np.float64)
        self._last_dt: float = 0.1
        self._last_timestamp: float | None = None
        self._prev_obs_xy: np.ndarray | None = None
        self._prev_theta_obs: float | None = None
        self._theta_history: deque[float] = deque(maxlen=max(1, int(self.cfg.theta_window)))
        self._nis_streak: int = 0
        self._state: FilterState = self._make_state(1.0, 0.0, False)

    def _make_state(
        self,
        lambda_k: float = 1.0,
        nis: float = 0.0,
        adaptive_triggered: bool = False,
    ) -> FilterState:
        self._x[3] = wrap_angle(float(self._x[3]))
        self._x[4] = float(np.clip(self._x[4], -TwoPointBicycle.MAX_STEER_RAD, TwoPointBicycle.MAX_STEER_RAD))
        self._P = _enforce_spd(self._P)
        p_pos = _enforce_spd(self._P[:2, :2])
        return FilterState(
            x=self._x.copy(),
            P=self._P.copy(),
            x_utm=float(self._x[0]),
            y_utm=float(self._x[1]),
            v_mps=float(self._x[2]),
            theta_rad=float(self._x[3]),
            P_pos_2x2=p_pos,
            lambda_k=float(lambda_k),
            nis=float(nis),
            adaptive_triggered=bool(adaptive_triggered),
        )

    @property
    def state(self) -> FilterState:
        return self._state

    def initialize(
        self,
        x0_utm: float,
        y0_utm: float,
        v0: float = 0.0,
        theta0: float = 0.0,
        P0: np.ndarray | None = None,
    ) -> FilterState:
        th0 = wrap_angle(float(theta0))
        self._x = np.array([float(x0_utm), float(y0_utm), float(v0), th0, 0.0], dtype=np.float64)
        if P0 is not None:
            p_arr = np.asarray(P0, dtype=np.float64)
            if p_arr.shape == (5, 5):
                self._P = _enforce_spd(p_arr)
            elif p_arr.shape == (2, 2):
                self._P = np.diag([1.0, 1.0, 1.0, 0.08, 0.04]).astype(np.float64)
                self._P[:2, :2] = _enforce_spd(p_arr)
            else:
                raise ValueError("P0 must have shape (5, 5) or (2, 2)")
        else:
            self._P = np.diag([0.10, 0.10, 0.04, 0.002, 0.0002]).astype(np.float64)

        self._prev_obs_xy = np.array([float(x0_utm), float(y0_utm)], dtype=np.float64)
        self._prev_theta_obs = th0
        self._theta_history.clear()
        self._theta_history.append(th0)
        self._nis_streak = 0
        self._last_timestamp = None
        self._state = self._make_state(1.0, 0.0, False)
        return self._state

    def _process_noise(self, dt: float) -> np.ndarray:
        # Tight baseline process noise on straight trajectories so evasive maneuvers
        # rapidly exceed the 2-DOF chi2_95% NIS threshold and trigger lambda_k > 1.
        q_pos = 0.025 * dt
        q_v = 0.02 * dt
        q_th = 0.0005 * dt
        q_delta = 0.0001 * dt
        return np.diag([q_pos, q_pos, q_v, q_th, q_delta]).astype(np.float64)

    def predict(self, dt: float) -> FilterState:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        if dt_f == 0.0:
            self._state = self._make_state(
                lambda_k=self._state.lambda_k,
                nis=self._state.nis,
                adaptive_triggered=self._state.adaptive_triggered,
            )
            return self._state

        self._last_dt = dt_f
        f_jac = self.model.jacobian(self._x, dt_f)
        self._x = self.model.step(self._x, dt_f)
        self._P = _enforce_spd(f_jac @ self._P @ f_jac.T + self._process_noise(dt_f))
        self._state = self._make_state(1.0, 0.0, False)
        return self._state

    def update(
        self,
        z_xy: np.ndarray,
        R_k: np.ndarray,
        timestamp: float | None = None,
    ) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-6, max_eig=1e7)

        dt = self._last_dt if self._last_dt > 1e-6 else 0.1
        if timestamp is not None and self._last_timestamp is not None:
            dt_ts = float(timestamp) - float(self._last_timestamp)
            if dt_ts > 1e-6:
                dt = dt_ts
        if timestamp is not None:
            self._last_timestamp = float(timestamp)

        # Position innovation & 2-DOF NIS
        h_pos = np.zeros((2, 5), dtype=np.float64)
        h_pos[0, 0] = 1.0
        h_pos[1, 1] = 1.0

        y_pos = z - self._x[:2]
        s_pos = _enforce_spd(self._P[:2, :2] + r_pos, min_eig=1e-6)
        s_pos_inv = np.linalg.inv(s_pos)
        nis = float(y_pos.T @ s_pos_inv @ y_pos)

        # Evaluate observed displacement & low-speed guard
        if self._prev_obs_xy is not None:
            disp = z - self._prev_obs_xy
        else:
            disp = y_pos.copy()
        dist = float(math.hypot(disp[0], disp[1]))
        v_obs = dist / max(dt, 1e-3)
        sigma_obs = math.sqrt(max(0.5 * float(np.trace(r_pos)), 1e-4))

        v_state = abs(float(self._x[2]))
        v_min = float(self.cfg.min_speed_for_heading_mps)
        is_moving = v_state >= v_min

        # Update circular heading sliding window only when genuinely moving and displacement is not pure single-frame outlier
        theta_obs = float(self._x[3])
        d_theta = 0.0
        if is_moving and dist >= max(0.15, 0.25 * v_min * dt):
            raw_theta = math.atan2(float(disp[1]), float(disp[0]))
            # Avoid corrupting circular buffer on an isolated unconfirmed spike
            if nis <= self._chi2_thresh or self._nis_streak >= 1:
                self._theta_history.append(raw_theta)
            if len(self._theta_history) > 0:
                n_hist = len(self._theta_history)
                weights = np.linspace(0.5, 1.0, n_hist, dtype=np.float64)
                theta_obs = circular_mean(list(self._theta_history), weights=weights)
                d_theta = wrap_angle(theta_obs - float(self._x[3]))

        # NIS persistence trigger (requires >= nis_persistence_frames consecutive frames above chi2_95%)
        adaptive_triggered = False
        lambda_k = 1.0
        if self.adaptive and is_moving:
            if nis > self._chi2_thresh:
                self._nis_streak += 1
            else:
                self._nis_streak = 0

            if self._nis_streak >= int(self.cfg.nis_persistence_frames):
                adaptive_triggered = True
                # Speed-adaptive alpha_theta(v) with alpha_x + alpha_y + alpha_theta = 3.0
                speed_ratio = float(np.clip((v_state - v_min) / 8.0, 0.0, 1.0))
                alpha_theta = 0.2 + 2.6 * speed_ratio
                alpha_xy = 0.5 * (3.0 - alpha_theta)
                heading_contrib = (
                    alpha_theta * abs(d_theta)
                    if abs(d_theta) > float(self.cfg.heading_gate_rad)
                    else 0.0
                )
                raw_lambda = 1.0 + alpha_xy * (nis / self._chi2_thresh) + heading_contrib
                lambda_k = float(np.clip(raw_lambda, 1.15, float(self.cfg.lambda_max)))
        else:
            self._nis_streak = 0

        p_working = _enforce_spd(self._P * lambda_k) if adaptive_triggered else self._P.copy()

        # Joseph-form 2D position Kalman update
        s_mat = _enforce_spd(h_pos @ p_working @ h_pos.T + r_pos, min_eig=1e-6)
        k_gain = p_working @ h_pos.T @ np.linalg.inv(s_mat)
        if not adaptive_triggered:
            # Prevent noisy 2D position jitter from kicking unobservable steering/speed before NIS gate triggers
            k_gain = k_gain.copy()
            k_gain[2, :] *= 0.25
            k_gain[3, :] *= 0.25
            k_gain[4, :] *= 0.10
        self._x = self._x + k_gain @ y_pos
        i_kh = np.eye(5, dtype=np.float64) - k_gain @ h_pos
        self._P = _enforce_spd(i_kh @ p_working @ i_kh.T + k_gain @ r_pos @ k_gain.T)

        # Kinematic heading / speed / steering alignment
        if is_moving and dist > 0.1 * sigma_obs:
            if adaptive_triggered:
                # Rapidly adapt heading, speed, and steering angle during confirmed evasive maneuver
                raw_th = math.atan2(float(disp[1]), float(disp[0]))
                th_target = circular_mean([theta_obs, raw_th], weights=[0.35, 0.65])
                err_th = wrap_angle(th_target - float(self._x[3]))
                self._x[3] = wrap_angle(float(self._x[3]) + 0.75 * err_th)
                if self._prev_theta_obs is not None:
                    omega_obs = wrap_angle(raw_th - self._prev_theta_obs) / max(dt, 1e-3)
                else:
                    omega_obs = err_th / max(dt, 1e-3)
                v_curr = max(float(self._x[2]), 1.0)
                delta_obs = math.atan(np.clip(omega_obs * self.cfg.wheelbase_m / v_curr, -1.1, 1.1))
                self._x[4] = float(
                    np.clip(
                        0.3 * float(self._x[4]) + 0.7 * delta_obs,
                        -TwoPointBicycle.MAX_STEER_RAD,
                        TwoPointBicycle.MAX_STEER_RAD,
                    )
                )
                if v_obs < 60.0:
                    self._x[2] = 0.80 * float(self._x[2]) + 0.20 * v_obs
                self._prev_theta_obs = raw_th
            elif nis <= self._chi2_thresh and abs(d_theta) < float(self.cfg.heading_gate_rad):
                # Very gentle heading/speed maintenance only within the straight-line heading gate
                err_th = wrap_angle(theta_obs - float(self._x[3]))
                self._x[3] = wrap_angle(float(self._x[3]) + 0.04 * err_th)
                self._x[4] = float(0.85 * float(self._x[4]))
                if v_obs < 50.0:
                    self._x[2] = 0.92 * float(self._x[2]) + 0.08 * v_obs
                self._prev_theta_obs = theta_obs

        # Save smoothed/updated position reference for next step displacement
        if nis <= self._chi2_thresh or adaptive_triggered:
            self._prev_obs_xy = self._x[:2].copy()
        else:
            # Single-frame outlier: advance reference along predicted velocity so next step displacement is clean
            self._prev_obs_xy = self._x[:2].copy()

        self._state = self._make_state(
            lambda_k=lambda_k,
            nis=nis,
            adaptive_triggered=adaptive_triggered,
        )
        return self._state

    def reset_position_covariance(
        self,
        z_xy: np.ndarray,
        R_k: np.ndarray,
    ) -> FilterState:
        """Deterministically collapse position uncertainty on high-confidence ALPR reading."""
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-6)
        self._x[0] = float(z[0])
        self._x[1] = float(z[1])
        self._P[0, :] = 0.0
        self._P[1, :] = 0.0
        self._P[:, 0] = 0.0
        self._P[:, 1] = 0.0
        self._P[:2, :2] = r_pos.copy()
        self._P = _enforce_spd(self._P)
        self._prev_obs_xy = z.copy()
        self._nis_streak = 0
        self._state = self._make_state(1.0, 0.0, False)
        return self._state


class CanonicalEKF(_BaseBicycleEKF):
    """Two-Point Bicycle EKF with fixed fading factor lambda_k = 1.0."""

    def __init__(self, config: EKFConfig | None = None) -> None:
        super().__init__(config=config, adaptive=False)


class AdaptiveEKF(_BaseBicycleEKF):
    """Stabilized Adaptive Two-Point Bicycle EKF with circular heading smoothing and NIS persistence."""

    def __init__(self, config: EKFConfig | None = None) -> None:
        super().__init__(config=config, adaptive=True)


class CTRVFilter:
    """5-state Coordinated Turn Rate and Velocity (CTRV) Extended Kalman Filter."""

    def __init__(self, config: EKFConfig | None = None) -> None:
        self.cfg = config if config is not None else EKFConfig(model_type="ctrv")
        self.model = CTRVModel()
        self._x = np.zeros(5, dtype=np.float64)
        self._P = np.eye(5, dtype=np.float64)
        self._last_dt: float = 0.1
        self._last_timestamp: float | None = None
        self._prev_xy: np.ndarray | None = None
        self._prev_theta: float | None = None
        self._state: FilterState = self._make_state(0.0)

    def _make_state(self, nis: float = 0.0) -> FilterState:
        self._x[3] = wrap_angle(float(self._x[3]))
        self._x[4] = float(np.clip(self._x[4], -2.5, 2.5))
        self._P = _enforce_spd(self._P)
        return FilterState(
            x=self._x.copy(),
            P=self._P.copy(),
            x_utm=float(self._x[0]),
            y_utm=float(self._x[1]),
            v_mps=float(self._x[2]),
            theta_rad=float(self._x[3]),
            P_pos_2x2=_enforce_spd(self._P[:2, :2]),
            lambda_k=1.0,
            nis=float(nis),
            adaptive_triggered=False,
        )

    @property
    def state(self) -> FilterState:
        return self._state

    def initialize(
        self,
        x0_utm: float,
        y0_utm: float,
        v0: float = 0.0,
        theta0: float = 0.0,
        P0: np.ndarray | None = None,
    ) -> FilterState:
        th0 = wrap_angle(float(theta0))
        self._x = np.array([float(x0_utm), float(y0_utm), float(v0), th0, 0.0], dtype=np.float64)
        if P0 is not None:
            p_arr = np.asarray(P0, dtype=np.float64)
            if p_arr.shape == (5, 5):
                self._P = _enforce_spd(p_arr)
            elif p_arr.shape == (2, 2):
                self._P = np.diag([1.0, 1.0, 1.0, 0.1, 0.1]).astype(np.float64)
                self._P[:2, :2] = _enforce_spd(p_arr)
            else:
                raise ValueError("P0 must have shape (5, 5) or (2, 2)")
        else:
            self._P = np.diag([0.25, 0.25, 1.0, 0.08, 0.08]).astype(np.float64)
        self._prev_xy = np.array([float(x0_utm), float(y0_utm)], dtype=np.float64)
        self._prev_theta = th0
        self._last_timestamp = None
        self._state = self._make_state(0.0)
        return self._state

    def predict(self, dt: float) -> FilterState:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        if dt_f == 0.0:
            self._state = self._make_state(self._state.nis)
            return self._state
        self._last_dt = dt_f
        f_jac = self.model.jacobian(self._x, dt_f)
        self._x = self.model.step(self._x, dt_f)
        q_mat = np.diag([0.15 * dt_f, 0.15 * dt_f, 0.5 * dt_f, 0.06 * dt_f, 0.08 * dt_f])
        self._P = _enforce_spd(f_jac @ self._P @ f_jac.T + q_mat)
        self._state = self._make_state(0.0)
        return self._state

    def update(
        self,
        z_xy: np.ndarray,
        R_k: np.ndarray,
        timestamp: float | None = None,
    ) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-6, max_eig=1e7)

        dt = self._last_dt if self._last_dt > 1e-6 else 0.1
        if timestamp is not None and self._last_timestamp is not None:
            dt_ts = float(timestamp) - float(self._last_timestamp)
            if dt_ts > 1e-6:
                dt = dt_ts
        if timestamp is not None:
            self._last_timestamp = float(timestamp)

        h_pos = np.zeros((2, 5), dtype=np.float64)
        h_pos[0, 0] = 1.0
        h_pos[1, 1] = 1.0

        y_pos = z - self._x[:2]
        s_mat = _enforce_spd(h_pos @ self._P @ h_pos.T + r_pos, min_eig=1e-6)
        s_inv = np.linalg.inv(s_mat)
        nis = float(y_pos.T @ s_inv @ y_pos)

        k_gain = self._P @ h_pos.T @ s_inv
        self._x = self._x + k_gain @ y_pos
        i_kh = np.eye(5, dtype=np.float64) - k_gain @ h_pos
        self._P = _enforce_spd(i_kh @ self._P @ i_kh.T + k_gain @ r_pos @ k_gain.T)

        if self._prev_xy is not None:
            disp = self._x[:2] - self._prev_xy
            dist = float(math.hypot(disp[0], disp[1]))
            if dist > 0.15 and abs(float(self._x[2])) >= 0.5:
                th_obs = math.atan2(float(disp[1]), float(disp[0]))
                err_th = wrap_angle(th_obs - float(self._x[3]))
                self._x[3] = wrap_angle(float(self._x[3]) + 0.35 * err_th)
                if self._prev_theta is not None:
                    omega_obs = wrap_angle(th_obs - self._prev_theta) / max(dt, 1e-3)
                    self._x[4] = float(np.clip(0.65 * float(self._x[4]) + 0.35 * omega_obs, -2.5, 2.5))
                self._prev_theta = float(self._x[3])
                v_obs = dist / max(dt, 1e-3)
                if v_obs < 60.0:
                    self._x[2] = 0.85 * float(self._x[2]) + 0.15 * v_obs
        self._prev_xy = self._x[:2].copy()
        self._state = self._make_state(nis)
        return self._state

    def reset_position_covariance(self, z_xy: np.ndarray, R_k: np.ndarray) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-6)
        self._x[0] = float(z[0])
        self._x[1] = float(z[1])
        self._P[0, :] = 0.0
        self._P[1, :] = 0.0
        self._P[:, 0] = 0.0
        self._P[:, 1] = 0.0
        self._P[:2, :2] = r_pos.copy()
        self._P = _enforce_spd(self._P)
        self._prev_xy = z.copy()
        self._state = self._make_state(0.0)
        return self._state


class LinearCVFilter:
    """4-state Constant Velocity (LinearCV) Kalman Filter using Joseph covariance update."""

    def __init__(self, config: EKFConfig | None = None) -> None:
        self.cfg = config if config is not None else EKFConfig(model_type="linear_cv")
        self.model = LinearCVModel()
        self._x = np.zeros(4, dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64)
        self._last_heading: float = 0.0
        self._state: FilterState = self._make_state(0.0)

    def _make_state(self, nis: float = 0.0) -> FilterState:
        self._P = _enforce_spd(self._P)
        vx, vy = float(self._x[2]), float(self._x[3])
        v_mps = float(math.hypot(vx, vy))
        if v_mps > 1e-3:
            self._last_heading = float(math.atan2(vy, vx))
        return FilterState(
            x=self._x.copy(),
            P=self._P.copy(),
            x_utm=float(self._x[0]),
            y_utm=float(self._x[1]),
            v_mps=v_mps,
            theta_rad=wrap_angle(self._last_heading),
            P_pos_2x2=_enforce_spd(self._P[:2, :2]),
            lambda_k=1.0,
            nis=float(nis),
            adaptive_triggered=False,
        )

    @property
    def state(self) -> FilterState:
        return self._state

    def initialize(
        self,
        x0_utm: float,
        y0_utm: float,
        v0: float = 0.0,
        theta0: float = 0.0,
        P0: np.ndarray | None = None,
    ) -> FilterState:
        th0 = wrap_angle(float(theta0))
        self._last_heading = th0
        vx0 = float(v0) * math.cos(th0)
        vy0 = float(v0) * math.sin(th0)
        self._x = np.array([float(x0_utm), float(y0_utm), vx0, vy0], dtype=np.float64)
        if P0 is not None:
            p_arr = np.asarray(P0, dtype=np.float64)
            if p_arr.shape == (4, 4):
                self._P = _enforce_spd(p_arr)
            elif p_arr.shape in ((2, 2), (5, 5)):
                self._P = np.diag([1.0, 1.0, 2.0, 2.0]).astype(np.float64)
                self._P[:2, :2] = _enforce_spd(p_arr[:2, :2])
            else:
                raise ValueError("P0 must have shape (4, 4), (5, 5), or (2, 2)")
        else:
            self._P = np.diag([0.25, 0.25, 1.5, 1.5]).astype(np.float64)
        self._state = self._make_state(0.0)
        return self._state

    def predict(self, dt: float) -> FilterState:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        if dt_f == 0.0:
            self._state = self._make_state(self._state.nis)
            return self._state

        f_mat = self.model.transition_matrix(dt_f)
        self._x = f_mat @ self._x
        # Conservative CV process noise so straight-line tracking is smooth while turns lag CTRV
        q_acc = 0.35
        dt2 = dt_f * dt_f
        dt3 = dt2 * dt_f
        dt4 = dt2 * dt2
        q_mat = q_acc * np.array(
            [
                [0.25 * dt4, 0.0, 0.5 * dt3, 0.0],
                [0.0, 0.25 * dt4, 0.0, 0.5 * dt3],
                [0.5 * dt3, 0.0, dt2, 0.0],
                [0.0, 0.5 * dt3, 0.0, dt2],
            ],
            dtype=np.float64,
        ) + np.diag([1e-4 * dt_f, 1e-4 * dt_f, 1e-3 * dt_f, 1e-3 * dt_f])
        self._P = _enforce_spd(f_mat @ self._P @ f_mat.T + q_mat)
        self._state = self._make_state(0.0)
        return self._state

    def update(
        self,
        z_xy: np.ndarray,
        R_k: np.ndarray,
        timestamp: float | None = None,
    ) -> FilterState:
        _ = timestamp
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-6, max_eig=1e7)

        h_pos = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
        y_pos = z - self._x[:2]
        s_mat = _enforce_spd(h_pos @ self._P @ h_pos.T + r_pos, min_eig=1e-6)
        s_inv = np.linalg.inv(s_mat)
        nis = float(y_pos.T @ s_inv @ y_pos)

        k_gain = self._P @ h_pos.T @ s_inv
        self._x = self._x + k_gain @ y_pos
        i_kh = np.eye(4, dtype=np.float64) - k_gain @ h_pos
        self._P = _enforce_spd(i_kh @ self._P @ i_kh.T + k_gain @ r_pos @ k_gain.T)
        self._state = self._make_state(nis)
        return self._state

    def reset_position_covariance(self, z_xy: np.ndarray, R_k: np.ndarray) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-6)
        self._x[0] = float(z[0])
        self._x[1] = float(z[1])
        self._P[0, :] = 0.0
        self._P[1, :] = 0.0
        self._P[:, 0] = 0.0
        self._P[:, 1] = 0.0
        self._P[:2, :2] = r_pos.copy()
        self._P = _enforce_spd(self._P)
        self._state = self._make_state(0.0)
        return self._state


class IMMFilter:
    """Interacting Multiple Model (IMM) filter combining AdaptiveEKF and CTRVFilter."""

    def __init__(self, config: EKFConfig | None = None) -> None:
        self.cfg = config if config is not None else EKFConfig(model_type="imm")
        self.m_bicycle = AdaptiveEKF(self.cfg)
        self.m_ctrv = CTRVFilter(self.cfg)
        self.models = [self.m_bicycle, self.m_ctrv]
        self.weights = np.array([0.5, 0.5], dtype=np.float64)
        self.trans = np.array([[0.95, 0.05], [0.05, 0.95]], dtype=np.float64)
        self._state: FilterState = self._fuse_states()

    def _fuse_states(self) -> FilterState:
        w0, w1 = float(self.weights[0]), float(self.weights[1])
        s0 = self.m_bicycle.state
        s1 = self.m_ctrv.state

        x_utm = w0 * s0.x_utm + w1 * s1.x_utm
        y_utm = w0 * s0.y_utm + w1 * s1.y_utm
        v_mps = w0 * s0.v_mps + w1 * s1.v_mps
        theta_rad = circular_mean([s0.theta_rad, s1.theta_rad], weights=[w0, w1])

        x_fused = np.array(
            [x_utm, y_utm, v_mps, theta_rad, w0 * float(s0.x[4]) + w1 * float(s1.x[4])],
            dtype=np.float64,
        )

        # Moment-matched covariance fusion
        p_fused = np.zeros((5, 5), dtype=np.float64)
        for w, st in ((w0, s0), (w1, s1)):
            diff = st.x.copy()
            diff[0] -= x_utm
            diff[1] -= y_utm
            diff[2] -= v_mps
            diff[3] = wrap_angle(float(st.theta_rad) - theta_rad)
            diff[4] -= float(x_fused[4])
            p_fused += w * (st.P + np.outer(diff, diff))

        p_fused = _enforce_spd(p_fused)
        return FilterState(
            x=x_fused,
            P=p_fused,
            x_utm=x_utm,
            y_utm=y_utm,
            v_mps=v_mps,
            theta_rad=theta_rad,
            P_pos_2x2=_enforce_spd(p_fused[:2, :2]),
            lambda_k=max(s0.lambda_k, s1.lambda_k),
            nis=w0 * s0.nis + w1 * s1.nis,
            adaptive_triggered=bool(s0.adaptive_triggered or s1.adaptive_triggered),
        )

    @property
    def state(self) -> FilterState:
        return self._state

    def initialize(
        self,
        x0_utm: float,
        y0_utm: float,
        v0: float = 0.0,
        theta0: float = 0.0,
        P0: np.ndarray | None = None,
    ) -> FilterState:
        self.m_bicycle.initialize(x0_utm, y0_utm, v0=v0, theta0=theta0, P0=P0)
        self.m_ctrv.initialize(x0_utm, y0_utm, v0=v0, theta0=theta0, P0=P0)
        self.weights = np.array([0.5, 0.5], dtype=np.float64)
        self._state = self._fuse_states()
        return self._state

    def predict(self, dt: float) -> FilterState:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        self.weights = self.trans.T @ self.weights
        self.weights = np.clip(self.weights, 1e-3, 1.0)
        self.weights /= np.sum(self.weights)
        self.m_bicycle.predict(dt_f)
        self.m_ctrv.predict(dt_f)
        self._state = self._fuse_states()
        return self._state

    def update(
        self,
        z_xy: np.ndarray,
        R_k: np.ndarray,
        timestamp: float | None = None,
    ) -> FilterState:
        s0 = self.m_bicycle.update(z_xy, R_k, timestamp=timestamp)
        s1 = self.m_ctrv.update(z_xy, R_k, timestamp=timestamp)

        # Update mode probabilities via Gaussian innovation likelihood
        l0 = math.exp(-0.5 * min(s0.nis, 50.0))
        l1 = math.exp(-0.5 * min(s1.nis, 50.0))
        post = np.array([self.weights[0] * l0, self.weights[1] * l1], dtype=np.float64) + 1e-6
        self.weights = post / np.sum(post)
        self._state = self._fuse_states()
        return self._state

    def reset_position_covariance(self, z_xy: np.ndarray, R_k: np.ndarray) -> FilterState:
        self.m_bicycle.reset_position_covariance(z_xy, R_k)
        self.m_ctrv.reset_position_covariance(z_xy, R_k)
        self._state = self._fuse_states()
        return self._state


# Required aliases per interface contract
CTRV = CTRVFilter
LinearCV = LinearCVFilter
IMM = IMMFilter


def create_filter(config: EKFConfig | None = None) -> AdaptiveEKF | CanonicalEKF | CTRVFilter | LinearCVFilter | IMMFilter:
    """Factory creating a kinematic filter instance from EKFConfig.model_type."""
    cfg = config if config is not None else EKFConfig()
    mapping = {
        "adaptive_bicycle": AdaptiveEKF,
        "canonical_bicycle": CanonicalEKF,
        "ctrv": CTRVFilter,
        "linear_cv": LinearCVFilter,
        "imm": IMMFilter,
    }
    cls = mapping.get(cfg.model_type, AdaptiveEKF)
    return cls(cfg)
