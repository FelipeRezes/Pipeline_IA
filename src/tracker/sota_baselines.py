"""State-of-the-art comparative baselines: SIR Particle Filter, LSTM, GRU, and Temporal GNN (T-GNN).

Implements deterministic, causal, FilterState-compatible baselines for controlled empirical
comparison against the Adaptive Two-Point-Bicycle EKF and physics-constrained multi-hypothesis
road-graph propagation proposed in this work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Literal

import numpy as np

from tracker.config import EKFConfig
from tracker.ekf import FilterState, _enforce_spd
from tracker.kinematics import TwoPointBicycle, circular_mean, wrap_angle
from tracker.osm_loader import RoadNetwork


# =============================================================================
# 1. Sequential Importance Resampling (SIR) Particle Filter
# =============================================================================


class ParticleFilter:
    """Nonlinear 5-state SIR Particle Filter with Two-Point-Bicycle kinematics.

    Implements bootstrap sequential importance sampling with Gaussian likelihood
    weighting and low-variance systematic resampling triggered when N_eff < 0.5 * N_p.
    References: Gordon et al. (1993); Arulampalam et al. (2002).
    """

    def __init__(
        self,
        config: EKFConfig | None = None,
        *,
        num_particles: int = 500,
        seed: int = 42,
    ) -> None:
        self.cfg = config if config is not None else EKFConfig()
        self.num_particles = max(50, int(num_particles))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)
        self.wheelbase_m = float(self.cfg.wheelbase_m)
        self.max_steer_rad = float(TwoPointBicycle.MAX_STEER_RAD)

        self._particles = np.zeros((self.num_particles, 5), dtype=np.float64)
        self._weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        self._last_dt: float = 0.1
        self._last_timestamp: float | None = None
        self._neff: float = float(self.num_particles)
        self._state: FilterState = self._estimate_state(nis=0.0)

    @property
    def state(self) -> FilterState:
        return self._state

    @property
    def neff(self) -> float:
        return self._neff

    def _estimate_state(self, nis: float = 0.0) -> FilterState:
        w = self._weights
        p = self._particles
        x_mean = float(np.sum(w * p[:, 0]))
        y_mean = float(np.sum(w * p[:, 1]))
        v_mean = float(np.sum(w * p[:, 2]))
        th_mean = float(math.atan2(float(np.sum(w * np.sin(p[:, 3]))), float(np.sum(w * np.cos(p[:, 3])))))
        d_mean = float(np.clip(float(np.sum(w * p[:, 4])), -self.max_steer_rad, self.max_steer_rad))

        x_vec = np.array([x_mean, y_mean, v_mean, th_mean, d_mean], dtype=np.float64)
        diffs = p - x_vec[None, :]
        # Circular angle difference for heading column
        diffs[:, 3] = (diffs[:, 3] + math.pi) % (2.0 * math.pi) - math.pi
        cov = (diffs.T * w[None, :]) @ diffs
        cov = _enforce_spd(cov, min_eig=1e-4, max_eig=1e6)

        return FilterState(
            x=x_vec,
            P=cov,
            x_utm=x_mean,
            y_utm=y_mean,
            v_mps=v_mean,
            theta_rad=wrap_angle(th_mean),
            P_pos_2x2=_enforce_spd(cov[:2, :2], min_eig=1e-4),
            lambda_k=1.0,
            nis=float(nis),
            adaptive_triggered=False,
        )

    def initialize(
        self,
        x0_utm: float,
        y0_utm: float,
        v0: float = 0.0,
        theta0: float = 0.0,
        P0: np.ndarray | None = None,
    ) -> FilterState:
        self._rng = np.random.default_rng(self.seed)
        th0 = wrap_angle(float(theta0))
        mean = np.array([float(x0_utm), float(y0_utm), float(v0), th0, 0.0], dtype=np.float64)
        if P0 is not None:
            p_arr = np.asarray(P0, dtype=np.float64)
            if p_arr.shape == (5, 5):
                cov0 = _enforce_spd(p_arr, min_eig=1e-4)
            elif p_arr.shape == (2, 2):
                cov0 = np.diag([0.25, 0.25, 0.5, 0.05, 0.04]).astype(np.float64)
                cov0[:2, :2] = _enforce_spd(p_arr, min_eig=1e-4)
            else:
                raise ValueError("P0 must have shape (5, 5) or (2, 2)")
        else:
            cov0 = np.diag([0.16, 0.16, 0.36, 0.02, 0.02]).astype(np.float64)

        self._particles = self._rng.multivariate_normal(mean, cov0, size=self.num_particles)
        self._particles[:, 2] = np.clip(self._particles[:, 2], 0.0, 60.0)
        self._particles[:, 3] = (self._particles[:, 3] + math.pi) % (2.0 * math.pi) - math.pi
        self._particles[:, 4] = np.clip(self._particles[:, 4], -self.max_steer_rad, self.max_steer_rad)
        self._weights = np.full(self.num_particles, 1.0 / self.num_particles, dtype=np.float64)
        self._neff = float(self.num_particles)
        self._last_timestamp = None
        self._state = self._estimate_state(nis=0.0)
        return self._state

    def predict(self, dt: float) -> FilterState:
        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        if dt_f == 0.0:
            return self._state

        self._last_dt = dt_f
        n = self.num_particles
        # Stochastic longitudinal acceleration and steering-rate excitations
        accel = self._rng.normal(0.0, 1.6, size=n)
        steer_rate = self._rng.normal(0.0, 0.65, size=n)
        pos_jitter = self._rng.normal(0.0, 0.08 * math.sqrt(dt_f), size=(n, 2))

        v = np.clip(self._particles[:, 2] + accel * dt_f, 0.0, 60.0)
        delta = np.clip(
            self._particles[:, 4] + steer_rate * dt_f,
            -self.max_steer_rad,
            self.max_steer_rad,
        )
        theta = self._particles[:, 3]
        omega = (v / self.wheelbase_m) * np.tan(delta)

        small = np.abs(omega) < 1e-5
        nx = np.empty(n, dtype=np.float64)
        ny = np.empty(n, dtype=np.float64)
        nth = theta + omega * dt_f

        if np.any(small):
            th_mid = theta[small] + 0.5 * omega[small] * dt_f
            nx[small] = self._particles[small, 0] + v[small] * np.cos(th_mid) * dt_f
            ny[small] = self._particles[small, 1] + v[small] * np.sin(th_mid) * dt_f
        if np.any(~small):
            m = ~small
            r = v[m] / omega[m]
            nx[m] = self._particles[m, 0] + r * (np.sin(nth[m]) - np.sin(theta[m]))
            ny[m] = self._particles[m, 1] + r * (-np.cos(nth[m]) + np.cos(theta[m]))

        self._particles[:, 0] = nx + pos_jitter[:, 0]
        self._particles[:, 1] = ny + pos_jitter[:, 1]
        self._particles[:, 2] = v
        self._particles[:, 3] = (nth + math.pi) % (2.0 * math.pi) - math.pi
        self._particles[:, 4] = delta
        self._state = self._estimate_state(nis=0.0)
        return self._state

    def _systematic_resample(self) -> None:
        n = self.num_particles
        positions = (self._rng.random() + np.arange(n, dtype=np.float64)) / float(n)
        cumsum = np.cumsum(self._weights)
        cumsum[-1] = 1.0
        idx = np.searchsorted(cumsum, positions)
        self._particles = self._particles[idx].copy()
        # Mild roughening jitter to prevent sample impoverishment
        self._particles[:, 0] += self._rng.normal(0.0, 0.03, size=n)
        self._particles[:, 1] += self._rng.normal(0.0, 0.03, size=n)
        self._particles[:, 3] = (
            self._particles[:, 3] + self._rng.normal(0.0, 0.015, size=n) + math.pi
        ) % (2.0 * math.pi) - math.pi
        self._particles[:, 4] = np.clip(
            self._particles[:, 4] + self._rng.normal(0.0, 0.02, size=n),
            -self.max_steer_rad,
            self.max_steer_rad,
        )
        self._weights.fill(1.0 / n)

    def update(
        self,
        z_xy: np.ndarray,
        R_k: np.ndarray,
        timestamp: float | None = None,
    ) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        if z.size != 2 or not np.all(np.isfinite(z)):
            raise ValueError("z_xy must be a finite 2-element array")
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-4, max_eig=1e7)
        if timestamp is not None:
            self._last_timestamp = float(timestamp)

        # Prior NIS relative to weighted mean
        y_prior = z - self._state.x[:2]
        s_prior = _enforce_spd(self._state.P_pos_2x2 + r_pos, min_eig=1e-4)
        nis = float(y_prior.T @ np.linalg.inv(s_prior) @ y_prior)

        # Measurement likelihood for all particles
        # Slightly inflate likelihood covariance floor for numerical stability under high noise
        r_eff = _enforce_spd(r_pos + np.diag([0.04, 0.04]), min_eig=1e-3)
        r_eff_inv = np.linalg.inv(r_eff)
        innov = z[None, :] - self._particles[:, :2]
        quad = np.einsum("ni,ij,nj->n", innov, r_eff_inv, innov)
        log_w = np.log(np.maximum(self._weights, 1e-300)) - 0.5 * quad
        log_w -= float(np.max(log_w))
        w_unnorm = np.exp(log_w)
        w_sum = float(np.sum(w_unnorm))
        if not math.isfinite(w_sum) or w_sum <= 1e-300:
            self._weights.fill(1.0 / self.num_particles)
        else:
            self._weights = w_unnorm / w_sum

        self._neff = 1.0 / max(float(np.sum(self._weights**2)), 1e-12)
        if self._neff < 0.5 * self.num_particles:
            self._systematic_resample()

        self._state = self._estimate_state(nis=nis)
        return self._state

    def reset_position_covariance(self, z_xy: np.ndarray, R_k: np.ndarray) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-4)
        pts = self._rng.multivariate_normal(z, r_pos, size=self.num_particles)
        self._particles[:, :2] = pts
        self._weights.fill(1.0 / self.num_particles)
        self._neff = float(self.num_particles)
        self._state = self._estimate_state(nis=0.0)
        return self._state


# =============================================================================
# 2. Recurrent Sequence Models (LSTM & GRU) in PyTorch
# =============================================================================


_RNN_MODEL_CACHE: dict[tuple[str, int, int, int], Any] = {}


def _generate_urban_kinematic_training_data(
    num_trajectories: int = 360,
    steps_per_traj: int = 45,
    dt: float = 0.1,
    seq_len: int = 6,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate SE(2)-invariant local-frame urban vehicle motion training sequences.

    Each input step has 6 features in the vehicle's ego frame at time t:
        [ds_lon / 3.0, ds_lat / 1.5, v / 20.0, omega, accel / 5.0, obs_flag]
    Target is next-step local-frame increment:
        [next_ds_lon / 3.0, next_ds_lat / 1.5, next_v / 20.0, next_dtheta]
    """
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    for traj_idx in range(num_trajectories):
        v = float(rng.uniform(4.0, 22.0))
        theta = float(rng.uniform(-math.pi, math.pi))
        omega = 0.0
        accel = 0.0
        x, y = 0.0, 0.0
        regime = traj_idx % 4  # 0: straight+accel, 1: sharp turn, 2: S-curve, 3: turn+brake
        turn_start = int(rng.integers(8, 18))
        turn_dur = int(rng.integers(10, 22))
        omega_mag = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.4, 1.35))
        sigma_obs = float(rng.uniform(0.05, 0.45))

        hist_feats: list[np.ndarray] = []
        gt_states: list[tuple[float, float, float, float]] = [(x, y, v, theta)]

        for k in range(1, steps_per_traj + 1):
            if regime == 0:
                omega = 0.0
                accel = float(rng.uniform(-1.5, 1.5)) if k % 10 == 0 else accel
            elif regime == 1:
                omega = omega_mag if turn_start <= k < turn_start + turn_dur else 0.0
                accel = 0.0
            elif regime == 2:
                if turn_start <= k < turn_start + turn_dur // 2:
                    omega = omega_mag
                elif turn_start + turn_dur // 2 <= k < turn_start + turn_dur:
                    omega = -omega_mag
                else:
                    omega = 0.0
                accel = 0.0
            else:
                if turn_start <= k < turn_start + turn_dur:
                    omega = omega_mag
                    accel = -1.8 if k < turn_start + turn_dur // 2 else 1.5
                else:
                    omega = 0.0
                    accel = 0.0

            v_next = float(np.clip(v + accel * dt, 1.0, 28.0))
            if abs(omega) < 1e-6:
                nx = x + v * math.cos(theta) * dt
                ny = y + v * math.sin(theta) * dt
                nth = wrap_angle(theta)
            else:
                nx = x + (v / omega) * (math.sin(theta + omega * dt) - math.sin(theta))
                ny = y + (v / omega) * (-math.cos(theta + omega * dt) + math.cos(theta))
                nth = wrap_angle(theta + omega * dt)

            # Rotate displacement into previous heading frame
            dx = nx - x + float(rng.normal(0.0, sigma_obs * 0.25))
            dy = ny - y + float(rng.normal(0.0, sigma_obs * 0.25))
            c, s = math.cos(theta), math.sin(theta)
            ds_lon = c * dx + s * dy
            ds_lat = -s * dx + c * dy
            dtheta = wrap_angle(nth - theta)
            a_step = (v_next - v) / dt

            feat = np.array(
                [ds_lon / 3.0, ds_lat / 1.5, v_next / 20.0, dtheta / dt, a_step / 5.0, 1.0],
                dtype=np.float32,
            )
            hist_feats.append(feat)
            gt_states.append((nx, ny, v_next, nth))
            x, y, v, theta = nx, ny, v_next, nth

        for t in range(seq_len, len(hist_feats)):
            window = np.stack(hist_feats[t - seq_len : t], axis=0)
            # Clean ground-truth next step increment in frame t-1
            x0, y0, _v0, th0 = gt_states[t]
            x1, y1, v1, th1 = gt_states[t + 1]
            c, s = math.cos(th0), math.sin(th0)
            dx_gt = x1 - x0
            dy_gt = y1 - y0
            target = np.array(
                [
                    (c * dx_gt + s * dy_gt) / 3.0,
                    (-s * dx_gt + c * dy_gt) / 1.5,
                    v1 / 20.0,
                    wrap_angle(th1 - th0),
                ],
                dtype=np.float32,
            )
            xs.append(window)
            ys.append(target)

    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def _get_or_train_rnn_model(
    cell_type: Literal["lstm", "gru"],
    hidden_size: int = 48,
    seq_len: int = 6,
    seed: int = 42,
) -> Any:
    """Train (once and cache) a lightweight PyTorch LSTM or GRU kinematic model on CPU."""
    import torch
    from torch import nn

    key = (cell_type, int(hidden_size), int(seq_len), int(seed))
    if key in _RNN_MODEL_CACHE:
        return _RNN_MODEL_CACHE[key]

    torch.manual_seed(seed)
    np.random.seed(seed)

    class _SeqTrajectoryNet(nn.Module):
        def __init__(self, kind: str, h_dim: int) -> None:
            super().__init__()
            rnn_cls = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = rnn_cls(input_size=6, hidden_size=h_dim, num_layers=2, batch_first=True)
            self.norm = nn.LayerNorm(h_dim)
            self.head = nn.Sequential(
                nn.Linear(h_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
            )

        def forward(self, seq: torch.Tensor) -> torch.Tensor:
            out, _ = self.rnn(seq)
            h_last = self.norm(out[:, -1, :])
            return self.head(h_last)

    X_np, Y_np = _generate_urban_kinematic_training_data(
        num_trajectories=320,
        steps_per_traj=42,
        dt=0.1,
        seq_len=seq_len,
        seed=seed + 100,
    )

    model = _SeqTrajectoryNet(cell_type, hidden_size)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    loss_fn = nn.SmoothL1Loss()

    X_t = torch.from_numpy(X_np)
    Y_t = torch.from_numpy(Y_np)
    n_samples = X_t.shape[0]
    batch_size = 256
    gen = torch.Generator().manual_seed(seed)

    for _epoch in range(16):
        perm = torch.randperm(n_samples, generator=gen)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad(set_to_none=True)
            pred = model(X_t[idx])
            loss = loss_fn(pred, Y_t[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    model.eval()
    _RNN_MODEL_CACHE[key] = model
    return model


class _BaseRecurrentTracker:
    """Shared causal wrapper for LSTM and GRU trajectory predictors."""

    def __init__(
        self,
        cell_type: Literal["lstm", "gru"],
        config: EKFConfig | None = None,
        *,
        hidden_size: int = 48,
        seq_len: int = 6,
        seed: int = 42,
    ) -> None:
        self.cell_type = cell_type
        self.cfg = config if config is not None else EKFConfig()
        self.hidden_size = int(hidden_size)
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self._model = _get_or_train_rnn_model(
            cell_type=self.cell_type,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            seed=self.seed,
        )
        self._x = np.zeros(5, dtype=np.float64)
        self._P = np.eye(5, dtype=np.float64)
        self._history: deque[np.ndarray] = deque(maxlen=self.seq_len)
        self._last_dt: float = 0.1
        self._last_obs_xy: np.ndarray | None = None
        self._last_obs_theta: float = 0.0
        self._last_v: float = 0.0
        self._blind_steps: int = 0
        self._state = self._make_state(0.0)

    @property
    def state(self) -> FilterState:
        return self._state

    @property
    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self._model.parameters())

    def _make_state(self, nis: float = 0.0) -> FilterState:
        self._x[3] = wrap_angle(float(self._x[3]))
        self._P = _enforce_spd(self._P, min_eig=1e-4, max_eig=1e6)
        return FilterState(
            x=self._x.copy(),
            P=self._P.copy(),
            x_utm=float(self._x[0]),
            y_utm=float(self._x[1]),
            v_mps=float(self._x[2]),
            theta_rad=float(self._x[3]),
            P_pos_2x2=_enforce_spd(self._P[:2, :2], min_eig=1e-4),
            lambda_k=1.0,
            nis=float(nis),
            adaptive_triggered=False,
        )

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
        if P0 is not None and np.asarray(P0).shape == (5, 5):
            self._P = _enforce_spd(np.asarray(P0, dtype=np.float64))
        else:
            self._P = np.diag([0.25, 0.25, 0.5, 0.05, 0.05]).astype(np.float64)
        self._history.clear()
        init_feat = np.array([float(v0) * 0.1 / 3.0, 0.0, float(v0) / 20.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(self.seq_len):
            self._history.append(init_feat.copy())
        self._last_obs_xy = np.array([float(x0_utm), float(y0_utm)], dtype=np.float64)
        self._last_obs_theta = th0
        self._last_v = float(v0)
        self._blind_steps = 0
        self._state = self._make_state(0.0)
        return self._state

    def predict(self, dt: float) -> FilterState:
        import torch

        dt_f = float(dt)
        if not math.isfinite(dt_f) or dt_f < 0.0:
            raise ValueError(f"dt must be finite and >= 0 (got {dt})")
        if dt_f == 0.0:
            return self._state
        self._last_dt = dt_f
        self._blind_steps += 1

        seq_arr = np.stack(list(self._history), axis=0)[None, :, :]
        with torch.no_grad():
            pred = self._model(torch.from_numpy(seq_arr)).numpy()[0]

        scale_dt = dt_f / 0.1
        ds_lon = float(pred[0]) * 3.0 * scale_dt
        ds_lat = float(pred[1]) * 1.5 * scale_dt
        v_pred = float(np.clip(float(pred[2]) * 20.0, 0.0, 50.0))
        dtheta = float(np.clip(float(pred[3]) * scale_dt, -0.6, 0.6))

        th = float(self._x[3])
        c, s = math.cos(th), math.sin(th)
        dx = c * ds_lon - s * ds_lat
        dy = s * ds_lon + c * ds_lat
        new_th = wrap_angle(th + dtheta)

        self._x[0] += dx
        self._x[1] += dy
        self._x[2] = v_pred
        self._x[3] = new_th
        self._x[4] = dtheta / max(dt_f, 1e-3)

        # Autoregressive feature feedback when running without observations
        a_est = (v_pred - self._last_v) / max(dt_f, 1e-3)
        self._last_v = v_pred
        if self._blind_steps > 1:
            self._history.append(
                np.array(
                    [ds_lon / 3.0, ds_lat / 1.5, v_pred / 20.0, dtheta / max(dt_f, 1e-3), a_est / 5.0, 0.0],
                    dtype=np.float32,
                )
            )

        # Grow uncertainty monotonically during prediction / blind rollout
        self._P = _enforce_spd(self._P + np.diag([0.35 * dt_f, 0.35 * dt_f, 0.4 * dt_f, 0.04 * dt_f, 0.05 * dt_f]))
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
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-4)
        dt = max(self._last_dt, 1e-3)

        y_pos = z - self._x[:2]
        s_pos = _enforce_spd(self._P[:2, :2] + r_pos, min_eig=1e-4)
        nis = float(y_pos.T @ np.linalg.inv(s_pos) @ y_pos)

        # Inverse-variance fusion between RNN prior and sensor observation
        k_xy = self._P[:2, :2] @ np.linalg.inv(s_pos)
        fused_xy = self._x[:2] + k_xy @ y_pos
        self._P[:2, :2] = _enforce_spd((np.eye(2) - k_xy) @ self._P[:2, :2])

        if self._last_obs_xy is not None:
            disp = fused_xy - self._last_obs_xy
            dist = float(math.hypot(disp[0], disp[1]))
            v_obs = float(np.clip(dist / dt, 0.0, 50.0))
            th_prev = self._last_obs_theta
            th_obs = math.atan2(float(disp[1]), float(disp[0])) if dist > 0.15 else th_prev
            c, s = math.cos(th_prev), math.sin(th_prev)
            ds_lon = c * float(disp[0]) + s * float(disp[1])
            ds_lat = -s * float(disp[0]) + c * float(disp[1])
            dtheta = wrap_angle(th_obs - th_prev)
            a_est = (v_obs - self._last_v) / dt

            self._x[0] = float(fused_xy[0])
            self._x[1] = float(fused_xy[1])
            self._x[2] = 0.6 * float(self._x[2]) + 0.4 * v_obs
            self._x[3] = circular_mean([float(self._x[3]), th_obs], weights=[0.45, 0.55])
            self._x[4] = dtheta / dt

            self._history.append(
                np.array(
                    [ds_lon / 3.0, ds_lat / 1.5, self._x[2] / 20.0, dtheta / dt, a_est / 5.0, 1.0],
                    dtype=np.float32,
                )
            )
            self._last_obs_theta = float(self._x[3])
            self._last_v = float(self._x[2])
        else:
            self._x[0] = float(fused_xy[0])
            self._x[1] = float(fused_xy[1])

        self._last_obs_xy = self._x[:2].copy()
        self._blind_steps = 0
        self._state = self._make_state(nis)
        return self._state

    def reset_position_covariance(self, z_xy: np.ndarray, R_k: np.ndarray) -> FilterState:
        z = np.asarray(z_xy, dtype=np.float64).ravel()
        r_pos = _enforce_spd(np.asarray(R_k, dtype=np.float64), min_eig=1e-4)
        self._x[0] = float(z[0])
        self._x[1] = float(z[1])
        self._P[:2, :2] = r_pos.copy()
        self._last_obs_xy = z.copy()
        self._blind_steps = 0
        self._state = self._make_state(0.0)
        return self._state


class LSTMTrajectoryPredictor(_BaseRecurrentTracker):
    """Long Short-Term Memory (LSTM) causal trajectory tracker and blind-zone predictor."""

    def __init__(
        self,
        config: EKFConfig | None = None,
        *,
        hidden_size: int = 48,
        seq_len: int = 6,
        seed: int = 42,
    ) -> None:
        super().__init__(
            cell_type="lstm",
            config=config,
            hidden_size=hidden_size,
            seq_len=seq_len,
            seed=seed,
        )


class GRUTrajectoryPredictor(_BaseRecurrentTracker):
    """Gated Recurrent Unit (GRU) causal trajectory tracker and blind-zone predictor."""

    def __init__(
        self,
        config: EKFConfig | None = None,
        *,
        hidden_size: int = 48,
        seq_len: int = 6,
        seed: int = 42,
    ) -> None:
        super().__init__(
            cell_type="gru",
            config=config,
            hidden_size=hidden_size,
            seq_len=seq_len,
            seed=seed,
        )


# =============================================================================
# 3. Temporal Graph Neural Network (T-GNN) for Road-Network Route & State Prediction
# =============================================================================


@dataclass
class TGNNRoutePrediction:
    """Predicted outgoing edge probabilities and blind-zone state from T-GNN."""

    edge_probabilities: list[tuple[tuple[int, int, int], float]]
    predicted_x_utm: float
    predicted_y_utm: float
    predicted_v_mps: float


class TemporalGNNPredictor:
    """Spatio-Temporal Graph Neural Network (T-GNN) over a directed urban RoadNetwork.

    Architecture:
      1. Spatial 2-layer normalized Graph Convolution (GCN) over G = (V, E) producing
         contextual intersection embeddings H_V from structural features (UTM coords,
         in/out degree, mean speed limit, lanes).
      2. Temporal GRU encoding recent vehicle kinematic sequence (v, cos(theta), sin(theta),
         turn rate, longitudinal progress).
      3. Edge-transition scoring MLP predicting calibrated next-edge probabilities over
         Out(u) at urban intersections + blind-zone curvilinear rollout.

    References: Yu et al. (IJCAI 2018); Zhao et al. (IEEE T-ITS 2020, T-GCN).
    """

    def __init__(
        self,
        network: RoadNetwork,
        *,
        hidden_dim: int = 32,
        seed: int = 42,
    ) -> None:
        self.network = network
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self._node_ids = sorted(network.nodes_utm.keys())
        self._node_to_idx = {nid: idx for idx, nid in enumerate(self._node_ids)}
        self._outgoing: dict[int, list[tuple[int, int, int]]] = {}
        for eid, edge in network.edges.items():
            self._outgoing.setdefault(edge.u, []).append(eid)
        for uid in self._outgoing:
            self._outgoing[uid].sort()

        self._adj_norm, self._node_feats, self._edge_feats = self._build_graph_tensors()
        self._model = self._build_and_train(seed=self.seed)

    @property
    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self._model.parameters())

    def _build_graph_tensors(self) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int, int], np.ndarray]]:
        n_nodes = len(self._node_ids)
        coords = np.array([self.network.nodes_utm[nid] for nid in self._node_ids], dtype=np.float64)
        c_mean = np.mean(coords, axis=0)
        c_std = np.maximum(np.std(coords, axis=0), 1.0)
        norm_coords = (coords - c_mean) / c_std

        in_deg = np.zeros(n_nodes, dtype=np.float64)
        out_deg = np.zeros(n_nodes, dtype=np.float64)
        mean_spd = np.zeros(n_nodes, dtype=np.float64)
        mean_lanes = np.zeros(n_nodes, dtype=np.float64)
        adj = np.eye(n_nodes, dtype=np.float64)

        edge_feats: dict[tuple[int, int, int], np.ndarray] = {}
        for eid, edge in self.network.edges.items():
            iu = self._node_to_idx[edge.u]
            iv = self._node_to_idx[edge.v]
            adj[iu, iv] = 1.0
            adj[iv, iu] = 1.0  # undirected spatial context message passing
            out_deg[iu] += 1.0
            in_deg[iv] += 1.0
            mean_spd[iu] += edge.maxspeed_mps
            mean_lanes[iu] += float(edge.lanes)

            curv_proxy = 1.0 / min(max(edge.min_curve_radius_m, 5.0), 200.0)
            edge_feats[eid] = np.array(
                [
                    edge.length_m / 200.0,
                    edge.maxspeed_mps / 20.0,
                    float(edge.lanes) / 4.0,
                    math.cos(edge.heading_start_rad),
                    math.sin(edge.heading_start_rad),
                    curv_proxy * 20.0,
                ],
                dtype=np.float32,
            )

        deg_safe = np.maximum(out_deg, 1.0)
        mean_spd = (mean_spd / deg_safe) / 20.0
        mean_lanes = (mean_lanes / deg_safe) / 4.0

        node_feats = np.column_stack(
            [
                norm_coords[:, 0],
                norm_coords[:, 1],
                in_deg / 4.0,
                out_deg / 4.0,
                mean_spd,
                mean_lanes,
            ]
        ).astype(np.float32)

        d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(np.sum(adj, axis=1), 1.0)))
        adj_norm = (d_inv_sqrt @ adj @ d_inv_sqrt).astype(np.float32)
        return adj_norm, node_feats, edge_feats

    def _build_and_train(self, seed: int) -> Any:
        import torch
        import torch.nn as nn

        torch.manual_seed(seed)
        rng = np.random.default_rng(seed + 77)
        h_dim = self.hidden_dim

        class _TGNNModule(nn.Module):
            def __init__(self, hidden: int) -> None:
                super().__init__()
                self.gcn1 = nn.Linear(6, hidden)
                self.gcn2 = nn.Linear(hidden, hidden)
                self.temp_gru = nn.GRU(input_size=5, hidden_size=hidden, num_layers=1, batch_first=True)
                # Edge scorer: [h_temp (H), h_u (H), h_v (H), edge_feat (6), cos_dth, sin_dth, v_ratio]
                self.edge_scorer = nn.Sequential(
                    nn.Linear(3 * hidden + 9, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, 1),
                )

            def encode_graph(self, adj_t: torch.Tensor, x_v: torch.Tensor) -> torch.Tensor:
                h1 = torch.relu(adj_t @ self.gcn1(x_v))
                h2 = torch.relu(adj_t @ self.gcn2(h1))
                return h2

            def score_candidates(
                self,
                h_nodes: torch.Tensor,
                temp_seq: torch.Tensor,
                u_idx: int,
                cand_v_idxs: list[int],
                cand_edge_feats: torch.Tensor,
                cand_kin_feats: torch.Tensor,
            ) -> torch.Tensor:
                _, h_t = self.temp_gru(temp_seq)  # [1, 1, H]
                ht = h_t[-1, 0]  # [H]
                hu = h_nodes[u_idx]
                k = len(cand_v_idxs)
                ht_rep = ht.unsqueeze(0).expand(k, -1)
                hu_rep = hu.unsqueeze(0).expand(k, -1)
                hv = h_nodes[cand_v_idxs]
                inp = torch.cat([ht_rep, hu_rep, hv, cand_edge_feats, cand_kin_feats], dim=-1)
                return self.edge_scorer(inp).squeeze(-1)

        model = _TGNNModule(h_dim)
        adj_t = torch.from_numpy(self._adj_norm)
        xv_t = torch.from_numpy(self._node_feats)

        # Build out-of-sample training samples directly from road graph physics & driver preferences
        samples: list[tuple[np.ndarray, int, list[int], np.ndarray, np.ndarray, int]] = []
        valid_edges = [e for e in self.network.edges.values() if len(self._outgoing.get(e.v, [])) >= 1]
        if valid_edges:
            for _ in range(320):
                prev_edge = valid_edges[int(rng.integers(0, len(valid_edges)))]
                cands = [
                    eid
                    for eid in self._outgoing.get(prev_edge.v, [])
                    if self.network.edges[eid].v != prev_edge.u or len(self._outgoing.get(prev_edge.v, [])) == 1
                ]
                if not cands:
                    continue
                v_mps = float(rng.uniform(4.0, 24.0))
                # Synthetic temporal sequence approaching node prev_edge.v
                seq = np.zeros((5, 5), dtype=np.float32)
                for step_i in range(5):
                    v_noise = float(np.clip(v_mps + rng.normal(0.0, 0.5), 1.0, 30.0))
                    th_noise = wrap_angle(prev_edge.heading_end_rad + float(rng.normal(0.0, 0.05)))
                    seq[step_i] = [
                        v_noise / 20.0,
                        math.cos(th_noise),
                        math.sin(th_noise),
                        float(rng.normal(0.0, 0.05)),
                        0.6 + 0.1 * step_i,
                    ]

                # Compute physics-consistent driver choice distribution for training supervision
                u_idx = self._node_to_idx[prev_edge.v]
                v_idxs: list[int] = []
                e_feats: list[np.ndarray] = []
                k_feats: list[np.ndarray] = []
                logits_gt: list[float] = []

                for eid in cands:
                    ne = self.network.edges[eid]
                    v_idxs.append(self._node_to_idx[ne.v])
                    e_feats.append(self._edge_feats[eid])
                    dth = wrap_angle(ne.heading_start_rad - prev_edge.heading_end_rad)
                    turn_deg = math.degrees(abs(dth))
                    r_eff = ne.min_curve_radius_m if ne.min_curve_radius_m < 1e4 else (9.0 if turn_deg >= 30.0 else 1e6)
                    v_crit = math.sqrt(0.7 * 9.81 * r_eff)
                    v_ratio = min(v_mps / max(v_crit, 1.0), 3.0)
                    k_feats.append(np.array([math.cos(dth), math.sin(dth), v_ratio], dtype=np.float32))
                    # Ground-truth stochastic driver preference respecting corner physics
                    if turn_deg >= 30.0 and v_mps > v_crit * 1.15:
                        score = -6.0
                    else:
                        score = (
                            0.5 * math.log(max(float(ne.lanes), 1.0))
                            + 0.4 * (ne.maxspeed_mps / 14.0)
                            - 0.85 * abs(dth)
                        )
                    logits_gt.append(score)

                probs = np.exp(np.array(logits_gt) - max(logits_gt))
                probs /= np.sum(probs)
                target_idx = int(rng.choice(len(cands), p=probs))
                samples.append(
                    (
                        seq,
                        u_idx,
                        v_idxs,
                        np.stack(e_feats, axis=0),
                        np.stack(k_feats, axis=0),
                        target_idx,
                    )
                )

        if samples:
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-4)
            ce_loss = nn.CrossEntropyLoss()
            for _epoch in range(12):
                h_nodes = model.encode_graph(adj_t, xv_t)
                batch_loss = torch.tensor(0.0)
                count = 0
                opt.zero_grad(set_to_none=True)
                for seq, u_idx, v_idxs, ef, kf, tgt in samples:
                    scores = model.score_candidates(
                        h_nodes,
                        torch.from_numpy(seq).unsqueeze(0),
                        u_idx,
                        v_idxs,
                        torch.from_numpy(ef),
                        torch.from_numpy(kf),
                    )
                    batch_loss = batch_loss + ce_loss(scores.unsqueeze(0), torch.tensor([tgt], dtype=torch.long))
                    count += 1
                    if count % 32 == 0:
                        (batch_loss / 32.0).backward()
                        opt.step()
                        opt.zero_grad(set_to_none=True)
                        h_nodes = model.encode_graph(adj_t, xv_t)
                        batch_loss = torch.tensor(0.0)
                if count % 32 != 0:
                    (batch_loss / float(count % 32)).backward()
                    opt.step()

        model.eval()
        with torch.no_grad():
            self._cached_h_nodes = model.encode_graph(adj_t, xv_t).detach()
        return model

    def predict_next_edge(
        self,
        current_edge_id: tuple[int, int, int],
        state_history: list[tuple[float, float, float]],
    ) -> list[tuple[tuple[int, int, int], float]]:
        """Predict probability distribution over outgoing edges at current_edge.v.

        Args:
            current_edge_id: Edge (u, v, key) the vehicle is exiting.
            state_history: List of recent (v_mps, theta_rad, progress_frac) tuples.
        """
        import torch

        if current_edge_id not in self.network.edges:
            return []
        curr_edge = self.network.edges[current_edge_id]
        cands = [
            eid
            for eid in self._outgoing.get(curr_edge.v, [])
            if self.network.edges[eid].v != curr_edge.u or len(self._outgoing.get(curr_edge.v, [])) == 1
        ]
        if not cands:
            return []

        hist = state_history[-5:] if state_history else [(10.0, curr_edge.heading_end_rad, 0.9)]
        while len(hist) < 5:
            hist = [hist[0], *hist]

        seq = np.zeros((5, 5), dtype=np.float32)
        for idx, (v_val, th_val, prog) in enumerate(hist):
            prev_th = hist[max(0, idx - 1)][1]
            dth_step = wrap_angle(th_val - prev_th)
            seq[idx] = [v_val / 20.0, math.cos(th_val), math.sin(th_val), dth_step, prog]

        v_now = float(hist[-1][0])
        u_idx = self._node_to_idx[curr_edge.v]
        v_idxs: list[int] = []
        e_feats: list[np.ndarray] = []
        k_feats: list[np.ndarray] = []
        for eid in cands:
            ne = self.network.edges[eid]
            v_idxs.append(self._node_to_idx[ne.v])
            e_feats.append(self._edge_feats[eid])
            dth = wrap_angle(ne.heading_start_rad - curr_edge.heading_end_rad)
            turn_deg = math.degrees(abs(dth))
            r_eff = ne.min_curve_radius_m if ne.min_curve_radius_m < 1e4 else (9.0 if turn_deg >= 30.0 else 1e6)
            v_crit = math.sqrt(0.7 * 9.81 * r_eff)
            v_ratio = min(v_now / max(v_crit, 1.0), 3.0)
            k_feats.append(np.array([math.cos(dth), math.sin(dth), v_ratio], dtype=np.float32))

        with torch.no_grad():
            scores = self._model.score_candidates(
                self._cached_h_nodes,
                torch.from_numpy(seq).unsqueeze(0),
                u_idx,
                v_idxs,
                torch.from_numpy(np.stack(e_feats, axis=0)),
                torch.from_numpy(np.stack(k_feats, axis=0)),
            )
            probs = torch.softmax(scores, dim=0).numpy()

        ranked = sorted(
            [(eid, float(p)) for eid, p in zip(cands, probs)],
            key=lambda item: -item[1],
        )
        return ranked

    def predict_blind_position(
        self,
        current_edge_id: tuple[int, int, int],
        s_m: float,
        state_history: list[tuple[float, float, float]],
        elapsed_s: float,
    ) -> TGNNRoutePrediction:
        """Predict most probable blind-zone position and edge distribution after elapsed_s."""
        from tracker.roadgraph import _interpolate_polyline

        curr_edge = self.network.edges[current_edge_id]
        v_now = float(state_history[-1][0]) if state_history else 10.0
        dist_rem = max(0.0, v_now * float(elapsed_s))
        rem_on_curr = max(0.0, curr_edge.length_m - float(s_m))

        ranked = self.predict_next_edge(current_edge_id, state_history)
        if dist_rem <= rem_on_curr or not ranked:
            px, py, _ = _interpolate_polyline(curr_edge.polyline_utm, min(curr_edge.length_m, s_m + dist_rem))
            return TGNNRoutePrediction(
                edge_probabilities=ranked,
                predicted_x_utm=px,
                predicted_y_utm=py,
                predicted_v_mps=v_now,
            )

        # Follow highest-probability predicted edge chain
        rem_dist = dist_rem - rem_on_curr
        edge_id = ranked[0][0]
        hist_cur = list(state_history)
        for _ in range(4):
            e_obj = self.network.edges[edge_id]
            if rem_dist <= e_obj.length_m:
                px, py, _ = _interpolate_polyline(e_obj.polyline_utm, rem_dist)
                return TGNNRoutePrediction(
                    edge_probabilities=ranked,
                    predicted_x_utm=px,
                    predicted_y_utm=py,
                    predicted_v_mps=min(v_now, e_obj.maxspeed_mps),
                )
            rem_dist -= e_obj.length_m
            hist_cur.append((min(v_now, e_obj.maxspeed_mps), e_obj.heading_end_rad, 1.0))
            nxt_ranked = self.predict_next_edge(edge_id, hist_cur)
            if not nxt_ranked:
                px, py, _ = _interpolate_polyline(e_obj.polyline_utm, e_obj.length_m)
                return TGNNRoutePrediction(
                    edge_probabilities=ranked,
                    predicted_x_utm=px,
                    predicted_y_utm=py,
                    predicted_v_mps=min(v_now, e_obj.maxspeed_mps),
                )
            edge_id = nxt_ranked[0][0]

        e_last = self.network.edges[edge_id]
        px, py, _ = _interpolate_polyline(e_last.polyline_utm, min(rem_dist, e_last.length_m))
        return TGNNRoutePrediction(
            edge_probabilities=ranked,
            predicted_x_utm=px,
            predicted_y_utm=py,
            predicted_v_mps=min(v_now, e_last.maxspeed_mps),
        )
