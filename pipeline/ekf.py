"""Laboratório de Filtragem Estocástica: EKF Adaptativo, Canônico, CTRV, CV Linear e IMM.

Implementa as Seções 2.1, 2.3 e 2.4 do artigo + modelos comparativos para os
Experimentos Científicos de Ablação (Testes A1 e A2):

1. AdaptiveEKF (Lou et al., 2002 - Seção 2.4):
   - Modelo Two-Point-Bicycle 5D: X = [x, y, v, theta, phi]^T;
   - Critério de ortogonalidade dos resíduos com pesos alpha = (0.1, 0.1, 2.8);
   - Fator de desvanecimento escalar lambda_k = trace(W N_k W) / trace(W M_k W) >= 1;
   - Proteção contra observabilidade fraca de phi e gate contra jitter espúrio em theta.
2. CanonicalBicycleEKF:
   - Mesma dinâmica Two-Point-Bicycle, mas com lambda_k = 1 fixo (EKF tradicional).
3. CoordinatedTurnEKF (Bradler et al., 2021 - Seção 2.2):
   - EKF com modelo CTRV [x, y, v, psi, omega]^T e arco exato R = v / omega.
4. LinearKalmanCV (Kalman, 1960 - Seção 2.1):
   - Filtro de Kalman puramente linear de Velocidade Constante [x, y, vx, vy]^T
     exposto na mesma interface 5D para comparação direta.
5. IMMFilter (Interacting Multiple Model):
   - Combina probabilisticamente um filtro Linear CV (movimento reto uniforme) e
     um AdaptiveEKF / CTRV (manobras evasivas e curvas acentuadas).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .geometry import angle_diff, wrap_angle
from .kinematics import (
    STATE_DIM,
    bicycle_jacobian,
    bicycle_step,
    coordinated_turn_jacobian,
    coordinated_turn_step,
    process_noise,
    velocity_vector,
)

Array = np.ndarray

# Matrizes de observação: apenas posição (x, y) ou posição + heading (x, y, theta)
H2 = np.zeros((2, STATE_DIM), dtype=float)
H2[0, 0] = 1.0
H2[1, 1] = 1.0

H3 = np.zeros((3, STATE_DIM), dtype=float)
H3[0, 0] = 1.0
H3[1, 1] = 1.0
H3[2, 3] = 1.0


def wrap_innovation(gamma: Array) -> Array:
    out = gamma.copy()
    if len(out) >= 3:
        out[2] = wrap_angle(out[2])
    return out


class AdaptiveEKF:
    """Filtro de Kalman Estendido Adaptativo com modelo Two-Point-Bicycle (Lou et al., 2002)."""

    def __init__(self,
                 x0: Sequence[float],
                 P0: Optional[Array] = None,
                 cfg: Optional[PipelineConfig] = None,
                 adaptive: Optional[bool] = None):
        self.cfg = cfg or PipelineConfig()
        self.adaptive = self.cfg.adaptive_fading if adaptive is None else bool(adaptive)
        self.x = np.asarray(x0, dtype=float).reshape(STATE_DIM).copy()
        self.P = (
            np.diag([2.5, 2.5, 6.0, 0.6, 0.15]).astype(float)
            if P0 is None
            else np.asarray(P0, dtype=float).reshape(STATE_DIM, STATE_DIM).copy()
        )
        self.V: Optional[Array] = None
        self.F: Optional[Array] = None
        self.Q: Optional[Array] = None
        self.P_prev: Optional[Array] = None
        self.x_prev: Array = self.x.copy()
        self.dt_last: float = self.cfg.dt
        self.last_lambda: float = 1.0
        self.lambda_history: list[float] = []
        self.innovation_history: list[float] = []

    def predict(self,
                dt: Optional[float] = None,
                accel: float = 0.0,
                steer_rate: float = 0.0) -> Array:
        dt_val = float(dt or self.cfg.dt)
        self.dt_last = dt_val
        max_steer_rad = math.radians(self.cfg.max_steer_deg)
        damping = self.cfg.steer_damping

        F = bicycle_jacobian(self.x, dt_val, self.cfg.wheelbase, steer_damping=damping)
        Q = process_noise(dt_val, self.cfg)
        self.P_prev = self.P.copy()
        self.F, self.Q = F, Q
        self.x_prev = self.x.copy()

        self.x = bicycle_step(
            self.x,
            dt_val,
            self.cfg.wheelbase,
            accel=accel,
            steer_rate=steer_rate,
            steer_damping=damping,
            max_steer_rad=max_steer_rad,
        )
        self.x[2] = float(np.clip(self.x[2], 0.0, self.cfg.max_speed_mps))
        self.x[3] = wrap_angle(self.x[3])
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self._updated_since_predict = False
        return self.x.copy()

    def propagate_only(self, dt: Optional[float] = None) -> Array:
        """Modo Out-of-Sight (Seção 3.2): dead reckoning cinemático sem observações."""
        return self.predict(dt)

    def update(self,
               z_position: Sequence[float],
               R_position: Array,
               heading: Optional[float] = None,
               sigma_heading: Optional[float] = None) -> Array:
        """Atualiza o estado a posteriori com fator de desvanecimento lambda_k (Seção 2.4)."""
        if self.F is None or self.Q is None or self.P_prev is None:
            self.predict(self.cfg.dt)

        z_pos = np.asarray(z_position, dtype=float).reshape(2)
        R_pos = np.asarray(R_position, dtype=float).reshape(2, 2)

        # Gate de confiabilidade para o canal de orientação theta:
        use_heading = (
            heading is not None
            and abs(self.x[2]) >= self.cfg.min_speed_for_heading
        )
        if use_heading:
            d_th = abs(angle_diff(float(heading), self.x[3]))
            if d_th > self.cfg.heading_gate_rad and len(self.innovation_history) > 3:
                sh = float(sigma_heading or self.cfg.sigma_theta_meas) * 2.5
            else:
                sh = float(sigma_heading or self.cfg.sigma_theta_meas)
            H = H3
            z = np.array([z_pos[0], z_pos[1], wrap_angle(float(heading))], dtype=float)
            R = np.zeros((3, 3), dtype=float)
            R[:2, :2] = R_pos
            R[2, 2] = sh * sh
        else:
            H, z, R = H2, z_pos, R_pos

        gamma = z - H @ self.x
        if len(gamma) == 3:
            gamma[2] = wrap_angle(gamma[2])
        self.innovation_history.append(float(np.linalg.norm(gamma[:2])))

        # ---- Fator de Desvanecimento Adaptativo (Lou et al., 2002) ----------
        outer = np.outer(gamma, gamma)
        if self.V is not None and self.V.shape != outer.shape:
            self.V = None
        self.V = (
            outer
            if self.V is None
            else self.cfg.fading_forget * self.V + (1.0 - self.cfg.fading_forget) * outer
        )

        if self.adaptive:
            alpha = np.asarray(self.cfg.alpha[: len(gamma)], dtype=float)
            alpha = alpha / alpha.sum() * float(len(gamma))
            W = np.diag(np.sqrt(alpha))

            N = self.V - H @ self.Q @ H.T - self.cfg.fading_beta * R
            M = H @ self.F @ self.P @ self.F.T @ H.T
            denom = float(np.trace(W @ M @ W))
            lam = 1.0 if denom <= 1e-12 else float(np.trace(W @ N @ W)) / denom
            lam = float(np.clip(max(self.cfg.min_lambda, lam),
                                self.cfg.min_lambda,
                                self.cfg.lambda_max))
        else:
            lam = 1.0

        self.last_lambda = lam
        self.lambda_history.append(lam)

        # ---- Atualização de Kalman (Forma de Joseph para estabilidade SPD) ---
        # Usa self.P acumulado (incluindo todos os passos propagate_only da zona cega!)
        P_pred = lam * (self.P - self.Q) + self.Q
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        S_inv = np.linalg.pinv(S + np.eye(S.shape[0]) * 1e-9)
        K = P_pred @ H.T @ S_inv

        dx = K @ gamma
        self.x = self.x + dx
        self.x[2] = float(np.clip(self.x[2], 0.0, self.cfg.max_speed_mps))
        self.x[3] = wrap_angle(self.x[3])
        max_steer_rad = math.radians(self.cfg.max_steer_deg)

        if len(gamma) == 3 and self.x[2] > 1.5 and self.dt_last <= 0.5:
            # Em observações visuais contínuas (ex: 10-30 fps), apenas amortecemos suavemente
            # o esterçamento phi dentro de [-0.25, 0.25] rad (~14 graus) para evitar chicoteamento
            # por ruído de diferenciação finita de heading
            dtheta = angle_diff(self.x[3], self.x_prev[3])
            if abs(dtheta) < 0.35:
                omega_est = dtheta / max(self.dt_last, 0.05)
                phi_ack = math.atan2(self.cfg.wheelbase * omega_est, max(self.x[2], 3.0))
                phi_ack = float(np.clip(phi_ack, -0.22, 0.22))
                self.x[4] = 0.85 * self.x[4] + 0.15 * phi_ack

        self.x[4] = float(np.clip(self.x[4], -max_steer_rad, max_steer_rad))

        I = np.eye(STATE_DIM)
        IKH = I - K @ H
        P_new = IKH @ P_pred @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P_new + P_new.T)
        if np.allclose(gamma, 0.0):
            self.V = None
        return self.x.copy()

    @property
    def velocity(self) -> Array:
        return velocity_vector(self.x, is_ctrv=False)

    @property
    def position(self) -> Array:
        return self.x[:2].copy()

    @property
    def heading(self) -> float:
        return wrap_angle(self.x[3])

    def uncertainty_ellipse(self) -> Tuple[float, float, float]:
        vals, vecs = np.linalg.eigh(self.P[:2, :2])
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        return (
            float(np.sqrt(max(vals[0], 0.0))),
            float(np.sqrt(max(vals[1], 0.0))),
            float(np.arctan2(vecs[1, 0], vecs[0, 0])),
        )

    def copy(self) -> "AdaptiveEKF":
        out = AdaptiveEKF(self.x, self.P, self.cfg, adaptive=self.adaptive)
        out.V = None if self.V is None else self.V.copy()
        out.last_lambda = self.last_lambda
        out.lambda_history = list(self.lambda_history)
        out.innovation_history = list(self.innovation_history)
        return out


class CoordinatedTurnEKF(AdaptiveEKF):
    """Filtro EKF baseado no modelo de Curva Coordenada (CTRV - Bradler et al., 2021).

    Estado 5D: X = [x, y, v, psi, omega]^T.
    """

    def predict(self,
                dt: Optional[float] = None,
                accel: float = 0.0,
                steer_rate: float = 0.0) -> Array:
        dt_val = float(dt or self.cfg.dt)
        self.dt_last = dt_val
        F = coordinated_turn_jacobian(self.x, dt_val)
        Q = process_noise(dt_val, self.cfg)
        Q[4, 4] = (self.cfg.sigma_yaw_rate * dt_val) ** 2 + 1e-5
        self.P_prev = self.P.copy()
        self.F, self.Q = F, Q
        self.x_prev = self.x.copy()

        self.x = coordinated_turn_step(self.x, dt_val)
        self.x[2] = float(np.clip(self.x[2] + accel * dt_val, 0.0, self.cfg.max_speed_mps))
        self.x[3] = wrap_angle(self.x[3])
        self.x[4] = float(np.clip(self.x[4], -1.8, 1.8))
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        return self.x.copy()

    @property
    def velocity(self) -> Array:
        return velocity_vector(self.x, is_ctrv=True)

    def copy(self) -> "CoordinatedTurnEKF":
        out = CoordinatedTurnEKF(self.x, self.P, self.cfg, adaptive=self.adaptive)
        out.V = None if self.V is None else self.V.copy()
        out.last_lambda = self.last_lambda
        return out


class LinearKalmanCV:
    """Filtro de Kalman Linear Convencional de Velocidade Constante (Seção 2.1).

    Opera internamente no espaço linear cartesiano 4D: X_lin = [x, y, vx, vy]^T,
    mas expõe `self.x` como [x, y, v, theta, 0] para compatibilidade transparente
    com a pipeline e com os experimentos comparativos (Teste A1).
    """

    def __init__(self,
                 x0: Sequence[float],
                 P0: Optional[Array] = None,
                 cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        x0_arr = np.asarray(x0, dtype=float).reshape(STATE_DIM)
        v0, th0 = float(x0_arr[2]), float(x0_arr[3])
        self._x4 = np.array([
            x0_arr[0],
            x0_arr[1],
            v0 * math.cos(th0),
            v0 * math.sin(th0),
        ], dtype=float)
        self._P4 = np.diag([4.0, 4.0, 9.0, 9.0]).astype(float)
        self.x = x0_arr.copy()
        self.P = (
            np.diag([4.0, 4.0, 9.0, 1.0, 0.1]).astype(float)
            if P0 is None
            else np.asarray(P0, dtype=float).copy()
        )
        self.last_lambda: float = 1.0
        self.lambda_history: list[float] = []
        self.innovation_history: list[float] = []
        self._sync_to_5d()

    def _sync_to_5d(self) -> None:
        vx, vy = float(self._x4[2]), float(self._x4[3])
        v = math.hypot(vx, vy)
        theta = math.atan2(vy, vx) if v > 1e-3 else float(self.x[3])
        self.x = np.array([self._x4[0], self._x4[1], v, wrap_angle(theta), 0.0], dtype=float)
        self.P[:2, :2] = self._P4[:2, :2]

    def _sync_from_5d(self) -> None:
        v, theta = float(self.x[2]), float(self.x[3])
        self._x4[0] = self.x[0]
        self._x4[1] = self.x[1]
        self._x4[2] = v * math.cos(theta)
        self._x4[3] = v * math.sin(theta)
        self._P4[:2, :2] = self.P[:2, :2]

    def predict(self, dt: Optional[float] = None, **kwargs) -> Array:
        self._sync_from_5d()
        dt_val = float(dt or self.cfg.dt)
        F4 = np.array([
            [1.0, 0.0, dt_val, 0.0],
            [0.0, 1.0, 0.0, dt_val],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=float)
        # No modelo linear CV tradicional (Seção 2.1), o ruído de processo Q é estático e baixo
        q_pos = (0.15 * (dt_val ** 2)) ** 2 + 1e-5
        q_vel = (0.35 * dt_val) ** 2 + 1e-5
        Q4 = np.diag([q_pos, q_pos, q_vel, q_vel])
        self._x4 = F4 @ self._x4
        self._P4 = F4 @ self._P4 @ F4.T + Q4
        self._sync_to_5d()
        return self.x.copy()

    def propagate_only(self, dt: Optional[float] = None) -> Array:
        return self.predict(dt)

    def update(self,
               z_position: Sequence[float],
               R_position: Array,
               heading: Optional[float] = None,
               sigma_heading: Optional[float] = None) -> Array:
        z_pos = np.asarray(z_position, dtype=float).reshape(2)
        R_pos = np.asarray(R_position, dtype=float).reshape(2, 2)
        H4 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=float)
        gamma = z_pos - H4 @ self._x4
        self.innovation_history.append(float(np.linalg.norm(gamma)))
        self.lambda_history.append(1.0)

        S = H4 @ self._P4 @ H4.T + R_pos
        K = self._P4 @ H4.T @ np.linalg.pinv(S + np.eye(2) * 1e-9)
        self._x4 = self._x4 + K @ gamma
        I4 = np.eye(4)
        IKH = I4 - K @ H4
        self._P4 = IKH @ self._P4 @ IKH.T + K @ R_pos @ K.T
        self._P4 = 0.5 * (self._P4 + self._P4.T)
        self._sync_to_5d()
        return self.x.copy()

    @property
    def velocity(self) -> Array:
        return np.array([self._x4[2], self._x4[3]], dtype=float)

    @property
    def position(self) -> Array:
        return self._x4[:2].copy()

    @property
    def heading(self) -> float:
        return wrap_angle(self.x[3])

    def uncertainty_ellipse(self) -> Tuple[float, float, float]:
        vals, vecs = np.linalg.eigh(self.P[:2, :2])
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        return (
            float(np.sqrt(max(vals[0], 0.0))),
            float(np.sqrt(max(vals[1], 0.0))),
            float(np.arctan2(vecs[1, 0], vecs[0, 0])),
        )

    def copy(self) -> "LinearKalmanCV":
        out = LinearKalmanCV(self.x, self.P, self.cfg)
        out._x4 = self._x4.copy()
        out._P4 = self._P4.copy()
        return out


class IMMFilter:
    """Interacting Multiple Model (IMM) combinando CV Linear + Two-Point-Bicycle Adaptativo.

    Alterna suavemente entre regime retilíneo de baixa variância e regime de manobra
    não linear evasiva com base na verossimilhança da inovação a cada quadro.
    """

    def __init__(self,
                 x0: Sequence[float],
                 P0: Optional[Array] = None,
                 cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.m_bicycle = AdaptiveEKF(x0, P0, self.cfg, adaptive=True)
        self.m_ctrv = CoordinatedTurnEKF(x0, P0, self.cfg, adaptive=True)
        self.mu = np.array([0.55, 0.45], dtype=float)  # Probabilidades dos modos
        self.PI = np.array([[0.92, 0.08], [0.08, 0.92]], dtype=float)
        self.x = self.m_bicycle.x.copy()
        self.P = self.m_bicycle.P.copy()
        self.last_lambda: float = 1.0
        self.lambda_history: list[float] = []
        self.innovation_history: list[float] = []

    def _combine(self) -> None:
        w0, w1 = float(self.mu[0]), float(self.mu[1])
        x0, x1 = self.m_bicycle.x, self.m_ctrv.x
        pos = w0 * x0[:2] + w1 * x1[:2]
        v = w0 * x0[2] + w1 * x1[2]
        # Média circular para theta
        s_th = w0 * math.sin(x0[3]) + w1 * math.sin(x1[3])
        c_th = w0 * math.cos(x0[3]) + w1 * math.cos(x1[3])
        theta = math.atan2(s_th, c_th)
        phi = x0[4]
        self.x = np.array([pos[0], pos[1], v, theta, phi], dtype=float)
        self.P = w0 * self.m_bicycle.P + w1 * self.m_ctrv.P

    def predict(self, dt: Optional[float] = None, **kwargs) -> Array:
        # Interação (Mixing)
        c_bar = self.PI.T @ self.mu
        c_bar = np.clip(c_bar, 1e-6, 1.0)
        c_bar /= c_bar.sum()
        self.mu = c_bar

        self.m_bicycle.predict(dt)
        self.m_ctrv.predict(dt)
        self._combine()
        return self.x.copy()

    def propagate_only(self, dt: Optional[float] = None) -> Array:
        return self.predict(dt)

    def update(self,
               z_position: Sequence[float],
               R_position: Array,
               heading: Optional[float] = None,
               sigma_heading: Optional[float] = None) -> Array:
        z_pos = np.asarray(z_position, dtype=float).reshape(2)
        e0 = float(np.linalg.norm(z_pos - self.m_bicycle.position))
        e1 = float(np.linalg.norm(z_pos - self.m_ctrv.position))

        self.m_bicycle.update(z_position, R_position, heading, sigma_heading)
        self.m_ctrv.update(z_position, R_position, heading, sigma_heading)

        sig2 = max(1.0, float(np.trace(R_position)))
        L0 = math.exp(-0.5 * (e0 * e0) / sig2) + 1e-9
        L1 = math.exp(-0.5 * (e1 * e1) / sig2) + 1e-9
        self.mu = self.mu * np.array([L0, L1])
        self.mu /= self.mu.sum()

        self._combine()
        self.last_lambda = float(
            self.mu[0] * self.m_bicycle.last_lambda + self.mu[1] * self.m_ctrv.last_lambda
        )
        self.lambda_history.append(self.last_lambda)
        self.innovation_history.append(min(e0, e1))
        return self.x.copy()

    @property
    def velocity(self) -> Array:
        return self.mu[0] * self.m_bicycle.velocity + self.mu[1] * self.m_ctrv.velocity

    @property
    def position(self) -> Array:
        return self.x[:2].copy()

    @property
    def heading(self) -> float:
        return wrap_angle(self.x[3])

    def uncertainty_ellipse(self) -> Tuple[float, float, float]:
        return self.m_bicycle.uncertainty_ellipse()

    def copy(self) -> "IMMFilter":
        out = IMMFilter(self.x, self.P, self.cfg)
        out.m_bicycle = self.m_bicycle.copy()
        out.m_ctrv = self.m_ctrv.copy()
        out.mu = self.mu.copy()
        return out


def create_filter(x0: Sequence[float],
                  P0: Optional[Array] = None,
                  cfg: Optional[PipelineConfig] = None):
    """Fábrica polimórfica de filtros cinemáticos segundo `cfg.filter_model`."""
    cfg = cfg or PipelineConfig()
    model = cfg.filter_model.lower()
    if model == "adaptive_bicycle":
        return AdaptiveEKF(x0, P0, cfg, adaptive=True)
    if model == "canonical_bicycle":
        return AdaptiveEKF(x0, P0, cfg, adaptive=False)
    if model == "ctrv":
        return CoordinatedTurnEKF(x0, P0, cfg, adaptive=True)
    if model == "linear_cv":
        return LinearKalmanCV(x0, P0, cfg)
    if model == "imm":
        return IMMFilter(x0, P0, cfg)
    raise ValueError(f"Modelo de filtro desconhecido: {cfg.filter_model}")
