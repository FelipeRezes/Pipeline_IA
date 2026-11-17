"""Filtro de Kalman Estendido adaptativo (Secs. 2.3 e 2.4 do artigo).

Novidades em relação ao EKF canônico:
1. Jacobiana analítica F do modelo Two-Point-Bicycle (kinematics.bicycle_jacobian);
2. critério de ortogonalidade dos resíduos com pesos alpha=(0,1; 0,1; 2,8);
3. fator de desvanecimento escalar lambda >= 1 obtido por trace(N)/trace(M),
   que infla P durante manobras bruscas e devolve prioridade às medições.

Quando lambda = 1 o filtro se comporta exatamente como o EKF padrão.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Sequence

from .config import PipelineConfig
from .geometry import wrap_angle
from .kinematics import (STATE_DIM, bicycle_jacobian, bicycle_step,
                         process_noise, velocity_vector)

Array = np.ndarray

# linhas do estado observadas: x, y e (opcionalmente) theta
H2 = np.zeros((2, STATE_DIM)); H2[0, 0] = 1.0; H2[1, 1] = 1.0
H3 = np.zeros((3, STATE_DIM)); H3[0, 0] = 1.0; H3[1, 1] = 1.0; H3[2, 3] = 1.0


class AdaptiveEKF:
    """EKF com desvanecimento adaptativo para manobras evasivas."""

    def __init__(self, x0: Sequence[float], P0: Optional[Array] = None,
                 cfg: Optional[PipelineConfig] = None):
        self.cfg = cfg or PipelineConfig()
        self.x = np.asarray(x0, dtype=float).reshape(STATE_DIM).copy()
        self.P = (np.eye(STATE_DIM) * 2.0 if P0 is None
                  else np.asarray(P0, dtype=float).copy())
        self.V: Optional[Array] = None       # covariância empírica dos resíduos
        self.F: Optional[Array] = None
        self.Q: Optional[Array] = None
        self.P_prev: Optional[Array] = None  # P anterior à propagação
        self.dt_last: float = self.cfg.dt
        self.last_lambda: float = 1.0
        self.lambda_history: list[float] = []
        self.innovation_history: list[float] = []

    # ------------------------------------------------------------------ predição
    def predict(self, dt: Optional[float] = None, accel: float = 0.0,
                steer_rate: float = 0.0) -> Array:
        dt = float(dt or self.cfg.dt)
        self.dt_last = dt
        F = bicycle_jacobian(self.x, dt, self.cfg.wheelbase)
        Q = process_noise(dt, self.cfg)
        self.P_prev = self.P.copy()
        self.F, self.Q = F, Q
        self.x = bicycle_step(self.x, dt, self.cfg.wheelbase, accel, steer_rate)
        self.x[3] = wrap_angle(self.x[3])
        self.x[2] = max(0.0, self.x[2])  # velocidade escalar não negativa
        self.P = F @ self.P @ F.T + Q    # propagação nominal (sem desvanecimento)
        return self.x.copy()

    def propagate_only(self, dt: Optional[float] = None) -> Array:
        """Modo out-of-sight: apenas dead reckoning, sem inovação (Sec. 3.2)."""
        return self.predict(dt)

    # ----------------------------------------------------------------- atualização
    def update(self, z_position: Sequence[float], R_position: Array,
               heading: Optional[float] = None,
               sigma_heading: Optional[float] = None) -> Array:
        """Corrige o estado com medição no plano de solo e, opcionalmente, heading."""
        if self.F is None or self.Q is None or self.P_prev is None:
            raise RuntimeError("predict() deve ser chamado antes de update()")

        z_pos = np.asarray(z_position, dtype=float).reshape(2)
        R_pos = np.asarray(R_position, dtype=float).reshape(2, 2)
        if heading is None or abs(self.x[2]) < self.cfg.min_speed_for_heading:
            H, z, R = H2, z_pos, R_pos
        else:
            sh = float(sigma_heading or self.cfg.sigma_theta_meas)
            H, z = H3, np.array([z_pos[0], z_pos[1], wrap_angle(heading)])
            R = np.zeros((3, 3))
            R[:2, :2] = R_pos
            R[2, 2] = sh ** 2

        gamma = z - H @ self.x
        if len(gamma) == 3:
            gamma[2] = wrap_angle(gamma[2])
        self.innovation_history.append(float(np.linalg.norm(gamma[:2])))

        # ---- fator de desvanecimento (trace(N) / trace(M)) ---------------------
        outer = np.outer(gamma, gamma)
        if self.V is not None and self.V.shape != outer.shape:
            self.V = None  # dimensionalidade da medição mudou (heading entrou/saiu)
        self.V = (outer if self.V is None else
                  self.cfg.fading_forget * self.V +
                  (1.0 - self.cfg.fading_forget) * outer)
        alpha = np.asarray(self.cfg.alpha[:len(gamma)], dtype=float)
        alpha = alpha / alpha.sum() * 3.0
        W = np.diag(np.sqrt(alpha))

        N = self.V - H @ self.Q @ H.T - self.cfg.fading_beta * R
        M = H @ self.F @ self.P_prev @ self.F.T @ H.T
        denom = float(np.trace(W @ M @ W))
        lam = 1.0 if denom <= 1e-12 else float(np.trace(W @ N @ W)) / denom
        lam = float(np.clip(max(self.cfg.min_lambda, lam),
                            self.cfg.min_lambda, self.cfg.lambda_max))
        self.last_lambda = lam
        self.lambda_history.append(lam)

        # ---- ganho e correção com P inflado -----------------------------------
        P_pred = lam * self.F @ self.P_prev @ self.F.T + self.Q
        S = H @ P_pred @ H.T + R
        S = 0.5 * (S + S.T)
        S_inv = np.linalg.pinv(S + np.eye(S.shape[0]) * 1e-9)
        K = P_pred @ H.T @ S_inv

        self.x = self.x + K @ gamma
        self.x[2] = max(0.0, self.x[2])
        self.x[3] = wrap_angle(self.x[3])
        self.x[4] = float(np.clip(self.x[4],
                                  -np.deg2rad(self.cfg.max_steer_deg),
                                  np.deg2rad(self.cfg.max_steer_deg)))

        I = np.eye(STATE_DIM)
        IKH = I - K @ H
        P_new = IKH @ P_pred @ IKH.T + K @ R @ K.T
        self.P = 0.5 * (P_new + P_new.T)
        self.V = None if np.allclose(gamma, 0.0) else self.V
        return self.x.copy()

    # ------------------------------------------------------------------ utilidades
    @property
    def velocity(self) -> Array:
        return velocity_vector(self.x)

    @property
    def position(self) -> Array:
        return self.x[:2].copy()

    @property
    def heading(self) -> float:
        return wrap_angle(self.x[3])

    def uncertainty_ellipse(self) -> tuple[float, float, float]:
        """(semi-eixo maior, semi-eixo menor, ângulo) da covariância de posição."""
        vals, vecs = np.linalg.eigh(self.P[:2, :2])
        idx = np.argsort(vals)[::-1]
        vals, vecs = vals[idx], vecs[:, idx]
        return (float(np.sqrt(max(vals[0], 0.0))),
                float(np.sqrt(max(vals[1], 0.0))),
                float(np.arctan2(vecs[1, 0], vecs[0, 0])))

    def copy(self) -> "AdaptiveEKF":
        out = AdaptiveEKF(self.x, self.P, self.cfg)
        out.V = None if self.V is None else self.V.copy()
        out.last_lambda = self.last_lambda
        return out
