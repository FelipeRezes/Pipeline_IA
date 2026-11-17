"""Cinemática não linear veicular (Sec. 2.2 do artigo).

Estado: X = [x, y, v, theta, phi]^T, com:
    x, y : centro do veículo no plano de solo (m)
    v    : velocidade escalar do eixo traseiro (m/s)
    theta: orientação do chassi (rad)
    phi  : esterçamento das rodas dianteiras (rad)

Dinâmica contínua:
    dx/dt   = v cos(theta + phi/2)
    dy/dt   = v sin(theta + phi/2)
    dv/dt   = a(t)                     (aceleração longitudinal)
    dtheta  = (v / l) tan(phi)
    dphi/dt = b(t)                     (taxa de esterçamento do volante)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .config import PipelineConfig
from .geometry import wrap_angle

Array = np.ndarray
STATE_DIM = 5


def bicycle_step(x: Array, dt: float, wheelbase: float,
                 accel: float = 0.0, steer_rate: float = 0.0) -> Array:
    """Propagação discreta do modelo Two-Point-Bicycle (Euler sem implícitos)."""
    x = np.asarray(x, dtype=float).reshape(STATE_DIM)
    px, py, v, theta, phi = x
    alpha = theta + 0.5 * phi
    return np.array([
        px + v * dt * np.cos(alpha),
        py + v * dt * np.sin(alpha),
        v + accel * dt,
        wrap_angle(theta + (v / wheelbase) * np.tan(phi) * dt),
        phi + steer_rate * dt,
    ])


def bicycle_jacobian(x: Array, dt: float, wheelbase: float) -> Array:
    """Jacobiana analítica F_k = d f_d / d X no estado x (matriz 5x5)."""
    x = np.asarray(x, dtype=float).reshape(STATE_DIM)
    _, _, v, theta, phi = x
    alpha = theta + 0.5 * phi
    c, s = np.cos(alpha), np.sin(alpha)
    F = np.eye(STATE_DIM)
    F[0, 2] = dt * c
    F[0, 3] = -v * dt * s
    F[0, 4] = -0.5 * v * dt * s
    F[1, 2] = dt * s
    F[1, 3] = v * dt * c
    F[1, 4] = 0.5 * v * dt * c
    F[3, 2] = (dt / wheelbase) * np.tan(phi)
    sec2 = 1.0 / max(np.cos(phi) ** 2, 1e-6)
    F[3, 4] = (v * dt / wheelbase) * sec2
    return F


def process_noise(dt: float, cfg: PipelineConfig) -> Array:
    """Covariância de processo Q (ruído branco em a e em b, mais deriva de heading)."""
    Q = np.zeros((STATE_DIM, STATE_DIM))
    Q[0, 0] = Q[1, 1] = (0.15 * cfg.sigma_accel * dt ** 2) ** 2 + 1e-6
    Q[2, 2] = (cfg.sigma_accel * dt) ** 2
    Q[3, 3] = (cfg.sigma_steer_rate * dt) ** 2 + (cfg.sigma_accel * dt / cfg.wheelbase) ** 2
    Q[4, 4] = (cfg.sigma_steer_rate * dt) ** 2
    return Q + np.eye(STATE_DIM) * 1e-8


def coordinated_turn_step(x: Array, dt: float) -> Array:
    """Modelo de curva coordenada (Sec. 2.2), para comparação/telemetria.

    Estado alternativo [x, y, v, psi, omega]; arco de raio R = v / omega.
    """
    x = np.asarray(x, dtype=float).reshape(STATE_DIM)
    px, py, v, psi, omega = x
    if abs(omega) < 1e-5:
        return np.array([px + v * dt * np.cos(psi),
                         py + v * dt * np.sin(psi), v, psi, omega])
    R = v / omega
    return np.array([
        px + R * (np.sin(psi + omega * dt) - np.sin(psi)),
        py + R * (np.cos(psi) - np.cos(psi + omega * dt)),
        v, wrap_angle(psi + omega * dt), omega,
    ])


def velocity_vector(x: Array) -> Array:
    """Vetor [vx, vy] no plano de solo derivado do estado do EKF."""
    x = np.asarray(x, dtype=float).reshape(STATE_DIM)
    alpha = x[3] + 0.5 * x[4]
    return np.array([x[2] * np.cos(alpha), x[2] * np.sin(alpha)])
