"""Modelos cinemáticos veiculares lineares e não lineares (Seção 2.2 do artigo).

Implementa 3 famílias de equações de transição de estado e suas Jacobianas analíticas:
1. Two-Point-Bicycle Model (Lou et al., 2002 - Seção 2.2):
   Estado 5D: X = [x, y, v, theta, phi]^T
     dx/dt    = v * cos(theta + phi/2)
     dy/dt    = v * sin(theta + phi/2)
     dv/dt    = a(t)
     dtheta/dt= (v / l) * tan(phi)
     dphi/dt  = b(t)

2. Coordinated Turn Polar Velocity Model - CTRV (Bradler et al., 2021 - Seção 2.2):
   Estado 5D: X = [x, y, v, psi, omega]^T
     Propaga exatamente ao longo de um arco circular de raio R = v / omega.

3. Constant Velocity Linear Model - CV (Kalman, 1960 - Seção 2.1):
   Estado 4D (ou embutido em 5D para compatibilidade de interface):
   Representa o baseline linear que colapsa em manobras evasivas e curvas acentuadas.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from .config import PipelineConfig
from .geometry import wrap_angle

Array = np.ndarray
STATE_DIM = 5


# ==============================================================================
# 1. TWO-POINT-BICYCLE MODEL (Lou et al., 2002)
# ==============================================================================
def bicycle_step(x: Array,
                 dt: float,
                 wheelbase: float,
                 accel: float = 0.0,
                 steer_rate: float = 0.0,
                 steer_damping: float = 1.0,
                 max_steer_rad: float = 0.610865) -> Array:
    """Propagação discreta do modelo Two-Point-Bicycle com integração de ponto médio (RK2).

    Nota: Mantém compatibilidade de 1ª ordem com Euler quando steer_damping=1.0,
    mas clampa phi dentro do envelope mecânico [-max_steer_rad, max_steer_rad].
    """
    x_arr = np.asarray(x, dtype=float).reshape(STATE_DIM)
    px, py, v, theta, phi = x_arr
    phi_clamped = float(np.clip(phi, -max_steer_rad, max_steer_rad))
    alpha = theta + 0.5 * phi_clamped

    nx = px + v * dt * math.cos(alpha)
    ny = py + v * dt * math.sin(alpha)
    nv = v + accel * dt
    ntheta = wrap_angle(theta + (v / max(wheelbase, 0.5)) * math.tan(phi_clamped) * dt)
    nphi = float(np.clip(phi_clamped * steer_damping + steer_rate * dt,
                         -max_steer_rad, max_steer_rad))
    return np.array([nx, ny, nv, ntheta, nphi], dtype=float)


def bicycle_jacobian(x: Array,
                     dt: float,
                     wheelbase: float,
                     steer_damping: float = 1.0) -> Array:
    """Jacobiana analítica F_k = df / dX (5x5) do modelo Two-Point-Bicycle."""
    x_arr = np.asarray(x, dtype=float).reshape(STATE_DIM)
    _, _, v, theta, phi = x_arr
    alpha = theta + 0.5 * phi
    c, s = math.cos(alpha), math.sin(alpha)
    l = max(float(wheelbase), 0.5)
    cos_phi = math.cos(phi)
    sec2 = 1.0 / max(cos_phi * cos_phi, 1e-6)

    F = np.eye(STATE_DIM, dtype=float)
    F[0, 2] = dt * c
    F[0, 3] = -v * dt * s
    F[0, 4] = -0.5 * v * dt * s

    F[1, 2] = dt * s
    F[1, 3] = v * dt * c
    F[1, 4] = 0.5 * v * dt * c

    F[3, 2] = (dt / l) * math.tan(phi)
    F[3, 4] = (v * dt / l) * sec2
    F[4, 4] = float(steer_damping)
    return F


def process_noise(dt: float, cfg: PipelineConfig) -> Array:
    """Matriz de covariância de ruído de processo Q (5x5) projetada pelas entradas [a(t), b(t)].

    Mapeia as perturbações estocásticas do condutor (aceleração longitudinal sigma_accel
    e velocidade angular do volante sigma_steer_rate) para o espaço de estado 5D.
    """
    dt = float(dt)
    sa = float(cfg.sigma_accel)
    sb = float(cfg.sigma_steer_rate)
    l = max(float(cfg.wheelbase), 0.5)

    Q = np.zeros((STATE_DIM, STATE_DIM), dtype=float)
    Q[0, 0] = Q[1, 1] = (0.15 * sa * (dt ** 2)) ** 2 + 1e-6
    Q[2, 2] = (sa * dt) ** 2 + 1e-6
    Q[3, 3] = (0.35 * sb * dt) ** 2 + (0.25 * sa * dt / l) ** 2 + 1e-6
    Q[4, 4] = (0.25 * sb * dt) ** 2 + 1e-6
    return Q


# ==============================================================================
# 2. COORDINATED TURN MODEL - CTRV (Bradler et al., 2021)
# ==============================================================================
def coordinated_turn_step(x: Array, dt: float, omega_damping: float = 0.95) -> Array:
    """Transição exata por arco circular do modelo CTRV: X = [x, y, v, psi, omega]^T."""
    x_arr = np.asarray(x, dtype=float).reshape(STATE_DIM)
    px, py, v, psi, omega = x_arr
    if abs(omega) < 1e-4:
        return np.array([
            px + v * dt * math.cos(psi),
            py + v * dt * math.sin(psi),
            v,
            wrap_angle(psi + omega * dt),
            omega * omega_damping,
        ], dtype=float)
    R = v / omega
    psi_new = psi + omega * dt
    return np.array([
        px + R * (math.sin(psi_new) - math.sin(psi)),
        py + R * (math.cos(psi) - math.cos(psi_new)),
        v,
        wrap_angle(psi_new),
        omega * omega_damping,
    ], dtype=float)


def coordinated_turn_jacobian(x: Array, dt: float, omega_damping: float = 0.95) -> Array:
    """Jacobiana analítica 5x5 do modelo CTRV."""
    x_arr = np.asarray(x, dtype=float).reshape(STATE_DIM)
    _, _, v, psi, omega = x_arr
    F = np.eye(STATE_DIM, dtype=float)
    F[4, 4] = omega_damping

    if abs(omega) < 1e-4:
        c, s = math.cos(psi), math.sin(psi)
        F[0, 2] = dt * c
        F[0, 3] = -v * dt * s
        F[0, 4] = -0.5 * v * (dt ** 2) * s
        F[1, 2] = dt * s
        F[1, 3] = v * dt * c
        F[1, 4] = 0.5 * v * (dt ** 2) * c
        F[3, 4] = dt
        return F

    psi_new = psi + omega * dt
    s_psi, c_psi = math.sin(psi), math.cos(psi)
    s_new, c_new = math.sin(psi_new), math.cos(psi_new)

    F[0, 2] = (s_new - s_psi) / omega
    F[0, 3] = (v / omega) * (c_new - c_psi)
    F[0, 4] = (v / omega) * dt * c_new - (v / (omega * omega)) * (s_new - s_psi)

    F[1, 2] = (c_psi - c_new) / omega
    F[1, 3] = (v / omega) * (s_new - s_psi)
    F[1, 4] = (v / omega) * dt * s_new - (v / (omega * omega)) * (c_psi - c_new)

    F[3, 4] = dt
    return F


# ==============================================================================
# 3. CONSTANT VELOCITY LINEAR MODEL - CV (Baseline Linear - Seção 2.1)
# ==============================================================================
def linear_cv_step(x_cv: Array, dt: float) -> Array:
    """Propagação puramente linear de velocidade constante: X_cv = [x, y, vx, vy]^T."""
    x_arr = np.asarray(x_cv, dtype=float).reshape(4)
    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)
    return F @ x_arr


def velocity_vector(x: Array, is_ctrv: bool = False) -> Array:
    """Retorna o vetor velocidade cartesiana [vx, vy] no plano de solo."""
    x_arr = np.asarray(x, dtype=float).reshape(STATE_DIM)
    v = float(x_arr[2])
    theta = float(x_arr[3])
    alpha = theta if is_ctrv else (theta + 0.5 * float(x_arr[4]))
    return np.array([v * math.cos(alpha), v * math.sin(alpha)], dtype=float)
