"""Geometria projetiva, plano de solo unificado e calibração volumétrica 3D.

Implementa os fundamentos da Seção 3.1 do artigo:
1. Transformação Homográfica Planar (Chilgunde et al., 2004):
   - Estimativa via DLT normalizado (Hartley) com condicionamento numérico;
   - Projeção direta (pixel -> plano de solo em metros) e inversa (solo -> pixel);
   - Jacobiana analítica d(X, Y)/d(u, v) e propagação rigorosa da covariância
     de medição R_k = J * Sigma_px * J^T.
2. Projeção Volumétrica e Correção de Perspectiva (Bradler et al., 2021):
   - Decomposição / uso de matrizes intrínseca K (3x3) e extrínseca P = [R | t] (3x4);
   - Correção do erro sistemático de projeção da bounding box 2D (o centro da base
     da caixa 2D não coincide com a projeção do centro do eixo traseiro devido à
     altura do veículo e ao ângulo de inclinação da câmera).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


def wrap_angle(angle: float) -> float:
    """Normaliza um ângulo em radianos para o intervalo (-pi, pi]."""
    return float((float(angle) + np.pi) % (2.0 * np.pi) - np.pi)


def angle_diff(a: float, b: float) -> float:
    """Diferença angular assinada mínima a - b em (-pi, pi]."""
    return wrap_angle(float(a) - float(b))


def apply_homography(H: Array, points: Sequence[float] | Array) -> Array:
    """Aplica matriz de homografia 3x3 H a ponto(s) (2,) ou (N, 2)."""
    H_mat = np.asarray(H, dtype=float).reshape(3, 3)
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    p = pts.reshape(-1, 2)
    ph = np.column_stack([p, np.ones(len(p))]) @ H_mat.T
    denom = np.where(np.abs(ph[:, 2:3]) < 1e-12, 1e-12, ph[:, 2:3])
    out = ph[:, :2] / denom
    return out[0] if single else out


def apply_inverse_homography(H: Array, world_points: Sequence[float] | Array) -> Array:
    """Projeta coordenadas métricas do plano de solo (X, Y) de volta para pixels (u, v)."""
    H_inv = np.linalg.inv(np.asarray(H, dtype=float).reshape(3, 3))
    return apply_homography(H_inv, world_points)


def fit_homography(src: Sequence[Sequence[float]],
                   dst: Sequence[Sequence[float]]) -> Array:
    """Estima homografia planar H (src -> dst) via DLT normalizado de Hartley.

    Requer no mínimo 4 pares de pontos não colineares.
    """
    s = np.asarray(src, dtype=float)
    d = np.asarray(dst, dtype=float)
    if s.shape != d.shape or s.ndim != 2 or s.shape[1] != 2 or len(s) < 4:
        raise ValueError("fit_homography requer >= 4 pares de pontos 2D (u,v) -> (X,Y)")

    def _normalize(pts: Array) -> Tuple[Array, Array]:
        centroid = pts.mean(axis=0)
        shifted = pts - centroid
        mean_dist = float(np.mean(np.linalg.norm(shifted, axis=1)))
        scale = math.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
        T = np.array([
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ])
        pts_h = np.column_stack([pts, np.ones(len(pts))]) @ T.T
        return T, pts_h

    Ts, sn = _normalize(s)
    Td, dn = _normalize(d)
    A = np.zeros((2 * len(s), 9), dtype=float)
    for i, ((u, v), (x, y)) in enumerate(zip(sn[:, :2], dn[:, :2])):
        A[2 * i] = [-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y]

    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    scale_factor = H[2, 2] if abs(H[2, 2]) > 1e-12 else 1.0
    return H / scale_factor


def homography_from_intrinsics_extrinsics(K: Array, R: Array, t: Array) -> Array:
    """Deriva a homografia plano de solo (Z=0) -> imagem e retorna sua inversa (imagem -> solo).

    Seção 3.1 (Bradler et al., 2021):
        x_ics ~ K * [r_1, r_2, r_3, t] * [X, Y, 0, 1]^T = K * [r_1, r_2, t] * [X, Y, 1]^T.
    Portanto, H_{world->img} = K @ [r_1, r_2, t], e H_{img->world} = inv(H_{world->img}).
    """
    K_mat = np.asarray(K, dtype=float).reshape(3, 3)
    R_mat = np.asarray(R, dtype=float).reshape(3, 3)
    t_vec = np.asarray(t, dtype=float).reshape(3, 1)
    H_w2i = K_mat @ np.hstack([R_mat[:, :2], t_vec])
    H_i2w = np.linalg.inv(H_w2i)
    return H_i2w / (H_i2w[2, 2] if abs(H_i2w[2, 2]) > 1e-12 else 1.0)


def bbox_bottom_center(box: Sequence[float],
                       fraction: float = 1.0,
                       camera_center_px: Optional[Tuple[float, float]] = None,
                       perspective_correction: float = 0.0) -> Array:
    """Calcula o ponto representativo de contato com o solo a partir da bbox 2D [x, y, w, h].

    Parâmetros:
    -----------
    box : [x, y, w, h] em pixels.
    fraction : fração vertical da altura da caixa (1.0 = borda inferior exata;
               ~0.90-0.95 aproxima o centro do eixo traseiro evitando sombra/para-choque).
    camera_center_px : centro óptico (cx, cy) opcional para correção radial de perspectiva.
    perspective_correction : ganho [0.0, 0.25] que desloca o ponto ligeiramente em direção
                             ao centro da caixa para compensar a projeção 3D do teto/capô
                             em câmeras oblíquas (Teste A3).
    """
    x, y, w, h = map(float, box)
    u = x + 0.5 * w
    v = y + fraction * h
    if perspective_correction > 0.0:
        # Desloca da borda inferior em direção ao centro geométrico da bbox
        # proporcionalmente ao fator de correção de perspectiva 3D
        cy_box = y + 0.5 * h
        v = (1.0 - perspective_correction) * v + perspective_correction * cy_box
        if camera_center_px is not None:
            cx, _ = camera_center_px
            dx = (u - cx) / max(cx, 1.0)
            u -= 0.12 * perspective_correction * dx * w
    return np.array([u, v], dtype=float)


def homography_jacobian(H: Array, u: float, v: float) -> Array:
    """Jacobiana analítica J = d(X, Y) / d(u, v) de H avaliada no pixel (u, v)."""
    h = np.asarray(H, dtype=float).reshape(3, 3)
    d = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    if abs(d) < 1e-12:
        d = 1e-12 if d >= 0 else -1e-12
    nx = h[0, 0] * u + h[0, 1] * v + h[0, 2]
    ny = h[1, 0] * u + h[1, 1] * v + h[1, 2]
    inv_d2 = 1.0 / (d * d)
    return np.array([
        [(h[0, 0] * d - nx * h[2, 0]) * inv_d2,
         (h[0, 1] * d - nx * h[2, 1]) * inv_d2],
        [(h[1, 0] * d - ny * h[2, 0]) * inv_d2,
         (h[1, 1] * d - ny * h[2, 1]) * inv_d2],
    ], dtype=float)


def measurement_covariance(H: Array,
                           pixel: Sequence[float],
                           sigma_px: float | Tuple[float, float],
                           floor_m: float = 0.25) -> Array:
    """Propaga a incerteza em pixels para covariância métrica no plano de solo:
        R = J * Sigma_px * J^T + floor_m^2 * I_2
    """
    u, v = float(pixel[0]), float(pixel[1])
    J = homography_jacobian(H, u, v)
    if isinstance(sigma_px, (tuple, list, np.ndarray)):
        sx, sy = float(sigma_px[0]), float(sigma_px[1])
        Sigma_px = np.diag([sx * sx, sy * sy])
    else:
        s = float(sigma_px)
        Sigma_px = np.eye(2) * (s * s)
    R = J @ Sigma_px @ J.T + np.eye(2) * (float(floor_m) ** 2)
    return 0.5 * (R + R.T)


def metric_size(H: Array,
                pixel: Sequence[float],
                w_px: float,
                h_px: float,
                vehicle_height_m: float = 1.45) -> Tuple[float, float]:
    """Estima dimensões físicas aparentes (largura W, comprimento L) em metros.

    Corrige a distorção de homografia planar (que inflaria L ao projetar a altura
    da bounding box 2D no chão) aplicando regularização anatômica veicular.
    """
    J = homography_jacobian(H, float(pixel[0]), float(pixel[1]))
    sx = float(np.linalg.norm(J @ np.array([1.0, 0.0])))
    sy = float(np.linalg.norm(J @ np.array([0.0, 1.0])))
    raw_w = abs(float(w_px) * sx)
    raw_l = abs(float(h_px) * sy)
    # Em câmeras elevadas, a altura vertical da bbox 2D mistura comprimento proyectado
    # e altura do teto (vehicle_height_m). Suavizamos para manter invariância geométrica:
    corrected_l = max(1.8, raw_l - 0.35 * vehicle_height_m)
    return float(np.clip(raw_w, 1.2, 4.5)), float(np.clip(corrected_l, 2.2, 12.0))
