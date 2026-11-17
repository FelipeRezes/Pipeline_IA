"""Geometria projetiva: homografia de plano de solo e incerteza de medição.

Implementação sem OpenCV: ajuste de homografia por DLT normalizado (SVD/NumPy) e
projeção analítica de pixels -> metros. Segue a Sec. 3.1 do artigo (ground-plane
constraint), que unifica câmeras monoculares estáticas em um plano de solo comum.
"""

from __future__ import annotations

import numpy as np
from typing import Iterable, Sequence, Tuple

Array = np.ndarray


def apply_homography(H: Array, points: Array) -> Array:
    """Aplica H (3x3) a pontos (N,2) ou (2,), retornando coordenadas (N,2)/(2,)."""
    H = np.asarray(H, dtype=float)
    pts = np.asarray(points, dtype=float)
    single = pts.ndim == 1
    p = pts.reshape(-1, 2)
    ph = np.column_stack([p, np.ones(len(p))]) @ H.T
    out = ph[:, :2] / ph[:, 2:3]
    return out[0] if single else out


def fit_homography(src: Sequence[Sequence[float]],
                   dst: Sequence[Sequence[float]]) -> Array:
    """Estima H com pelo menos 4 correspondências, via DLT normalizado (Hartley).

    Parameters
    ----------
    src : pontos na imagem (u, v).
    dst : pontos no plano de solo métrico (X, Y).
    """
    s = np.asarray(src, dtype=float)
    d = np.asarray(dst, dtype=float)
    if s.shape != d.shape or s.ndim != 2 or s.shape[1] != 2 or len(s) < 4:
        raise ValueError("fit_homography requer >=4 pares (u,v) -> (X,Y)")

    def normalize(pts: Array) -> Tuple[Array, Array]:
        centroid = pts.mean(axis=0)
        shifted = pts - centroid
        mean_dist = np.mean(np.linalg.norm(shifted, axis=1))
        scale = np.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
        T = np.array([[scale, 0.0, -scale * centroid[0]],
                      [0.0, scale, -scale * centroid[1]],
                      [0.0, 0.0, 1.0]])
        return T, np.column_stack([pts, np.ones(len(pts))]) @ T.T

    Ts, sn = normalize(s)
    Td, dn = normalize(d)
    A = np.zeros((2 * len(s), 9))
    for i, ((u, v), (x, y)) in enumerate(zip(sn[:, :2], dn[:, :2])):
        A[2 * i] = [-u, -v, -1.0, 0.0, 0.0, 0.0, x * u, x * v, x]
        A[2 * i + 1] = [0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y]
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ Hn @ Ts
    return H / H[2, 2]


def bbox_bottom_center(box: Sequence[float], fraction: float = 1.0) -> Array:
    """Ponto de contato com o solo da bbox [x, y, w, h]: (x + w/2, y + f*h)."""
    x, y, w, h = map(float, box)
    return np.array([x + 0.5 * w, y + fraction * h])


def homography_jacobian(H: Array, u: float, v: float) -> Array:
    """Jacobiana analítica d(X,Y)/d(u,v) de H em (u,v)."""
    h = np.asarray(H, dtype=float)
    d = h[2, 0] * u + h[2, 1] * v + h[2, 2]
    if abs(d) < 1e-12:
        d = 1e-12
    nx = h[0, 0] * u + h[0, 1] * v + h[0, 2]
    ny = h[1, 0] * u + h[1, 1] * v + h[1, 2]
    return np.array([
        [(h[0, 0] * d - nx * h[2, 0]) / d ** 2,
         (h[0, 1] * d - nx * h[2, 1]) / d ** 2],
        [(h[1, 0] * d - ny * h[2, 0]) / d ** 2,
         (h[1, 1] * d - ny * h[2, 1]) / d ** 2],
    ])


def measurement_covariance(H: Array, pixel: Sequence[float],
                           sigma_px: float, floor_m: float = 0.25) -> Array:
    """Propaga ruído de pixel para covariância métrica R = J Σ_px Jᵀ (+ piso)."""
    J = homography_jacobian(H, float(pixel[0]), float(pixel[1]))
    R = J @ (np.eye(2) * float(sigma_px) ** 2) @ J.T
    return R + np.eye(2) * floor_m ** 2


def metric_size(H: Array, pixel: Sequence[float], w_px: float,
                h_px: float) -> Tuple[float, float]:
    """Largura/comprimento aparentes em metros via escala local da homografia."""
    J = homography_jacobian(H, float(pixel[0]), float(pixel[1]))
    sx = np.linalg.norm(J @ np.array([1.0, 0.0]))
    sy = np.linalg.norm(J @ np.array([0.0, 1.0]))
    return abs(float(w_px) * sx), abs(float(h_px) * sy)


def wrap_angle(angle: float) -> float:
    """Normaliza ângulo para (-pi, pi]."""
    return (float(angle) + np.pi) % (2.0 * np.pi) - np.pi
