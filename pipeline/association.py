"""Associação probabilística intercâmeras em regiões cegas (Sec. 3.3 do artigo).

Para um alvo a que sai da câmara 1 em t_a e um veículo b detetado na câmara 2
em t_b, o vetor de diferenças é:

    O_ab = [ (X_b - X_a) - Vx*dt,
             (Y_b - Y_a) - Vy*dt,
             (W_b - C_w * W_a),
             (L_b - C_l * L_a) ]

Cada componente é normalizada pelo seu desvio e a verossimilhança composta é o
produto das gaussianas. A métrica do artigo é M = -log(produto) com limiar 350;
também é oferecida a estatística qui-quadrado (4 gl, p=0,01 => 13,2767), mais
adequada quando as componentes são expressas em metros.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .optimize import chi2_ppf, linear_sum_assignment

Array = np.ndarray


@dataclass
class AssociationCandidate:
    """Resumo do estado necessário para comparar duas detecções entre câmeras."""
    key: str
    position: Array          # última posição conhecida no plano de solo
    velocity: Array          # vetor [vx, vy] estimado pelo EKF
    size: Array              # [largura, comprimento] aparentes em metros
    last_time: float
    camera_id: str = ""
    predicted_position: Optional[Array] = None  # predição já restrita ao grafo viário


@dataclass
class AssociationResult:
    metric: float
    chi2: float
    residuals: Array
    accepted: bool
    threshold: float
    dt: float

    def as_dict(self) -> dict:
        return {"M": round(self.metric, 3), "chi2": round(self.chi2, 3),
                "accepted": self.accepted, "dt_s": round(self.dt, 2)}


def association_metric(a: AssociationCandidate,
                       b: AssociationCandidate,
                       cfg: PipelineConfig) -> AssociationResult:
    """Calcula M(a=b | O_ab) e decide a continuidade de identidade."""
    dt = max(1e-6, float(b.last_time - a.last_time))
    expected = (np.asarray(a.predicted_position, dtype=float)
                if a.predicted_position is not None else
                np.asarray(a.position, dtype=float) +
                np.asarray(a.velocity, dtype=float) * dt)
    resid_pos = np.asarray(b.position, dtype=float) - expected
    resid_w = float(b.size[0] - cfg.cw_default * a.size[0])
    resid_l = float(b.size[1] - cfg.cl_default * a.size[1])

    z = np.array([
        resid_pos[0] / cfg.sigma_assoc_pos,
        resid_pos[1] / cfg.sigma_assoc_pos,
        resid_w / cfg.sigma_assoc_w,
        resid_l / cfg.sigma_assoc_l,
    ])
    chi2 = float(np.dot(z, z))
    # M = -sum log N(z_i | 0, 1) = 0.5*chi2 + n/2*log(2*pi)
    metric = 0.5 * chi2 + 0.5 * len(z) * math.log(2.0 * math.pi)
    threshold = (cfg.association_threshold if cfg.association_mode == "article"
                 else chi2_ppf(0.99, len(z)))
    score = metric if cfg.association_mode == "article" else chi2
    return AssociationResult(metric, chi2, z, bool(score <= threshold),
                             float(threshold), dt)


def associate(lost: Sequence[AssociationCandidate],
              appeared: Sequence[AssociationCandidate],
              cfg: PipelineConfig) -> List[Tuple[int, int, AssociationResult]]:
    """Emparelhamento ótimo (Hungarian) entre alvos perdidos e novas detecções."""
    if not lost or not appeared:
        return []
    M = np.zeros((len(lost), len(appeared)))
    results: dict[tuple[int, int], AssociationResult] = {}
    for i, a in enumerate(lost):
        for j, b in enumerate(appeared):
            res = association_metric(a, b, cfg)
            too_old = (b.last_time - a.last_time) > cfg.max_blind_time_s
            M[i, j] = res.chi2 + (1e6 if too_old or not res.accepted else 0.0)
            results[(i, j)] = res
    rows, cols = linear_sum_assignment(M)
    out: List[Tuple[int, int, AssociationResult]] = []
    for i, j in zip(rows, cols):
        res = results[(int(i), int(j))]
        if res.accepted and (res.dt <= cfg.max_blind_time_s):
            out.append((int(i), int(j), res))
    return out
