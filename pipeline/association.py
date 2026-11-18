"""Associação probabilística intercâmeras em regiões cegas (Seção 3.3 + Fusão Híbrida).

Implementa e compara 4 paradigmas de correspondência multicâmera (Teste A4):
1. Modo "article" (Chilgunde et al., 2004 - Seção 3.3):
   - Vetor de diferenças O_ab = [dX - Vx*dt, dY - Vy*dt, W_b - C_w*W_a, L_b - C_l*L_a]^T;
   - Métrica M = -log(prod P(z_i)) com limiar M <= 350.
2. Modo "chi2" (Estatística Qui-Quadrado 4 GL + Propagação Unimodal no Grafo):
   - Teste de hipótese chi^2 <= 13.2767 (p = 0.01).
3. Modo "reid_only" (Apenas Aparência Visual Re-ID):
   - Emparelhamento por similaridade cosseno de embeddings visuais com gate temporal básico.
4. Modo "hybrid" (Fusão Multi-Hipótese no Grafo + EKF + Aparência Re-ID):
   - Avalia todas as hipóteses propagadas na Árvore de Rotas de Fuga (BlindHypothesis),
     selecionando o ramo r* que maximiza a verossimilhança espaço-temporal na chegada
     à câmera B e fundindo com a similaridade cosseno do embedding Re-ID;
   - Realiza automaticamente o Backtracking Forense (identifica qual rota da árvore o
     veículo efetivamente percorreu durante o apagão visual).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .optimize import chi2_ppf, linear_sum_assignment
from .reid import cosine_similarity
from .roadgraph import BlindHypothesis, Route

Array = np.ndarray


@dataclass
class AssociationCandidate:
    """Estado consolidado de um tracklet que saiu de visão (lost) ou surgiu (appeared)."""
    key: str
    position: Array                  # Posição no plano de solo (X, Y) em metros
    velocity: Array                  # Vetor velocidade [Vx, Vy] estimado pelo EKF
    size: Array                      # [Largura W, Comprimento L] aparentes em metros
    last_time: float                 # Timestamp em segundos
    camera_id: str = ""
    predicted_position: Optional[Array] = None
    hypotheses: List[BlindHypothesis] = field(default_factory=list)
    embedding: Optional[Array] = None


@dataclass
class AssociationResult:
    metric: float                    # Métrica log-verossimilhança composta M
    chi2: float                      # Estatística de Mahalanobis / Qui-quadrado
    residuals: Array                 # Vetor de resíduos normalizados z (4D)
    accepted: bool                   # Decisão binária de associação
    threshold: float                 # Limiar utilizado
    dt: float                        # Tempo de trânsito na região cega (s)
    reid_similarity: float = 0.0     # Similaridade cosseno de aparência Re-ID [-1, 1]
    best_route: Optional[Route] = None  # Rota reconstruída por Backtracking Forense

    def as_dict(self) -> dict:
        return {
            "M": round(self.metric, 3),
            "chi2": round(self.chi2, 3),
            "reid_sim": round(self.reid_similarity, 3),
            "accepted": self.accepted,
            "dt_s": round(self.dt, 2),
        }


def _best_hypothesis_residual(a: AssociationCandidate,
                              b_pos: Array,
                              dt: float,
                              cfg: PipelineConfig) -> Tuple[Array, Optional[Route], float]:
    """Seleciona a hipótese da árvore de rotas que melhor explica o surgimento em b_pos."""
    if cfg.use_multipath_propagation and a.hypotheses:
        best_z2 = math.inf
        best_res = np.zeros(2, dtype=float)
        best_route: Optional[Route] = None
        best_prior_bonus = 0.0
        for hyp in a.hypotheses:
            diff = b_pos - np.asarray(hyp.position, dtype=float)
            # Pondera distância espacial pela probabilidade a priori do ramo da árvore
            z2 = float(np.dot(diff, diff)) / (cfg.sigma_assoc_pos ** 2)
            prior_penalty = -0.35 * math.log(max(hyp.prob, 1e-4))
            if z2 + prior_penalty < best_z2:
                best_z2 = z2 + prior_penalty
                best_res = diff
                best_route = hyp.route
                best_prior_bonus = prior_penalty
        return best_res, best_route, best_prior_bonus

    expected = (
        np.asarray(a.predicted_position, dtype=float)
        if a.predicted_position is not None
        else np.asarray(a.position, dtype=float) + np.asarray(a.velocity, dtype=float) * dt
    )
    return b_pos - expected, None, 0.0


def association_metric(a: AssociationCandidate,
                       b: AssociationCandidate,
                       cfg: PipelineConfig) -> AssociationResult:
    """Calcula a verossimilhança de associação M(a = b | O_ab) segundo o modo configurado."""
    raw_dt = float(b.last_time - a.last_time)
    if raw_dt < 0.0:
        # Causalidade temporal violada: b apareceu antes de a desaparecer
        return AssociationResult(
            metric=1e6,
            chi2=1e6,
            residuals=np.full(4, 999.0),
            accepted=False,
            threshold=0.0,
            dt=raw_dt,
        )

    dt = max(1e-6, raw_dt)
    b_pos = np.asarray(b.position, dtype=float)

    resid_pos, best_route, prior_penalty = _best_hypothesis_residual(a, b_pos, dt, cfg)
    resid_w = float(b.size[0] - cfg.cw_default * a.size[0])
    resid_l = float(b.size[1] - cfg.cl_default * a.size[1])

    # A incerteza espacial cresce suavemente com o tempo de trânsito na região cega
    sigma_pos_eff = cfg.sigma_assoc_pos * (1.0 + 0.025 * max(0.0, dt - 5.0))
    z = np.array([
        resid_pos[0] / sigma_pos_eff,
        resid_pos[1] / sigma_pos_eff,
        resid_w / cfg.sigma_assoc_w,
        resid_l / cfg.sigma_assoc_l,
    ], dtype=float)

    chi2_kin = float(np.dot(z, z)) + prior_penalty
    metric = 0.5 * chi2_kin + 0.5 * len(z) * math.log(2.0 * math.pi)
    reid_sim = cosine_similarity(a.embedding, b.embedding)
    has_reid = (a.embedding is not None) and (b.embedding is not None)

    mode = cfg.association_mode.lower()
    if mode == "article":
        threshold = float(cfg.association_threshold)
        accepted = bool(metric <= threshold)
        eff_chi2 = chi2_kin
    elif mode == "chi2":
        threshold = float(chi2_ppf(0.99, len(z)))
        accepted = bool(chi2_kin <= threshold)
        eff_chi2 = chi2_kin
    elif mode == "reid_only":
        threshold = float(cfg.min_reid_cosine)
        if has_reid:
            accepted = bool(reid_sim >= threshold and dt <= cfg.max_blind_time_s)
            eff_chi2 = max(0.0, 20.0 * (1.0 - reid_sim))
        else:
            threshold = float(chi2_ppf(0.99, len(z)))
            accepted = bool(chi2_kin <= threshold)
            eff_chi2 = chi2_kin
    else:  # "hybrid": Fusão Espaço-Temporal (EKF + Grafo Multi-Hipótese) + Aparência Re-ID
        threshold = float(chi2_ppf(0.99, len(z)))
        if has_reid:
            # Converte similaridade cosseno [0, 1] em desconto/penalidade na distância de Mahalanobis
            reid_dist2 = ((1.0 - reid_sim) / 0.25) ** 2
            w_app = float(np.clip(cfg.reid_weight, 0.0, 0.9))
            eff_chi2 = (1.0 - w_app) * chi2_kin + w_app * reid_dist2
            # Gate duplo: não aceita se a aparência for incompatível OU se for fisicamente impossível chegar
            spatial_gate = chi2_kin <= (threshold * 2.2)
            reid_gate = reid_sim >= cfg.min_reid_cosine
            accepted = bool((eff_chi2 <= threshold) and spatial_gate and reid_gate)
        else:
            eff_chi2 = chi2_kin
            accepted = bool(eff_chi2 <= threshold)

    return AssociationResult(
        metric=metric,
        chi2=eff_chi2,
        residuals=z,
        accepted=accepted,
        threshold=threshold,
        dt=dt,
        reid_similarity=reid_sim,
        best_route=best_route,
    )


def associate(lost: Sequence[AssociationCandidate],
              appeared: Sequence[AssociationCandidate],
              cfg: PipelineConfig) -> List[Tuple[int, int, AssociationResult]]:
    """Emparelhamento global ótimo (Hungarian / Kuhn-Munkres) entre alvos perdidos e novos."""
    if not lost or not appeared:
        return []
    M = np.zeros((len(lost), len(appeared)), dtype=float)
    results: dict[tuple[int, int], AssociationResult] = {}
    for i, a in enumerate(lost):
        for j, b in enumerate(appeared):
            res = association_metric(a, b, cfg)
            invalid_time = (b.last_time < a.last_time) or (
                (b.last_time - a.last_time) > cfg.max_blind_time_s
            )
            M[i, j] = res.chi2 + (1e6 if invalid_time or not res.accepted else 0.0)
            results[(i, j)] = res

    rows, cols = linear_sum_assignment(M)
    out: List[Tuple[int, int, AssociationResult]] = []
    for i, j in zip(rows, cols):
        res = results[(int(i), int(j))]
        if res.accepted and (0.0 <= res.dt <= cfg.max_blind_time_s):
            out.append((int(i), int(j), res))
    return out
