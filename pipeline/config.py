"""Configuração central da pipeline.

Os valores padrão seguem as ordens do artigo:
- modelo Two-Point-Bicycle com entre-eixos l ~= 2,5 m (Sec. 2.2);
- pesos de ortogonalidade alpha_x = 0,1 / alpha_y = 0,1 / alpha_theta = 2,8
  com soma 3 (Sec. 2.4);
- fator de desvanecimento lambda >= 1 com picos > 25 em manobras (Sec. 2.4);
- limiar de associação intercâmeras M = 350 (Sec. 3.3);
- poda por conectividade/sentido e envelope de esterçamento (Sec. 4.2);
- operação em tempo real (~10 fps no CityFlow, >20 fps no artigo, Sec. 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class PipelineConfig:
    # --- cinemática / filtragem -------------------------------------------------
    dt: float = 0.1                      # passo de tempo; CityFlow usa ~10 fps
    wheelbase: float = 2.5               # l do Two-Point-Bicycle (m)
    max_steer_deg: float = 35.0          # phi_max para poda de conversões
    sigma_accel: float = 2.0             # desvio do ruído de aceleração (m/s²)
    sigma_steer_rate: float = 0.35       # desvio da taxa do volante (rad/s)
    sigma_pos_meas: float = 3.0         # ruído de medição no plano de solo (m)
    sigma_theta_meas: float = 0.25       # ruído da orientação derivada (rad)
    min_speed_for_heading: float = 1.5   # abaixo disso, orientação é descartada

    # --- EKF adaptativo (critério de ortogonalidade / desvanecimento) ----------
    alpha: Tuple[float, float, float] = (0.1, 0.1, 2.8)
    fading_forget: float = 0.95          # fator de esquecimento dos resíduos
    fading_beta: float = 1.5             # fator de amolecimento de R
    lambda_max: float = 60.0             # teto numérico para lambda
    min_lambda: float = 1.0

    # --- associação intercâmeras ------------------------------------------------
    association_mode: str = "chi2"       # "chi2" (recomendado) ou "article"
    association_threshold: float = 350.0  # M do artigo (Sec. 3.3)
    chi2_threshold_4dof_99: float = 13.2767  # qui-quadrado 4 gl, p = 0,01
    sigma_assoc_pos: float = 12.0        # sigma das componentes de posição (m)
                                         # calibrado para trechos cegos de 80-160 m
    sigma_assoc_w: float = 0.35          # sigma da largura normalizada
    sigma_assoc_l: float = 0.45          # sigma do comprimento normalizado
    cw_default: float = 1.0              # C_w^{12} padrão (calibrável)
    cl_default: float = 1.0              # C_l^{12} padrão (calibrável)

    # --- regiões cegas / grafo viário ------------------------------------------
    miss_timeout_frames: int = 10        # quadros sem detecção -> modo cego
    max_blind_time_s: float = 30.0       # descarta hipóteses muito antigas
    route_horizon: int = 4               # nº de arestas na árvore de rotas
    beam_width: int = 12                 # largura do feixe na expansão
    max_routes: int = 6                  # rotas devolvidas ao despacho
    min_route_prob: float = 1e-4         # poda de rotas improváveis
    turn_margin_deg: float = 8.0         # margem no envelope de esterçamento
    speed_bonus: float = 0.08            # vias mais rápidas = fuga mais provável
    graph_snap_weight: float = 0.35      # força de atração para a aresta viária

    # --- dados ------------------------------------------------------------------
    fps: float = 10.0
    det_score_min: float = 0.0
    bbox_bottom_fraction: float = 1.0    # bottom-center da bbox (1.0 = base)

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt deve ser positivo")
        if self.wheelbase <= 0:
            raise ValueError("wheelbase deve ser positivo")
        if abs(sum(self.alpha) - 3.0) > 1e-6:
            raise ValueError("alpha deve somar 3 (critério de ortogonalidade)")
        if self.association_mode not in {"chi2", "article"}:
            raise ValueError("association_mode deve ser 'chi2' ou 'article'")
