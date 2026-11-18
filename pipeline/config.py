"""Configuração central e hiperparâmetros físicos/estocásticos da pipeline.

Contém todas as constantes físicas, parâmetros dos modelos cinemáticos
(Linear CV, Two-Point-Bicycle EKF, CTRV, IMM), parâmetros do fator de
desvanecimento adaptativo (Lou et al., 2002), restrições de malha viária
e velocidade crítica de curva, bem como hiperparâmetros de associação
multicâmera (Chilgunde et al., 2004 + fusão Re-ID + Grafo).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class PipelineConfig:
    # --- 1. Parâmetros Temporais e Físicos do Veículo ------------------------
    fps: float = 10.0
    dt: float = 0.1                      # Passo temporal padrão (1 / fps)
    wheelbase: float = 2.5               # Entre-eixos l (m) - Two-Point-Bicycle
    max_steer_deg: float = 35.0          # Ângulo máximo de esterçamento phi_max (graus)
    tire_friction_mu: float = 0.75       # Coeficiente de atrito pneu-asfalto seco
    gravity_g: float = 9.81              # Aceleração da gravidade (m/s^2)
    max_speed_mps: float = 45.0          # Teto físico de velocidade urbana (~162 km/h)

    # --- 2. Ruídos de Processo e Medição (EKF / CTRV / CV / IMM) -------------
    sigma_accel: float = 2.2             # Desvio padrão da aceleração longitudinal (m/s^2)
    sigma_steer_rate: float = 0.30       # Desvio padrão da taxa do volante b(t) (rad/s)
    sigma_yaw_rate: float = 0.35         # Desvio padrão da taxa de giro omega (CTRV)
    sigma_pos_meas: float = 1.5          # Piso de incerteza métrica no plano de solo (m)
    sigma_theta_meas: float = 0.22       # Incerteza da orientação observada (rad)
    min_speed_for_heading: float = 1.8   # Velocidade mínima (m/s) para confiar no heading delta
    steer_damping: float = 0.85          # Amortecimento natural do volante quando sem medição de phi

    # --- 3. Adaptação Estocástica (Lou et al., 2002 - Seção 2.4) -------------
    adaptive_fading: bool = True         # Ativa/desativa fator de desvanecimento lambda_k
    alpha: Tuple[float, float, float] = (0.1, 0.1, 2.8)  # Pesos de ortogonalidade (soma = 3.0)
    fading_forget: float = 0.90          # Fator de esquecimento exponencial da matriz V_k
    fading_beta: float = 1.5             # Fator de suavização da covariância de medição R
    min_lambda: float = 1.0              # Limite inferior de lambda_k (1.0 = EKF canônico)
    lambda_max: float = 40.0             # Teto numérico de lambda_k em manobras extremas
    heading_gate_rad: float = 1.2        # Gate contra jitter espúrio de bbox no canal theta

    # --- 4. Seleção de Modelo Cinemático -------------------------------------
    # Opções: "adaptive_bicycle" | "canonical_bicycle" | "ctrv" | "linear_cv" | "imm"
    filter_model: str = "adaptive_bicycle"

    # --- 5. Geometria e Projeção no Plano de Solo (Seção 3.1) ----------------
    bbox_bottom_fraction: float = 0.92   # Ponto de contato do eixo traseiro na bbox (corrige overhang)
    perspective_height_m: float = 1.45   # Altura média de veículo de passeio para correção 3D->2D

    # --- 6. Regiões Cegas, Map-Matching e Poda Física (Seção 4) --------------
    miss_timeout_frames: int = 5         # Quadros sem detecção para entrar em modo Out-of-Sight
    max_blind_time_s: float = 35.0       # Tempo máximo de sobrevivência de hipóteses na zona cega
    route_horizon: int = 5               # Profundidade máxima de arestas na Árvore de Rotas
    beam_width: int = 16                 # Largura do feixe de hipóteses no propagador MHT
    max_routes: int = 6                  # Número de rotas prioritárias enviadas ao despacho CICC
    min_route_prob: float = 1e-4         # Limiar de poda de ramos com probabilidade desprezível
    turn_margin_deg: float = 8.0         # Margem angular adicional no envelope de conversão
    speed_bonus: float = 0.10            # Peso de preferência por vias arteriais de maior capacidade
    graph_snap_weight: float = 0.40      # Ganho de projeção lateral no map-matching contínuo
    use_multipath_propagation: bool = True  # Propaga EKF ao longo de múltiplos ramos na zona cega

    # --- 7. Associação Intercâmeras e Re-ID Visual (Seção 3.3 + Fusão) -------
    # Modos: "hybrid" (EKF+Grafo+ReID) | "chi2" (EKF+Grafo) | "article" (M <= 350) | "reid_only"
    association_mode: str = "hybrid"
    association_threshold: float = 350.0
    chi2_threshold_4dof_99: float = 13.2767
    sigma_assoc_pos: float = 11.0        # Incerteza espacial base na transição cega (m)
    sigma_assoc_w: float = 0.35          # Desvio de largura aparente normalizada
    sigma_assoc_l: float = 0.45          # Desvio de comprimento aparente normalizado
    cw_default: float = 1.0              # Constante de escala intercâmera C_w^{12}
    cl_default: float = 1.0              # Constante de escala intercâmera C_l^{12}
    reid_weight: float = 0.55            # Peso da similaridade de aparência Re-ID na fusão híbrida
    min_reid_cosine: float = 0.35        # Similaridade cosseno mínima para aceitar correspondência

    # --- 8. Detecção Visual e Rastreamento Intra-Câmera (SCT) ----------------
    det_score_min: float = 0.25          # Confiança mínima de detecção
    sct_high_thresh: float = 0.50        # Limiar alto do ByteTrack
    sct_low_thresh: float = 0.15         # Limiar baixo do ByteTrack (recuperação de oclusão)
    sct_min_iou: float = 0.15            # IoU mínimo para associação intra-câmera
    sct_max_miss: int = 12               # Quadros de memória do rastreador intra-câmera

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt deve ser positivo")
        if self.wheelbase <= 0:
            raise ValueError("wheelbase deve ser positivo")
        if abs(sum(self.alpha) - 3.0) > 1e-6:
            raise ValueError("alpha deve somar 3 (critério de ortogonalidade de Lou et al.)")
        valid_assocs = {"hybrid", "chi2", "article", "reid_only"}
        if self.association_mode not in valid_assocs:
            raise ValueError(f"association_mode deve ser um de {valid_assocs}")
        valid_filters = {
            "adaptive_bicycle",
            "canonical_bicycle",
            "ctrv",
            "linear_cv",
            "imm",
        }
        if self.filter_model not in valid_filters:
            raise ValueError(f"filter_model deve ser um de {valid_filters}")
