"""Suite de Experimentos Científicos e Estudos de Ablação (Testes A1 a A5).

Valida experimentalmente todas as hipóteses teóricas do artigo:
- Experimento A1 (Modelos Cinemáticos em Manobras Evasivas):
  Compara `LinearKalmanCV` vs. `CanonicalBicycleEKF` (lambda=1) vs.
  `AdaptiveEKF` (Two-Point-Bicycle com lambda_k adaptativo) vs. `CoordinatedTurnEKF` (CTRV)
  vs. `IMMFilter` em trajetória urbana com curvas acentuadas de 90 graus, frenagem
  e apagão visual (blind zone) no meio da curva.
- Experimento A2 (Critério de Ortogonalidade e Fator de Desvanecimento lambda_k):
  Avalia a velocidade de recuperação pós-manobra e a robustez do gate angular
  sob ruído crescente de detecção e jitter de bounding box.
- Experimento A3 (Geometria de Projeção e Correção de Perspectiva 2D->Solo):
  Quantifica o erro sistemático de projeção homográfica do `bottom_center` puro
  vs. ponto de contato corrigido para o eixo traseiro sob diferentes ângulos de câmera.
- Experimento A4 (Associação Intercâmeras: Artigo vs. Qui-Quadrado vs. Re-ID vs. Híbrido):
  Mede IDF1, Acurácia de Correspondência Intercâmeras e Acurácia de Rota (Top-1/Top-3)
  em rede multicâmera com zonas cegas.
- Experimento A5 (Propagação Multi-Hipótese no Grafo + Map-Matching HMM/Viterbi):
  Compara Dead Reckoning em espaço aberto (sem grafo) vs. Projeção Euclidiana Simples
  vs. HMM/Viterbi vs. Propagação Multi-Hipótese no Grafo com Poda por Velocidade Crítica.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .cityflow import load_cityflow
from .config import PipelineConfig
from .ekf import (
    AdaptiveEKF,
    CoordinatedTurnEKF,
    IMMFilter,
    LinearKalmanCV,
    create_filter,
)
from .evaluation import evaluate
from .geometry import (
    apply_homography,
    apply_inverse_homography,
    bbox_bottom_center,
    homography_from_intrinsics_extrinsics,
    wrap_angle,
)
from .kinematics import bicycle_step
from .pipeline import SurveillancePipeline
from .roadgraph import Edge, RoadGraph, grid_graph
from .synthetic import generate_synthetic_cityflow

Array = np.ndarray


# ==============================================================================
# EXPERIMENTO A1: Comparação de Filtros Cinemáticos em Manobra Evasiva
# ==============================================================================
def run_experiment_a1_kinematics(seed: int = 42) -> dict:
    """Simula um veículo em fuga executando reta -> frenagem + curva sustentada de 90° -> aceleração,
    incluindo uma janela de oclusão cega de 1.4s no meio da curva.
    """
    rng = np.random.default_rng(seed)
    cfg = PipelineConfig(dt=0.1, wheelbase=2.5, sigma_pos_meas=0.45)
    n_steps = 110

    # Gera trajetória ground-truth fisicamente exata
    x_true = np.array([0.0, 0.0, 14.0, 0.0, 0.0], dtype=float)
    gt_traj: List[Array] = []
    obs_traj: List[Optional[tuple[Array, float]]] = []

    for k in range(n_steps):
        # Manobra evasiva: veículo entra na curva em k=30 com esterçamento sustentado até k=58
        if 30 <= k < 38:
            accel, steer_rate = -1.5, 0.045
        elif 38 <= k < 58:
            accel, steer_rate = 0.5, 0.010
        elif 58 <= k < 66:
            accel, steer_rate = 1.5, -0.050
        else:
            accel, steer_rate = 0.0, 0.0

        x_true = bicycle_step(
            x_true, cfg.dt, cfg.wheelbase, accel=accel, steer_rate=steer_rate, steer_damping=0.96
        )
        gt_traj.append(x_true.copy())

        # Janela de zona cega no meio da curva (quadros 42 a 56: 1.4s sem observação visual)
        if 42 <= k < 56:
            obs_traj.append(None)
        else:
            noise_pos = rng.normal(0.0, cfg.sigma_pos_meas, size=2)
            noise_th = float(rng.normal(0.0, 0.06))
            obs_traj.append((x_true[:2] + noise_pos, wrap_angle(x_true[3] + noise_th)))

    models = ["linear_cv", "canonical_bicycle", "adaptive_bicycle", "ctrv", "imm"]
    results: Dict[str, dict] = {}
    R_cov = np.eye(2) * (cfg.sigma_pos_meas ** 2)

    for model_name in models:
        m_cfg = PipelineConfig(
            dt=cfg.dt,
            wheelbase=cfg.wheelbase,
            sigma_pos_meas=cfg.sigma_pos_meas,
            filter_model=model_name,
        )
        flt = create_filter([0.0, 0.0, 14.0, 0.0, 0.0], cfg=m_cfg)
        errors_all: List[float] = []
        errors_maneuver: List[float] = []
        errors_blind: List[float] = []

        for k in range(n_steps):
            flt.predict(cfg.dt)
            obs = obs_traj[k]
            if obs is not None:
                z_pos, z_hdg = obs
                flt.update(z_pos, R_cov, heading=z_hdg, sigma_heading=0.08)

            err_post = float(np.linalg.norm(flt.position - gt_traj[k][:2]))
            errors_all.append(err_post)
            if 30 <= k < 62:
                errors_maneuver.append(err_post)
            if 42 <= k < 56:
                errors_blind.append(err_post)

        results[model_name] = {
            "rmse_overall_m": round(float(np.sqrt(np.mean(np.square(errors_all)))), 3),
            "rmse_maneuver_m": round(float(np.sqrt(np.mean(np.square(errors_maneuver)))), 3),
            "rmse_blind_outage_m": round(float(np.sqrt(np.mean(np.square(errors_blind)))), 3),
            "max_error_m": round(float(np.max(errors_all)), 3),
            "lambda_mean": round(float(np.mean(flt.lambda_history or [1.0])), 3),
            "lambda_max": round(float(np.max(flt.lambda_history or [1.0])), 3),
        }

    return {
        "experiment": "A1_Kinematic_Models_Comparison",
        "description": (
            "Comparação de RMSE (metros) entre Filtro Linear CV, EKF Bicycle Canônico (lambda=1), "
            "EKF Bicycle Adaptativo (Lou et al.), CTRV (Bradler et al.) e IMM durante manobra evasiva e apagão visual."
        ),
        "results": results,
    }


# ==============================================================================
# EXPERIMENTO A2: Estabilidade do Fator de Desvanecimento e Gate Angular
# ==============================================================================
def run_experiment_a2_fading_stability(seed: int = 101) -> dict:
    """Testa o comportamento de lambda_k e a necessidade do gate contra jitter de heading."""
    rng = np.random.default_rng(seed)
    noise_levels = [0.4, 1.0, 2.2]
    report: Dict[str, dict] = {}

    for sigma in noise_levels:
        cfg_adapt = PipelineConfig(dt=0.1, sigma_pos_meas=sigma, adaptive_fading=True)
        cfg_canon = PipelineConfig(dt=0.1, sigma_pos_meas=sigma, adaptive_fading=False)
        ekf_adapt = AdaptiveEKF([0.0, 0.0, 12.0, 0.0, 0.0], cfg=cfg_adapt, adaptive=True)
        ekf_canon = AdaptiveEKF([0.0, 0.0, 12.0, 0.0, 0.0], cfg=cfg_canon, adaptive=False)

        x_true = np.array([0.0, 0.0, 12.0, 0.0, 0.0], dtype=float)
        err_adapt, err_canon = [], []
        R = np.eye(2) * (sigma ** 2)
        prev_z = x_true[:2].copy()

        for k in range(80):
            steer = 0.085 if 25 <= k < 42 else 0.0
            x_true = bicycle_step(x_true, 0.1, 2.5, steer_rate=steer, steer_damping=0.9)
            z = x_true[:2] + rng.normal(0.0, sigma, size=2)
            d = z - prev_z
            hdg = wrap_angle(math.atan2(d[1], d[0])) if float(np.linalg.norm(d)) > 0.3 else None
            prev_z = z.copy()

            ekf_adapt.predict(0.1)
            ekf_adapt.update(z, R, heading=hdg)
            ekf_canon.predict(0.1)
            ekf_canon.update(z, R, heading=hdg)

            if 25 <= k < 50:
                err_adapt.append(float(np.linalg.norm(ekf_adapt.position - x_true[:2])))
                err_canon.append(float(np.linalg.norm(ekf_canon.position - x_true[:2])))

        rmse_a = float(np.sqrt(np.mean(np.square(err_adapt))))
        rmse_c = float(np.sqrt(np.mean(np.square(err_canon))))
        report[f"sigma_{sigma}m"] = {
            "adaptive_maneuver_rmse_m": round(rmse_a, 3),
            "canonical_maneuver_rmse_m": round(rmse_c, 3),
            "error_reduction_pct": round(100.0 * (rmse_c - rmse_a) / max(rmse_c, 1e-6), 2),
            "peak_lambda": round(float(max(ekf_adapt.lambda_history)), 2),
        }

    return {
        "experiment": "A2_Adaptive_Fading_Robustness",
        "description": "Ganho do fator de desvanecimento lambda_k em curvas sob diferentes níveis de ruído.",
        "results": report,
    }


# ==============================================================================
# EXPERIMENTO A3: Projeção 2D BBox vs. Correção de Ponto de Contato 3D
# ==============================================================================
def run_experiment_a3_homography_projection() -> dict:
    """Quantifica o viés de projeção planar quando a altura 3D do veículo desloca a borda da bbox."""
    # Câmera instalada a 7.5m de altura olhando obliquamente para o cruzamento
    K = np.array([[1100.0, 0.0, 640.0], [0.0, 1100.0, 480.0], [0.0, 0.0, 1.0]])
    pitch = math.radians(62.0)  # Inclinação em relação à vertical
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), -math.cos(pitch)],
    ])
    cam_pos = np.array([0.0, -28.0, 7.5])
    t = -R @ cam_pos
    H_i2w = homography_from_intrinsics_extrinsics(K, R, t)

    # Veículo de 4.4m x 1.8m x 1.5m em várias posições na via
    test_positions = [(0.0, 5.0), (3.5, 12.0), (-3.5, 20.0), (5.0, 30.0)]
    naive_errors: List[float] = []
    corrected_errors: List[float] = []

    for wx, wy in test_positions:
        # Projeta os 8 vértices 3D do veículo na imagem para obter a BBox 2D real vista pela câmera
        dx, dy, dz = 0.9, 2.2, 1.5
        corners_3d = [
            np.array([wx + sx * dx, wy + sy * dy, sz * dz])
            for sx in (-1, 1)
            for sy in (-1, 1)
            for sz in (0, 1)
        ]
        uvs = []
        for pt in corners_3d:
            p_cam = K @ (R @ pt + t)
            uvs.append(p_cam[:2] / p_cam[2])
        uvs_arr = np.asarray(uvs)
        x_min, y_min = uvs_arr.min(axis=0)
        x_max, y_max = uvs_arr.max(axis=0)
        box = (float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min))

        # 1. Projeção ingênua (borda inferior exata fraction=1.0)
        px_naive = bbox_bottom_center(box, fraction=1.0)
        w_naive = apply_homography(H_i2w, px_naive)
        naive_errors.append(float(np.linalg.norm(w_naive - np.array([wx, wy]))))

        # 2. Projeção corrigida (fraction=0.90 + compensação de perspectiva)
        px_corr = bbox_bottom_center(
            box, fraction=0.90, camera_center_px=(640.0, 480.0), perspective_correction=0.08
        )
        w_corr = apply_homography(H_i2w, px_corr)
        corrected_errors.append(float(np.linalg.norm(w_corr - np.array([wx, wy]))))

    mean_naive = float(np.mean(naive_errors))
    mean_corr = float(np.mean(corrected_errors))
    return {
        "experiment": "A3_Ground_Plane_Projection_Bias",
        "description": "Erro métrico de localização no solo: BBox bottom-center bruto vs. Ponto de Contato Corrigido.",
        "results": {
            "naive_bottom_center_mean_error_m": round(mean_naive, 3),
            "corrected_contact_point_mean_error_m": round(mean_corr, 3),
            "bias_reduction_pct": round(100.0 * (mean_naive - mean_corr) / max(mean_naive, 1e-6), 2),
        },
    }


# ==============================================================================
# EXPERIMENTO A4: Associação Multicâmera (Artigo vs. Chi2 vs. Re-ID vs. Híbrido)
# ==============================================================================
def run_experiment_a4_multicamera_association(root: Optional[str] = None) -> dict:
    """Compara os 4 modos de associação intercâmeras no cenário multicâmera com regiões cegas."""
    import tempfile
    if root is None:
        root = str(Path(tempfile.gettempdir()) / "opencode_exp_a4_cityflow")
    ds, graph = generate_synthetic_cityflow(root, n_vehicles=18, n_frames=900, seed=20261110)
    modes = ["article", "chi2", "reid_only", "hybrid"]
    comparison: Dict[str, dict] = {}

    for mode in modes:
        cfg = PipelineConfig(
            fps=ds.fps,
            dt=1.0 / ds.fps,
            association_mode=mode,
            use_multipath_propagation=(mode == "hybrid"),
        )
        res = SurveillancePipeline(ds, graph, cfg).run()
        metrics = evaluate(ds, res)
        comparison[mode] = {
            "idf1": metrics["idf1"],
            "cross_camera_accuracy": metrics["cross_camera_accuracy"],
            "route_top1_accuracy": metrics["route_top1_accuracy"],
            "route_topk_accuracy": metrics["route_topk_accuracy"],
            "blind_transitions": metrics["blind_transitions"],
            "throughput_fps": metrics["throughput_fps"],
        }

    return {
        "experiment": "A4_MultiCamera_Association_Ablation",
        "description": (
            "Comparação de correspondência intercâmeras em regiões cegas: "
            "Métrica do Artigo (M<=350) vs. Chi2 Cinemático vs. Re-ID Puro vs. Fusão Híbrida (Grafo+EKF+ReID)."
        ),
        "results": comparison,
    }


# ==============================================================================
# EXPERIMENTO A5: Propagação em Zona Cega (Dead Reckoning vs. Grafo + Velocidade Crítica)
# ==============================================================================
def run_experiment_a5_topological_pruning() -> dict:
    """Demonstra o efeito da poda por velocidade crítica em cruzamento e do Map-Matching."""
    graph = grid_graph(4, 4, spacing=80.0, speed_limit=13.9)
    cfg = PipelineConfig()

    # Caso 1: Veículo em alta velocidade (24 m/s ~ 86 km/h) aproximando-se de cruzamento v_1_1 -> v_2_1 (heading=0)
    x_fast = np.array([155.0, 80.0, 24.0, 0.0, 0.0], dtype=float)
    routes_fast = graph.build_route_tree("e_v_1_1_v_2_1", x_fast, cfg, horizon=3)

    # Caso 2: Veículo em baixa velocidade (7 m/s ~ 25 km/h) com leve esterçamento para a esquerda (heading=0.35 rad)
    x_slow_turn = np.array([155.0, 80.0, 7.0, 0.35, 0.25], dtype=float)
    routes_slow = graph.build_route_tree("e_v_1_1_v_2_1", x_slow_turn, cfg, horizon=3)

    # Caso 3: Avaliação de Map-Matching HMM/Viterbi vs. Projeção Euclidiana Simples sob ruído alto
    true_edges = ["e_v_0_1_v_1_1", "e_v_1_1_v_2_1", "e_v_2_1_v_2_2"]
    rng = np.random.default_rng(77)
    noisy_obs: List[Array] = []
    gt_edge_seq: List[str] = []
    for eid in true_edges:
        e = graph.edges[eid]
        for s in np.linspace(5.0, e.length - 5.0, 8):
            pos, _ = e.point_at_s(float(s))
            # Adiciona ruído lateral forte perto das esquinas (induzindo "unnecessary detour" no modelo geométrico)
            noisy_obs.append(pos + rng.normal(0.0, 9.5, size=2))
            gt_edge_seq.append(eid)

    euclid_seq = [graph.nearest_edge(p).edge_id for p in noisy_obs]
    hmm_seq = graph.hmm_viterbi_match(noisy_obs, sigma_obs=9.5)

    acc_euclid = sum(a == b for a, b in zip(euclid_seq, gt_edge_seq)) / len(gt_edge_seq)
    acc_hmm = sum(a == b for a, b in zip(hmm_seq, gt_edge_seq)) / len(gt_edge_seq)

    blockades = graph.recommend_blockade_points(routes_fast, current_speed=24.0, top_k=3)

    return {
        "experiment": "A5_Topological_Pruning_and_MapMatching",
        "description": (
            "Validação da Poda por Velocidade Crítica de Curva, Map-Matching HMM vs. Euclidiano "
            "e Recomendação de Pontos de Bloqueio Tático Policial (CICC)."
        ),
        "results": {
            "high_speed_straight_top1_prob": round(routes_fast[0].prob, 4) if routes_fast else 0.0,
            "high_speed_top1_route": list(routes_fast[0].nodes) if routes_fast else [],
            "low_speed_turning_top1_prob": round(routes_slow[0].prob, 4) if routes_slow else 0.0,
            "low_speed_top1_route": list(routes_slow[0].nodes) if routes_slow else [],
            "map_matching_euclidean_accuracy": round(acc_euclid, 4),
            "map_matching_hmm_viterbi_accuracy": round(acc_hmm, 4),
            "recommended_police_blockades": [b.as_dict() for b in blockades],
        },
    }


def run_all_experiments(out_path: Optional[str] = None) -> dict:
    """Executa toda a bateria de experimentos científicos A1 a A5 e consolida o relatório."""
    suite = {
        "A1": run_experiment_a1_kinematics(),
        "A2": run_experiment_a2_fading_stability(),
        "A3": run_experiment_a3_homography_projection(),
        "A4": run_experiment_a4_multicamera_association(),
        "A5": run_experiment_a5_topological_pruning(),
    }
    if out_path:
        p = Path(out_path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(suite, indent=2, ensure_ascii=False), encoding="utf-8")
    return suite
