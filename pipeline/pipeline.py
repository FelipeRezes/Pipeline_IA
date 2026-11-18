"""Orquestração causal em tempo real do pipeline EKF + Map-Matching + Re-ID (Seções 3, 4 e 5.3).

Refatoração completa (Zero-Leakage & Multi-Hypothesis):
1. Inicialização 100% Causal (Online):
   - Nenhum dado do futuro (`_build_seeds` legado removido!) é lido antes do frame atual;
   - No 1º frame de um veículo, a orientação inicial theta_0 vem do alinhamento
     com a via mais próxima no grafo; no 2º frame consecutivo, a velocidade v_1 e
     o heading theta_1 são refinados causalmente pelo deslocamento observado.
2. Polimorfismo de Filtros Cinemáticos:
   - Suporta `AdaptiveEKF` (Two-Point-Bicycle com desvanecimento lambda_k),
     `CanonicalBicycleEKF` (lambda=1), `CoordinatedTurnEKF` (CTRV),
     `LinearKalmanCV` (Baseline linear) e `IMMFilter`.
3. Propagação Real em Zona Cega (Out-of-Sight):
   - Enquanto um veículo está em modo `"blind"`, o filtro continua propagando
     a incerteza P_k e gera hipóteses cinemáticas ao longo de cada ramo da
     Árvore de Rotas de Fuga (`propagate_route_tree`);
   - Calcula automaticamente os Pontos de Bloqueio Policial (Cerco Tático CICC).
4. Associação Multicâmera Híbrida + Backtracking Forense:
   - Combina verossimilhança espaço-temporal multi-hipótese no grafo com
     similaridade cosseno de aparência Re-ID;
   - Ao reidentificar o veículo na câmera de destino, reconstrói a rota exata
     percorrida na região cega (Backtracking Forense).
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .association import AssociationCandidate, associate
from .cityflow import CityFlowDataset, Detection
from .config import PipelineConfig
from .ekf import AdaptiveEKF, create_filter
from .geometry import (
    apply_homography,
    bbox_bottom_center,
    measurement_covariance,
    metric_size,
    wrap_angle,
)
from .optimize import linear_sum_assignment
from .reid import l2_normalize
from .roadgraph import BlindHypothesis, RoadGraph, Route, TacticalBlockPoint

Array = np.ndarray


@dataclass
class Track:
    key: str
    camera_id: str
    local_id: int
    global_id: int
    ekf: Any
    size: Array
    last_time: float
    last_world: Array
    prev_world: Optional[Array] = None
    prev_time: Optional[float] = None
    obs_count: int = 1
    mode: str = "visible"           # visible | blind | closed
    miss_count: int = 0
    graph_edge: Optional[str] = None
    blind_node: Optional[str] = None
    blind_entry_pos: Optional[Array] = None
    blind_entry_speed: float = 0.0
    routes: List[Route] = field(default_factory=list)
    blockade_points: List[TacticalBlockPoint] = field(default_factory=list)
    embedding: Optional[Array] = None
    history: List[dict] = field(default_factory=list)

    @property
    def position(self) -> Array:
        return self.ekf.position


@dataclass
class MatchEvent:
    lost_key: str
    appeared_key: str
    global_id: int
    dt: float
    metric: float
    chi2: float
    predicted_nodes: Tuple[str, ...]
    actual_node: Optional[str]
    top1_correct: bool
    topk_correct: bool
    cross_camera: bool = False
    reid_similarity: float = 0.0
    reconstructed_route: Optional[List[str]] = None

    def as_dict(self) -> dict:
        return {
            "de": self.lost_key,
            "para": self.appeared_key,
            "id_global": self.global_id,
            "dt_s": round(self.dt, 2),
            "M": round(self.metric, 2),
            "chi2": round(self.chi2, 2),
            "reid_sim": round(self.reid_similarity, 3),
            "cross_camera": self.cross_camera,
            "top1": self.top1_correct,
            "topk": self.topk_correct,
            "forensic_route": self.reconstructed_route,
        }


@dataclass
class PipelineResult:
    tracks: Dict[str, Track]
    assignments: Dict[Tuple[str, int, int], int]
    matches: List[MatchEvent]
    routes: Dict[str, List[Route]]
    blockades: Dict[str, List[TacticalBlockPoint]]
    elapsed_s: float
    n_updates: int
    n_predictions: int
    fps_processed: float
    config: PipelineConfig

    def route_report(self) -> List[dict]:
        out = []
        for k, v in self.routes.items():
            blks = [b.as_dict() for b in self.blockades.get(k, [])]
            out.append({
                "track": k,
                "top": [r.as_dict() for r in v[:3]],
                "tactical_blockades": blks,
            })
        return out

    def match_report(self) -> List[dict]:
        return [m.as_dict() for m in self.matches]


class SurveillancePipeline:
    """Pipeline causal de rastreamento multicâmera, predição de rotas de fuga e cerco tático."""

    def __init__(self,
                 dataset: CityFlowDataset,
                 graph: RoadGraph,
                 cfg: Optional[PipelineConfig] = None):
        self.ds = dataset
        self.graph = graph
        self.cfg = cfg or PipelineConfig()
        self.cfg.validate()

    # --------------------------------------------------------------- geometria
    def _contact_pixel(self, det: Detection) -> Array:
        cal = self.ds.calibrations[det.camera_id]
        return bbox_bottom_center(
            det.box,
            fraction=self.cfg.bbox_bottom_fraction,
            camera_center_px=(cal.width / 2.0, cal.height / 2.0),
        )

    def _world_of(self, det: Detection) -> Array:
        H = self.ds.calibrations[det.camera_id].H
        return apply_homography(H, self._contact_pixel(det))

    def _world_size(self, det: Detection) -> Array:
        H = self.ds.calibrations[det.camera_id].H
        px = self._contact_pixel(det)
        return np.array(
            metric_size(H, px, det.w, det.h, vehicle_height_m=self.cfg.perspective_height_m),
            dtype=float,
        )

    def _covariance_of(self, det: Detection) -> Array:
        H = self.ds.calibrations[det.camera_id].H
        px = self._contact_pixel(det)
        sigma_px = max(1.5, 0.035 * min(det.w, det.h))
        return measurement_covariance(H, px, sigma_px, floor_m=self.cfg.sigma_pos_meas)

    # --------------------------------------------------- inicialização causal
    def _new_track(self, det: Detection, frame: int) -> Track:
        cam = det.camera_id
        p = self._world_of(det)
        proj = self.graph.nearest_edge(p, max_dist=90.0)
        theta0 = proj.heading if proj is not None else 0.0
        v0 = 8.5  # Prior urbano moderado (~30 km/h) até a 2ª observação confirmar velocidade
        x0 = np.array([p[0], p[1], v0, theta0, 0.0], dtype=float)
        P0 = np.diag([3.0, 3.0, 25.0, 0.8, 0.2]).astype(float)
        ekf = create_filter(x0, P0, self.cfg)
        t = self.ds.calibrations[cam].timestamp(frame, self.ds.fps)
        emb = l2_normalize(det.embedding) if det.embedding is not None else None
        tr = Track(
            key=f"{cam}:{det.track_id}",
            camera_id=cam,
            local_id=det.track_id,
            global_id=-1,
            ekf=ekf,
            size=self._world_size(det),
            last_time=t,
            last_world=p.copy(),
            prev_world=p.copy(),
            prev_time=t,
            obs_count=1,
            embedding=emb,
        )
        tr.history.append({
            "frame": frame,
            "time": round(t, 3),
            "mode": "visible",
            "x": round(float(p[0]), 2),
            "y": round(float(p[1]), 2),
            "v": round(float(v0), 2),
            "theta": round(float(theta0), 3),
            "lambda": 1.0,
        })
        return tr

    def _blind_hypotheses(self, tr: Track, dt: float) -> Tuple[Array, List[BlindHypothesis]]:
        """Propaga o estado de um veículo na zona cega por dt segundos ao longo da árvore de rotas."""
        speed = float(max(tr.blind_entry_speed or tr.ekf.x[2], 1.0))
        entry_pos = tr.blind_entry_pos if tr.blind_entry_pos is not None else tr.last_world
        if tr.routes:
            hyps = self.graph.propagate_route_tree(
                tr.routes, entry_pos, speed, max(dt, 0.0), self.cfg
            )
            if hyps:
                return hyps[0].position.copy(), hyps
        # Fallback: dead-reckoning vetorial puro se o veículo estiver fora da malha viária
        fallback_pos = entry_pos + tr.ekf.velocity * max(dt, 0.0)
        return fallback_pos, []

    # ------------------------------------------------------------------- execução
    def run(self) -> PipelineResult:
        t0 = time.perf_counter()
        cfg, ds = self.cfg, self.ds

        by_frame_cam: Dict[Tuple[int, str], List[Detection]] = defaultdict(list)
        for cam, dets in ds.detections.items():
            for d in dets:
                if d.score >= cfg.det_score_min:
                    by_frame_cam[(d.frame_id, cam)].append(d)

        frames = sorted({f for f, _ in by_frame_cam})
        active: Dict[str, Track] = {}
        all_tracks: Dict[str, Track] = {}
        assignments: Dict[Tuple[str, int, int], int] = {}
        matches: List[MatchEvent] = []
        routes: Dict[str, List[Route]] = {}
        blockades: Dict[str, List[TacticalBlockPoint]] = {}
        next_id = 1
        n_updates = n_predictions = 0

        for frame in frames:
            for cam in ds.camera_ids():
                cal = ds.calibrations[cam]
                now = cal.timestamp(frame, ds.fps)
                dets = by_frame_cam.get((frame, cam), [])
                cam_tracks = [
                    tr for tr in active.values()
                    if tr.camera_id == cam and tr.mode != "closed"
                ]

                # ---- 1. Associação detecção <-> tracklet dentro da mesma câmera --
                pair_map: Dict[int, int] = {}
                if dets and cam_tracks:
                    cost = np.array([
                        [
                            (
                                float(np.linalg.norm(self._world_of(d) - tr.position))
                                if tr.local_id == d.track_id
                                else 1e6
                            )
                            for d in dets
                        ]
                        for tr in cam_tracks
                    ], dtype=float)
                    rows, cols = linear_sum_assignment(cost)
                    for i, j in zip(rows, cols):
                        if cost[i, j] <= max(18.0, 5.0 * cfg.sigma_pos_meas):
                            pair_map[int(i)] = int(j)

                # ---- 2. Predição e Atualização EKF dos tracks existentes ---------
                for i, tr in enumerate(cam_tracks):
                    if i in pair_map:
                        d = dets[pair_map[i]]
                        world = self._world_of(d)
                        assignments[(cam, frame, d.track_id)] = tr.global_id
                        dt_obs = max(now - (tr.last_time or now), cfg.dt)

                        # No 2º ponto observado, calibramos causalmente a velocidade e o heading reais
                        if tr.obs_count == 1 and tr.prev_world is not None:
                            delta_init = world - tr.prev_world
                            dist_init = float(np.linalg.norm(delta_init))
                            if dist_init > 0.15:
                                v_real = float(np.clip(dist_init / dt_obs, 1.0, cfg.max_speed_mps))
                                th_real = wrap_angle(math.atan2(delta_init[1], delta_init[0]))
                                tr.ekf.x[2] = v_real
                                tr.ekf.x[3] = th_real

                        tr.ekf.predict(cfg.dt)
                        n_predictions += 1

                        heading = None
                        if tr.prev_world is not None and tr.miss_count == 0:
                            delta = world - tr.prev_world
                            if float(np.linalg.norm(delta)) > 0.35:
                                heading = wrap_angle(math.atan2(delta[1], delta[0]))

                        tr.ekf.update(world, self._covariance_of(d), heading=heading)
                        n_updates += 1

                        # Atualização EMA da assinatura visual Re-ID
                        if d.embedding is not None:
                            emb_new = l2_normalize(d.embedding)
                            tr.embedding = (
                                emb_new
                                if tr.embedding is None
                                else l2_normalize(0.8 * tr.embedding + 0.2 * emb_new)
                            )

                        tr.prev_world = tr.last_world.copy()
                        tr.prev_time = tr.last_time
                        tr.last_world = world.copy()
                        tr.last_time = now
                        tr.size = 0.7 * tr.size + 0.3 * self._world_size(d)
                        tr.obs_count += 1
                        tr.miss_count = 0
                        if tr.mode == "blind":
                            tr.mode = "visible"
                            tr.graph_edge = None

                        tr.history.append({
                            "frame": frame,
                            "time": round(now, 3),
                            "mode": "visible",
                            "x": round(float(tr.ekf.x[0]), 2),
                            "y": round(float(tr.ekf.x[1]), 2),
                            "v": round(float(tr.ekf.x[2]), 2),
                            "theta": round(float(tr.ekf.x[3]), 3),
                            "lambda": round(float(tr.ekf.last_lambda), 3),
                        })
                    else:
                        # Sem detecção neste frame: propaga incerteza (Out-of-Sight / Oclusão)
                        tr.ekf.propagate_only(cfg.dt)
                        n_predictions += 1
                        tr.miss_count += 1
                        if tr.miss_count >= cfg.miss_timeout_frames and tr.mode != "blind":
                            tr.mode = "blind"
                            tr.blind_entry_pos = tr.last_world.copy()
                            tr.blind_entry_speed = float(max(tr.ekf.x[2], 3.0))
                            proj = self.graph.nearest_edge(
                                tr.last_world,
                                max_dist=90.0,
                                heading=float(tr.ekf.x[3]),
                            )
                            if proj is not None:
                                tr.graph_edge = proj.edge_id
                                edge_obj = self.graph.edges[proj.edge_id]
                                tr.blind_node = edge_obj.target
                                tr.ekf.x, tr.ekf.P = self.graph.constrain_to_edge(
                                    tr.ekf.x, tr.ekf.P, edge_obj, cfg
                                )
                                tr.routes = self.graph.build_route_tree(
                                    proj.edge_id, tr.ekf.x, cfg
                                )
                                tr.blockade_points = self.graph.recommend_blockade_points(
                                    tr.routes, tr.blind_entry_speed
                                )
                                routes[tr.key] = tr.routes
                                blockades[tr.key] = tr.blockade_points

                # ---- 3. Instanciação de novos tracklets --------------------------
                appeared: List[Track] = []
                used_dets = set(pair_map.values())
                for j, d in enumerate(dets):
                    if j in used_dets:
                        continue
                    tr = self._new_track(d, frame)
                    if tr.global_id < 0:
                        tr.global_id = next_id
                        next_id += 1
                    active[tr.key] = tr
                    all_tracks[tr.key] = tr
                    appeared.append(tr)
                    assignments[(tr.camera_id, frame, tr.local_id)] = tr.global_id

                # ---- 4. Associação Intercâmeras e Backtracking Forense -----------
                if appeared:
                    lost_candidates = [
                        tr for tr in active.values()
                        if tr.mode == "blind"
                        and 0.0 <= (now - tr.last_time) <= cfg.max_blind_time_s
                    ]
                    if lost_candidates:
                        lost_assoc: List[AssociationCandidate] = []
                        for tr in lost_candidates:
                            dt_blind = max(0.0, now - tr.last_time)
                            pred_pos, hyps = self._blind_hypotheses(tr, dt_blind)
                            lost_assoc.append(
                                AssociationCandidate(
                                    key=tr.key,
                                    position=(
                                        tr.blind_entry_pos.copy()
                                        if tr.blind_entry_pos is not None
                                        else tr.last_world.copy()
                                    ),
                                    velocity=tr.ekf.velocity,
                                    size=tr.size.copy(),
                                    last_time=tr.last_time,
                                    camera_id=tr.camera_id,
                                    predicted_position=pred_pos,
                                    hypotheses=hyps,
                                    embedding=tr.embedding,
                                )
                            )

                        app_assoc: List[AssociationCandidate] = [
                            AssociationCandidate(
                                key=tr.key,
                                position=tr.last_world.copy(),
                                velocity=tr.ekf.velocity,
                                size=tr.size.copy(),
                                last_time=tr.last_time,
                                camera_id=tr.camera_id,
                                embedding=tr.embedding,
                            )
                            for tr in appeared
                        ]

                        pairs = associate(lost_assoc, app_assoc, cfg)
                        for li, aj, res in pairs:
                            lost, new = lost_candidates[li], appeared[aj]
                            new.global_id = lost.global_id
                            new.routes = lost.routes
                            if new.embedding is None and lost.embedding is not None:
                                new.embedding = lost.embedding.copy()
                            assignments[(new.camera_id, frame, new.local_id)] = new.global_id

                            node = self.graph.nearest_node(new.last_world, max_dist=90.0)
                            actual_node = node.id if node is not None else None
                            pred_nodes = [
                                n for r in lost.routes[: cfg.max_routes]
                                for n in r.nodes[1:]
                            ]
                            top1 = bool(
                                lost.routes and actual_node in lost.routes[0].nodes[1:]
                            )
                            topk = bool(actual_node in pred_nodes)
                            forensic_edges = (
                                list(res.best_route.edge_ids)
                                if res.best_route is not None
                                else (list(lost.routes[0].edge_ids) if lost.routes else None)
                            )

                            matches.append(
                                MatchEvent(
                                    lost_key=lost.key,
                                    appeared_key=new.key,
                                    global_id=new.global_id,
                                    dt=res.dt,
                                    metric=res.metric,
                                    chi2=res.chi2,
                                    predicted_nodes=tuple(pred_nodes),
                                    actual_node=actual_node,
                                    top1_correct=top1,
                                    topk_correct=topk,
                                    cross_camera=(lost.camera_id != new.camera_id),
                                    reid_similarity=res.reid_similarity,
                                    reconstructed_route=forensic_edges,
                                )
                            )
                            lost.mode = "closed"
                            trans_key = f"{lost.key}->{new.key}"
                            routes[trans_key] = lost.routes
                            if lost.blockade_points:
                                blockades[trans_key] = lost.blockade_points

                # ---- 5. Limpeza de hipóteses expiradas ---------------------------
                for tr in list(active.values()):
                    if tr.mode == "blind" and (now - tr.last_time) > cfg.max_blind_time_s:
                        tr.mode = "closed"
                        active.pop(tr.key, None)

        elapsed = time.perf_counter() - t0
        n_det = sum(len(v) for v in ds.detections.values())
        return PipelineResult(
            tracks=all_tracks,
            assignments=assignments,
            matches=matches,
            routes=routes,
            blockades=blockades,
            elapsed_s=elapsed,
            n_updates=n_updates,
            n_predictions=n_predictions,
            fps_processed=n_det / max(elapsed, 1e-9),
            config=cfg,
        )
