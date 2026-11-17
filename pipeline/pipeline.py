"""Orquestração completa do pipeline (Secs. 3, 4 e 5.3 do artigo).

Fluxo por quadro:
1. detecção 2D -> bottom-center -> homografia -> plano de solo comum;
2. EKF adaptativo por tracklet de câmera única (com desvanecimento);
3. perda de visão -> dead reckoning + restrição à aresta viária (map-matching);
4. surgimento em outra câmera -> associação gaussiana e unificação de ID global;
5. árvore de rotas prováveis elaborada no instante da entrada na região cega.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .association import AssociationCandidate, associate
from .cityflow import CityFlowDataset, Detection
from .config import PipelineConfig
from .ekf import AdaptiveEKF
from .geometry import apply_homography, measurement_covariance, metric_size, wrap_angle
from .optimize import linear_sum_assignment
from .roadgraph import RoadGraph, Route

Array = np.ndarray


@dataclass
class Track:
    key: str
    camera_id: str
    local_id: int
    global_id: int
    ekf: AdaptiveEKF
    size: Array
    last_time: float
    last_world: Array
    prev_world: Optional[Array] = None
    mode: str = "visible"           # visible | blind | closed
    miss_count: int = 0
    graph_edge: Optional[str] = None
    blind_node: Optional[str] = None
    routes: List[Route] = field(default_factory=list)
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

    def as_dict(self) -> dict:
        return {"de": self.lost_key, "para": self.appeared_key,
                "id_global": self.global_id, "dt_s": round(self.dt, 2),
                "M": round(self.metric, 2), "cross_camera": self.cross_camera,
                "top1": self.top1_correct, "topk": self.topk_correct}


@dataclass
class PipelineResult:
    tracks: Dict[str, Track]
    assignments: Dict[Tuple[str, int, int], int]
    matches: List[MatchEvent]
    routes: Dict[str, List[Route]]
    elapsed_s: float
    n_updates: int
    n_predictions: int
    fps_processed: float
    config: PipelineConfig

    def route_report(self) -> List[dict]:
        return [{"track": k, "top": [r.as_dict() for r in v[:3]]}
                for k, v in self.routes.items()]

    def match_report(self) -> List[dict]:
        return [m.as_dict() for m in self.matches]


class SurveillancePipeline:
    """Pipeline EKF + homografia + associação + map-matching para CityFlow."""

    def __init__(self, dataset: CityFlowDataset, graph: RoadGraph,
                 cfg: Optional[PipelineConfig] = None):
        self.ds = dataset
        self.graph = graph
        self.cfg = cfg or PipelineConfig()
        self.cfg.validate()
        self._seed: Dict[Tuple[str, int], tuple[Array, float, float]] = {}

    # --------------------------------------------------------------- preparação
    def _world_of(self, det: Detection) -> Array:
        H = self.ds.calibrations[det.camera_id].H
        return apply_homography(H, det.bottom_center)

    def _world_size(self, det: Detection) -> Array:
        H = self.ds.calibrations[det.camera_id].H
        return np.array(metric_size(H, det.bottom_center, det.w, det.h))

    def _covariance_of(self, det: Detection) -> Array:
        H = self.ds.calibrations[det.camera_id].H
        sigma_px = max(1.5, 0.04 * min(det.w, det.h))
        return measurement_covariance(H, det.bottom_center, sigma_px,
                                      floor_m=self.cfg.sigma_pos_meas)

    def _build_seeds(self) -> None:
        """Velocidade/heading iniciais de cada tracklet a partir dos 2 primeiros pontos."""
        by_track: Dict[Tuple[str, int], List[Detection]] = defaultdict(list)
        for cam, dets in self.ds.detections.items():
            for d in dets:
                by_track[(cam, d.track_id)].append(d)
        for key, dets in by_track.items():
            dets.sort(key=lambda d: d.frame_id)
            p1 = self._world_of(dets[0])
            if len(dets) >= 2:
                p2 = self._world_of(dets[1])
                dt = max((dets[1].frame_id - dets[0].frame_id) / self.ds.fps, 1e-3)
                v = float(np.linalg.norm(p2 - p1) / dt)
                theta = wrap_angle(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            else:
                v = 4.0
                proj = self.graph.nearest_edge(p1, max_dist=80.0)
                theta = proj.heading if proj else 0.0
            self._seed[key] = (p1, v, theta)

    def _new_track(self, det: Detection, frame: int) -> Track:
        cam = det.camera_id
        key = (cam, det.track_id)
        p, v, theta = self._seed.get(key, (self._world_of(det), 4.0, 0.0))
        x0 = np.array([p[0], p[1], v, theta, 0.0])
        ekf = AdaptiveEKF(x0, np.diag([4.0, 4.0, 12.0, 1.5, 0.3]), self.cfg)
        t = self.ds.calibrations[cam].timestamp(frame, self.ds.fps)
        return Track(key=f"{cam}:{det.track_id}", camera_id=cam,
                     local_id=det.track_id, global_id=-1, ekf=ekf,
                     size=self._world_size(det), last_time=t,
                     last_world=self._world_of(det))

    def _predicted_blind_position(self, tr: Track, dt: float) -> Array:
        """Posição prevista no fim da região cega, propagada pela rota mais provável.

        Se não houver grafo/árvore disponível, retorna a propagação linear clássica
        X + v*dt (Sec. 3.2 do artigo).
        """
        speed = float(max(tr.ekf.x[2], 0.0))
        if tr.routes:
            distance = speed * max(dt, 0.0)
            for route in tr.routes[:1]:
                for eid in route.edge_ids:
                    edge = self.graph.edges.get(eid)
                    if edge is None:
                        continue
                    if distance <= edge.length:
                        a = np.asarray(edge.polyline[0])
                        b = np.asarray(edge.polyline[-1])
                        return a + (distance / edge.length) * (b - a)
                    distance -= edge.length
                return np.asarray(self.graph.edges[route.edge_ids[-1]].polyline[-1])
        return tr.last_world + tr.ekf.velocity * max(dt, 0.0)

    # ------------------------------------------------------------------- execução
    def run(self) -> PipelineResult:
        t0 = time.perf_counter()
        self._build_seeds()
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
        next_id = 1
        n_updates = n_predictions = 0

        for frame in frames:
            for cam in ds.camera_ids():
                cal = ds.calibrations[cam]
                now = cal.timestamp(frame, ds.fps)
                dets = by_frame_cam.get((frame, cam), [])
                cam_tracks = [tr for tr in active.values()
                              if tr.camera_id == cam and tr.mode != "closed"]

                # ---- associação detecção <-> tracklet dentro da câmera --------
                pair_map: Dict[int, int] = {}   # índice do track -> índice do det
                if dets and cam_tracks:
                    # o ID do tracklet local é uma restrição dura: nunca trocar
                    # a identidade de dois tracklets que se cruzam na mesma câmera
                    cost = np.array([[
                        (np.linalg.norm(self._world_of(d) - tr.position)
                         if tr.local_id == d.track_id else 1e6)
                        for d in dets] for tr in cam_tracks])
                    rows, cols = linear_sum_assignment(cost)
                    for i, j in zip(rows, cols):
                        if cost[i, j] <= max(12.0, 4.0 * cfg.sigma_pos_meas):
                            pair_map[int(i)] = int(j)

                # ---- atualização / propagação dos tracks existentes ----------
                for i, tr in enumerate(cam_tracks):
                    tr.ekf.predict(cfg.dt)
                    n_predictions += 1
                    if i in pair_map:
                        d = dets[pair_map[i]]
                        world = self._world_of(d)
                        assignments[(cam, frame, d.track_id)] = tr.global_id
                        heading = None
                        if tr.prev_world is not None:
                            delta = world - tr.prev_world
                            if np.linalg.norm(delta) > 0.3:
                                heading = wrap_angle(math.atan2(delta[1], delta[0]))
                        tr.ekf.update(world, self._covariance_of(d), heading=heading)
                        n_updates += 1
                        tr.prev_world = tr.last_world.copy()
                        tr.last_world = world.copy()
                        tr.last_time = now
                        tr.size = self._world_size(d)
                        tr.miss_count = 0
                        if tr.mode == "blind":
                            tr.mode = "visible"
                            tr.graph_edge = None
                    else:
                        tr.miss_count += 1
                        if tr.miss_count >= cfg.miss_timeout_frames:
                            if tr.mode != "blind":
                                tr.mode = "blind"
                                proj = self.graph.nearest_edge(tr.last_world, max_dist=90.0)
                                if proj:
                                    tr.graph_edge = proj.edge_id
                                    tr.blind_node = self.graph.edges[proj.edge_id].target
                                    tr.ekf.x, tr.ekf.P = self.graph.constrain_to_edge(
                                        tr.ekf.x, tr.ekf.P,
                                        self.graph.edges[proj.edge_id], cfg)
                                    tr.routes = self.graph.build_route_tree(
                                        proj.edge_id, tr.ekf.x, cfg)
                                    routes[tr.key] = tr.routes

                # ---- novos tracklets (inclui reentrada após região cega) -------
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
                for tr in appeared:
                    assignments[(tr.camera_id, frame, tr.local_id)] = tr.global_id

                # ---- associação intercâmeras ----------------------------------
                if appeared:
                    lost_candidates = [tr for tr in active.values()
                                       if tr.mode == "blind" and
                                       now - tr.last_time <= cfg.max_blind_time_s]
                    if lost_candidates:
                        def cand(tr: Track) -> AssociationCandidate:
                            return AssociationCandidate(
                                key=tr.key, position=tr.last_world.copy(),
                                velocity=tr.ekf.velocity, size=tr.size.copy(),
                                last_time=tr.last_time, camera_id=tr.camera_id,
                                predicted_position=self._predicted_blind_position(
                                    tr, now - tr.last_time))
                        pairs = associate([cand(t) for t in lost_candidates],
                                          [cand(t) for t in appeared], cfg)
                        for li, aj, res in pairs:
                            lost, new = lost_candidates[li], appeared[aj]
                            new.global_id = lost.global_id
                            new.routes = lost.routes
                            assignments[(new.camera_id, frame, new.local_id)] = new.global_id
                            node = self.graph.nearest_node(new.last_world, max_dist=90.0)
                            actual_node = node.id if node else None
                            pred_nodes = [n for r in lost.routes[:cfg.max_routes]
                                          for n in r.nodes[1:]]
                            top1 = bool(lost.routes and actual_node in lost.routes[0].nodes[1:])
                            topk = bool(actual_node in pred_nodes)
                            matches.append(MatchEvent(
                                lost.key, new.key, new.global_id, res.dt,
                                res.metric, res.chi2, tuple(pred_nodes),
                                actual_node, top1, topk,
                                cross_camera=lost.camera_id != new.camera_id))
                            lost.mode = "closed"
                            routes[f"{lost.key}->{new.key}"] = lost.routes

                # ---- limpeza de hipóteses expiradas ---------------------------
                for tr in list(active.values()):
                    if tr.mode == "blind" and now - tr.last_time > cfg.max_blind_time_s:
                        tr.mode = "closed"
                        active.pop(tr.key, None)

        elapsed = time.perf_counter() - t0
        n_det = sum(len(v) for v in ds.detections.values())
        return PipelineResult(all_tracks, assignments, matches, routes, elapsed,
                              n_updates, n_predictions,
                              n_det / max(elapsed, 1e-9), cfg)
