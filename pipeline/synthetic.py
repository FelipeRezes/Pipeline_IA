"""Gerador de dataset sintético no formato CityFlow para demonstração.

Evita a necessidade de baixar ~16 GB da CityFlowV2 apenas para validar a pipeline.
O cenário é uma malha ortogonal com 4 câmeras estáticas não sobrepostas: veículos
percorrem rotas passando por múltiplas câmeras e ficam invisíveis nas regiões
cegas entre elas — exatamente o regimemodelado no artigo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cityflow import Calibration, CityFlowDataset, Detection, write_cityflow
from .config import PipelineConfig
from .geometry import wrap_angle
from .roadgraph import RoadGraph, grid_graph

Array = np.ndarray


def _camera_homography(center: Sequence[float], yaw_deg: float,
                       patch_w: float = 70.0, patch_h: float = 55.0,
                       width: int = 1280, height: int = 960) -> Array:
    """H pixels->solo: retângulo de solo rotacionado centrado na interseção."""
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    sx, sy = patch_w / width, patch_h / height
    A = np.array([[c * sx, -s * sy], [s * sx, c * sy]])
    t = np.asarray(center, dtype=float) - A @ np.array([width / 2.0, height / 2.0])
    return np.array([[A[0, 0], A[0, 1], t[0]],
                     [A[1, 0], A[1, 1], t[1]],
                     [0.0, 0.0, 1.0]])


def _pixel_from_world(H: Array, point: Sequence[float]) -> Array:
    Hi = np.linalg.inv(H)
    p = Hi @ np.array([point[0], point[1], 1.0])
    return p[:2] / p[2]


def generate_synthetic_cityflow(
    root: str,
    n_vehicles: int = 14,
    n_frames: int = 900,
    seed: int = 20261110,
    cfg: Optional[PipelineConfig] = None,
) -> Tuple[CityFlowDataset, RoadGraph]:
    """Gera detections/GT/calib/grafo e grava em ``root`` no formato CityFlow."""
    root_dir = Path(root).expanduser()
    root_dir.mkdir(parents=True, exist_ok=True)
    root = str(root_dir)
    cfg = cfg or PipelineConfig()
    rng = np.random.default_rng(seed)
    graph = grid_graph(4, 4, spacing=80.0, speed_limit=13.9)

    cam_specs = [("v_1_1", 15.0), ("v_2_1", 105.0),
                 ("v_1_2", 195.0), ("v_2_2", 285.0)]
    ds = CityFlowDataset(fps=cfg.fps, graph_path="graph.json")
    for cam, yaw in cam_specs:
        node = graph.nodes[cam]
        ds.calibrations[str(int(cam.split("_")[1])) + str(int(cam.split("_")[2]))] = Calibration(
            camera_id=str(int(cam.split("_")[1])) + str(int(cam.split("_")[2])),
            H=_camera_homography((node.x, node.y), yaw),
        )
    # IDs de câmera legíveis e estáveis (11, 21, 12, 22)
    cam_of_node = {node: str(int(node.split("_")[1])) + str(int(node.split("_")[2]))
                   for node, _ in cam_specs}
    node_of_cam = {v: k for k, v in cam_of_node.items()}
    ds.calibrations = {cid: Calibration(cid, ds.calibrations[cid].H) for cid in cam_of_node.values()}

    node_ids = sorted(graph.nodes)
    cam_counter: Dict[str, int] = {}      # IDs únicos por câmera (como um SCT real)
    local_assigned: Dict[Tuple[int, str], int] = {}
    last_seen: Dict[Tuple[int, str], int] = {}
    vehicles: List[dict] = []

    # Trajetos que atravessam a região central, passando por 3-4 câmeras
    # separadas por quarteirões sem cobertura (regiões cegas de ~80 m).
    templates = [
        ["v_0_1", "v_1_1", "v_1_2", "v_2_2", "v_2_3", "v_3_3"],
        ["v_3_1", "v_2_1", "v_2_2", "v_1_2", "v_1_1", "v_0_1"],
        ["v_1_0", "v_1_1", "v_2_1", "v_2_2", "v_3_2", "v_3_3"],
        ["v_2_0", "v_2_1", "v_1_1", "v_1_2", "v_0_2", "v_0_3"],
        ["v_0_2", "v_1_2", "v_1_1", "v_2_1", "v_2_2", "v_3_2"],
        ["v_3_3", "v_2_3", "v_2_2", "v_1_2", "v_1_1", "v_0_1"],
    ]
    for vid in range(1, n_vehicles + 1):
        route = list(templates[(vid - 1) % len(templates)])
        if vid % 2 == 0:
            route = route[::-1]
        start_frame = int(rng.integers(1, max(2, n_frames // 3)))
        speed = float(rng.uniform(9.0, 15.0))
        vehicles.append({"id": vid, "route": route, "start": start_frame,
                         "speed": speed, "pos": None, "edge_idx": 0, "t": 0.0,
                         "finished": False})

    for frame in range(1, n_frames + 1):
        for veh in vehicles:
            if veh["finished"] or frame < veh["start"]:
                continue
            edge_idx = veh["edge_idx"]
            if edge_idx >= len(veh["route"]) - 1:
                veh["finished"] = True
                continue
            src, dst = veh["route"][edge_idx], veh["route"][edge_idx + 1]
            candidates = [e for e in graph.out_edges(src) if e.target == dst]
            if not candidates:
                veh["finished"] = True
                continue
            edge = candidates[0]
            veh["t"] += veh["speed"] * cfg.dt
            if veh["t"] >= edge.length:
                veh["t"] -= edge.length
                veh["edge_idx"] += 1
                continue
            frac = veh["t"] / edge.length
            a, b = np.asarray(edge.polyline[0]), np.asarray(edge.polyline[-1])
            pos = a + frac * (b - a)
            veh["pos"] = pos

            for node_id, cam_id in cam_of_node.items():
                cal = ds.calibrations[cam_id]
                px = _pixel_from_world(cal.H, pos)
                if not (70 <= px[0] <= 1210 and 70 <= px[1] <= 890):
                    continue
                w_px = 4.4 / 0.055 + float(rng.normal(0, 2.5))
                h_px = 2.1 / 0.045 + float(rng.normal(0, 1.8))
                w_px, h_px = max(20.0, w_px), max(12.0, h_px)
                jitter = rng.normal(0, 1.0, size=2)
                u, v = px + jitter
                x, y = u - w_px / 2.0, v - h_px
                key = (veh["id"], cam_id)
                if (key not in local_assigned or
                        frame - last_seen.get(key, -999) > 3):
                    cam_counter[cam_id] = cam_counter.get(cam_id, 0) + 1
                    local_assigned[key] = cam_counter[cam_id]
                last_seen[key] = frame
                det = Detection(cam_id, local_assigned[key], frame,
                                float(x), float(y), float(w_px), float(h_px),
                                score=0.95)
                ds.detections.setdefault(cam_id, []).append(det)
                gt_det = Detection(cam_id, veh["id"], frame,
                                   float(x), float(y), float(w_px), float(h_px),
                                   score=1.0)
                ds.gt.setdefault(cam_id, {}).setdefault(veh["id"], []).append(gt_det)

    for cam in ds.detections:
        ds.detections[cam].sort(key=lambda d: (d.frame_id, d.track_id))
    for cam in ds.gt:
        for dets in ds.gt[cam].values():
            dets.sort(key=lambda d: d.frame_id)

    graph.save_json(str(Path(root) / "graph.json"))
    write_cityflow(ds, root)
    return ds, graph
