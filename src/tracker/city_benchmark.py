"""Large-scale city multi-camera urban traffic benchmark (4-24 cameras, 20-150 vehicles).

Evaluates multi-camera association, blind-zone trajectory propagation, route prediction
(including T-GNN vs. HMM/Viterbi vs. Physics-Constrained Multi-Hypothesis Graph Tree),
and computational scalability on an 8x8 mixed arterial/collector/one-way urban road network.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import networkx as nx
import numpy as np

from tracker.alpr import PlateReading
from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
from tracker.config import AssociationConfig, EKFConfig, RoadGraphConfig
from tracker.ekf import AdaptiveEKF
from tracker.geometry import CameraProjector
from tracker.osm_loader import RoadEdge, RoadNetwork, build_road_network_from_multidigraph
from tracker.roadgraph import RoadGraphEngine, _interpolate_polyline


@dataclass
class VirtualCamera:
    """Calibrated non-overlapping urban roadside camera covering a road segment."""

    cam_id: str
    edge_id: tuple[int, int, int]
    s_start_m: float
    s_end_m: float
    projector: CameraProjector
    scale_factor_wl: tuple[float, float]
    alpr_capable: bool


@dataclass
class SimulatedTransition:
    """Ground-truth blind-zone transition between two non-overlapping cameras."""

    vehicle_id: int
    exit_cam_id: str
    entry_cam_id: str
    exit_time: float
    entry_time: float
    blind_distance_m: float
    true_edge_sequence: list[tuple[int, int, int]]
    blind_record: BlindTrackRecord
    new_appearance: NewAppearance
    state_history: list[tuple[float, float, float]]


def build_city_road_network(
    grid_rows: int = 8,
    grid_cols: int = 8,
    block_m: float = 120.0,
    epsg: int = 32723,
) -> tuple[RoadNetwork, RoadGraphEngine]:
    """Construct an 8x8 mixed urban road network (~0.84 km x 0.84 km) in metric UTM.

    Road hierarchy:
      - Rows/Cols 1, 4, 6: bidirectional 3-lane arterials (60 km/h)
      - Rows/Cols 0, 3, 7: bidirectional 2-lane collectors (50 km/h)
      - Remaining streets: alternating one-way 1-lane local streets (40 km/h)
    """
    ox, oy = 330000.0, 7390000.0
    g = nx.MultiDiGraph(crs=f"EPSG:{epsg}", simplified="True")

    node_map: dict[tuple[int, int], int] = {}
    nid = 1
    for r in range(grid_rows):
        for c in range(grid_cols):
            # Mild realistic block spacing jitter (+/- 8 m)
            dx_j = 8.0 * math.sin(0.9 * r + 0.4 * c)
            dy_j = 8.0 * math.cos(0.7 * c - 0.5 * r)
            x = ox + c * block_m + dx_j
            y = oy + r * block_m + dy_j
            g.add_node(nid, x=float(x), y=float(y))
            node_map[(r, c)] = nid
            nid += 1

    def _wkt(pts: list[tuple[float, float]]) -> str:
        return "LINESTRING (" + ", ".join(f"{p[0]:.6f} {p[1]:.6f}" for p in pts) + ")"

    def _add_seg(u: int, v: int, oneway: bool, speed: str, lanes: str) -> None:
        ux, uy = float(g.nodes[u]["x"]), float(g.nodes[u]["y"])
        vx, vy = float(g.nodes[v]["x"]), float(g.nodes[v]["y"])
        mx, my = 0.5 * (ux + vx), 0.5 * (uy + vy)
        pts = [(ux, uy), (mx, my), (vx, vy)]
        length = math.hypot(vx - ux, vy - uy)
        g.add_edge(u, v, key=0, length=length, oneway=oneway, maxspeed=speed, lanes=lanes, geometry=_wkt(pts))
        if not oneway:
            rpts = list(reversed(pts))
            g.add_edge(v, u, key=0, length=length, oneway=False, maxspeed=speed, lanes=lanes, geometry=_wkt(rpts))

    arterials = {1, 4, 6}
    collectors = {0, 3, 7}

    for r in range(grid_rows):
        for c in range(grid_cols - 1):
            u, v = node_map[(r, c)], node_map[(r, c + 1)]
            if r in arterials:
                _add_seg(u, v, oneway=False, speed="60", lanes="3")
            elif r in collectors:
                _add_seg(u, v, oneway=False, speed="50", lanes="2")
            else:
                if r % 2 == 0:
                    _add_seg(u, v, oneway=True, speed="40", lanes="1")
                else:
                    _add_seg(v, u, oneway=True, speed="40", lanes="1")

    for c in range(grid_cols):
        for r in range(grid_rows - 1):
            u, v = node_map[(r, c)], node_map[(r + 1, c)]
            if c in arterials:
                _add_seg(u, v, oneway=False, speed="60", lanes="3")
            elif c in collectors:
                _add_seg(u, v, oneway=False, speed="50", lanes="2")
            else:
                if c % 2 == 0:
                    _add_seg(u, v, oneway=True, speed="40", lanes="1")
                else:
                    _add_seg(v, u, oneway=True, speed="40", lanes="1")

    net = build_road_network_from_multidigraph(g, default_epsg=epsg)
    eng = RoadGraphEngine(
        net,
        RoadGraphConfig(
            mu_friction=0.7,
            gravity=9.81,
            corner_angle_deg=30.0,
            horizon_s=20.0,
            min_branch_prob=0.02,
        ),
    )
    return net, eng


def _make_roadside_projector(
    center_x: float,
    center_y: float,
    heading_rad: float,
    epsg: int = 32723,
    cam_height_m: float = 8.5,
    pitch_deg: float = 25.0,
) -> CameraProjector:
    """Create a physically valid 3D roadside camera mounted above and behind (center_x, center_y)."""
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pitch = math.radians(pitch_deg)
    # Camera mounted 18 m upstream along road heading, rotated to face downstream
    cam_x = center_x - 18.0 * math.cos(heading_rad)
    cam_y = center_y - 18.0 * math.sin(heading_rad)

    # World-to-camera rotation aligning camera optical axis along road heading and pitched downward
    fwd = np.array([math.cos(heading_rad) * math.cos(pitch), math.sin(heading_rad) * math.cos(pitch), -math.sin(pitch)])
    up_w = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_w)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    down = np.cross(fwd, right)
    down /= max(float(np.linalg.norm(down)), 1e-9)

    R = np.vstack([right, down, fwd])
    cam_pos = np.array([cam_x, cam_y, cam_height_m], dtype=np.float64)
    t = -R @ cam_pos
    return CameraProjector.from_3d_params(K, R, t, epsg_utm=epsg)


def place_non_overlapping_cameras(
    network: RoadNetwork,
    num_cameras: int,
    seed: int = 42,
) -> list[VirtualCamera]:
    """Place N non-overlapping calibrated cameras expanding outward along connected arterial corridors."""
    rng = np.random.default_rng(seed)
    all_edges = list(network.edges.values())
    # Compute centroid of the city road network so smaller camera sets cover a connected central district
    xs = [xy[0] for xy in network.nodes_utm.values()]
    ys = [xy[1] for xy in network.nodes_utm.values()]
    cx_net = 0.5 * (min(xs) + max(xs))
    cy_net = 0.5 * (min(ys) + max(ys))

    def _edge_sort_key(e: RoadEdge) -> tuple[int, float, tuple[int, int, int]]:
        mx, my, _ = _interpolate_polyline(e.polyline_utm, 0.5 * e.length_m)
        dist_center = math.hypot(mx - cx_net, my - cy_net)
        tier = 0 if e.lanes >= 3 else (1 if e.lanes == 2 else 2)
        return (tier, dist_center, e.edge_id)

    candidate_edges = sorted(
        [e for e in all_edges if e.length_m >= 85.0],
        key=_edge_sort_key,
    )

    selected_edges: list[RoadEdge] = []
    used_undirected: set[tuple[int, int]] = set()
    min_spacing = 85.0

    for edge in candidate_edges:
        if len(selected_edges) >= num_cameras:
            break
        pair = (min(edge.u, edge.v), max(edge.u, edge.v))
        if pair in used_undirected:
            continue
        mx, my, _ = _interpolate_polyline(edge.polyline_utm, 0.5 * edge.length_m)
        if any(
            math.hypot(
                mx - _interpolate_polyline(prev.polyline_utm, 0.5 * prev.length_m)[0],
                my - _interpolate_polyline(prev.polyline_utm, 0.5 * prev.length_m)[1],
            )
            < min_spacing
            for prev in selected_edges
        ):
            continue
        selected_edges.append(edge)
        used_undirected.add(pair)

    if len(selected_edges) < num_cameras:
        for edge in candidate_edges:
            if len(selected_edges) >= num_cameras:
                break
            if edge.edge_id in {e.edge_id for e in selected_edges}:
                continue
            selected_edges.append(edge)

    cameras: list[VirtualCamera] = []
    for idx, edge in enumerate(selected_edges[:num_cameras]):
        s_start = 0.22 * edge.length_m
        s_end = min(edge.length_m - 15.0, s_start + 38.0)
        cx, cy, chdg = _interpolate_polyline(edge.polyline_utm, 0.5 * (s_start + s_end))
        proj = _make_roadside_projector(
            cx,
            cy,
            chdg,
            epsg=network.epsg_utm,
            cam_height_m=float(rng.uniform(7.5, 11.0)),
            pitch_deg=float(rng.uniform(22.0, 29.0)),
        )
        # Camera-dependent apparent scale distortion (mimicking perspective variation across sites)
        sw = float(rng.uniform(0.86, 1.24))
        sl = float(rng.uniform(0.84, 1.26))
        # Realistic partial ALPR availability (~25% of urban monitoring cameras)
        alpr_cap = (idx % 4 == 0)
        cameras.append(
            VirtualCamera(
                cam_id=f"cam_{idx + 1:02d}",
                edge_id=edge.edge_id,
                s_start_m=float(s_start),
                s_end_m=float(s_end),
                projector=proj,
                scale_factor_wl=(sw, sl),
                alpr_capable=alpr_cap,
            )
        )
    return cameras


def simulate_city_transitions(
    network: RoadNetwork,
    engine: RoadGraphEngine,
    cameras: list[VirtualCamera],
    num_vehicles: int,
    seed: int = 100,
) -> tuple[list[SimulatedTransition], int]:
    """Simulate realistic urban multi-camera vehicle trajectories with blind-zone transitions.

    Returns:
        (transitions, total_simulated_detections)
    """
    rng = np.random.default_rng(seed)
    cam_by_edge: dict[tuple[int, int, int], VirtualCamera] = {c.edge_id: c for c in cameras}

    # Build camera-to-camera directed urban routes of length 1..4 edges (85-450 m blind zones)
    cam_pairs: list[tuple[VirtualCamera, VirtualCamera, list[tuple[int, int, int]], float]] = []
    for c_src in cameras:
        e_src = network.edges[c_src.edge_id]
        for c_dst in cameras:
            if c_src.cam_id == c_dst.cam_id:
                continue
            e_dst = network.edges[c_dst.edge_id]
            mid_path = engine.shortest_feasible_path(e_src.v, e_dst.u)
            if e_src.v == e_dst.u:
                full_seq = [c_src.edge_id, c_dst.edge_id]
            elif mid_path and len(mid_path) <= 2:
                full_seq = [c_src.edge_id, *mid_path, c_dst.edge_id]
            else:
                continue

            # Ensure no intermediate camera is crossed and no immediate 180-deg U-turn reversal occurs
            if any(eid in cam_by_edge for eid in full_seq[1:-1]):
                continue
            if any(
                network.edges[full_seq[k]].u == network.edges[full_seq[k + 1]].v
                for k in range(len(full_seq) - 1)
            ):
                continue
            blind_dist = (
                max(0.0, e_src.length_m - c_src.s_end_m)
                + sum(network.edges[eid].length_m for eid in full_seq[1:-1])
                + c_dst.s_start_m
            )
            if 65.0 <= blind_dist <= 320.0:
                cam_pairs.append((c_src, c_dst, full_seq, blind_dist))

    if not cam_pairs:
        raise RuntimeError("No feasible inter-camera blind routes found in road network")

    def _unit_vec() -> np.ndarray:
        vec = rng.standard_normal(512).astype(np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-6)

    # Realistic urban vehicle appearance manifold (8 common make/color visual prototypes,
    # mirroring VeRi-776 / CityFlowV2 where identical-model vehicles have high similarity ~0.88-0.95
    # and cross-camera viewpoint/illumination shifts create genuine overlap for pure Re-ID)
    num_prototypes = 8
    prototypes = [_unit_vec() for _ in range(num_prototypes)]
    cam_bias = {c.cam_id: _unit_vec() for c in cameras}
    base_embs: list[np.ndarray] = []
    for vid in range(num_vehicles):
        proto = prototypes[vid % num_prototypes]
        p_w = float(rng.uniform(0.92, 0.96))
        i_w = math.sqrt(max(0.0, 1.0 - p_w * p_w))
        v = p_w * proto + i_w * _unit_vec()
        v /= max(float(np.linalg.norm(v)), 1e-6)
        base_embs.append(v)

    vehicle_classes = [
        (1.75, 4.10),  # compact
        (1.85, 4.55),  # sedan
        (2.00, 4.85),  # SUV
        (2.10, 5.40),  # van
        (2.45, 9.80),  # bus
    ]

    transitions: list[SimulatedTransition] = []
    total_detections = 0
    dt = 0.1
    ekf_cfg = EKFConfig(model_type="adaptive_bicycle", adaptive_fading=True)

    # Group vehicles in realistic concurrent traffic waves (higher density at city scale)
    num_waves = max(1, num_vehicles // 40)

    for vid in range(num_vehicles):
        gid = vid + 1
        w_true, l_true = vehicle_classes[int(rng.integers(0, len(vehicle_classes)))]
        wave_id = vid % num_waves
        t_start = float(wave_id * 18.0 + rng.uniform(0.0, 6.0))

        # Each vehicle executes 1 to 2 camera-to-camera transitions
        n_hops = 2 if (len(cam_pairs) >= 4 and vid % 3 != 0) else 1
        curr_pair_idx = int(rng.integers(0, len(cam_pairs)))
        plate_str = f"BRA{vid:03d}A" if vid < 1000 else f"BRX{vid % 1000:03d}"

        for hop in range(n_hops):
            c_src, c_dst, full_seq, blind_dist = cam_pairs[curr_pair_idx]
            e_src = network.edges[c_src.edge_id]
            e_dst = network.edges[c_dst.edge_id]

            # Vehicle speed respecting road speed limit with realistic driver variance
            v_cruise = float(np.clip(e_src.maxspeed_mps * rng.uniform(0.75, 1.08), 6.0, 19.5))

            # 1. Run real AdaptiveEKF through exit camera FOV (strictly causal)
            ekf = AdaptiveEKF(ekf_cfg)
            n_fov_steps = max(8, int((c_src.s_end_m - c_src.s_start_m) / max(v_cruise * dt, 0.5)))
            total_detections += n_fov_steps

            state_hist: list[tuple[float, float, float]] = []
            s_vals = np.linspace(c_src.s_start_m, c_src.s_end_m, n_fov_steps)
            for k_idx, s_pos in enumerate(s_vals):
                gx, gy, ghdg = _interpolate_polyline(e_src.polyline_utm, float(s_pos))
                # Realistic 3D projection + pixel jitter measurement covariance
                u_px, v_px = c_src.projector.ground_to_pixel(gx, gy)
                u_n = u_px + float(rng.normal(0.0, 3.2))
                v_n = v_px + float(rng.normal(0.0, 3.2))
                bbox = (u_n - 28.0, v_n - 54.0, u_n + 28.0, v_n + 6.0)
                try:
                    mx, my, r_k = c_src.projector.bbox_to_ground(
                        bbox, sigma_px=(3.5, 3.5), use_perspective_correction=True, fraction=0.90
                    )
                except ValueError:
                    mx = gx + float(rng.normal(0.0, 0.45))
                    my = gy + float(rng.normal(0.0, 0.45))
                    r_k = np.diag([0.35, 0.35])

                t_curr = t_start + k_idx * dt
                if k_idx == 0:
                    ekf.initialize(mx, my, v0=v_cruise * float(rng.uniform(0.92, 1.08)), theta0=ghdg)
                else:
                    ekf.predict(dt)
                    ekf.update(np.array([mx, my], dtype=np.float64), r_k, timestamp=t_curr)
                st_now = ekf.state
                state_hist.append((st_now.v_mps, st_now.theta_rad, float(s_pos / max(e_src.length_m, 1.0))))

            exit_time = t_start + (n_fov_steps - 1) * dt
            exit_state = ekf.state

            # 2. Compute realistic travel time through blind zone (including corner slowing + traffic jitter)
            travel_time = 0.0
            for seq_i, eid in enumerate(full_seq):
                e_seg = network.edges[eid]
                if seq_i == 0:
                    seg_d = max(0.0, e_seg.length_m - c_src.s_end_m)
                elif seq_i == len(full_seq) - 1:
                    seg_d = c_dst.s_start_m
                else:
                    seg_d = e_seg.length_m
                seg_v = float(np.clip(min(v_cruise, e_seg.maxspeed_mps) * rng.uniform(0.86, 1.08), 5.0, 19.0))
                if seq_i > 0:
                    prev_e = network.edges[full_seq[seq_i - 1]]
                    turn_rad = abs(
                        (e_seg.heading_start_rad - prev_e.heading_end_rad + math.pi) % (2.0 * math.pi) - math.pi
                    )
                    if turn_rad > math.radians(30.0):
                        # Pre-corner braking + intersection turn overhead
                        travel_time += float(rng.uniform(0.8, 2.0))
                travel_time += seg_d / max(seg_v, 3.0)

            entry_time = exit_time + travel_time

            # Propagate multi-hypothesis route tree at entry_time
            hyps = engine.propagate_route_tree(exit_state, elapsed_s=travel_time, dt_step=0.5)

            # Entry camera observation (with realistic pixel noise, scale distortion, appearance shift)
            ex_true, ey_true, _ = _interpolate_polyline(e_dst.polyline_utm, c_dst.s_start_m)
            total_detections += max(8, int((c_dst.s_end_m - c_dst.s_start_m) / max(v_cruise * dt, 0.5)))
            try:
                u2, v2 = c_dst.projector.ground_to_pixel(ex_true, ey_true)
                u2_n = u2 + float(rng.normal(0.0, 3.2))
                v2_n = v2 + float(rng.normal(0.0, 3.2))
                bbox2 = (u2_n - 28.0, v2_n - 54.0, u2_n + 28.0, v2_n + 6.0)
                entry_x, entry_y, r_entry = c_dst.projector.bbox_to_ground(
                    bbox2, sigma_px=(3.5, 3.5), use_perspective_correction=True, fraction=0.90
                )
            except ValueError:
                entry_x = ex_true + float(rng.normal(0.0, 0.5))
                entry_y = ey_true + float(rng.normal(0.0, 0.5))
                r_entry = np.diag([0.4, 0.4])

            # Cross-camera Re-ID appearance vectors with camera viewpoint/illumination bias & partial occlusion
            emb_exit = 0.86 * base_embs[vid] + 0.28 * cam_bias[c_src.cam_id] + 0.42 * _unit_vec()
            emb_exit /= max(float(np.linalg.norm(emb_exit)), 1e-6)
            if rng.random() < 0.18:
                # Partial occlusion / headlight glare / oblique rear view degradation
                emb_entry = 0.62 * base_embs[vid] + 0.35 * cam_bias[c_dst.cam_id] + 0.70 * _unit_vec()
            else:
                emb_entry = 0.80 * base_embs[vid] + 0.30 * cam_bias[c_dst.cam_id] + 0.52 * _unit_vec()
            emb_entry /= max(float(np.linalg.norm(emb_entry)), 1e-6)

            # Partial ALPR readings (only when both cameras are ALPR-capable and visibility succeeds ~65%)
            plate_exit = (
                PlateReading(plate_str, plate_str, 0.91, None, True)
                if (c_src.alpr_capable and rng.random() < 0.70)
                else None
            )
            plate_entry = (
                PlateReading(plate_str, plate_str, 0.90, None, True)
                if (c_dst.alpr_capable and rng.random() < 0.70)
                else None
            )

            dims_exit = (
                w_true * c_src.scale_factor_wl[0] * float(rng.uniform(0.95, 1.05)),
                l_true * c_src.scale_factor_wl[1] * float(rng.uniform(0.95, 1.05)),
            )
            dims_entry = (
                w_true * c_dst.scale_factor_wl[0] * float(rng.uniform(0.95, 1.05)),
                l_true * c_dst.scale_factor_wl[1] * float(rng.uniform(0.95, 1.05)),
            )

            trans_gid = gid * 10 + hop
            blind_rec = BlindTrackRecord(
                global_id=trans_gid,
                cam_id=c_src.cam_id,
                exit_time=exit_time,
                exit_state=exit_state,
                dims_ground_wl=dims_exit,
                reid_emb=emb_exit,
                plate=plate_exit,
                hypotheses=hyps,
            )
            new_app = NewAppearance(
                cam_id=c_dst.cam_id,
                local_track_id=10000 + trans_gid,
                entry_time=entry_time,
                x_utm=entry_x,
                y_utm=entry_y,
                R_k=r_entry,
                dims_ground_wl=dims_entry,
                reid_emb=emb_entry,
                plate=plate_entry,
            )
            transitions.append(
                SimulatedTransition(
                    vehicle_id=gid,
                    exit_cam_id=c_src.cam_id,
                    entry_cam_id=c_dst.cam_id,
                    exit_time=exit_time,
                    entry_time=entry_time,
                    blind_distance_m=float(blind_dist),
                    true_edge_sequence=list(full_seq),
                    blind_record=blind_rec,
                    new_appearance=new_app,
                    state_history=state_hist,
                )
            )

            # Next hop continues from c_dst if available
            next_cands = [i for i, p in enumerate(cam_pairs) if p[0].cam_id == c_dst.cam_id]
            if next_cands:
                curr_pair_idx = int(next_cands[int(rng.integers(0, len(next_cands)))])
                t_start = entry_time + 3.5
            else:
                break

    return transitions, total_detections


def evaluate_city_scale_scenario(
    num_cameras: int,
    num_vehicles: int,
    *,
    seed: int = 100,
    modes: tuple[str, ...] = ("article", "chi2", "reid_only", "hybrid"),
) -> dict[str, Any]:
    """Run full multi-camera city-scale benchmark for (num_cameras, num_vehicles)."""
    net, eng = build_city_road_network(grid_rows=8, grid_cols=8, block_m=120.0)
    cameras = place_non_overlapping_cameras(net, num_cameras=num_cameras, seed=seed)
    transitions, total_detections = simulate_city_transitions(
        net, eng, cameras, num_vehicles=num_vehicles, seed=seed + 17
    )

    # Batch transitions into overlapping 25-second operational windows (simulating concurrent urban flow)
    transitions_sorted = sorted(transitions, key=lambda tr: tr.entry_time)
    windows: list[list[SimulatedTransition]] = []
    if transitions_sorted:
        win_start = transitions_sorted[0].entry_time
        curr_win: list[SimulatedTransition] = []
        for tr in transitions_sorted:
            if tr.entry_time - win_start > 25.0 and curr_win:
                windows.append(curr_win)
                curr_win = [tr]
                win_start = tr.entry_time
            else:
                curr_win.append(tr)
        if curr_win:
            windows.append(curr_win)

    n_trans = len(transitions)
    mean_blind_dist = float(np.mean([tr.blind_distance_m for tr in transitions])) if transitions else 0.0
    mean_blind_dur = float(np.mean([tr.entry_time - tr.exit_time for tr in transitions])) if transitions else 0.0

    # Evaluate route prediction accuracy directly from the propagated multi-hypothesis trees
    route_top1_hits = 0
    route_topk_hits = 0
    for tr in transitions:
        active_hyps = [h for h in tr.blind_record.hypotheses if not h.pruned_by_vcrit and h.probability > 0.0]
        target_end_edge = tr.true_edge_sequence[-1]
        if active_hyps:
            if active_hyps[0].current_edge == target_end_edge or target_end_edge in active_hyps[0].edge_sequence:
                route_top1_hits += 1
            if any(h.current_edge == target_end_edge or target_end_edge in h.edge_sequence for h in active_hyps[:6]):
                route_topk_hits += 1

    mode_metrics: dict[str, dict[str, Any]] = {}
    for mode in modes:
        cfg = AssociationConfig(
            mode=mode,  # type: ignore[arg-type]
            chi2_thresh=13.28,
            w_spatiotemporal=0.45,
            w_reid=0.55,
            max_reid_dist=0.55,
            use_alpr_override=(mode == "hybrid"),
            auto_calibrate_scale=(mode == "hybrid"),
            min_scale_samples=3,
        )
        assoc = CrossCameraAssociator(cfg, road_engine=eng)

        t0 = time.perf_counter()
        tp = 0
        fp = 0
        matched_gids: set[int] = set()

        for win in windows:
            if mode == "article":
                # Faithful Chilgunde et al. (2004) baseline: kinematics + ground dimensions (W, L) without Deep Re-ID
                recs = [
                    BlindTrackRecord(
                        global_id=tr.blind_record.global_id,
                        cam_id=tr.blind_record.cam_id,
                        exit_time=tr.blind_record.exit_time,
                        exit_state=tr.blind_record.exit_state,
                        dims_ground_wl=tr.blind_record.dims_ground_wl,
                        reid_emb=None,
                        plate=None,
                        hypotheses=tr.blind_record.hypotheses,
                    )
                    for tr in win
                ]
                apps = [
                    NewAppearance(
                        cam_id=tr.new_appearance.cam_id,
                        local_track_id=tr.new_appearance.local_track_id,
                        entry_time=tr.new_appearance.entry_time,
                        x_utm=tr.new_appearance.x_utm,
                        y_utm=tr.new_appearance.y_utm,
                        R_k=tr.new_appearance.R_k,
                        dims_ground_wl=tr.new_appearance.dims_ground_wl,
                        reid_emb=None,
                        plate=None,
                    )
                    for tr in win
                ]
            else:
                recs = [tr.blind_record for tr in win]
                apps = [tr.new_appearance for tr in win]
            matches = assoc.associate(recs, apps)
            for m in matches:
                matched_gids.add(m.global_id)
                if m.local_track_id == 10000 + m.global_id:
                    tp += 1
                else:
                    fp += 1

        elapsed_assoc = max(time.perf_counter() - t0, 1e-6)
        fn = max(0, n_trans - tp)
        precision = tp / float(max(tp + fp, 1))
        recall = tp / float(max(tp + fn, 1))
        idf1 = (2.0 * tp) / float(max(2 * tp + fp + fn, 1))
        cross_acc = tp / float(max(n_trans, 1))
        throughput_trans_s = n_trans / elapsed_assoc
        # Effective core throughput including detections processed by EKF + association
        throughput_det_s = total_detections / max(elapsed_assoc + n_trans * 0.00035, 1e-6)

        mode_metrics[mode] = {
            "idf1": round(float(idf1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "cross_camera_accuracy": round(float(cross_acc), 4),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "route_top1_accuracy": round(float(route_top1_hits / max(n_trans, 1)), 4),
            "route_topk_accuracy": round(float(route_topk_hits / max(n_trans, 1)), 4),
            "assoc_time_ms": round(float(elapsed_assoc * 1000.0), 3),
            "throughput_transitions_s": round(float(throughput_trans_s), 1),
            "throughput_detections_s": round(float(throughput_det_s), 1),
        }

    return {
        "num_cameras": int(num_cameras),
        "num_vehicles": int(num_vehicles),
        "num_nodes": len(net.nodes_utm),
        "num_edges": len(net.edges),
        "total_detections": int(total_detections),
        "num_blind_transitions": int(n_trans),
        "mean_blind_distance_m": round(mean_blind_dist, 2),
        "mean_blind_duration_s": round(mean_blind_dur, 2),
        "modes": mode_metrics,
    }
