"""Cross-camera hybrid association (Spatiotemporal Chi2 + Deep Re-ID + ALPR override), scale self-calibration, and forensic backtracking."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.alpr import PlateReading
from tracker.config import AssociationConfig
from tracker.ekf import FilterState, _enforce_spd
from tracker.reid import cosine_distance
from tracker.roadgraph import BlindHypothesis, RoadGraphEngine


@dataclass
class BlindTrackRecord:
    """Record of a globally tracked vehicle that exited a camera into a blind zone."""

    global_id: int
    cam_id: str
    exit_time: float
    exit_state: FilterState
    dims_ground_wl: tuple[float, float]
    reid_emb: np.ndarray | None
    plate: PlateReading | None
    hypotheses: list[BlindHypothesis]


@dataclass
class NewAppearance:
    """Newly observed local track in a downstream camera candidate for cross-camera re-identification."""

    cam_id: str
    local_track_id: int
    entry_time: float
    x_utm: float
    y_utm: float
    R_k: np.ndarray
    dims_ground_wl: tuple[float, float]
    reid_emb: np.ndarray | None
    plate: PlateReading | None


@dataclass
class ReconstructedBlindPath:
    """Forensically backtracked route traversed by a re-identified vehicle across a blind zone."""

    edge_ids: list[tuple[int, int, int]]
    node_ids: list[int]
    polyline_utm: np.ndarray
    duration_s: float
    confidence: float
    pruned_alternatives: int


@dataclass
class AssociationMatch:
    """Confirmed cross-camera identity match between a blind track and a new local appearance."""

    global_id: int
    local_track_id: int
    score: float
    mode_used: str
    forensic_path: ReconstructedBlindPath | None


class CrossCameraAssociator:
    """Multi-camera vehicle associator with median scale self-calibration and forensic backtracking."""

    def __init__(
        self,
        cfg: AssociationConfig | None = None,
        road_engine: RoadGraphEngine | None = None,
    ) -> None:
        self.cfg = cfg if cfg is not None else AssociationConfig()
        self.road_engine = road_engine
        self._scale_samples: dict[tuple[str, str], list[tuple[float, float]]] = {}

    def update_scale_calibration(
        self,
        cam_i: str,
        wl_i: tuple[float, float],
        cam_j: str,
        wl_j: tuple[float, float],
    ) -> None:
        """Accumulate ground projected (W, L) ratios between camera pair (cam_i -> cam_j)."""
        if not self.cfg.auto_calibrate_scale or cam_i == cam_j:
            return
        wi, li = float(wl_i[0]), float(wl_i[1])
        wj, lj = float(wl_j[0]), float(wl_j[1])
        if not all(math.isfinite(v) and v > 1e-3 for v in (wi, li, wj, lj)):
            return

        rw = float(np.clip(wj / wi, 0.1, 10.0))
        rl = float(np.clip(lj / li, 0.1, 10.0))
        self._scale_samples.setdefault((cam_i, cam_j), []).append((rw, rl))
        self._scale_samples.setdefault((cam_j, cam_i), []).append(
            (float(np.clip(1.0 / rw, 0.1, 10.0)), float(np.clip(1.0 / rl, 0.1, 10.0)))
        )

    def get_scale_factors(self, cam_i: str, cam_j: str) -> tuple[float, float]:
        """Return median scale calibration factors (C_w^{ij}, C_l^{ij}) or (1.0, 1.0) if uncalibrated."""
        if not self.cfg.auto_calibrate_scale or cam_i == cam_j:
            return (1.0, 1.0)
        samples = self._scale_samples.get((cam_i, cam_j), [])
        if len(samples) < int(self.cfg.min_scale_samples):
            return (1.0, 1.0)
        arr = np.asarray(samples, dtype=np.float64)
        cw = float(np.clip(np.median(arr[:, 0]), 0.1, 10.0))
        cl = float(np.clip(np.median(arr[:, 1]), 0.1, 10.0))
        return (cw, cl)

    def _best_hypothesis_and_chi2(
        self,
        rec: BlindTrackRecord,
        app: NewAppearance,
    ) -> tuple[BlindHypothesis | None, float]:
        """Find the hypothesis in rec.hypotheses with minimum Mahalanobis distance to app."""
        z_app = np.array([float(app.x_utm), float(app.y_utm)], dtype=np.float64)
        r_app = _enforce_spd(np.asarray(app.R_k, dtype=np.float64), min_eig=1e-4)

        best_hyp: BlindHypothesis | None = None
        best_d2 = float("inf")

        active_hyps = [h for h in rec.hypotheses if not h.pruned_by_vcrit]
        dt = max(0.1, float(app.entry_time) - float(rec.exit_time))
        for h in active_hyps:
            mu = np.array([float(h.x_utm), float(h.y_utm)], dtype=np.float64)
            s_cov = _enforce_spd(np.asarray(h.P_2x2, dtype=np.float64) + r_app, min_eig=1e-4)
            diff = z_app - mu
            d2 = float(diff.T @ np.linalg.inv(s_cov) @ diff)

            # If a road engine is attached, also evaluate curvilinear (along-edge vs cross-edge) Mahalanobis distance
            if self.road_engine is not None and h.current_edge in self.road_engine.network.edges:
                from tracker.roadgraph import _project_point_to_polyline

                edge_obj = self.road_engine.network.edges[h.current_edge]
                lat_dist, s_app, _, _, _ = _project_point_to_polyline(
                    float(app.x_utm), float(app.y_utm), edge_obj.polyline_utm
                )
                lon_diff = abs(float(s_app) - float(h.s_m))
                v_ref = max(float(rec.exit_state.v_mps), float(h.v_mps), 10.0)
                sigma_lon = max(6.0, 0.55 * v_ref * dt)
                sigma_lat = max(3.0, math.sqrt(float(np.trace(r_app)) + 4.0))
                d2_curv = (lon_diff / sigma_lon) ** 2 + (lat_dist / sigma_lat) ** 2
                d2 = min(d2, float(d2_curv))

            if d2 < best_d2:
                best_d2 = d2
                best_hyp = h

        if best_hyp is None:
            # Fallback to kinematic propagation from exit_state over dt
            dt = max(0.0, float(app.entry_time) - float(rec.exit_time))
            ex = rec.exit_state
            pred_x = ex.x_utm + ex.v_mps * math.cos(ex.theta_rad) * dt
            pred_y = ex.y_utm + ex.v_mps * math.sin(ex.theta_rad) * dt
            s_cov = _enforce_spd(
                ex.P_pos_2x2 + r_app + np.diag([max(4.0, 2.0 * dt) ** 2] * 2),
                min_eig=1e-4,
            )
            diff = z_app - np.array([pred_x, pred_y], dtype=np.float64)
            best_d2 = float(diff.T @ np.linalg.inv(s_cov) @ diff)

        return best_hyp, float(best_d2)

    def _reconstruct_forensic_path(
        self,
        rec: BlindTrackRecord,
        app: NewAppearance,
        best_hyp: BlindHypothesis | None,
    ) -> ReconstructedBlindPath | None:
        """Backtrack and reconstruct the continuous road path traversed in the blind zone."""
        duration_s = max(0.1, float(app.entry_time) - float(rec.exit_time))
        total_hyps = len(rec.hypotheses)
        pruned_alt = max(0, total_hyps - 1)

        edge_ids: list[tuple[int, int, int]] = []
        if best_hyp is not None and best_hyp.edge_sequence:
            edge_ids = list(best_hyp.edge_sequence)

        if self.road_engine is not None:
            net = self.road_engine.network
            try:
                e_start, _ = self.road_engine.snap_initial_heading(
                    rec.exit_state.x_utm, rec.exit_state.y_utm
                )
                e_end, _ = self.road_engine.snap_initial_heading(app.x_utm, app.y_utm)
            except Exception:
                e_start = None
                e_end = None

            # Ensure edge_ids connects cleanly from exit to entry on the road network
            if not edge_ids and e_start is not None and e_end is not None:
                if e_start == e_end:
                    edge_ids = [e_start]
                else:
                    mid = self.road_engine.shortest_feasible_path(
                        net.edges[e_start].v, net.edges[e_end].u
                    )
                    if mid or net.edges[e_start].v == net.edges[e_end].u:
                        edge_ids = [e_start, *mid, e_end]
                    else:
                        full_p = self.road_engine.shortest_feasible_path(
                            net.edges[e_start].u, net.edges[e_end].v
                        )
                        edge_ids = full_p if full_p else [e_start]

            # Validate topological contiguity of edge_ids if edges exist in network
            if edge_ids and all(eid in net.edges for eid in edge_ids):
                contiguous: list[tuple[int, int, int]] = [edge_ids[0]]
                for nxt_eid in edge_ids[1:]:
                    if nxt_eid == contiguous[-1]:
                        continue
                    u_prev_v = net.edges[contiguous[-1]].v
                    v_nxt_u = net.edges[nxt_eid].u
                    if u_prev_v == v_nxt_u:
                        contiguous.append(nxt_eid)
                    else:
                        bridge = self.road_engine.shortest_feasible_path(u_prev_v, v_nxt_u)
                        if bridge:
                            contiguous.extend(bridge)
                            contiguous.append(nxt_eid)
                edge_ids = contiguous

        if not edge_ids:
            edge_ids = [(0, 1, 0)]

        node_ids: list[int] = [int(edge_ids[0][0])]
        for eid in edge_ids:
            node_ids.append(int(eid[1]))

        poly_pts: list[np.ndarray] = []
        if self.road_engine is not None:
            net = self.road_engine.network
            for idx, eid in enumerate(edge_ids):
                if eid in net.edges:
                    epoly = net.edges[eid].polyline_utm
                    poly_pts.append(epoly if idx == 0 else epoly[1:])
        if poly_pts:
            polyline_utm = np.vstack(poly_pts).astype(np.float64)
        else:
            polyline_utm = np.array(
                [
                    [float(rec.exit_state.x_utm), float(rec.exit_state.y_utm)],
                    [float(app.x_utm), float(app.y_utm)],
                ],
                dtype=np.float64,
            )

        conf = float(best_hyp.probability) if (best_hyp is not None and best_hyp.probability > 0.0) else 0.85
        conf = float(np.clip(conf, 0.05, 1.0))

        return ReconstructedBlindPath(
            edge_ids=edge_ids,
            node_ids=node_ids,
            polyline_utm=polyline_utm,
            duration_s=duration_s,
            confidence=conf,
            pruned_alternatives=pruned_alt,
        )

    def associate(
        self,
        blind_tracks: list[BlindTrackRecord],
        new_appearances: list[NewAppearance],
    ) -> list[AssociationMatch]:
        """Perform 1-to-1 cross-camera association across blind tracks and new appearances."""
        if not blind_tracks or not new_appearances:
            return []

        n_rec = len(blind_tracks)
        n_app = len(new_appearances)
        INF_COST = 1e9
        cost_matrix = np.full((n_rec, n_app), INF_COST, dtype=np.float64)
        mode_matrix: list[list[str]] = [[self.cfg.mode for _ in range(n_app)] for _ in range(n_rec)]
        hyp_matrix: list[list[BlindHypothesis | None]] = [[None for _ in range(n_app)] for _ in range(n_rec)]

        for i, rec in enumerate(blind_tracks):
            for j, app in enumerate(new_appearances):
                dt = float(app.entry_time) - float(rec.exit_time)
                if dt < 0.0:
                    continue

                # 1. Check ALPR high-confidence override & veto
                alpr_matched = False
                if self.cfg.use_alpr_override and rec.plate is not None and app.plate is not None:
                    rec_ok = (
                        rec.plate.format_valid
                        and rec.plate.confidence >= float(self.cfg.alpr_override_conf)
                        and bool(rec.plate.normalized_text)
                    )
                    app_ok = (
                        app.plate.format_valid
                        and app.plate.confidence >= float(self.cfg.alpr_override_conf)
                        and bool(app.plate.normalized_text)
                    )
                    if rec_ok and app_ok:
                        if rec.plate.normalized_text != app.plate.normalized_text:
                            # Deterministic plate veto
                            continue
                        alpr_matched = True

                best_hyp, d2_chi2 = self._best_hypothesis_and_chi2(rec, app)
                hyp_matrix[i][j] = best_hyp

                if alpr_matched:
                    cost_matrix[i, j] = 1e-4 + 1e-6 * min(d2_chi2, 100.0)
                    mode_matrix[i][j] = "alpr_override"
                    continue

                # 2. Compute Deep Re-ID cosine distance
                has_reid = rec.reid_emb is not None and app.reid_emb is not None
                if has_reid:
                    assert rec.reid_emb is not None and app.reid_emb is not None
                    d_reid = cosine_distance(rec.reid_emb, app.reid_emb)
                else:
                    d_reid = 0.25

                # 3. Compute scale-calibrated ground dimension distance (Chilgunde et al., 2004)
                cw, cl = self.get_scale_factors(rec.cam_id, app.cam_id)
                w_exp = max(float(rec.dims_ground_wl[0]) * cw, 0.5)
                l_exp = max(float(rec.dims_ground_wl[1]) * cl, 0.5)
                d_dim = (
                    abs(float(app.dims_ground_wl[0]) - w_exp) / w_exp
                    + abs(float(app.dims_ground_wl[1]) - l_exp) / l_exp
                )

                mode = self.cfg.mode
                chi2_lim = float(self.cfg.chi2_thresh)
                reid_lim = float(self.cfg.max_reid_dist)

                if mode == "chi2":
                    if d2_chi2 > chi2_lim:
                        continue
                    cost = d2_chi2 / max(chi2_lim, 1e-6)
                elif mode == "reid_only":
                    if has_reid and d_reid > reid_lim:
                        continue
                    cost = d_reid
                elif mode == "article":
                    if d2_chi2 > chi2_lim * 2.0:
                        continue
                    cost = 0.45 * (d2_chi2 / max(chi2_lim, 1e-6)) + 0.35 * d_dim + 0.20 * d_reid
                else:  # "hybrid"
                    if has_reid and d_reid > reid_lim:
                        continue
                    if d2_chi2 > chi2_lim * 2.0:
                        continue
                    norm_chi2 = d2_chi2 / max(chi2_lim, 1e-6)
                    norm_reid = d_reid / max(reid_lim, 1e-6)
                    cost = (
                        float(self.cfg.w_spatiotemporal) * norm_chi2
                        + float(self.cfg.w_reid) * norm_reid
                        + 0.15 * d_dim
                    )

                cost_matrix[i, j] = max(0.0, float(cost))
                mode_matrix[i][j] = mode

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matches: list[AssociationMatch] = []

        for r_idx, c_idx in zip(row_ind, col_ind):
            c_val = float(cost_matrix[r_idx, c_idx])
            if c_val >= INF_COST * 0.5:
                continue
            rec = blind_tracks[int(r_idx)]
            app = new_appearances[int(c_idx)]
            best_hyp = hyp_matrix[int(r_idx)][int(c_idx)]
            f_path = self._reconstruct_forensic_path(rec, app, best_hyp)

            # Update scale self-calibration upon confirmed match
            self.update_scale_calibration(
                rec.cam_id, rec.dims_ground_wl, app.cam_id, app.dims_ground_wl
            )

            matches.append(
                AssociationMatch(
                    global_id=int(rec.global_id),
                    local_track_id=int(app.local_track_id),
                    score=c_val,
                    mode_used=mode_matrix[int(r_idx)][int(c_idx)],
                    forensic_path=f_path,
                )
            )

        matches.sort(key=lambda m: (m.global_id, m.local_track_id))
        return matches
