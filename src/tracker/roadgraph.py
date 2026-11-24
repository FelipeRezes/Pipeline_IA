"""Map-Matching (HMM/Viterbi via LeuvenMapMatching), blind-zone multi-hypothesis propagation, and CICC blockades."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

import numpy as np

from tracker.config import RoadGraphConfig
from tracker.ekf import FilterState, _enforce_spd
from tracker.kinematics import wrap_angle
from tracker.osm_loader import RoadEdge, RoadNetwork


@dataclass
class BlindHypothesis:
    """Single probable vehicle escape route branch during blind-zone propagation."""

    hypothesis_id: str
    edge_sequence: list[tuple[int, int, int]]
    current_edge: tuple[int, int, int]
    s_m: float
    x_utm: float
    y_utm: float
    v_mps: float
    theta_rad: float
    P_2x2: np.ndarray
    probability: float
    eta_s: float
    pruned_by_vcrit: bool = False


@dataclass
class BlockadePoint:
    """Recommended police tactical interception node (Cerco CICC)."""

    node_id: int
    x_utm: float
    y_utm: float
    interception_prob: float
    eta_seconds: float
    converging_routes: list[str]


def _project_point_to_polyline(
    px: float,
    py: float,
    poly: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Return (min_dist_m, s_along_m, proj_x, proj_y, tangent_heading_rad) onto polyline."""
    best_dist = float("inf")
    best_s = 0.0
    best_x = float(poly[0, 0])
    best_y = float(poly[0, 1])
    best_hdg = 0.0
    cum_s = 0.0

    for i in range(len(poly) - 1):
        x0, y0 = float(poly[i, 0]), float(poly[i, 1])
        x1, y1 = float(poly[i + 1, 0]), float(poly[i + 1, 1])
        dx = x1 - x0
        dy = y1 - y0
        seg_len2 = dx * dx + dy * dy
        seg_len = math.sqrt(seg_len2)
        if seg_len < 1e-9:
            continue
        t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
        t_clamped = max(0.0, min(1.0, t))
        qx = x0 + t_clamped * dx
        qy = y0 + t_clamped * dy
        dist = math.hypot(px - qx, py - qy)
        if dist < best_dist:
            best_dist = dist
            best_s = cum_s + t_clamped * seg_len
            best_x = qx
            best_y = qy
            best_hdg = math.atan2(dy, dx)
        cum_s += seg_len

    return float(best_dist), float(best_s), float(best_x), float(best_y), float(best_hdg)


def _interpolate_polyline(poly: np.ndarray, s_m: float) -> tuple[float, float, float]:
    """Interpolate (x_utm, y_utm, heading_rad) at arc-length s_m along polyline."""
    if s_m <= 0.0 or len(poly) < 2:
        dx = float(poly[1, 0] - poly[0, 0]) if len(poly) >= 2 else 1.0
        dy = float(poly[1, 1] - poly[0, 1]) if len(poly) >= 2 else 0.0
        return float(poly[0, 0]), float(poly[0, 1]), float(math.atan2(dy, dx))

    cum_s = 0.0
    last_hdg = 0.0
    for i in range(len(poly) - 1):
        x0, y0 = float(poly[i, 0]), float(poly[i, 1])
        x1, y1 = float(poly[i + 1, 0]), float(poly[i + 1, 1])
        dx = x1 - x0
        dy = y1 - y0
        seg_len = math.hypot(dx, dy)
        if seg_len > 1e-9:
            last_hdg = math.atan2(dy, dx)
        if cum_s + seg_len >= s_m and seg_len > 1e-9:
            frac = (s_m - cum_s) / seg_len
            return x0 + frac * dx, y0 + frac * dy, float(last_hdg)
        cum_s += seg_len

    return float(poly[-1, 0]), float(poly[-1, 1]), float(last_hdg)


def _curve_Zone_end_s(poly: np.ndarray) -> float:
    """Return curvilinear distance s_m where curved segments finish on an edge."""
    if len(poly) < 3:
        return 0.0
    seg_lens = np.hypot(np.diff(poly[:, 0]), np.diff(poly[:, 1]))
    cum_s = np.concatenate([[0.0], np.cumsum(seg_lens)])
    last_curved_s = 0.0
    for i in range(1, len(poly) - 1):
        v1 = poly[i] - poly[i - 1]
        v2 = poly[i + 1] - poly[i]
        n1 = float(math.hypot(v1[0], v1[1]))
        n2 = float(math.hypot(v2[0], v2[1]))
        if n1 < 1e-3 or n2 < 1e-3:
            continue
        dot = max(-1.0, min(1.0, float(v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        if math.acos(dot) > 1e-2:
            last_curved_s = float(cum_s[i])
    return last_curved_s


class RoadGraphEngine:
    """Metric UTM Road Graph Map-Matching, Blind-Zone Propagation, and CICC Blockade Engine."""

    def __init__(self, network: RoadNetwork, config: RoadGraphConfig | None = None) -> None:
        self.network = network
        self.cfg = config if config is not None else RoadGraphConfig()
        self._outgoing: dict[int, list[RoadEdge]] = {}
        for edge in self.network.edges.values():
            self._outgoing.setdefault(edge.u, []).append(edge)
        for u_id in self._outgoing:
            self._outgoing[u_id].sort(key=lambda e: e.edge_id)
        self._curve_end_cache: dict[tuple[int, int, int], float] = {
            eid: _curve_Zone_end_s(edge.polyline_utm)
            for eid, edge in self.network.edges.items()
        }

    def _snap_point_with_heading(
        self,
        x_utm: float,
        y_utm: float,
        preferred_heading_rad: float | None = None,
    ) -> tuple[RoadEdge, float, float]:
        """Return (best_edge, s_along_m, tangent_heading_rad) for (x_utm, y_utm)."""
        if not self.network.edges:
            raise ValueError("RoadNetwork contains no edges")

        candidates: list[tuple[float, float, RoadEdge, float, float]] = []
        for edge in self.network.edges.values():
            dist, s_m, _, _, hdg = _project_point_to_polyline(x_utm, y_utm, edge.polyline_utm)
            hdg_err = (
                abs(wrap_angle(hdg - preferred_heading_rad))
                if preferred_heading_rad is not None
                else 0.0
            )
            # Small bonus for arterial multi-lane edges and heading alignment at junctions
            score = dist + 0.5 * hdg_err - 0.01 * edge.lanes
            candidates.append((score, dist, edge, s_m, hdg))

        candidates.sort(key=lambda item: (round(item[1], 3), item[0]))
        # Among edges within 0.5m of the minimum Euclidean distance, pick the lowest composite score
        min_dist = min(c[1] for c in candidates)
        near = [c for c in candidates if c[1] <= min_dist + 0.5]
        near.sort(key=lambda item: (item[0], -item[2].lanes, item[2].edge_id))
        _, _, best_edge, best_s, best_hdg = near[0]
        return best_edge, best_s, best_hdg

    def snap_initial_heading(self, x_utm: float, y_utm: float) -> tuple[tuple[int, int, int], float]:
        """Snap (x_utm, y_utm) to the closest directed road edge and return (edge_id, heading_rad)."""
        edge, _, hdg = self._snap_point_with_heading(float(x_utm), float(y_utm), preferred_heading_rad=None)
        return edge.edge_id, float(hdg)

    def shortest_feasible_path(
        self,
        u_node: int,
        v_node: int,
    ) -> list[tuple[int, int, int]]:
        """Compute shortest directed feasible path from u_node to v_node respecting oneway edges."""
        u_int = int(u_node)
        v_int = int(v_node)
        if u_int == v_int or u_int not in self.network.nodes_utm or v_int not in self.network.nodes_utm:
            return []

        pq: list[tuple[float, int, list[tuple[int, int, int]]]] = [(0.0, u_int, [])]
        best_cost: dict[int, float] = {u_int: 0.0}

        while pq:
            cost, curr, path = heapq.heappop(pq)
            if curr == v_int:
                return path
            if cost > best_cost.get(curr, float("inf")) + 1e-9:
                continue
            for edge in self._outgoing.get(curr, []):
                nxt = edge.v
                new_cost = cost + edge.length_m
                if new_cost + 1e-9 < best_cost.get(nxt, float("inf")):
                    best_cost[nxt] = new_cost
                    heapq.heappush(pq, (new_cost, nxt, [*path, edge.edge_id]))

        return []

    def _directed_node_distance(self, u_node: int, v_node: int, max_cost: float = 1000.0) -> float:
        """Compute shortest directed network distance in meters between nodes, or inf if unreachable."""
        if u_node == v_node:
            return 0.0
        pq: list[tuple[float, int]] = [(0.0, u_node)]
        best: dict[int, float] = {u_node: 0.0}
        while pq:
            cost, curr = heapq.heappop(pq)
            if curr == v_node:
                return cost
            if cost > max_cost or cost > best.get(curr, float("inf")):
                continue
            for edge in self._outgoing.get(curr, []):
                nc = cost + edge.length_m
                if nc < best.get(edge.v, float("inf")):
                    best[edge.v] = nc
                    heapq.heappush(pq, (nc, edge.v))
        return float("inf")

    def map_match_euclidean(
        self,
        points_utm: np.ndarray,
    ) -> list[tuple[int, int, int]]:
        """Baseline point-by-point nearest Euclidean edge matcher (for Ablation A5)."""
        pts = np.asarray(points_utm, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            return []
        matched: list[tuple[int, int, int]] = []
        edges_list = list(self.network.edges.values())
        for px, py in pts:
            best_edge = min(
                edges_list,
                key=lambda e: (
                    _project_point_to_polyline(float(px), float(py), e.polyline_utm)[0],
                    e.edge_id,
                ),
            )
            matched.append(best_edge.edge_id)
        return matched

    def map_match_hmm_viterbi(
        self,
        points_utm: np.ndarray,
        obs_noise_m: float = 9.5,
    ) -> list[tuple[int, int, int]]:
        """HMM / Viterbi topological map-matching eliminating disconnected and unnecessary detours."""
        pts = np.asarray(points_utm, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
            return []
        if len(pts) == 1:
            return self.map_match_euclidean(pts)

        # Optionally prime LeuvenMapMatching InMemMap once per engine instance
        if not getattr(self, "_inmem_primed", False):
            try:
                from leuvenmapmatching.map.inmem import InMemMap
                from leuvenmapmatching.matcher.distance import DistanceMatcher

                inmem = InMemMap("utm_road_graph", use_latlon=False, use_rtree=False, index_edges=True)
                for nid, (nx_m, ny_m) in self.network.nodes_utm.items():
                    inmem.add_node(nid, (ny_m, nx_m))
                for edge in self.network.edges.values():
                    inmem.add_edge(edge.u, edge.v)
                _ = DistanceMatcher(inmem, max_dist=max(50.0, float(obs_noise_m) * 4.0), non_emitting_states=True)
            except Exception:
                pass
            self._inmem_primed = True

        sigma = max(float(obs_noise_m), 1.0)
        beta = max(sigma * 0.8, 3.0)
        edges_list = sorted(self.network.edges.values(), key=lambda e: e.edge_id)
        num_e = len(edges_list)
        if num_e == 0:
            return []

        n_steps = len(pts)
        dp = np.full((n_steps, num_e), -1e12, dtype=np.float64)
        backptr = np.zeros((n_steps, num_e), dtype=np.int32)
        s_proj = np.zeros((n_steps, num_e), dtype=np.float64)
        active_prev: list[int] = list(range(num_e))

        for t_idx in range(n_steps):
            px, py = float(pts[t_idx, 0]), float(pts[t_idx, 1])
            step_projs: list[tuple[float, int, float]] = []
            for e_idx, edge in enumerate(edges_list):
                dist, s_val, _, _, _ = _project_point_to_polyline(px, py, edge.polyline_utm)
                s_proj[t_idx, e_idx] = s_val
                step_projs.append((dist, e_idx, s_val))

            if num_e > 24:
                step_projs.sort(key=lambda item: item[0])
                step_cands = step_projs[:16]
            else:
                step_cands = step_projs

            active_curr: list[int] = []
            for dist, e_idx, s_val in step_cands:
                edge = edges_list[e_idx]
                active_curr.append(e_idx)
                # Gaussian emission log-likelihood + slight preference for primary arterial corridors
                emit_ll = -0.5 * ((dist / sigma) ** 2) + 0.15 * (edge.lanes - 1)
                if t_idx == 0:
                    dp[0, e_idx] = emit_ll
                else:
                    d_euc = float(
                        math.hypot(px - float(pts[t_idx - 1, 0]), py - float(pts[t_idx - 1, 1]))
                    )
                    best_prev_ll = -1e12
                    best_prev_idx = active_prev[0] if active_prev else 0
                    for p_idx in active_prev:
                        prev_edge = edges_list[p_idx]
                        prev_val = dp[t_idx - 1, p_idx]
                        if prev_val <= -1e11:
                            continue
                        s_prev = s_proj[t_idx - 1, p_idx]
                        if p_idx == e_idx:
                            if s_val + 2.0 >= s_prev:
                                d_route = max(0.0, s_val - s_prev)
                                trans_ll = -abs(d_route - d_euc) / beta
                            else:
                                # Backwards movement on a directed edge is heavily penalized
                                trans_ll = -25.0 - abs(s_prev - s_val) / beta
                        elif prev_edge.v == edge.u:
                            d_route = max(0.0, prev_edge.length_m - s_prev) + s_val
                            turn_defl = abs(wrap_angle(edge.heading_start_rad - prev_edge.heading_end_rad))
                            # Immediate U-turn penalty (u -> v -> u)
                            uturn_pen = 18.0 if prev_edge.u == edge.v else 0.0
                            turn_pen = 1.5 * turn_defl + uturn_pen
                            trans_ll = -abs(d_route - d_euc) / beta - turn_pen
                        else:
                            mid_dist = self._directed_node_distance(prev_edge.v, edge.u, max_cost=400.0)
                            if math.isfinite(mid_dist):
                                d_route = max(0.0, prev_edge.length_m - s_prev) + mid_dist + s_val
                                trans_ll = -abs(d_route - d_euc) / beta - 4.0
                            else:
                                # Disconnected jump penalty
                                trans_ll = -80.0

                        cand_ll = prev_val + trans_ll + emit_ll
                        if cand_ll > best_prev_ll:
                            best_prev_ll = cand_ll
                            best_prev_idx = p_idx

                    dp[t_idx, e_idx] = best_prev_ll
                    backptr[t_idx, e_idx] = best_prev_idx
            active_prev = active_curr

        last_idx = int(np.argmax(dp[-1]))
        seq_indices = [0] * n_steps
        seq_indices[-1] = last_idx
        for t_idx in range(n_steps - 1, 0, -1):
            seq_indices[t_idx - 1] = int(backptr[t_idx, seq_indices[t_idx]])

        return [edges_list[idx].edge_id for idx in seq_indices]

    def _edge_vcrit_and_turn_deg(
        self,
        prev_edge: RoadEdge | None,
        next_edge: RoadEdge,
    ) -> tuple[float, float]:
        """Compute (v_crit_mps, turn_angle_deg) for entering and traversing next_edge."""
        internal_defl = abs(wrap_angle(next_edge.heading_end_rad - next_edge.heading_start_rad))
        junction_defl = (
            abs(wrap_angle(next_edge.heading_start_rad - prev_edge.heading_end_rad))
            if prev_edge is not None
            else 0.0
        )
        turn_deg = math.degrees(max(internal_defl, junction_defl))

        if next_edge.min_curve_radius_m < 1e4:
            r_eff = next_edge.min_curve_radius_m
        elif turn_deg >= float(self.cfg.corner_angle_deg):
            r_eff = 9.0
        else:
            r_eff = 1e6

        v_crit = math.sqrt(float(self.cfg.mu_friction) * float(self.cfg.gravity) * r_eff)
        return float(v_crit), float(turn_deg)

    def _grow_anisotropic_cov(
        self,
        p_2x2: np.ndarray,
        theta_rad: float,
        dt: float,
        v_mps: float,
    ) -> np.ndarray:
        """Expand 2x2 covariance anisotropically along road tangent and normal directions."""
        if dt <= 0.0:
            return _enforce_spd(p_2x2)
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)
        rot = np.array([[c, -s], [s, c]], dtype=np.float64)
        q_lon = (1.2 + 0.25 * abs(v_mps)) * dt
        q_lat = (0.35 + 0.05 * abs(v_mps)) * dt
        q_local = np.diag([q_lon, q_lat])
        return _enforce_spd(p_2x2 + rot @ q_local @ rot.T)

    def propagate_route_tree(
        self,
        state_at_exit: FilterState,
        elapsed_s: float,
        dt_step: float = 0.5,
    ) -> list[BlindHypothesis]:
        """Propagate multi-hypothesis escape route tree through blind zone with corner braking & v_crit pruning."""
        elapsed = max(0.0, float(elapsed_s))
        step_dt = max(0.05, float(dt_step))
        start_edge, s0, hdg0 = self._snap_point_with_heading(
            state_at_exit.x_utm,
            state_at_exit.y_utm,
            preferred_heading_rad=state_at_exit.theta_rad,
        )
        v0 = max(0.0, float(state_at_exit.v_mps))
        p0 = _enforce_spd(state_at_exit.P_pos_2x2)

        if elapsed == 0.0:
            ix, iy, ih = _interpolate_polyline(start_edge.polyline_utm, s0)
            rem_len = max(0.0, start_edge.length_m - s0)
            eta0 = rem_len / max(v0, 1.0) if v0 > 1e-3 else 0.0
            return [
                BlindHypothesis(
                    hypothesis_id="hyp_0",
                    edge_sequence=[start_edge.edge_id],
                    current_edge=start_edge.edge_id,
                    s_m=float(s0),
                    x_utm=ix,
                    y_utm=iy,
                    v_mps=v0,
                    theta_rad=ih,
                    P_2x2=p0,
                    probability=1.0,
                    eta_s=float(eta0),
                    pruned_by_vcrit=False,
                )
            ]

        @dataclass
        class _ActiveBranch:
            edge_seq: list[tuple[int, int, int]]
            curr_edge: RoadEdge
            s_m: float
            v_mps: float
            p_2x2: np.ndarray
            weight: float
            rem_time_s: float
            pruned: bool = False
            in_corner_until_s: float = 0.0
            corner_vcrit: float = 1e3
            reaccel_after_corner: bool = False

        # Maximum physical tire-asphalt braking deceleration (mu * g) for pre-corner v_crit feasibility
        a_brake_max = float(self.cfg.mu_friction) * float(self.cfg.gravity)
        a_accel = 1.8

        queue: list[_ActiveBranch] = [
            _ActiveBranch(
                edge_seq=[start_edge.edge_id],
                curr_edge=start_edge,
                s_m=float(s0),
                v_mps=v0,
                p_2x2=p0,
                weight=1.0,
                rem_time_s=elapsed,
                pruned=False,
                in_corner_until_s=self._curve_end_cache.get(start_edge.edge_id, 0.0),
                corner_vcrit=math.sqrt(
                    float(self.cfg.mu_friction) * float(self.cfg.gravity) * start_edge.min_curve_radius_m
                ),
                reaccel_after_corner=self._curve_end_cache.get(start_edge.edge_id, 0.0) > 0.0,
            )
        ]
        finished: list[_ActiveBranch] = []

        while queue and (len(queue) + len(finished) < 128):
            br = queue.pop(0)
            if br.pruned or br.rem_time_s <= 1e-6:
                finished.append(br)
                continue

            # Advance along current edge in increments of step_dt until reaching node v or time expires
            edge = br.curr_edge
            while br.rem_time_s > 1e-6:
                dt = min(step_dt, br.rem_time_s)
                # Speed profile on current edge:
                # 1) If still inside curved geometry zone (s_m <= in_corner_until_s), hold speed <= 0.92 * corner_vcrit
                if br.in_corner_until_s > 0.0 and br.s_m <= br.in_corner_until_s:
                    target_v = min(edge.maxspeed_mps, 0.92 * br.corner_vcrit)
                    if br.v_mps > target_v:
                        br.v_mps = max(target_v, br.v_mps - a_brake_max * dt)
                elif br.reaccel_after_corner:
                    # Re-accelerate toward road maxspeed after exiting a sharp corner
                    if br.v_mps < edge.maxspeed_mps:
                        br.v_mps = min(edge.maxspeed_mps, br.v_mps + a_accel * dt)
                else:
                    # Maintain steady cruising speed v0 along straight corridors (clamped by road maxspeed)
                    cruise_v = min(max(v0, br.v_mps), edge.maxspeed_mps)
                    if br.v_mps < cruise_v:
                        br.v_mps = min(cruise_v, br.v_mps + a_accel * dt)

                ds = br.v_mps * dt
                rem_edge = max(0.0, edge.length_m - br.s_m)
                _, _, hdg_here = _interpolate_polyline(edge.polyline_utm, br.s_m)

                if ds < rem_edge or rem_edge <= 1e-6 and not self._outgoing.get(edge.v):
                    br.s_m = min(edge.length_m, br.s_m + ds)
                    br.rem_time_s -= dt
                    br.p_2x2 = self._grow_anisotropic_cov(br.p_2x2, hdg_here, dt, br.v_mps)
                    if br.rem_time_s <= 1e-6 or not self._outgoing.get(edge.v):
                        finished.append(br)
                        break
                else:
                    # Reaches end of current edge (junction node edge.v)
                    t_to_node = rem_edge / max(br.v_mps, 1e-3) if br.v_mps > 1e-3 else dt
                    t_used = min(br.rem_time_s, t_to_node)
                    br.rem_time_s -= t_used
                    br.p_2x2 = self._grow_anisotropic_cov(br.p_2x2, hdg_here, t_used, br.v_mps)
                    br.s_m = edge.length_m

                    # Available braking distance on the edge leading into this junction
                    d_approach = max(0.0, edge.length_m - (s0 if len(br.edge_seq) == 1 else 0.0))
                    v_entry_reference = v0 if len(br.edge_seq) == 1 else br.v_mps
                    v_min_possible = math.sqrt(
                        max(0.0, v_entry_reference * v_entry_reference - 2.0 * a_brake_max * d_approach)
                    )

                    outgoing = [
                        e_next
                        for e_next in self._outgoing.get(edge.v, [])
                        # Prevent immediate U-turn reversal onto identical undirected pair unless dead-end
                        if e_next.v != edge.u or len(self._outgoing.get(edge.v, [])) == 1
                    ]
                    if not outgoing:
                        finished.append(br)
                        break

                    for e_next in outgoing:
                        v_crit, turn_deg = self._edge_vcrit_and_turn_deg(edge, e_next)
                        is_sharp_corner = (
                            turn_deg >= float(self.cfg.corner_angle_deg)
                            or e_next.min_curve_radius_m < 100.0
                        )
                        if is_sharp_corner and v_min_possible > v_crit * 1.05:
                            # Pruned by v_crit: vehicle arrived too fast to brake for this sharp turn
                            finished.append(
                                _ActiveBranch(
                                    edge_seq=[*br.edge_seq, e_next.edge_id],
                                    curr_edge=e_next,
                                    s_m=0.0,
                                    v_mps=br.v_mps,
                                    p_2x2=br.p_2x2.copy(),
                                    weight=0.0,
                                    rem_time_s=0.0,
                                    pruned=True,
                                    in_corner_until_s=self._curve_end_cache.get(e_next.edge_id, 0.0),
                                    corner_vcrit=v_crit,
                                )
                            )
                        else:
                            # Valid turn or straight continuation
                            if is_sharp_corner:
                                v_into_next = min(br.v_mps, 0.90 * v_crit)
                            else:
                                v_into_next = min(max(br.v_mps, v0), e_next.maxspeed_mps)
                            # Weight branches by lanes, maxspeed, and smoother deflection
                            defl_rad = abs(wrap_angle(e_next.heading_start_rad - edge.heading_end_rad))
                            branch_pref = (
                                float(e_next.lanes)
                                * (e_next.maxspeed_mps / 13.89)
                                * math.exp(-0.6 * defl_rad)
                            )
                            curve_end_s = self._curve_end_cache.get(e_next.edge_id, 0.0)
                            if is_sharp_corner and curve_end_s <= 0.0:
                                curve_end_s = min(12.0, 0.4 * e_next.length_m)
                            queue.append(
                                _ActiveBranch(
                                    edge_seq=[*br.edge_seq, e_next.edge_id],
                                    curr_edge=e_next,
                                    s_m=0.0,
                                    v_mps=v_into_next,
                                    p_2x2=br.p_2x2.copy(),
                                    weight=br.weight * max(branch_pref, 0.05),
                                    rem_time_s=br.rem_time_s,
                                    pruned=False,
                                    in_corner_until_s=curve_end_s,
                                    corner_vcrit=v_crit,
                                    reaccel_after_corner=bool(br.reaccel_after_corner or is_sharp_corner),
                                )
                            )
                    break

        # Convert finished branches into BlindHypothesis objects and normalize surviving probabilities
        surviving_weights = sum(b.weight for b in finished if not b.pruned and b.weight > 0.0)
        hypotheses: list[BlindHypothesis] = []

        for idx, b in enumerate(finished):
            ix, iy, ih = _interpolate_polyline(b.curr_edge.polyline_utm, b.s_m)
            rem_len = max(0.0, b.curr_edge.length_m - b.s_m)
            eta_val = rem_len / max(b.v_mps, 1.0) if not b.pruned else 0.0
            if not b.pruned and eta_val <= 0.0:
                eta_val = max(0.5, b.curr_edge.length_m / max(b.v_mps, 1.0))
            prob = (
                float(b.weight / surviving_weights)
                if (not b.pruned and surviving_weights > 0.0)
                else 0.0
            )
            hypotheses.append(
                BlindHypothesis(
                    hypothesis_id=f"hyp_{idx}",
                    edge_sequence=list(b.edge_seq),
                    current_edge=b.curr_edge.edge_id,
                    s_m=float(b.s_m),
                    x_utm=float(ix),
                    y_utm=float(iy),
                    v_mps=float(b.v_mps),
                    theta_rad=wrap_angle(float(ih)),
                    P_2x2=_enforce_spd(b.p_2x2),
                    probability=prob,
                    eta_s=float(eta_val),
                    pruned_by_vcrit=bool(b.pruned),
                )
            )

        # Filter out minuscule non-pruned branches below min_branch_prob and re-normalize
        min_p = float(self.cfg.min_branch_prob)
        active = [h for h in hypotheses if not h.pruned_by_vcrit and h.probability >= min_p]
        if active:
            total_p = sum(h.probability for h in active)
            for h in hypotheses:
                if not h.pruned_by_vcrit:
                    if h.probability >= min_p and total_p > 0.0:
                        h.probability = float(h.probability / total_p)
                    else:
                        h.probability = 0.0

        hypotheses.sort(key=lambda h: (h.pruned_by_vcrit, -h.probability, h.hypothesis_id))
        return hypotheses

    def recommend_blockade_points(
        self,
        hypotheses: list[BlindHypothesis],
        top_k: int = 3,
    ) -> list[BlockadePoint]:
        """Rank downstream road vertices by cumulative escape interception probability and ETA."""
        active_hyps = [
            h
            for h in hypotheses
            if not h.pruned_by_vcrit and math.isfinite(h.probability) and h.probability > 0.0
        ]
        if not active_hyps or top_k <= 0:
            return []

        node_probs: dict[int, float] = {}
        node_etas: dict[int, list[float]] = {}
        node_routes: dict[int, list[str]] = {}

        for h in active_hyps:
            curr_edge = self.network.edges.get(h.current_edge)
            v_eff = max(float(h.v_mps), 2.0)
            if curr_edge is not None:
                target_node = curr_edge.v
                rem_m = max(0.0, curr_edge.length_m - float(h.s_m))
                base_eta = float(h.eta_s) if h.eta_s > 0.0 else max(0.5, rem_m / v_eff)
            else:
                target_node = int(h.current_edge[1])
                base_eta = max(0.5, float(h.eta_s))

            if target_node not in self.network.nodes_utm:
                continue

            # Reachable downstream nodes (0 hops = target_node, plus up to 2 downstream hops)
            visited_for_hyp: dict[int, float] = {target_node: base_eta}
            frontier: list[tuple[int, float, int]] = [(target_node, base_eta, 0)]
            while frontier:
                nid, eta_here, depth = frontier.pop(0)
                if depth >= 2:
                    continue
                for out_e in self._outgoing.get(nid, []):
                    nxt_id = out_e.v
                    nxt_eta = eta_here + out_e.length_m / max(v_eff, out_e.maxspeed_mps * 0.7)
                    if nxt_id not in visited_for_hyp or nxt_eta < visited_for_hyp[nxt_id]:
                        visited_for_hyp[nxt_id] = nxt_eta
                        frontier.append((nxt_id, nxt_eta, depth + 1))

            for nid, eta_node in visited_for_hyp.items():
                if nid not in self.network.nodes_utm:
                    continue
                node_probs[nid] = min(1.0, node_probs.get(nid, 0.0) + float(h.probability))
                node_etas.setdefault(nid, []).append(max(0.1, float(eta_node)))
                routes = node_routes.setdefault(nid, [])
                if h.hypothesis_id not in routes:
                    routes.append(h.hypothesis_id)

        blockades: list[BlockadePoint] = []
        for nid, prob in node_probs.items():
            nx_m, ny_m = self.network.nodes_utm[nid]
            etas = node_etas[nid]
            routes = node_routes[nid]
            mean_eta = float(sum(etas) / len(etas))
            blockades.append(
                BlockadePoint(
                    node_id=int(nid),
                    x_utm=float(nx_m),
                    y_utm=float(ny_m),
                    interception_prob=float(prob),
                    eta_seconds=max(0.1, mean_eta),
                    converging_routes=routes,
                )
            )

        blockades.sort(
            key=lambda b: (
                -round(b.interception_prob, 6),
                -len(b.converging_routes),
                b.eta_seconds,
                b.node_id,
            )
        )
        return blockades[: int(top_k)]
