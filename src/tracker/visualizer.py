"""Split-Screen CICC Tactical Dashboard Renderer (Synchronized Cameras + 2D Metric UTM Map)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tracker.config import CameraConfig, VisualizerConfig
from tracker.osm_loader import RoadNetwork
from tracker.pipeline import PipelineFrameState


class CICCVisualizer:
    """Headless-safe OpenCV Split-Screen CICC tactical dashboard renderer and MP4 video writer."""

    def __init__(
        self,
        cfg: VisualizerConfig,
        road_network: RoadNetwork | None = None,
        cameras: list[CameraConfig] | None = None,
    ) -> None:
        self.cfg = cfg
        self.road_network = road_network
        self.cameras = list(cameras) if cameras is not None else []
        self._writer: cv2.VideoWriter | None = None
        self._is_open: bool = False

    def open(self) -> None:
        """Open the MP4 video writer if output_video is configured."""
        if self.cfg.output_video is None:
            self._is_open = True
            return
        out_path = Path(self.cfg.output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc,
            float(self.cfg.fps),
            (int(self.cfg.width), int(self.cfg.height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {out_path}")
        self._writer = writer
        self._is_open = True

    def write(self, state: PipelineFrameState) -> np.ndarray:
        """Render frame and write to video output if open."""
        if self.cfg.output_video is not None and not self._is_open:
            raise RuntimeError("CICCVisualizer.write() called before open()")
        frame = self.render_frame(state)
        if self._writer is not None:
            self._writer.write(frame)
        return frame

    def close(self) -> None:
        """Idempotently close and release the video writer."""
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None
        self._is_open = False

    def _utm_to_map_px(
        self,
        x_utm: float,
        y_utm: float,
        bounds: tuple[float, float, float, float],
        panel_w: int,
        panel_h: int,
        margin: int = 44,
    ) -> tuple[int, int, float]:
        min_x, min_y, max_x, max_y = bounds
        span_x = max(max_x - min_x, 20.0)
        span_y = max(max_y - min_y, 20.0)
        usable_w = max(panel_w - 2 * margin, 40)
        usable_h = max(panel_h - 2 * margin - 40, 40)
        scale = min(usable_w / span_x, usable_h / span_y)

        cx_utm = 0.5 * (min_x + max_x)
        cy_utm = 0.5 * (min_y + max_y)
        px = int(round(panel_w * 0.5 + (x_utm - cx_utm) * scale))
        py = int(round((panel_h + 36) * 0.5 - (y_utm - cy_utm) * scale))
        px = max(-2000, min(panel_w + 2000, px))
        py = max(-2000, min(panel_h + 2000, py))
        return px, py, float(scale)

    def _render_left_cameras(
        self,
        state: PipelineFrameState,
        panel_w: int,
        panel_h: int,
    ) -> np.ndarray:
        panel = np.full((panel_h, panel_w, 3), (22, 24, 28), dtype=np.uint8)
        header_h = 38
        cv2.rectangle(panel, (0, 0), (panel_w, header_h), (36, 40, 48), -1)

        is_blind = bool(state.blind_hypotheses) or any("ZONA CEGA" in ev.upper() for ev in state.events)
        has_visible = any(
            any(tr.get("status") == "ACTIVE" for tr in tr_list)
            for tr_list in state.camera_tracks.values()
        )
        status_txt = "ZONA CEGA (ALERTA CICC)" if (is_blind and not has_visible) else ("VISIVEL" if has_visible else "MONITORANDO")
        status_col = (40, 90, 250) if "ZONA CEGA" in status_txt else ((70, 215, 90) if has_visible else (175, 185, 195))

        cv2.putText(
            panel,
            f"CICC CAMERAS | t={state.timestamp_s:.2f}s",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 240, 245),
            1,
            cv2.LINE_AA,
        )
        badge_w = min(230, panel_w // 2)
        cv2.rectangle(panel, (panel_w - badge_w - 8, 6), (panel_w - 8, header_h - 6), status_col, -1)
        cv2.putText(
            panel,
            status_txt[:22],
            (panel_w - badge_w, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (15, 15, 15),
            1,
            cv2.LINE_AA,
        )

        cam_ids = [c.cam_id for c in self.cameras] if self.cameras else sorted(state.frames_bgr.keys())
        if not cam_ids:
            cam_ids = ["cam_1"]

        n_cams = len(cam_ids)
        grid_h = max(panel_h - header_h, 40)
        if n_cams == 1:
            rows, cols = 1, 1
        elif n_cams == 2:
            rows, cols = 2, 1
        else:
            cols = int(math.ceil(math.sqrt(n_cams)))
            rows = int(math.ceil(n_cams / cols))

        tile_w = max(panel_w // cols, 32)
        tile_h = max(grid_h // rows, 32)
        cam_cfg_map = {c.cam_id: c for c in self.cameras}

        for idx, cid in enumerate(cam_ids):
            r_idx = idx // cols
            c_idx = idx % cols
            x0 = c_idx * tile_w
            y0 = header_h + r_idx * tile_h
            tw = panel_w - x0 if c_idx == cols - 1 else tile_w
            th = (panel_h - y0) if r_idx == rows - 1 else tile_h
            if tw <= 4 or th <= 4:
                continue

            raw_frame = state.frames_bgr.get(cid)
            if raw_frame is not None and isinstance(raw_frame, np.ndarray) and raw_frame.size > 0:
                src_h, src_w = raw_frame.shape[:2]
                tile = cv2.resize(raw_frame, (tw, th), interpolation=cv2.INTER_LINEAR)
            else:
                src_w, src_h = (640, 480)
                if cid in cam_cfg_map:
                    src_w, src_h = cam_cfg_map[cid].resolution
                tile = np.full((th, tw, 3), (32, 36, 42), dtype=np.uint8)

            sx = tw / max(float(src_w), 1.0)
            sy = th / max(float(src_h), 1.0)

            # Draw camera exit zones
            if cid in cam_cfg_map and cam_cfg_map[cid].exit_zones:
                for poly in cam_cfg_map[cid].exit_zones:
                    pts = np.array(
                        [[int(round(px * sx)), int(round(py * sy))] for px, py in poly],
                        dtype=np.int32,
                    ).reshape((-1, 1, 2))
                    cv2.polylines(tile, [pts], isClosed=True, color=(0, 195, 255), thickness=2)

            # Draw tracked vehicles in this camera
            for tr in state.camera_tracks.get(cid, []):
                bbox = tr.get("bbox_xyxy")
                if bbox is None or len(bbox) < 4:
                    continue
                bx1 = max(0, min(tw - 1, int(round(float(bbox[0]) * sx))))
                by1 = max(0, min(th - 1, int(round(float(bbox[1]) * sy))))
                bx2 = max(0, min(tw - 1, int(round(float(bbox[2]) * sx))))
                by2 = max(0, min(th - 1, int(round(float(bbox[3]) * sy))))
                gid = tr.get("global_id", tr.get("local_track_id", 0))
                v_kmh = float(tr.get("v_mps", 0.0)) * 3.6
                tr_status = str(tr.get("status", "ACTIVE"))
                box_col = (50, 230, 80) if tr_status == "ACTIVE" else (0, 180, 255)
                cv2.rectangle(tile, (bx1, by1), (bx2, by2), box_col, 2)

                plate_str = tr.get("plate")
                lbl = f"ID:{gid} {v_kmh:.0f}km/h"
                if plate_str:
                    lbl += f" [{plate_str}]"
                cv2.rectangle(tile, (bx1, max(0, by1 - 20)), (min(tw - 1, bx1 + len(lbl) * 8 + 6), by1), box_col, -1)
                cv2.putText(
                    tile,
                    lbl,
                    (bx1 + 3, max(14, by1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (10, 10, 10),
                    1,
                    cv2.LINE_AA,
                )

            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (75, 85, 98), 1)
            cv2.putText(
                tile,
                f"[{cid}]",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (245, 225, 110),
                1,
                cv2.LINE_AA,
            )
            panel[y0 : y0 + th, x0 : x0 + tw] = tile

        return panel

    def _render_right_map(
        self,
        state: PipelineFrameState,
        panel_w: int,
        panel_h: int,
    ) -> np.ndarray:
        panel = np.full((panel_h, panel_w, 3), (18, 22, 30), dtype=np.uint8)
        header_h = 38
        cv2.rectangle(panel, (0, 0), (panel_w, header_h), (28, 35, 48), -1)
        cv2.putText(
            panel,
            "MAPA TATICO 2D UTM - MALHA VIARIA & CERCO CICC",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (225, 235, 250),
            1,
            cv2.LINE_AA,
        )

        # Subtle metric grid lines
        for gx in range(40, panel_w, 60):
            cv2.line(panel, (gx, header_h), (gx, panel_h), (26, 31, 40), 1)
        for gy in range(header_h + 40, panel_h, 60):
            cv2.line(panel, (0, gy), (panel_w, gy), (26, 31, 40), 1)

        if self.road_network is not None and self.road_network.edges:
            bounds = self.road_network.bounds_utm
        else:
            bounds = (330000.0, 7389800.0, 330400.0, 7390150.0)

        # 1. Draw base road network edges and nodes
        if self.road_network is not None:
            for edge in self.road_network.edges.values():
                pts = [
                    self._utm_to_map_px(float(x), float(y), bounds, panel_w, panel_h)[:2]
                    for x, y in edge.polyline_utm
                ]
                if len(pts) >= 2:
                    arr_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                    thickness = 3 if edge.lanes >= 2 else 2
                    cv2.polylines(panel, [arr_pts], isClosed=False, color=(82, 92, 110), thickness=thickness, lineType=cv2.LINE_AA)
            for nx_m, ny_m in self.road_network.nodes_utm.values():
                px, py, _ = self._utm_to_map_px(nx_m, ny_m, bounds, panel_w, panel_h)
                cv2.circle(panel, (px, py), 3, (125, 135, 150), -1)

        # 2. Draw Forensic Backtracked Paths
        for fp in state.forensic_paths:
            if fp.polyline_utm is not None and len(fp.polyline_utm) >= 2:
                pts = [
                    self._utm_to_map_px(float(x), float(y), bounds, panel_w, panel_h)[:2]
                    for x, y in fp.polyline_utm
                ]
                arr_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(panel, [arr_pts], isClosed=False, color=(220, 80, 255), thickness=3, lineType=cv2.LINE_AA)

        # 3. Draw Blind-Zone Escape Route Tree colored by probability
        if self.cfg.show_escape_tree and state.blind_hypotheses:
            for gid, hyps in state.blind_hypotheses.items():
                for h in hyps:
                    if h.pruned_by_vcrit or h.probability <= 0.0:
                        continue
                    p_clamped = float(np.clip(h.probability, 0.0, 1.0))
                    # Color gradient: high prob -> bright red-orange (40, 110, 255); low prob -> gold/cyan (220, 190, 50)
                    col_b = int(round(220 * (1.0 - p_clamped) + 35 * p_clamped))
                    col_g = int(round(200 * (1.0 - 0.5 * p_clamped)))
                    col_r = int(round(80 + 175 * p_clamped))
                    branch_col = (col_b, col_g, col_r)
                    thick = max(2, int(round(2 + 4 * p_clamped)))

                    if self.road_network is not None:
                        for eid in h.edge_sequence:
                            if eid in self.road_network.edges:
                                e_poly = self.road_network.edges[eid].polyline_utm
                                pts = [
                                    self._utm_to_map_px(float(x), float(y), bounds, panel_w, panel_h)[:2]
                                    for x, y in e_poly
                                ]
                                arr_pts = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                                cv2.polylines(panel, [arr_pts], isClosed=False, color=branch_col, thickness=thick, lineType=cv2.LINE_AA)

                    hx, hy, sc = self._utm_to_map_px(h.x_utm, h.y_utm, bounds, panel_w, panel_h)
                    if self.cfg.show_cov_ellipse:
                        self._draw_cov_ellipse(panel, hx, hy, h.P_2x2, sc, branch_col)
                    cv2.circle(panel, (hx, hy), 6, branch_col, -1)
                    cv2.putText(
                        panel,
                        f"G{gid}:{h.probability*100:.0f}%",
                        (hx + 8, hy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        branch_col,
                        1,
                        cv2.LINE_AA,
                    )

        # 4. Draw Global Vehicle Positions & EKF Covariance Ellipses P_k
        for gid, st in state.global_states.items():
            px, py, sc = self._utm_to_map_px(st.x_utm, st.y_utm, bounds, panel_w, panel_h)
            in_blind = gid in state.blind_hypotheses and len(state.blind_hypotheses[gid]) > 0
            v_col = (40, 140, 255) if in_blind else (60, 240, 100)

            if self.cfg.show_cov_ellipse and st.P_pos_2x2 is not None:
                self._draw_cov_ellipse(panel, px, py, st.P_pos_2x2, sc, v_col)

            cv2.circle(panel, (px, py), 6, v_col, -1)
            cv2.circle(panel, (px, py), 8, (245, 245, 245), 1)
            # Heading vector arrow
            ax = int(round(px + 16.0 * math.cos(st.theta_rad)))
            ay = int(round(py - 16.0 * math.sin(st.theta_rad)))
            cv2.arrowedLine(panel, (px, py), (ax, ay), v_col, 2, tipLength=0.35)
            cv2.putText(
                panel,
                f"G{gid} ({st.v_mps*3.6:.0f}km/h)",
                (px + 10, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (235, 245, 235),
                1,
                cv2.LINE_AA,
            )

        # 5. Draw Recommended CICC Police Blockade Points with ETA
        if self.cfg.show_blockades and state.blockade_points:
            for rank, blk in enumerate(state.blockade_points):
                bx, by, _ = self._utm_to_map_px(blk.x_utm, blk.y_utm, bounds, panel_w, panel_h)
                b_col = (40, 50, 255) if rank == 0 else (0, 165, 255)
                cv2.circle(panel, (bx, by), 11, b_col, 2)
                cv2.drawMarker(panel, (bx, by), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
                lbl = f"BLOQUEIO N{blk.node_id} ({blk.interception_prob*100:.0f}% | ETA {blk.eta_seconds:.1f}s)"
                cv2.putText(
                    panel,
                    lbl,
                    (max(10, min(panel_w - 240, bx + 12)), max(header_h + 18, min(panel_h - 12, by + 16))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (110, 220, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.rectangle(panel, (0, 0), (panel_w - 1, panel_h - 1), (65, 75, 92), 1)
        return panel

    def _draw_cov_ellipse(
        self,
        panel: np.ndarray,
        cx_px: int,
        cy_px: int,
        p_2x2: np.ndarray,
        scale_px_per_m: float,
        color: tuple[int, int, int],
    ) -> None:
        """Safely render a 95% confidence covariance ellipse for P_2x2 onto the 2D UTM map panel."""
        arr = np.asarray(p_2x2, dtype=np.float64)
        if arr.shape != (2, 2) or not np.all(np.isfinite(arr)):
            return
        sym = 0.5 * (arr + arr.T)
        try:
            vals, vecs = np.linalg.eigh(sym)
        except np.linalg.LinAlgError:
            return

        # Clamp eigenvalues so degenerate (1e-12) or huge (1e7) matrices render cleanly
        vals_clamped = np.clip(vals, 1e-4, 2500.0)
        chi2_95_sqrt = 2.4477  # sqrt(5.991)
        # Convert semi-axes in meters to pixels (with readable visual magnification)
        a_m = chi2_95_sqrt * math.sqrt(float(vals_clamped[1]))
        b_m = chi2_95_sqrt * math.sqrt(float(vals_clamped[0]))
        a_px = int(round(np.clip(a_m * scale_px_per_m * 1.5, 4.0, max(panel.shape[:2]) * 0.45)))
        b_px = int(round(np.clip(b_m * scale_px_per_m * 1.5, 3.0, max(panel.shape[:2]) * 0.45)))

        vx = float(vecs[0, 1])
        vy = float(-vecs[1, 1])  # image y-axis is inverted relative to UTM y_north
        angle_deg = float(math.degrees(math.atan2(vy, vx)))

        overlay = panel.copy()
        cv2.ellipse(
            overlay,
            (cx_px, cy_px),
            (a_px, b_px),
            angle_deg,
            0.0,
            360.0,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.22, panel, 0.78, 0.0, dst=panel)
        cv2.ellipse(
            panel,
            (cx_px, cy_px),
            (a_px, b_px),
            angle_deg,
            0.0,
            360.0,
            color,
            1,
            cv2.LINE_AA,
        )

    def render_frame(self, state: PipelineFrameState) -> np.ndarray:
        """Render a single (height, width, 3) uint8 BGR Split-Screen CICC dashboard frame."""
        total_w = int(self.cfg.width)
        total_h = int(self.cfg.height)
        if not self.cfg.split_screen:
            return self._render_left_cameras(state, total_w, total_h)

        left_w = total_w // 2
        right_w = total_w - left_w
        left_panel = self._render_left_cameras(state, left_w, total_h)
        right_panel = self._render_right_map(state, right_w, total_h)
        return np.hstack([left_panel, right_panel])
