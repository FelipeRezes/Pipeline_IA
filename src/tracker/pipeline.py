"""Strictly causal real-time multi-camera tracking orchestrator and CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable

import cv2
import numpy as np

from tracker.alpr import ALPREngine, PlateReading
from tracker.association import (
    BlindTrackRecord,
    CrossCameraAssociator,
    NewAppearance,
    ReconstructedBlindPath,
)
from tracker.config import CameraConfig, PipelineConfig, load_config
from tracker.detector import LocalPerception, LocalTrack
from tracker.ekf import FilterState, create_filter
from tracker.geometry import CameraProjector
from tracker.kinematics import circular_mean
from tracker.osm_loader import RoadNetwork, load_utm_road_graph
from tracker.reid import ReIDExtractor
from tracker.roadgraph import BlindHypothesis, BlockadePoint, RoadGraphEngine


@dataclass
class PipelineFrameState:
    """Complete causal multi-camera state snapshot at timestamp_s."""

    timestamp_s: float
    frames_bgr: dict[str, np.ndarray]
    camera_tracks: dict[str, list[dict[str, Any]]]
    global_states: dict[int, FilterState]
    blind_hypotheses: dict[int, list[BlindHypothesis]]
    blockade_points: list[BlockadePoint]
    forensic_paths: list[ReconstructedBlindPath]
    events: list[str]


def _build_camera_projector(cam: CameraConfig) -> CameraProjector:
    if cam.K is not None and cam.R is not None and cam.t is not None:
        return CameraProjector.from_3d_params(cam.K, cam.R, cam.t, epsg_utm=cam.epsg_utm)
    if cam.homography is not None:
        return CameraProjector.from_homography(cam.homography, epsg_utm=cam.epsg_utm)
    if cam.calib_path is not None and Path(cam.calib_path).is_file():
        return CameraProjector.from_yaml(cam.calib_path)
    # Default identity metric projector fallback if none configured
    h_default = np.array([[0.1, 0.0, 330000.0], [0.0, -0.1, 7390000.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return CameraProjector.from_homography(h_default, epsg_utm=cam.epsg_utm or 32723)


class CausalMultiCamPipeline:
    """Strictly causal frame-by-frame multi-camera vehicle tracking and blind-zone propagation pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        road_network: RoadNetwork | None = None,
        road_engine: RoadGraphEngine | None = None,
        perceptions: dict[str, LocalPerception] | None = None,
        local_perceptions: dict[str, LocalPerception] | None = None,
        detectors: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
        detector_fns: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
        reid_extractor: ReIDExtractor | None = None,
        alpr_engine: ALPREngine | None = None,
    ) -> None:
        self.config = config
        self.cam_configs: dict[str, CameraConfig] = {c.cam_id: c for c in config.cameras}
        self.projectors: dict[str, CameraProjector] = {
            c.cam_id: _build_camera_projector(c) for c in config.cameras
        }

        if road_network is not None:
            self.road_network: RoadNetwork | None = road_network
        elif config.road_graph.graphml_cache is not None and Path(config.road_graph.graphml_cache).is_file():
            self.road_network = load_utm_road_graph(
                place_name=config.road_graph.place_name,
                bbox_wgs84=config.road_graph.bbox_wgs84,
                graphml_cache=config.road_graph.graphml_cache,
                overpass_urls=config.road_graph.overpass_urls,
            )
        else:
            self.road_network = None

        if road_engine is not None:
            self.road_engine: RoadGraphEngine | None = road_engine
        elif self.road_network is not None:
            self.road_engine = RoadGraphEngine(self.road_network, config.road_graph)
        else:
            self.road_engine = None

        inj_perceptions = perceptions or local_perceptions
        inj_detectors = detectors or detector_fns or {}
        self.perceptions: dict[str, LocalPerception] = {}
        if inj_perceptions is not None:
            self.perceptions.update(inj_perceptions)
        for cam in config.cameras:
            if cam.cam_id not in self.perceptions:
                det_fn = inj_detectors.get(cam.cam_id)
                if det_fn is not None or Path(config.detector.model_path).is_file():
                    self.perceptions[cam.cam_id] = LocalPerception(
                        config.detector,
                        cam.exit_zones,
                        detector_fn=det_fn,
                    )

        if reid_extractor is not None:
            self.reid: ReIDExtractor | None = reid_extractor
        else:
            try:
                self.reid = ReIDExtractor(config.reid)
            except FileNotFoundError:
                self.reid = None

        self.alpr = alpr_engine if alpr_engine is not None else ALPREngine(config.alpr)
        self.associator = CrossCameraAssociator(config.association, road_engine=self.road_engine)

        self._last_timestamp_s: float | None = None
        self._next_global_id: int = 1
        self._local_to_global: dict[tuple[str, int], int] = {}
        self._filters: dict[int, Any] = {}
        self._hit_counts: dict[int, int] = {}
        self._first_pos_time: dict[int, tuple[float, float, float]] = {}
        self._last_active_time: dict[int, float] = {}
        self._last_active_state: dict[int, FilterState] = {}
        self._last_dims: dict[int, tuple[float, float]] = {}
        self._last_reid: dict[int, np.ndarray] = {}
        self._last_plate: dict[int, PlateReading] = {}
        self._blind_records: dict[int, BlindTrackRecord] = {}
        self._forensic_history: list[ReconstructedBlindPath] = []

    def _extract_appearance_and_plate(
        self,
        crop_bgr: np.ndarray | None,
    ) -> tuple[np.ndarray | None, PlateReading | None]:
        emb: np.ndarray | None = None
        plate: PlateReading | None = None
        if crop_bgr is not None and crop_bgr.size > 0 and crop_bgr.ndim == 3:
            if self.reid is not None:
                try:
                    emb = self.reid.extract(crop_bgr)
                except Exception:
                    emb = None
            if self.alpr is not None and self.config.alpr.enabled:
                try:
                    plate = self.alpr.recognize(crop_bgr)
                except Exception:
                    plate = None
        return emb, plate

    def step(
        self,
        timestamp_s: float,
        frames_bgr: dict[str, np.ndarray],
    ) -> PipelineFrameState:
        """Process a single synchronous multi-camera time step strictly causally (no future lookahead)."""
        ts = float(timestamp_s)
        if not math.isfinite(ts):
            raise ValueError(f"timestamp_s must be finite (got {timestamp_s})")
        if self._last_timestamp_s is not None and ts < self._last_timestamp_s - 1e-9:
            raise ValueError(
                f"Non-monotonic timestamp {ts:.4f}s < {self._last_timestamp_s:.4f}s violates strict causality"
            )

        if self._last_timestamp_s is None:
            default_fps = self.config.cameras[0].fps if self.config.cameras else 10.0
            dt = 1.0 / max(float(default_fps), 1.0)
        else:
            dt = max(0.0, ts - self._last_timestamp_s)
        self._last_timestamp_s = ts

        events: list[str] = []
        camera_tracks: dict[str, list[dict[str, Any]]] = {
            cam_id: [] for cam_id in frames_bgr
        }

        # 1. Run local perception per camera
        local_results: dict[str, list[LocalTrack]] = {}
        for cam_id, frame in frames_bgr.items():
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                local_results[cam_id] = []
                continue
            perc = self.perceptions.get(cam_id)
            if perc is None:
                local_results[cam_id] = []
                continue
            local_results[cam_id] = perc.process_frame(frame)

        # 2. Process EXITED_TO_BLIND transitions first so exited vehicles enter the blind pool
        for cam_id, ltracks in local_results.items():
            for ltr in ltracks:
                if ltr.status != "EXITED_TO_BLIND":
                    continue
                key = (cam_id, ltr.local_track_id)
                gid = self._local_to_global.get(key)
                if gid is None or gid not in self._filters:
                    continue
                exit_st = self._filters[gid].state
                hyps: list[BlindHypothesis] = []
                if self.road_engine is not None:
                    hyps = self.road_engine.propagate_route_tree(
                        exit_st, elapsed_s=max(dt, 0.2)
                    )
                self._blind_records[gid] = BlindTrackRecord(
                    global_id=gid,
                    cam_id=cam_id,
                    exit_time=ts,
                    exit_state=exit_st,
                    dims_ground_wl=self._last_dims.get(gid, (1.8, 4.3)),
                    reid_emb=self._last_reid.get(gid),
                    plate=self._last_plate.get(gid),
                    hypotheses=hyps,
                )
                events.append(f"G{gid} exited {cam_id} ({ltr.exit_zone_id}) -> ZONA CEGA")

        # 3. Propagate existing blind tracks up to current timestamp_s
        for gid, rec in self._blind_records.items():
            elapsed = max(0.0, ts - float(rec.exit_time))
            if elapsed > 0.0 and gid in self._filters and dt > 0.0:
                self._filters[gid].predict(dt)
            if self.road_engine is not None:
                rec.hypotheses = self.road_engine.propagate_route_tree(
                    rec.exit_state,
                    elapsed_s=max(elapsed, 0.2),
                )

        # 4. Separate ACTIVE tracks into known global IDs vs new candidate appearances
        known_active: list[tuple[str, LocalTrack, int, float, float, np.ndarray, tuple[float, float], np.ndarray | None, PlateReading | None]] = []
        new_candidates: list[tuple[str, LocalTrack, NewAppearance]] = []

        for cam_id, ltracks in local_results.items():
            proj = self.projectors.get(cam_id)
            if proj is None:
                continue
            for ltr in ltracks:
                if ltr.status == "EXITED_TO_BLIND":
                    continue
                if ltr.status == "OCCLUDED_LOCAL":
                    key = (cam_id, ltr.local_track_id)
                    gid = self._local_to_global.get(key)
                    if gid is not None and gid in self._filters and dt > 0.0:
                        st_occ = self._filters[gid].predict(dt)
                        # If a track experiences a multi-second blackout (>= 1.5s, e.g. UA-DETRAC 3-8s blind mask),
                        # start propagating blind-zone route hypotheses from its last visible state.
                        last_vis_t = self._last_active_time.get(gid, ts)
                        if (ts - last_vis_t) >= 1.5 and gid not in self._blind_records:
                            exit_st = self._last_active_state.get(gid, st_occ)
                            hyps_mask: list[BlindHypothesis] = []
                            if self.road_engine is not None:
                                hyps_mask = self.road_engine.propagate_route_tree(
                                    exit_st,
                                    elapsed_s=max(0.2, ts - last_vis_t),
                                )
                            self._blind_records[gid] = BlindTrackRecord(
                                global_id=gid,
                                cam_id=cam_id,
                                exit_time=last_vis_t,
                                exit_state=exit_st,
                                dims_ground_wl=self._last_dims.get(gid, (1.8, 4.3)),
                                reid_emb=self._last_reid.get(gid),
                                plate=self._last_plate.get(gid),
                                hypotheses=hyps_mask,
                            )
                        camera_tracks.setdefault(cam_id, []).append(
                            {
                                "global_id": gid,
                                "local_track_id": ltr.local_track_id,
                                "bbox_xyxy": ltr.bbox_xyxy.tolist(),
                                "conf": float(ltr.conf),
                                "class_id": int(ltr.class_id),
                                "status": "OCCLUDED_LOCAL",
                                "v_mps": float(st_occ.v_mps),
                                "x_utm": float(st_occ.x_utm),
                                "y_utm": float(st_occ.y_utm),
                                "plate": self._last_plate[gid].normalized_text if gid in self._last_plate else None,
                            }
                        )
                    continue

                # ACTIVE track
                try:
                    x_utm, y_utm, r_k = proj.bbox_to_ground(
                        ltr.bbox_xyxy,
                        sigma_px=self.config.ekf.sigma_px,
                        use_perspective_correction=True,
                    )
                    dims_wl = proj.bbox_dimensions_ground(ltr.bbox_xyxy)
                except ValueError:
                    continue

                emb, plate = self._extract_appearance_and_plate(ltr.crop_bgr)
                key = (cam_id, ltr.local_track_id)
                if key in self._local_to_global:
                    gid = self._local_to_global[key]
                    known_active.append((cam_id, ltr, gid, x_utm, y_utm, r_k, dims_wl, emb, plate))
                else:
                    app = NewAppearance(
                        cam_id=cam_id,
                        local_track_id=ltr.local_track_id,
                        entry_time=ts,
                        x_utm=x_utm,
                        y_utm=y_utm,
                        R_k=r_k,
                        dims_ground_wl=dims_wl,
                        reid_emb=emb,
                        plate=plate,
                    )
                    new_candidates.append((cam_id, ltr, app))

        # 5. Update known active tracks
        for cam_id, ltr, gid, x_utm, y_utm, r_k, dims_wl, emb, plate in known_active:
            flt = self._filters[gid]
            self._hit_counts[gid] = self._hit_counts.get(gid, 1) + 1
            hits = self._hit_counts[gid]

            if hits == 2 and gid in self._first_pos_time:
                x0, y0, t0 = self._first_pos_time[gid]
                dt_init = max(ts - t0, dt, 1e-2)
                dx = x_utm - x0
                dy = y_utm - y0
                dist = math.hypot(dx, dy)
                v_init = float(np.clip(dist / dt_init, 0.5, 45.0))
                th_obs = math.atan2(dy, dx) if dist > 0.05 else flt.state.theta_rad
                if self.road_engine is not None and dist > 0.05:
                    _, th_road = self.road_engine.snap_initial_heading(x_utm, y_utm)
                    if abs(math.cos(th_obs - th_road)) > 0.5:
                        th_init = circular_mean([th_obs, th_road], weights=[0.7, 0.3])
                    else:
                        th_init = th_obs
                else:
                    th_init = th_obs
                flt.initialize(x_utm, y_utm, v0=v_init, theta0=th_init)
                st = flt.state
            else:
                if dt > 0.0:
                    flt.predict(dt)
                st = flt.update(np.array([x_utm, y_utm], dtype=np.float64), r_k, timestamp=ts)

            self._last_active_time[gid] = ts
            self._last_active_state[gid] = st
            self._last_dims[gid] = dims_wl
            if emb is not None:
                if gid in self._last_reid:
                    blended = 0.7 * self._last_reid[gid] + 0.3 * emb
                    norm_b = float(np.linalg.norm(blended))
                    self._last_reid[gid] = (blended / max(norm_b, 1e-6)).astype(np.float32)
                else:
                    self._last_reid[gid] = emb
            if plate is not None and plate.format_valid:
                self._last_plate[gid] = plate
            if gid in self._blind_records:
                del self._blind_records[gid]

            camera_tracks.setdefault(cam_id, []).append(
                {
                    "global_id": gid,
                    "local_track_id": ltr.local_track_id,
                    "bbox_xyxy": ltr.bbox_xyxy.tolist(),
                    "conf": float(ltr.conf),
                    "class_id": int(ltr.class_id),
                    "status": "ACTIVE",
                    "v_mps": float(st.v_mps),
                    "x_utm": float(st.x_utm),
                    "y_utm": float(st.y_utm),
                    "plate": self._last_plate[gid].normalized_text if gid in self._last_plate else None,
                }
            )

        # 6. Cross-camera association for new appearances
        if new_candidates:
            apps = [item[2] for item in new_candidates]
            matched_by_local: dict[tuple[str, int], int] = {}

            if self._blind_records:
                matches = self.associator.associate(list(self._blind_records.values()), apps)
                # Map local_track_id back to (cam_id, local_track_id)
                for m in matches:
                    # Find corresponding candidate
                    for cam_id, ltr, app in new_candidates:
                        if app.local_track_id == m.local_track_id and (cam_id, ltr.local_track_id) not in matched_by_local:
                            matched_by_local[(cam_id, ltr.local_track_id)] = m.global_id
                            gid = m.global_id
                            self._local_to_global[(cam_id, ltr.local_track_id)] = gid
                            if gid in self._blind_records:
                                del self._blind_records[gid]
                            if m.forensic_path is not None:
                                self._forensic_history.append(m.forensic_path)
                            events.append(
                                f"Re-ID G{gid} in {cam_id} (local={ltr.local_track_id}, mode={m.mode_used})"
                            )
                            break

            for cam_id, ltr, app in new_candidates:
                key = (cam_id, ltr.local_track_id)
                if key in matched_by_local:
                    gid = matched_by_local[key]
                    flt = self._filters[gid]
                    st = flt.reset_position_covariance(
                        np.array([app.x_utm, app.y_utm], dtype=np.float64),
                        app.R_k,
                    )
                    self._first_pos_time[gid] = (app.x_utm, app.y_utm, ts)
                    self._hit_counts[gid] = 1
                else:
                    gid = self._next_global_id
                    self._next_global_id += 1
                    self._local_to_global[key] = gid
                    flt = create_filter(self.config.ekf)
                    th0 = 0.0
                    if self.road_engine is not None:
                        try:
                            _, th0 = self.road_engine.snap_initial_heading(app.x_utm, app.y_utm)
                        except Exception:
                            th0 = 0.0
                    st = flt.initialize(app.x_utm, app.y_utm, v0=8.0, theta0=th0)
                    self._filters[gid] = flt
                    self._first_pos_time[gid] = (app.x_utm, app.y_utm, ts)
                    self._hit_counts[gid] = 1

                self._last_active_time[gid] = ts
                self._last_active_state[gid] = st
                self._last_dims[gid] = app.dims_ground_wl
                if app.reid_emb is not None:
                    self._last_reid[gid] = app.reid_emb
                if app.plate is not None and app.plate.format_valid:
                    self._last_plate[gid] = app.plate

                camera_tracks.setdefault(cam_id, []).append(
                    {
                        "global_id": gid,
                        "local_track_id": ltr.local_track_id,
                        "bbox_xyxy": ltr.bbox_xyxy.tolist(),
                        "conf": float(ltr.conf),
                        "class_id": int(ltr.class_id),
                        "status": "ACTIVE",
                        "v_mps": float(st.v_mps),
                        "x_utm": float(st.x_utm),
                        "y_utm": float(st.y_utm),
                        "plate": self._last_plate[gid].normalized_text if gid in self._last_plate else None,
                    }
                )

        # 7. Assemble global states, blind hypotheses, and CICC police blockade points
        global_states: dict[int, FilterState] = {
            gid: flt.state for gid, flt in self._filters.items()
        }
        blind_hypotheses: dict[int, list[BlindHypothesis]] = {}
        all_active_hyps: list[BlindHypothesis] = []
        for gid, rec in self._blind_records.items():
            surviving = [h for h in rec.hypotheses if not h.pruned_by_vcrit and h.probability > 0.0]
            if surviving:
                blind_hypotheses[gid] = surviving
                all_active_hyps.extend(surviving)

        blockade_points: list[BlockadePoint] = []
        if self.road_engine is not None and all_active_hyps:
            blockade_points = self.road_engine.recommend_blockade_points(all_active_hyps, top_k=3)

        return PipelineFrameState(
            timestamp_s=ts,
            frames_bgr=frames_bgr,
            camera_tracks=camera_tracks,
            global_states=global_states,
            blind_hypotheses=blind_hypotheses,
            blockade_points=blockade_points,
            forensic_paths=list(self._forensic_history),
            events=events,
        )

    def run(self, max_frames: int | None = None) -> dict[str, Any]:
        """Run the causal pipeline over configured camera video sources and return a summary dict."""
        caps: dict[str, cv2.VideoCapture] = {}
        for cam in self.config.cameras:
            src_path = Path(cam.source)
            if src_path.is_file():
                cap = cv2.VideoCapture(str(src_path))
                if cap.isOpened():
                    caps[cam.cam_id] = cap

        fps = self.config.cameras[0].fps if self.config.cameras else 10.0
        dt = 1.0 / max(float(fps), 1.0)
        limit = int(max_frames) if max_frames is not None else 30
        frames_processed = 0
        last_state: PipelineFrameState | None = None

        try:
            for idx in range(limit):
                frames_bgr: dict[str, np.ndarray] = {}
                if caps:
                    for cam_id, cap in caps.items():
                        ok, frame = cap.read()
                        if ok and frame is not None:
                            frames_bgr[cam_id] = frame
                    if not frames_bgr:
                        break
                else:
                    for cam in self.config.cameras:
                        w, h = cam.resolution
                        frames_bgr[cam.cam_id] = np.full((h, w, 3), 40, dtype=np.uint8)

                last_state = self.step(idx * dt, frames_bgr)
                frames_processed += 1
        finally:
            for cap in caps.values():
                cap.release()

        return {
            "frames_processed": frames_processed,
            "num_global_tracks": len(self._filters),
            "num_forensic_paths": len(self._forensic_history),
            "last_timestamp_s": last_state.timestamp_s if last_state is not None else 0.0,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Pipeline_IA strictly causal multi-camera vehicle tracking pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to PipelineConfig YAML.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum number of synchronous frames to process.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        pipe = CausalMultiCamPipeline(cfg)
        summary = pipe.run(max_frames=args.max_frames)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary))
        return 0
    except Exception as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
