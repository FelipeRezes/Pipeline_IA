"""Single-camera vehicle detection (Ultralytics YOLO), tracking (BoxMOT), and Shapely Exit Zones."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
from shapely.geometry import Point, Polygon, box

from tracker.config import DetectorConfig


@dataclass
class LocalTrack:
    """Single-camera tracked vehicle state with Exit Zone status separation."""

    local_track_id: int
    bbox_xyxy: np.ndarray
    conf: float
    class_id: int
    status: Literal["ACTIVE", "OCCLUDED_LOCAL", "EXITED_TO_BLIND"]
    exit_zone_id: str | None = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    crop_bgr: np.ndarray | None = None


@dataclass
class _InternalTrackState:
    track_id: int
    bbox_xyxy: np.ndarray
    velocity_xyxy: np.ndarray
    conf: float
    class_id: int
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    last_exit_zone_id: str | None = None
    emitted_blind: bool = False


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return float(inter / max(union, 1e-6))


def _extract_roi_crop(frame_bgr: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (
        int(round(float(bbox_xyxy[0]))),
        int(round(float(bbox_xyxy[1]))),
        int(round(float(bbox_xyxy[2]))),
        int(round(float(bbox_xyxy[3]))),
    )
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
        return None
    cx1 = max(0, min(w, x1))
    cy1 = max(0, min(h, y1))
    cx2 = max(0, min(w, x2))
    cy2 = max(0, min(h, y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return frame_bgr[cy1:cy2, cx1:cx2].copy()


class LocalPerception:
    """Intra-camera vehicle detector + multi-object tracker + Exit Zone classifier."""

    MAX_LOCAL_OCCLUSION_FRAMES: int = 30

    def __init__(
        self,
        detector_cfg: DetectorConfig,
        exit_zones: list[list[tuple[float, float]]] | None = None,
        detector_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        detector: Callable[[np.ndarray], np.ndarray] | None = None,
        detect_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self.cfg = detector_cfg
        self._allowed_classes = {int(c) for c in self.cfg.classes}
        self._detector_fn = detector_fn or detector or detect_fn
        self._yolo_model: Any | None = None
        self._boxmot_tracker: Any | None = None

        # Validate and build Shapely Exit Zone polygons
        self._exit_polygons: list[tuple[str, Polygon]] = []
        if exit_zones:
            for idx, poly_pts in enumerate(exit_zones):
                if len(poly_pts) < 3:
                    raise ValueError(f"Exit zone polygon {idx} must have at least 3 vertices")
                poly = Polygon([(float(x), float(y)) for x, y in poly_pts])
                if not poly.is_valid or poly.area <= 0.0:
                    raise ValueError(f"Exit zone polygon {idx} is degenerate or invalid")
                self._exit_polygons.append((f"exit_zone_{idx}", poly))

        # If no custom detector_fn is injected, validate that the model weights exist
        if self._detector_fn is None:
            model_path = Path(self.cfg.model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"YOLO detector weights not found: {model_path}")

        self._tracks: dict[int, _InternalTrackState] = {}
        self._next_track_id: int = 1

    def _init_boxmot_backend(self) -> None:
        """Initialize optional BoxMOT tracker backend (ByteTrack / BoTSORT / DeepOCSORT)."""
        try:
            import boxmot

            t_type = self.cfg.tracker_type.lower()
            if t_type == "bytetrack" and hasattr(boxmot, "ByteTrack"):
                self._boxmot_tracker = boxmot.ByteTrack()
            elif t_type == "botsort" and hasattr(boxmot, "BoTSORT"):
                self._boxmot_tracker = boxmot.BoTSORT(
                    reid_weights=Path("osnet_x0_25_msmt17.pt"),
                    device="cpu",
                    half=False,
                )
        except Exception:
            self._boxmot_tracker = None

    def _run_detector(self, frame_bgr: np.ndarray) -> np.ndarray:
        if self._detector_fn is not None:
            raw = self._detector_fn(frame_bgr)
            if raw is None or len(raw) == 0:
                return np.empty((0, 6), dtype=np.float64)
            arr = np.asarray(raw, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] < 6:
                return np.empty((0, 6), dtype=np.float64)
            return arr[:, :6]

        if self._yolo_model is None:
            from ultralytics import YOLO

            model_path = Path(self.cfg.model_path)
            if not model_path.is_file():
                raise FileNotFoundError(f"YOLO detector weights not found: {model_path}")
            self._yolo_model = YOLO(str(model_path))

        device = "cpu" if self.cfg.device == "auto" else self.cfg.device
        results = self._yolo_model.predict(
            frame_bgr,
            conf=float(self.cfg.conf_thresh),
            iou=float(self.cfg.iou_thresh),
            classes=list(self._allowed_classes),
            device=device,
            verbose=False,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return np.empty((0, 6), dtype=np.float64)
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = boxes.cls.cpu().numpy().reshape(-1, 1)
        return np.hstack([xyxy, conf, cls]).astype(np.float64)

    def _match_exit_zone(self, bbox_xyxy: np.ndarray) -> str | None:
        if not self._exit_polygons:
            return None
        x1, y1, x2, y2 = float(bbox_xyxy[0]), float(bbox_xyxy[1]), float(bbox_xyxy[2]), float(bbox_xyxy[3])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        contact_y = y1 + 0.90 * (y2 - y1)
        pt_center = Point(cx, cy)
        pt_contact = Point(cx, contact_y)
        b_poly = box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        b_area = max(b_poly.area, 1e-6)

        for zone_id, poly in self._exit_polygons:
            if poly.contains(pt_center) or poly.touches(pt_center):
                return zone_id
            if poly.contains(pt_contact) or poly.touches(pt_contact):
                return zone_id
            if poly.intersects(b_poly) and (poly.intersection(b_poly).area / b_area) >= 0.35:
                return zone_id
        return None

    def process_frame(self, frame_bgr: np.ndarray) -> list[LocalTrack]:
        """Detect, track, and classify vehicle tracks into ACTIVE, OCCLUDED_LOCAL, or EXITED_TO_BLIND."""
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.size == 0:
            return []

        raw_dets = self._run_detector(frame_bgr)
        valid_dets: list[tuple[np.ndarray, float, int]] = []
        for row in raw_dets:
            bbox = np.asarray(row[:4], dtype=np.float64)
            conf = float(row[4])
            cls_id = int(round(float(row[5])))
            if not np.all(np.isfinite(bbox)) or not np.isfinite(conf):
                continue
            if conf < float(self.cfg.conf_thresh):
                continue
            if cls_id not in self._allowed_classes:
                continue
            valid_dets.append((bbox, conf, cls_id))

        # Predict existing tracks forward one frame
        predicted_boxes: dict[int, np.ndarray] = {}
        for tid, trk in list(self._tracks.items()):
            if trk.emitted_blind:
                del self._tracks[tid]
                continue
            predicted_boxes[tid] = trk.bbox_xyxy + trk.velocity_xyxy

        # Greedy / bipartite matching between existing tracks and valid detections
        unmatched_track_ids = set(predicted_boxes.keys())
        unmatched_det_indices = set(range(len(valid_dets)))
        matched_pairs: list[tuple[int, int]] = []

        if unmatched_track_ids and unmatched_det_indices:
            candidate_pairs: list[tuple[float, int, int]] = []
            for tid in unmatched_track_ids:
                trk = self._tracks[tid]
                pred_b = predicted_boxes[tid]
                pred_cx = 0.5 * (pred_b[0] + pred_b[2])
                pred_cy = 0.5 * (pred_b[1] + pred_b[3])
                pred_diag = max(
                    float(np.hypot(pred_b[2] - pred_b[0], pred_b[3] - pred_b[1])),
                    20.0,
                )
                for d_idx in unmatched_det_indices:
                    det_b, _, det_cls = valid_dets[d_idx]
                    if det_cls != trk.class_id:
                        continue
                    iou = _bbox_iou(pred_b, det_b)
                    det_cx = 0.5 * (det_b[0] + det_b[2])
                    det_cy = 0.5 * (det_b[1] + det_b[3])
                    c_dist = float(np.hypot(det_cx - pred_cx, det_cy - pred_cy))
                    if iou >= 0.05 or c_dist <= 1.75 * pred_diag:
                        score = iou - 0.25 * (c_dist / pred_diag)
                        candidate_pairs.append((score, tid, d_idx))

            candidate_pairs.sort(key=lambda item: -item[0])
            for _, tid, d_idx in candidate_pairs:
                if tid in unmatched_track_ids and d_idx in unmatched_det_indices:
                    matched_pairs.append((tid, d_idx))
                    unmatched_track_ids.remove(tid)
                    unmatched_det_indices.remove(d_idx)

        output_tracks: list[LocalTrack] = []

        # 1. Update matched tracks -> ACTIVE
        for tid, d_idx in matched_pairs:
            trk = self._tracks[tid]
            det_b, det_conf, det_cls = valid_dets[d_idx]
            dt_steps = max(1, trk.time_since_update + 1)
            inst_vel = (det_b - trk.bbox_xyxy) / float(dt_steps)
            trk.velocity_xyxy = 0.6 * trk.velocity_xyxy + 0.4 * inst_vel
            trk.bbox_xyxy = det_b.copy()
            trk.conf = det_conf
            trk.class_id = det_cls
            trk.age += 1
            trk.hits += 1
            trk.time_since_update = 0
            trk.last_exit_zone_id = self._match_exit_zone(det_b)

            output_tracks.append(
                LocalTrack(
                    local_track_id=trk.track_id,
                    bbox_xyxy=trk.bbox_xyxy.copy(),
                    conf=trk.conf,
                    class_id=trk.class_id,
                    status="ACTIVE",
                    exit_zone_id=trk.last_exit_zone_id,
                    age=trk.age,
                    hits=trk.hits,
                    time_since_update=0,
                    crop_bgr=_extract_roi_crop(frame_bgr, trk.bbox_xyxy),
                )
            )

        # 2. Initialize new tracks for unmatched detections -> ACTIVE
        for d_idx in sorted(unmatched_det_indices):
            det_b, det_conf, det_cls = valid_dets[d_idx]
            tid = self._next_track_id
            self._next_track_id += 1
            ez_id = self._match_exit_zone(det_b)
            trk = _InternalTrackState(
                track_id=tid,
                bbox_xyxy=det_b.copy(),
                velocity_xyxy=np.zeros(4, dtype=np.float64),
                conf=det_conf,
                class_id=det_cls,
                age=1,
                hits=1,
                time_since_update=0,
                last_exit_zone_id=ez_id,
                emitted_blind=False,
            )
            self._tracks[tid] = trk
            output_tracks.append(
                LocalTrack(
                    local_track_id=tid,
                    bbox_xyxy=trk.bbox_xyxy.copy(),
                    conf=trk.conf,
                    class_id=trk.class_id,
                    status="ACTIVE",
                    exit_zone_id=ez_id,
                    age=1,
                    hits=1,
                    time_since_update=0,
                    crop_bgr=_extract_roi_crop(frame_bgr, trk.bbox_xyxy),
                )
            )

        # 3. Handle unmatched existing tracks -> EXITED_TO_BLIND vs OCCLUDED_LOCAL
        for tid in sorted(unmatched_track_ids):
            trk = self._tracks[tid]
            trk.age += 1
            trk.time_since_update += 1
            coasted_box = trk.bbox_xyxy + trk.velocity_xyxy
            ez_now = trk.last_exit_zone_id or self._match_exit_zone(coasted_box)

            if ez_now is not None:
                # Vehicle departed through an Exit Zone -> transition to EXITED_TO_BLIND
                trk.emitted_blind = True
                output_tracks.append(
                    LocalTrack(
                        local_track_id=trk.track_id,
                        bbox_xyxy=trk.bbox_xyxy.copy(),
                        conf=trk.conf,
                        class_id=trk.class_id,
                        status="EXITED_TO_BLIND",
                        exit_zone_id=ez_now,
                        age=trk.age,
                        hits=trk.hits,
                        time_since_update=trk.time_since_update,
                        crop_bgr=None,
                    )
                )
                del self._tracks[tid]
            else:
                # Center-of-FOV temporary occlusion -> stay OCCLUDED_LOCAL up to max buffer frames
                if trk.time_since_update > self.MAX_LOCAL_OCCLUSION_FRAMES:
                    del self._tracks[tid]
                    continue
                trk.bbox_xyxy = coasted_box
                trk.velocity_xyxy *= 0.85
                output_tracks.append(
                    LocalTrack(
                        local_track_id=trk.track_id,
                        bbox_xyxy=trk.bbox_xyxy.copy(),
                        conf=trk.conf,
                        class_id=trk.class_id,
                        status="OCCLUDED_LOCAL",
                        exit_zone_id=None,
                        age=trk.age,
                        hits=trk.hits,
                        time_since_update=trk.time_since_update,
                        crop_bgr=None,
                    )
                )

        output_tracks.sort(key=lambda t: t.local_track_id)
        return output_tracks
