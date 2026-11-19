from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


def _make_blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 40, dtype=np.uint8)
    frame[140:360, :] = (70, 70, 70)
    return frame


def _draw_vehicle_box(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    bgr_color: tuple[int, int, int] = (30, 30, 220),
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    x1 = max(0, min(w, int(round(bbox_xyxy[0]))))
    y1 = max(0, min(h, int(round(bbox_xyxy[1]))))
    x2 = max(0, min(w, int(round(bbox_xyxy[2]))))
    y2 = max(0, min(h, int(round(bbox_xyxy[3]))))
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = bgr_color
    return out


def _build_local_perception(
    detector_cfg: Any,
    exit_zones: list[list[tuple[float, float]]],
    detector_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Any:
    from tracker.detector import LocalPerception

    if detector_fn is None:
        return LocalPerception(detector_cfg, exit_zones)

    sig = inspect.signature(LocalPerception.__init__)
    params = sig.parameters
    for kw_name in ("detector", "detector_fn", "detect_fn", "detection_fn", "predictor"):
        if kw_name in params:
            return LocalPerception(detector_cfg, exit_zones, **{kw_name: detector_fn})

    # Fallback: try positional 3rd argument if accepted, otherwise default init
    positional_params = [
        p
        for p in params.values()
        if p.name != "self"
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional_params) >= 3:
        return LocalPerception(detector_cfg, exit_zones, detector_fn)
    pytest.fail(
        f"LocalPerception.__init__ does not expose a compatible detector DI parameter; "
        f"found parameters: {list(params.keys())}"
    )


def _get_crops_and_zones(
    synthetic_bgr_frames_and_crops: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[tuple[float, float]]]]:
    if isinstance(synthetic_bgr_frames_and_crops, dict):
        crop_red_a = synthetic_bgr_frames_and_crops["crop_red_a"]
        crop_red_a_shifted = synthetic_bgr_frames_and_crops["crop_red_a_shifted"]
        crop_blue_b = synthetic_bgr_frames_and_crops["crop_blue_b"]
        exit_zones = synthetic_bgr_frames_and_crops["exit_zones_east"]
        return crop_red_a, crop_red_a_shifted, crop_blue_b, exit_zones
    raise TypeError("synthetic_bgr_frames_and_crops fixture must be a dict")


# =============================================================================
# Area 5: Detection & ExitZones (tracker.detector) — Tier 1 (6 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area5
def test_a5_t1_01_vehicle_detection_and_sct_id_persistence(
    tmp_path: Path,
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.config import DetectorConfig

    _, _, _, exit_zones = _get_crops_and_zones(synthetic_bgr_frames_and_crops)
    frame_idx = {"i": 0}

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        i = frame_idx["i"]
        x1 = 100.0 + i * 20.0
        return np.array([[x1, 200.0, x1 + 80.0, 260.0, 0.90, 2.0]], dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        iou_thresh=0.5,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, exit_zones, detector_fn=detect_fn)

    track_ids: list[int] = []
    hits_seq: list[int] = []
    for i in range(10):
        frame_idx["i"] = i
        x1 = 100.0 + i * 20.0
        frame = _draw_vehicle_box(_make_blank_frame(), (x1, 200.0, x1 + 80.0, 260.0))
        tracks = perception.process_frame(frame)
        active = [t for t in tracks if t.status == "ACTIVE"]
        if active:
            t = active[0]
            assert t.class_id == 2
            assert t.conf >= 0.35
            assert isinstance(t.bbox_xyxy, np.ndarray)
            assert t.bbox_xyxy.shape == (4,)
            assert t.time_since_update == 0
            track_ids.append(t.local_track_id)
            hits_seq.append(t.hits)

    assert len(track_ids) >= 7
    # After warm-up, track ID must remain identical and hits must increase monotonically
    assert len(set(track_ids[-5:])) == 1
    assert all(b >= a for a, b in zip(hits_seq, hits_seq[1:]))
    assert hits_seq[-1] > hits_seq[0]


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area5
def test_a5_t1_02_non_vehicle_classes_filtered(
    tmp_path: Path,
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.config import DetectorConfig

    _, _, _, exit_zones = _get_crops_and_zones(synthetic_bgr_frames_and_crops)

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [40.0, 180.0, 90.0, 280.0, 0.95, 0.0],   # person (filtered)
                [130.0, 190.0, 180.0, 260.0, 0.92, 1.0], # bicycle (filtered)
                [240.0, 200.0, 320.0, 260.0, 0.88, 2.0], # car (kept)
                [380.0, 180.0, 490.0, 270.0, 0.85, 7.0], # truck (kept)
            ],
            dtype=np.float32,
        )

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, exit_zones, detector_fn=detect_fn)

    observed_classes: set[int] = set()
    for _ in range(4):
        frame = _make_blank_frame()
        frame = _draw_vehicle_box(frame, (240.0, 200.0, 320.0, 260.0))
        frame = _draw_vehicle_box(frame, (380.0, 180.0, 490.0, 270.0), (200, 120, 40))
        tracks = perception.process_frame(frame)
        for t in tracks:
            observed_classes.add(t.class_id)
            assert t.class_id in {2, 3, 5, 7}

    assert 0 not in observed_classes
    assert 1 not in observed_classes
    assert {2, 7}.issubset(observed_classes)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area5
def test_a5_t1_03_center_fov_short_occlusion_stays_occluded_local(
    tmp_path: Path,
) -> None:
    from tracker.config import DetectorConfig

    exit_zones = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]
    step = {"i": 0}

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        i = step["i"]
        if 0 <= i <= 5:
            x1 = 240.0 + i * 6.0
            return np.array([[x1, 200.0, x1 + 60.0, 260.0, 0.92, 2.0]], dtype=np.float32)
        if 6 <= i <= 10:
            return np.empty((0, 6), dtype=np.float32)
        x1 = 276.0 + (i - 10) * 6.0
        return np.array([[x1, 200.0, x1 + 60.0, 260.0, 0.92, 2.0]], dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, exit_zones, detector_fn=detect_fn)

    pre_ids: list[int] = []
    post_ids: list[int] = []
    for i in range(16):
        step["i"] = i
        frame = _make_blank_frame()
        if i <= 5:
            x1 = 240.0 + i * 6.0
            frame = _draw_vehicle_box(frame, (x1, 200.0, x1 + 60.0, 260.0))
        elif i >= 11:
            x1 = 276.0 + (i - 10) * 6.0
            frame = _draw_vehicle_box(frame, (x1, 200.0, x1 + 60.0, 260.0))

        tracks = perception.process_frame(frame)
        for t in tracks:
            assert t.status != "EXITED_TO_BLIND"
            if i <= 5 and t.status == "ACTIVE":
                pre_ids.append(t.local_track_id)
            elif 6 <= i <= 10:
                assert t.status == "OCCLUDED_LOCAL"
            elif i >= 11 and t.status == "ACTIVE":
                post_ids.append(t.local_track_id)

    assert len(pre_ids) >= 1
    assert len(post_ids) >= 1
    assert pre_ids[-1] == post_ids[0]


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area5
def test_a5_t1_04_exit_zone_departure_triggers_exited_to_blind(
    tmp_path: Path,
) -> None:
    from tracker.config import DetectorConfig

    exit_zones = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]
    step = {"i": 0}

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 8:
            # Progressively moves from center x1=350 into exit zone x1=574
            x1 = 350.0 + i * 28.0
            return np.array([[x1, 210.0, x1 + 60.0, 275.0, 0.91, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, exit_zones, detector_fn=detect_fn)

    blind_events = []
    for i in range(13):
        step["i"] = i
        frame = _make_blank_frame()
        if i <= 8:
            x1 = 350.0 + i * 28.0
            frame = _draw_vehicle_box(frame, (x1, 210.0, x1 + 60.0, 275.0))
        tracks = perception.process_frame(frame)
        for t in tracks:
            if t.status == "EXITED_TO_BLIND":
                blind_events.append(t)

    assert len(blind_events) >= 1
    assert blind_events[0].exit_zone_id is not None


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area5
def test_a5_t1_05_bytetrack_and_botsort_backends(
    tmp_path: Path,
) -> None:
    from tracker.config import DetectorConfig

    exit_zones = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]

    for tracker_type in ("bytetrack", "botsort"):
        step = {"i": 0}

        def detect_fn(_frame: np.ndarray, _step: dict[str, int] = step) -> np.ndarray:
            i = _step["i"]
            ax1 = 80.0 + i * 20.0
            bx1 = 480.0 - i * 20.0
            return np.array(
                [
                    [ax1, 160.0, ax1 + 60.0, 210.0, 0.90, 2.0],
                    [bx1, 280.0, bx1 + 70.0, 340.0, 0.88, 7.0],
                ],
                dtype=np.float32,
            )

        cfg = DetectorConfig(
            model_path=tmp_path / f"yolo_{tracker_type}.pt",
            tracker_type=tracker_type,
            conf_thresh=0.35,
            classes=[2, 3, 5, 7],
            device="cpu",
        )
        perception = _build_local_perception(cfg, exit_zones, detector_fn=detect_fn)

        id_history_by_class: dict[int, list[int]] = {2: [], 7: []}
        for i in range(8):
            step["i"] = i
            ax1 = 80.0 + i * 20.0
            bx1 = 480.0 - i * 20.0
            frame = _make_blank_frame()
            frame = _draw_vehicle_box(frame, (ax1, 160.0, ax1 + 60.0, 210.0), (30, 30, 220))
            frame = _draw_vehicle_box(frame, (bx1, 280.0, bx1 + 70.0, 340.0), (220, 60, 30))
            tracks = perception.process_frame(frame)
            for t in tracks:
                if t.status == "ACTIVE" and t.class_id in id_history_by_class:
                    id_history_by_class[t.class_id].append(t.local_track_id)

        assert len(id_history_by_class[2]) >= 4
        assert len(id_history_by_class[7]) >= 4
        assert len(set(id_history_by_class[2][-4:])) == 1
        assert len(set(id_history_by_class[7][-4:])) == 1
        assert id_history_by_class[2][-1] != id_history_by_class[7][-1]


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area5
def test_a5_t1_06_crop_bgr_matches_vehicle_roi(
    tmp_path: Path,
) -> None:
    from tracker.config import DetectorConfig

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        return np.array([[120.0, 180.0, 220.0, 260.0, 0.94, 2.0]], dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, [], detector_fn=detect_fn)
    frame = _draw_vehicle_box(_make_blank_frame(), (120.0, 180.0, 220.0, 260.0), (30, 30, 220))

    active_tracks = []
    for _ in range(3):
        tracks = perception.process_frame(frame)
        active_tracks = [t for t in tracks if t.status == "ACTIVE"]

    assert len(active_tracks) == 1
    crop = active_tracks[0].crop_bgr
    assert crop is not None
    assert crop.dtype == np.uint8
    assert abs(crop.shape[0] - 80) <= 2
    assert abs(crop.shape[1] - 100) <= 2
    assert float(crop[:, :, 2].mean()) > 180.0
    assert float(crop[:, :, 0].mean()) < 60.0


# =============================================================================
# Area 5: Detection & ExitZones — Tier 2 Boundary & Corner Cases (6 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area5
def test_a5_t2_01_empty_frame_zero_detections(tmp_path: Path) -> None:
    from tracker.config import DetectorConfig

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, [], detector_fn=detect_fn)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    for _ in range(5):
        assert perception.process_frame(blank) == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area5
def test_a5_t2_02_empty_and_degenerate_exit_zones(tmp_path: Path) -> None:
    from tracker.config import DetectorConfig

    step = {"i": 0}

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        if step["i"] < 3:
            return np.array([[280.0, 200.0, 340.0, 260.0, 0.90, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )

    # (a) Empty exit_zones=[]
    p_empty = _build_local_perception(cfg, [], detector_fn=detect_fn)
    for i in range(6):
        step["i"] = i
        tracks = p_empty.process_frame(_make_blank_frame())
        assert all(t.status != "EXITED_TO_BLIND" for t in tracks)

    # (b) Degenerate <3-vertex polygon
    step["i"] = 0
    try:
        p_degen = _build_local_perception(
            cfg, [[(10.0, 10.0), (20.0, 20.0)]], detector_fn=detect_fn
        )
    except ValueError:
        return

    for i in range(6):
        step["i"] = i
        tracks = p_degen.process_frame(_make_blank_frame())
        assert isinstance(tracks, list)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area5
def test_a5_t2_03_boundary_touching_and_clipped_bbox(tmp_path: Path) -> None:
    from tracker.config import DetectorConfig

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [-6.0, 200.0, 50.0, 260.0, 0.90, 2.0],
                [600.0, 440.0, 650.0, 490.0, 0.91, 2.0],
                [650.0, 490.0, 700.0, 520.0, 0.92, 2.0],
            ],
            dtype=np.float32,
        )

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, [], detector_fn=detect_fn)
    frame = _make_blank_frame()
    tracks = []
    for _ in range(3):
        tracks = perception.process_frame(frame)

    assert len(tracks) >= 1
    for t in tracks:
        x1, y1, x2, y2 = t.bbox_xyxy
        if x2 <= 0 or y2 <= 0 or x1 >= 640 or y1 >= 480:
            assert t.crop_bgr is None or t.crop_bgr.size == 0
        else:
            assert t.crop_bgr is not None
            assert t.crop_bgr.size > 0
            assert t.crop_bgr.shape[0] <= 480 and t.crop_bgr.shape[1] <= 640


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area5
def test_a5_t2_04_confidence_threshold_boundary(tmp_path: Path) -> None:
    from tracker.config import DetectorConfig

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [80.0, 180.0, 150.0, 240.0, 0.349, 2.0],  # below 0.35
                [300.0, 180.0, 380.0, 240.0, 0.351, 2.0], # above 0.35
            ],
            dtype=np.float32,
        )

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, [], detector_fn=detect_fn)
    frame = _make_blank_frame()
    active_centers: list[float] = []
    for _ in range(5):
        tracks = perception.process_frame(frame)
        for t in tracks:
            if t.status == "ACTIVE":
                cx = float((t.bbox_xyxy[0] + t.bbox_xyxy[2]) * 0.5)
                active_centers.append(cx)
                assert t.conf >= 0.35

    assert len(active_centers) >= 1
    assert all(cx > 250.0 for cx in active_centers)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area5
def test_a5_t2_05_missing_weights_no_silent_dummy(tmp_path: Path) -> None:
    from tracker.config import DetectorConfig

    cfg = DetectorConfig(
        model_path=tmp_path / "nonexistent_yolo.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        perception = _build_local_perception(cfg, [], detector_fn=None)
        perception.process_frame(_make_blank_frame())


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area5
def test_a5_t2_06_prolonged_center_occlusion_expires_without_blind(
    tmp_path: Path,
) -> None:
    from tracker.config import DetectorConfig

    exit_zones = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]
    step = {"i": 0}

    def detect_fn(_frame: np.ndarray) -> np.ndarray:
        if step["i"] < 5:
            return np.array([[260.0, 200.0, 320.0, 260.0, 0.90, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    perception = _build_local_perception(cfg, exit_zones, detector_fn=detect_fn)
    frame = _make_blank_frame()

    last_tracks = []
    for i in range(65):
        step["i"] = i
        tracks = perception.process_frame(frame)
        assert all(t.status != "EXITED_TO_BLIND" for t in tracks)
        last_tracks = tracks

    assert last_tracks == []


# =============================================================================
# Area 6: Re-ID & ALPR (tracker.reid, tracker.alpr) — Tier 1 (6 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area6
def test_a6_t1_01_reid_onnx_512d_l2_norm_and_cosine_bounds(
    synthetic_reid_onnx_path: Path,
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    crop_red_a, _, _, _ = _get_crops_and_zones(synthetic_bgr_frames_and_crops)
    cfg = ReIDConfig(
        onnx_path=synthetic_reid_onnx_path,
        device="cpu",
        input_size=(256, 128),
        allow_hsv_fallback=False,
    )
    ext = ReIDExtractor(cfg)
    emb = ext.extract(crop_red_a)

    assert emb.shape == (512,)
    assert emb.dtype == np.float32
    assert np.all(np.isfinite(emb))
    assert float(np.linalg.norm(emb)) == pytest.approx(1.0, abs=1e-4)
    assert ext.cosine_distance(emb, emb) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area6
def test_a6_t1_02_reid_batch_and_appearance_discrimination(
    synthetic_reid_onnx_path: Path,
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    crop_red_a, crop_red_a_shifted, crop_blue_b, _ = _get_crops_and_zones(
        synthetic_bgr_frames_and_crops
    )
    cfg = ReIDConfig(
        onnx_path=synthetic_reid_onnx_path,
        device="cpu",
        input_size=(256, 128),
        allow_hsv_fallback=False,
    )
    ext = ReIDExtractor(cfg)
    embs = ext.extract_batch([crop_red_a, crop_red_a_shifted, crop_blue_b])

    assert embs.shape == (3, 512)
    assert embs.dtype == np.float32
    for row in embs:
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-4)

    single = ext.extract(crop_red_a)
    assert np.allclose(embs[0], single, atol=1e-5)

    d_same_car = ext.cosine_distance(embs[0], embs[1])
    d_diff_car = ext.cosine_distance(embs[0], embs[2])
    assert 0.0 <= d_diff_car <= 2.0
    assert d_same_car < d_diff_car


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area6
def test_a6_t1_03_reid_hsv_hog_test_explicit_opt_in(
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    crop_red_a, crop_red_a_shifted, crop_blue_b, _ = _get_crops_and_zones(
        synthetic_bgr_frames_and_crops
    )
    cfg = ReIDConfig(
        onnx_path=Path("hsv_hog_test"),
        device="cpu",
        input_size=(256, 128),
        allow_hsv_fallback=True,
    )
    ext = ReIDExtractor(cfg)
    e1 = ext.extract(crop_red_a)
    e2 = ext.extract(crop_red_a_shifted)
    e3 = ext.extract(crop_blue_b)

    assert e1.shape == (512,)
    assert e1.dtype == np.float32
    assert float(np.linalg.norm(e1)) == pytest.approx(1.0, abs=1e-4)
    assert ext.cosine_distance(e1, e2) < ext.cosine_distance(e1, e3)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area6
def test_a6_t1_04_alpr_normalize_br_legacy_and_mercosul() -> None:
    from tracker.alpr import ALPREngine
    from tracker.config import ALPRConfig

    engine = ALPREngine(ALPRConfig(enabled=True, backend="auto", min_conf=0.5))
    cases = [
        ("abc-1234", ("ABC1234", True)),
        ("XYZ 9876", ("XYZ9876", True)),
        ("abc1d23", ("ABC1D23", True)),
        ("BRA-2E19", ("BRA2E19", True)),
    ]
    for raw, expected in cases:
        assert engine.normalize_plate(raw) == expected


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area6
def test_a6_t1_05_alpr_recognize_synthetic_plate_crop(
    synthetic_plate_crops: dict[str, np.ndarray],
) -> None:
    from tracker.alpr import ALPREngine, PlateReading
    from tracker.config import ALPRConfig

    plate_crop = synthetic_plate_crops["plate_mercosul_abc1d23"]
    engine = ALPREngine(ALPRConfig(enabled=True, backend="auto", min_conf=0.3))
    res = engine.recognize(plate_crop)

    assert isinstance(res, PlateReading)
    assert res.normalized_text == "ABC1D23"
    assert res.format_valid is True
    assert res.confidence >= 0.3


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area6
def test_a6_t1_06_alpr_disabled_returns_none(
    synthetic_plate_crops: dict[str, np.ndarray],
) -> None:
    from tracker.alpr import ALPREngine
    from tracker.config import ALPRConfig

    plate_crop = synthetic_plate_crops["plate_mercosul_abc1d23"]
    engine = ALPREngine(ALPRConfig(enabled=False, backend="auto", min_conf=0.5))
    assert engine.recognize(plate_crop) is None


# =============================================================================
# Area 6: Re-ID & ALPR — Tier 2 Boundary & Corner Cases (7 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
def test_a6_t2_01_reid_default_rejects_missing_onnx(tmp_path: Path) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    cfg = ReIDConfig(
        onnx_path=tmp_path / "missing.onnx",
        device="cpu",
        allow_hsv_fallback=False,
    )
    with pytest.raises((FileNotFoundError, RuntimeError, OSError)):
        ReIDExtractor(cfg)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
@pytest.mark.parametrize("backend_mode", ["onnx", "hsv_hog"])
def test_a6_t2_02_reid_tiny_blank_and_empty_crops(
    backend_mode: str,
    synthetic_reid_onnx_path: Path,
) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    if backend_mode == "onnx":
        cfg = ReIDConfig(
            onnx_path=synthetic_reid_onnx_path,
            device="cpu",
            allow_hsv_fallback=False,
        )
    else:
        cfg = ReIDConfig(
            onnx_path=Path("hsv_hog_test"),
            device="cpu",
            allow_hsv_fallback=True,
        )
    ext = ReIDExtractor(cfg)

    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        ext.extract(empty_crop)

    for crop in (
        np.full((2, 2, 3), 120, dtype=np.uint8),
        np.zeros((128, 64, 3), dtype=np.uint8),
    ):
        try:
            vec = ext.extract(crop)
        except ValueError:
            continue
        assert vec.shape == (512,)
        assert vec.dtype == np.float32
        assert np.all(np.isfinite(vec))
        norm = float(np.linalg.norm(vec))
        assert math.isclose(norm, 1.0, abs_tol=1e-4) or math.isclose(norm, 0.0, abs_tol=1e-6)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
def test_a6_t2_03_reid_extract_batch_empty_list(
    synthetic_reid_onnx_path: Path,
) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    cfg = ReIDConfig(
        onnx_path=synthetic_reid_onnx_path,
        device="cpu",
        allow_hsv_fallback=False,
    )
    ext = ReIDExtractor(cfg)
    out = ext.extract_batch([])
    assert isinstance(out, np.ndarray)
    assert out.shape == (0, 512)
    assert out.dtype == np.float32


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
def test_a6_t2_04_reid_cosine_distance_zero_norm_and_clamping(
    synthetic_reid_onnx_path: Path,
) -> None:
    from tracker.config import ReIDConfig
    from tracker.reid import ReIDExtractor

    cfg = ReIDConfig(
        onnx_path=synthetic_reid_onnx_path,
        device="cpu",
        allow_hsv_fallback=False,
    )
    ext = ReIDExtractor(cfg)
    a = np.zeros(512, dtype=np.float32)
    b = np.zeros(512, dtype=np.float32)
    b[0] = 1.0

    d_same = ext.cosine_distance(b, b)
    assert d_same >= 0.0
    assert d_same == pytest.approx(0.0, abs=1e-6)

    b_eps = b.copy()
    b_eps[1] = 1e-7
    assert ext.cosine_distance(b, b_eps) >= 0.0

    try:
        d_zero = ext.cosine_distance(a, b)
    except ValueError:
        return
    assert math.isfinite(d_zero)
    assert 0.0 <= d_zero <= 2.0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
def test_a6_t2_05_alpr_invalid_and_corrupted_plate_strings() -> None:
    from tracker.alpr import ALPREngine
    from tracker.config import ALPRConfig

    engine = ALPREngine(ALPRConfig(enabled=True, backend="auto", min_conf=0.5))
    invalid_inputs = [
        "",
        "1234567",
        "ABCDEFG",
        "AB1234",
        "ABC12D3",
        "ABC12345",
        "@@#$$%",
        "0001111",
    ]
    for raw in invalid_inputs:
        _norm, valid = engine.normalize_plate(raw)
        assert valid is False, f"Expected invalid plate for {raw!r}"


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
def test_a6_t2_06_alpr_noise_blank_and_low_confidence_rejected(
    synthetic_plate_crops: dict[str, np.ndarray],
) -> None:
    from tracker.alpr import ALPREngine
    from tracker.config import ALPRConfig

    engine = ALPREngine(ALPRConfig(enabled=True, backend="auto", min_conf=0.50))
    for key in ("plate_blank_black", "plate_gaussian_noise"):
        crop = synthetic_plate_crops[key]
        res = engine.recognize(crop)
        assert res is None

    # Impossible confidence threshold guarantees low-confidence readings are filtered out
    strict_engine = ALPREngine(ALPRConfig(enabled=True, backend="auto", min_conf=0.9999))
    blurry = synthetic_plate_crops["plate_mercosul_abc1d23"].copy()
    # Downsample/upsample with pure numpy block averaging to degrade contrast without needing cv2
    small = blurry[::4, ::4]
    degraded = np.repeat(np.repeat(small, 4, axis=0), 4, axis=1)[: blurry.shape[0], : blurry.shape[1]]
    res_strict = strict_engine.recognize(degraded)
    assert res_strict is None or res_strict.confidence >= 0.9999


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area6
def test_a6_t2_07_alpr_whitespace_lowercase_punctuation_cleanup() -> None:
    from tracker.alpr import ALPREngine
    from tracker.config import ALPRConfig

    engine = ALPREngine(ALPRConfig(enabled=True, backend="auto", min_conf=0.5))
    assert engine.normalize_plate("  rIo-2a19 \n\t") == ("RIO2A19", True)
    assert engine.normalize_plate("abc·1234") == ("ABC1234", True)
