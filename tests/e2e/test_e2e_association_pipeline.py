from __future__ import annotations

import inspect
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


def _unit_vec(seed_idx: int, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed_idx)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _make_filter_state(
    x_utm: float,
    y_utm: float,
    v_mps: float = 10.0,
    theta_rad: float = 0.0,
    cov_diag: tuple[float, float] = (2.0, 2.0),
) -> Any:
    from tracker.ekf import FilterState

    P = np.diag([cov_diag[0], cov_diag[1], 1.0, 0.2, 0.1]).astype(np.float64)
    x = np.array([x_utm, y_utm, v_mps, theta_rad, 0.0], dtype=np.float64)
    return FilterState(
        x=x,
        P=P,
        x_utm=x_utm,
        y_utm=y_utm,
        v_mps=v_mps,
        theta_rad=theta_rad,
        P_pos_2x2=P[:2, :2].copy(),
    )


def _find_edge_near(
    road_network: Any,
    u_xy: tuple[float, float],
    v_xy: tuple[float, float],
    tol_m: float = 15.0,
) -> tuple[int, int, int]:
    best_eid = None
    best_dist = float("inf")
    for eid, edge in road_network.edges.items():
        ux, uy = road_network.nodes_utm[edge.u]
        vx, vy = road_network.nodes_utm[edge.v]
        d = math.hypot(ux - u_xy[0], uy - u_xy[1]) + math.hypot(vx - v_xy[0], vy - v_xy[1])
        if d < best_dist:
            best_dist = d
            best_eid = eid
    assert best_eid is not None and best_dist <= tol_m * 2.0, (
        f"No edge near {u_xy} -> {v_xy} (closest dist={best_dist:.2f}m)"
    )
    return best_eid


def _make_blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    frame[140:360, :] = (70, 70, 70)
    return frame


def _draw_vehicle(
    frame: np.ndarray,
    bbox_xyxy: tuple[float, float, float, float],
    color: tuple[int, int, int] = (30, 30, 220),
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    x1 = max(0, min(w, int(round(bbox_xyxy[0]))))
    y1 = max(0, min(h, int(round(bbox_xyxy[1]))))
    x2 = max(0, min(w, int(round(bbox_xyxy[2]))))
    y2 = max(0, min(h, int(round(bbox_xyxy[3]))))
    if x2 > x1 and y2 > y1:
        out[y1:y2, x1:x2] = color
    return out


def _make_homography_to_utm(x0_utm: float, y0_utm: float, scale_m_per_px: float = 0.125) -> list[list[float]]:
    """Maps 640x480 image coords around (u=320, v=240) to UTM meters around (x0_utm, y0_utm)."""
    # Pixel (u, v) -> Ground (x0 + (u - 320)*s, y0 - (v - 240)*s)
    # Note: y_utm increases North while pixel v increases South
    H = np.array(
        [
            [scale_m_per_px, 0.0, x0_utm - 320.0 * scale_m_per_px],
            [0.0, -scale_m_per_px, y0_utm + 240.0 * scale_m_per_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return H.tolist()


def _utm_to_bbox(
    x_utm: float,
    y_utm: float,
    x0_utm: float,
    y0_utm: float,
    scale_m_per_px: float = 0.125,
    box_w_px: float = 48.0,
    box_h_px: float = 36.0,
) -> tuple[float, float, float, float]:
    # Inverse of _make_homography_to_utm with contact point at y1 + 0.90*h
    u_c = 320.0 + (x_utm - x0_utm) / scale_m_per_px
    v_c = 240.0 - (y_utm - y0_utm) / scale_m_per_px
    x1 = u_c - box_w_px * 0.5
    x2 = u_c + box_w_px * 0.5
    y1 = v_c - 0.90 * box_h_px
    y2 = y1 + box_h_px
    return (x1, y1, x2, y2)


def _build_local_perception(
    detector_cfg: Any,
    exit_zones: list[list[tuple[float, float]]],
    detector_fn: Callable[[np.ndarray], np.ndarray],
) -> Any:
    from tracker.detector import LocalPerception

    sig = inspect.signature(LocalPerception.__init__)
    params = sig.parameters
    for kw in ("detector", "detector_fn", "detect_fn", "detection_fn", "predictor"):
        if kw in params:
            return LocalPerception(detector_cfg, exit_zones, **{kw: detector_fn})
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


def _build_pipeline_config(
    tmp_path: Path,
    synthetic_reid_onnx_path: Path,
    synthetic_graphml_path: Path,
    utm_origin: tuple[float, float, int],
    *,
    cam1_center_x: float | None = None,
    cam2_center_x: float | None = None,
) -> Any:
    from tracker.config import (
        ALPRConfig,
        AssociationConfig,
        CameraConfig,
        DetectorConfig,
        EKFConfig,
        PipelineConfig,
        ReIDConfig,
        RoadGraphConfig,
        VisualizerConfig,
    )

    ox, oy, _ = utm_origin
    c1x = cam1_center_x if cam1_center_x is not None else ox + 40.0
    c2x = cam2_center_x if cam2_center_x is not None else ox + 260.0

    cam1 = CameraConfig(
        cam_id="cam_1",
        source=str(tmp_path / "cam_1.mp4"),
        fps=10.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography_to_utm(c1x, oy),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(560.0, 140.0), (640.0, 140.0), (640.0, 340.0), (560.0, 340.0)]],
    )
    cam2 = CameraConfig(
        cam_id="cam_2",
        source=str(tmp_path / "cam_2.mp4"),
        fps=10.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography_to_utm(c2x, oy),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(560.0, 140.0), (640.0, 140.0), (640.0, 340.0), (560.0, 340.0)]],
    )

    det_cfg = DetectorConfig(
        model_path=tmp_path / "yolo_unused_di.pt",
        tracker_type="bytetrack",
        conf_thresh=0.35,
        iou_thresh=0.5,
        classes=[2, 3, 5, 7],
        device="cpu",
    )
    reid_cfg = ReIDConfig(
        onnx_path=synthetic_reid_onnx_path,
        device="cpu",
        input_size=(256, 128),
        allow_hsv_fallback=False,
    )
    alpr_cfg = ALPRConfig(enabled=False, backend="auto", min_conf=0.5)
    ekf_cfg = EKFConfig(model_type="adaptive_bicycle", sigma_px=(4.0, 4.0))
    rg_cfg = RoadGraphConfig(
        place_name=None,
        bbox_wgs84=None,
        graphml_cache=synthetic_graphml_path,
        horizon_s=15.0,
        min_branch_prob=0.02,
    )
    assoc_cfg = AssociationConfig(
        mode="hybrid",
        chi2_thresh=25.0,
        max_reid_dist=0.6,
        auto_calibrate_scale=True,
        min_scale_samples=3,
    )
    vis_cfg = VisualizerConfig(output_video=None, fps=10.0, width=1280, height=480)

    return PipelineConfig(
        cameras=[cam1, cam2],
        detector=det_cfg,
        reid=reid_cfg,
        alpr=alpr_cfg,
        ekf=ekf_cfg,
        road_graph=rg_cfg,
        association=assoc_cfg,
        visualizer=vis_cfg,
    )


def _instantiate_pipeline(
    cfg: Any,
    road_network: Any,
    road_engine: Any,
    perceptions: dict[str, Any],
    detectors: dict[str, Callable[[np.ndarray], np.ndarray]],
) -> Any:
    from tracker.pipeline import CausalMultiCamPipeline

    sig = inspect.signature(CausalMultiCamPipeline.__init__)
    params = sig.parameters
    kwargs: dict[str, Any] = {}
    if "road_network" in params:
        kwargs["road_network"] = road_network
    if "road_engine" in params:
        kwargs["road_engine"] = road_engine
    if "perceptions" in params:
        kwargs["perceptions"] = perceptions
    elif "local_perceptions" in params:
        kwargs["local_perceptions"] = perceptions
    if "detectors" in params:
        kwargs["detectors"] = detectors
    elif "detector_fns" in params:
        kwargs["detector_fns"] = detectors
    if not any(k in kwargs for k in ("perceptions", "local_perceptions", "detectors", "detector_fns")):
        pytest.fail(
            f"CausalMultiCamPipeline.__init__ does not expose a compatible perception/detector DI parameter; "
            f"found parameters: {list(params.keys())}"
        )
    return CausalMultiCamPipeline(cfg, **kwargs)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# =============================================================================
# Area 7: Association & Backtracking (tracker.association) — Tier 1 (6 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area7
def test_a7_t1_01_scale_self_calibration_median_ratio() -> None:
    from tracker.association import CrossCameraAssociator
    from tracker.config import AssociationConfig

    cfg = AssociationConfig(auto_calibrate_scale=True, min_scale_samples=5)
    assoc = CrossCameraAssociator(cfg)

    # 6 clean samples around W2/W1=1.25, L2/L1=0.80 + 1 gross outlier
    pairs = [
        ((2.0, 4.5), (2.50, 3.60)),
        ((1.8, 4.0), (2.25, 3.20)),
        ((2.2, 5.0), (2.75, 4.00)),
        ((2.0, 4.2), (2.52, 3.36)),
        ((1.9, 4.4), (2.37, 3.52)),
        ((2.1, 4.6), (2.63, 3.68)),
        ((2.0, 4.5), (8.00, 0.90)),  # outlier
    ]
    for wl1, wl2 in pairs:
        assoc.update_scale_calibration("cam_1", wl1, "cam_2", wl2)

    cw, cl = assoc.get_scale_factors("cam_1", "cam_2")
    assert cw == pytest.approx(1.25, abs=0.05)
    assert cl == pytest.approx(0.80, abs=0.05)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area7
def test_a7_t1_02_hybrid_association_matches_blind_track(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig

    ox, oy, _ = utm_origin
    cfg = AssociationConfig(mode="hybrid", chi2_thresh=13.28, max_reid_dist=0.45)
    assoc = CrossCameraAssociator(cfg)

    emb_a = _unit_vec(101)
    emb_b = (0.97 * emb_a + 0.03 * _unit_vec(102)).astype(np.float32)
    emb_b /= np.linalg.norm(emb_b)

    from tracker.roadgraph import BlindHypothesis

    hyp = BlindHypothesis(
        hypothesis_id="h_7",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=100.0,
        x_utm=ox + 100.0,
        y_utm=oy + 200.0,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([4.0, 4.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )

    rec = BlindTrackRecord(
        global_id=7,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox + 40.0, oy + 200.0),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb_a,
        plate=None,
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=12,
        entry_time=5.0,
        x_utm=ox + 100.8,
        y_utm=oy + 199.5,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb_b,
        plate=None,
    )

    matches = assoc.associate([rec], [app])
    assert len(matches) == 1
    m = matches[0]
    assert m.global_id == 7
    assert m.local_track_id == 12
    assert m.mode_used == "hybrid"
    assert math.isfinite(m.score) and m.score >= 0.0


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area7
def test_a7_t1_03_alpr_override_resolves_ambiguous_grid(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.alpr import PlateReading
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    cfg = AssociationConfig(
        mode="hybrid",
        use_alpr_override=True,
        alpr_override_conf=0.85,
        chi2_thresh=13.28,
    )
    assoc = CrossCameraAssociator(cfg)

    shared_emb = _unit_vec(201)
    hyp1 = BlindHypothesis(
        hypothesis_id="h1",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=50.0,
        x_utm=ox + 105.0,
        y_utm=oy + 100.0,
        v_mps=10.0,
        theta_rad=0.0,
        P_2x2=np.diag([25.0, 25.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )
    hyp2 = BlindHypothesis(
        hypothesis_id="h2",
        edge_sequence=[(0, 2, 0)],
        current_edge=(0, 2, 0),
        s_m=50.0,
        x_utm=ox + 101.0,
        y_utm=oy + 100.0,
        v_mps=10.0,
        theta_rad=0.0,
        P_2x2=np.diag([25.0, 25.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )

    rec1 = BlindTrackRecord(
        global_id=1,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox + 50.0, oy + 100.0),
        dims_ground_wl=(1.8, 4.4),
        reid_emb=shared_emb,
        plate=PlateReading("ABC-1D23", "ABC1D23", 0.94, None, True),
        hypotheses=[hyp1],
    )
    rec2 = BlindTrackRecord(
        global_id=2,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox + 50.0, oy + 100.0),
        dims_ground_wl=(1.8, 4.4),
        reid_emb=shared_emb,
        plate=PlateReading("DEF-4G56", "DEF4G56", 0.95, None, True),
        hypotheses=[hyp2],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=33,
        entry_time=5.0,
        x_utm=ox + 100.0,
        y_utm=oy + 100.0,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.4),
        reid_emb=shared_emb,
        plate=PlateReading("ABC1D23", "ABC1D23", 0.93, None, True),
    )

    matches = assoc.associate([rec1, rec2], [app])
    assert len(matches) == 1
    assert matches[0].global_id == 1
    assert matches[0].local_track_id == 33
    assert "alpr" in matches[0].mode_used.lower()


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area7
@pytest.mark.parametrize("mode", ["hybrid", "chi2", "reid_only", "article"])
def test_a7_t1_04_all_ablation_modes_execute_cleanly(
    mode: str,
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    assoc = CrossCameraAssociator(
        AssociationConfig(mode=mode, chi2_thresh=13.28, max_reid_dist=0.45)
    )
    emb = _unit_vec(301)
    hyp = BlindHypothesis(
        hypothesis_id="h_mode",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=40.0,
        x_utm=ox + 80.0,
        y_utm=oy,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([4.0, 4.0]).astype(np.float64),
        probability=1.0,
        eta_s=3.0,
    )
    rec = BlindTrackRecord(
        global_id=10,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox + 40.0, oy),
        dims_ground_wl=(1.8, 4.2),
        reid_emb=emb,
        plate=None,
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=20,
        entry_time=3.0,
        x_utm=ox + 80.5,
        y_utm=oy + 0.2,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.2),
        reid_emb=emb,
        plate=None,
    )

    matches = assoc.associate([rec], [app])
    assert len(matches) == 1
    assert matches[0].global_id == 10
    assert matches[0].local_track_id == 20


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area7
def test_a7_t1_05_forensic_backtracking_reconstructs_path(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
) -> None:
    from tracker.association import (
        BlindTrackRecord,
        CrossCameraAssociator,
        NewAppearance,
        ReconstructedBlindPath,
    )
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    e_01 = _find_edge_near(synthetic_road_network, (ox, oy), (ox + 100.0, oy))
    e_12 = _find_edge_near(synthetic_road_network, (ox + 100.0, oy), (ox + 200.0, oy))
    e_14 = _find_edge_near(synthetic_road_network, (ox + 100.0, oy), (ox + 100.0, oy - 120.0))

    hyp_main = BlindHypothesis(
        hypothesis_id="h_main",
        edge_sequence=[e_01, e_12],
        current_edge=e_12,
        s_m=100.0,
        x_utm=ox + 200.0,
        y_utm=oy,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([4.0, 4.0]).astype(np.float64),
        probability=0.7,
        eta_s=8.0,
    )
    hyp_alt = BlindHypothesis(
        hypothesis_id="h_alt",
        edge_sequence=[e_01, e_14],
        current_edge=e_14,
        s_m=120.0,
        x_utm=ox + 100.0,
        y_utm=oy - 120.0,
        v_mps=8.0,
        theta_rad=-math.pi / 2.0,
        P_2x2=np.diag([6.0, 6.0]).astype(np.float64),
        probability=0.3,
        eta_s=10.0,
    )

    emb = _unit_vec(401)
    rec = BlindTrackRecord(
        global_id=5,
        cam_id="cam_1",
        exit_time=1.0,
        exit_state=_make_filter_state(ox + 50.0, oy),
        dims_ground_wl=(1.8, 4.4),
        reid_emb=emb,
        plate=None,
        hypotheses=[hyp_main, hyp_alt],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=9,
        entry_time=9.0,
        x_utm=ox + 200.5,
        y_utm=oy + 0.3,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.4),
        reid_emb=emb,
        plate=None,
    )

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid"), road_engine=synthetic_road_engine)
    matches = assoc.associate([rec], [app])
    assert len(matches) == 1
    fp = matches[0].forensic_path
    assert isinstance(fp, ReconstructedBlindPath)
    assert fp.edge_ids == [e_01, e_12]
    assert len(fp.node_ids) >= 2
    assert fp.polyline_utm.ndim == 2 and fp.polyline_utm.shape[1] == 2
    assert fp.duration_s > 0.0
    assert 0.0 < fp.confidence <= 1.0
    assert fp.pruned_alternatives >= 1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area7
def test_a7_t1_06_scale_calibration_normalizes_dimensions_in_association(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    cfg_cal = AssociationConfig(
        mode="hybrid",
        auto_calibrate_scale=True,
        min_scale_samples=5,
        chi2_thresh=13.28,
    )
    assoc_cal = CrossCameraAssociator(cfg_cal)
    for _ in range(6):
        assoc_cal.update_scale_calibration("cam_1", (2.0, 4.0), "cam_2", (3.0, 6.0))

    assoc_uncal = CrossCameraAssociator(
        AssociationConfig(mode="hybrid", auto_calibrate_scale=False)
    )

    emb = _unit_vec(501)
    hyp = BlindHypothesis(
        hypothesis_id="h_scale",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=50.0,
        x_utm=ox + 60.0,
        y_utm=oy,
        v_mps=10.0,
        theta_rad=0.0,
        P_2x2=np.diag([4.0, 4.0]).astype(np.float64),
        probability=1.0,
        eta_s=4.0,
    )
    rec = BlindTrackRecord(
        global_id=14,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox + 20.0, oy),
        dims_ground_wl=(2.0, 4.0),
        reid_emb=emb,
        plate=None,
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=44,
        entry_time=4.0,
        x_utm=ox + 60.2,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(3.0, 6.0),  # 1.5x scaled in cam_2
        reid_emb=emb,
        plate=None,
    )

    m_cal = assoc_cal.associate([rec], [app])
    m_uncal = assoc_uncal.associate([rec], [app])
    assert len(m_cal) == 1
    assert m_cal[0].global_id == 14
    if m_uncal:
        assert m_cal[0].score <= m_uncal[0].score or math.isclose(m_cal[0].score, m_uncal[0].score, rel_tol=0.1)


# =============================================================================
# Area 7: Association & Backtracking — Tier 2 Boundary & Corner Cases (6 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area7
def test_a7_t2_01_conflicting_high_conf_alpr_veto(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.alpr import PlateReading
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    emb = _unit_vec(601)
    hyp = BlindHypothesis(
        hypothesis_id="h_veto",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=50.0,
        x_utm=ox + 50.0,
        y_utm=oy,
        v_mps=10.0,
        theta_rad=0.0,
        P_2x2=np.diag([2.0, 2.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )
    rec = BlindTrackRecord(
        global_id=1,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox, oy),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=PlateReading("ABC1D23", "ABC1D23", 0.95, None, True),
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=2,
        entry_time=5.0,
        x_utm=ox + 50.0,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=PlateReading("XYZ9W88", "XYZ9W88", 0.96, None, True),
    )

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid", use_alpr_override=True))
    assert assoc.associate([rec], [app]) == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area7
def test_a7_t2_02_scale_calibration_below_min_samples_and_zero_guard() -> None:
    from tracker.association import CrossCameraAssociator
    from tracker.config import AssociationConfig

    assoc = CrossCameraAssociator(
        AssociationConfig(auto_calibrate_scale=True, min_scale_samples=5)
    )
    assert assoc.get_scale_factors("cam_1", "cam_2") == (1.0, 1.0)

    for _ in range(4):
        assoc.update_scale_calibration("cam_1", (2.0, 4.0), "cam_2", (3.0, 6.0))
    assert assoc.get_scale_factors("cam_1", "cam_2") == (1.0, 1.0)

    # Zero / near-zero dimension guard
    assoc.update_scale_calibration("cam_1", (0.0, 2.0), "cam_2", (2.0, 0.0))
    cw, cl = assoc.get_scale_factors("cam_1", "cam_2")
    assert math.isfinite(cw) and math.isfinite(cl)
    assert 0.1 <= cw <= 10.0 and 0.1 <= cl <= 10.0


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area7
def test_a7_t2_03_max_reid_dist_threshold_rejection(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    emb_a = np.zeros(512, dtype=np.float32)
    emb_a[0] = 1.0
    emb_b = np.zeros(512, dtype=np.float32)
    emb_b[1] = 1.0  # orthogonal -> cosine distance = 1.0 > 0.45

    hyp = BlindHypothesis(
        hypothesis_id="h_reid_gate",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=50.0,
        x_utm=ox + 50.0,
        y_utm=oy,
        v_mps=10.0,
        theta_rad=0.0,
        P_2x2=np.diag([2.0, 2.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )
    rec = BlindTrackRecord(
        global_id=1,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox, oy),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb_a,
        plate=None,
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=2,
        entry_time=5.0,
        x_utm=ox + 50.0,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb_b,
        plate=None,
    )

    for mode in ("hybrid", "reid_only"):
        assoc = CrossCameraAssociator(AssociationConfig(mode=mode, max_reid_dist=0.45))
        assert assoc.associate([rec], [app]) == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area7
def test_a7_t2_04_empty_inputs_and_negative_dt_rejected(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    emb = _unit_vec(701)
    hyp = BlindHypothesis(
        hypothesis_id="h_dt",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=20.0,
        x_utm=ox + 20.0,
        y_utm=oy,
        v_mps=10.0,
        theta_rad=0.0,
        P_2x2=np.diag([2.0, 2.0]).astype(np.float64),
        probability=1.0,
        eta_s=2.0,
    )
    rec = BlindTrackRecord(
        global_id=1,
        cam_id="cam_1",
        exit_time=5.0,
        exit_state=_make_filter_state(ox, oy),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=None,
        hypotheses=[hyp],
    )
    app_past = NewAppearance(
        cam_id="cam_2",
        local_track_id=2,
        entry_time=3.5,  # dt = -1.5 s < 0
        x_utm=ox + 20.0,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=None,
    )

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid"))
    assert assoc.associate([], [app_past]) == []
    assert assoc.associate([rec], []) == []
    assert assoc.associate([rec], [app_past]) == []


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area7
def test_a7_t2_05_one_to_one_assignment_and_tie_consistency(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    e1 = _unit_vec(801)
    e2 = _unit_vec(802)

    def make_rec(gid: int, emb: np.ndarray) -> BlindTrackRecord:
        hyp = BlindHypothesis(
            hypothesis_id=f"h_{gid}",
            edge_sequence=[(0, 1, 0)],
            current_edge=(0, 1, 0),
            s_m=50.0,
            x_utm=ox + 50.0,
            y_utm=oy,
            v_mps=10.0,
            theta_rad=0.0,
            P_2x2=np.diag([4.0, 4.0]).astype(np.float64),
            probability=1.0,
            eta_s=5.0,
        )
        return BlindTrackRecord(
            global_id=gid,
            cam_id="cam_1",
            exit_time=0.0,
            exit_state=_make_filter_state(ox, oy),
            dims_ground_wl=(1.8, 4.3),
            reid_emb=emb,
            plate=None,
            hypotheses=[hyp],
        )

    rec1 = make_rec(1, e1)
    rec2 = make_rec(2, e2)
    app10 = NewAppearance(
        cam_id="cam_2",
        local_track_id=10,
        entry_time=5.0,
        x_utm=ox + 50.0,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=e2,  # matches rec2
        plate=None,
    )
    app11 = NewAppearance(
        cam_id="cam_2",
        local_track_id=11,
        entry_time=5.0,
        x_utm=ox + 50.0,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=e1,  # matches rec1
        plate=None,
    )

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid", max_reid_dist=0.5))
    matches = assoc.associate([rec1, rec2], [app10, app11])
    assert len(matches) == 2
    gids = [m.global_id for m in matches]
    lids = [m.local_track_id for m in matches]
    assert len(set(gids)) == 2
    assert len(set(lids)) == 2
    pair_map = {m.global_id: m.local_track_id for m in matches}
    assert pair_map[1] == 11
    assert pair_map[2] == 10


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area7
def test_a7_t2_06_forensic_backtracking_shortest_path_fallback(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
) -> None:
    from tracker.alpr import PlateReading
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig

    ox, oy, _ = utm_origin
    emb = _unit_vec(901)
    rec = BlindTrackRecord(
        global_id=77,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=_make_filter_state(ox + 20.0, oy),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=PlateReading("BRA2E19", "BRA2E19", 0.95, None, True),
        hypotheses=[],  # No surviving branch reaching cam_2
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=88,
        entry_time=12.0,
        x_utm=ox + 250.0,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=PlateReading("BRA2E19", "BRA2E19", 0.96, None, True),
    )

    assoc = CrossCameraAssociator(
        AssociationConfig(mode="hybrid", use_alpr_override=True, alpr_override_conf=0.85),
        road_engine=synthetic_road_engine,
    )
    matches = assoc.associate([rec], [app])
    assert len(matches) == 1
    assert matches[0].global_id == 77
    if matches[0].forensic_path is not None:
        assert len(matches[0].forensic_path.edge_ids) >= 1
        assert 0.0 <= matches[0].forensic_path.confidence <= 1.0


# =============================================================================
# Area 8: Causal Pipeline (tracker.pipeline) — Tier 1 (5 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area8
def test_a8_t1_01_pipeline_step_returns_valid_frame_state(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    from tracker.pipeline import PipelineFrameState

    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )

    def det_cam1(_frame: np.ndarray) -> np.ndarray:
        bbox = _utm_to_bbox(ox + 20.0, oy, ox + 40.0, oy)
        return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0.92, 2.0]], dtype=np.float32)

    def det_cam2(_frame: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    bbox = _utm_to_bbox(ox + 20.0, oy, ox + 40.0, oy)
    frame1 = _draw_vehicle(_make_blank_frame(), bbox)
    frame2 = _make_blank_frame()
    state = pipeline.step(timestamp_s=0.1, frames_bgr={"cam_1": frame1, "cam_2": frame2})

    assert isinstance(state, PipelineFrameState)
    assert state.timestamp_s == pytest.approx(0.1)
    assert "cam_1" in state.camera_tracks
    assert isinstance(state.global_states, dict)
    assert isinstance(state.blockade_points, list)
    assert isinstance(state.forensic_paths, list)
    assert isinstance(state.events, list)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area8
def test_a8_t1_02_causal_ekf_initialization_and_tracking(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )
    step = {"i": 0}
    dt = 0.2
    true_v = 10.0  # eastbound along N0->N1 (theta = 0.0)

    def det_cam1(_frame: np.ndarray) -> np.ndarray:
        x_utm = ox + 10.0 + step["i"] * true_v * dt
        bbox = _utm_to_bbox(x_utm, oy, ox + 40.0, oy)
        return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0.93, 2.0]], dtype=np.float32)

    def det_cam2(_frame: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    last_state = None
    for i in range(7):
        step["i"] = i
        x_utm = ox + 10.0 + i * true_v * dt
        bbox = _utm_to_bbox(x_utm, oy, ox + 40.0, oy)
        f1 = _draw_vehicle(_make_blank_frame(), bbox)
        last_state = pipeline.step(timestamp_s=i * dt, frames_bgr={"cam_1": f1, "cam_2": _make_blank_frame()})

    assert last_state is not None
    assert len(last_state.global_states) >= 1
    st = next(iter(last_state.global_states.values()))
    assert st.v_mps > 2.0
    assert abs(_wrap_angle(st.theta_rad - 0.0)) < 0.35


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area8
def test_a8_t1_03_exit_zone_triggers_blind_hypotheses_and_blockades(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )
    step = {"i": 0}
    dt = 0.2

    def det_cam1(_frame: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 6:
            # Moves into ExitZone at east border (u >= 560 => x_utm >= ox + 70)
            x_utm = ox + 45.0 + i * 5.0
            bbox = _utm_to_bbox(x_utm, oy, ox + 40.0, oy)
            return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0.93, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    def det_cam2(_frame: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    saw_blind = False
    saw_blockade = False
    for i in range(18):
        step["i"] = i
        f1 = _make_blank_frame()
        if i <= 6:
            x_utm = ox + 45.0 + i * 5.0
            f1 = _draw_vehicle(f1, _utm_to_bbox(x_utm, oy, ox + 40.0, oy))
        st = pipeline.step(timestamp_s=i * dt, frames_bgr={"cam_1": f1, "cam_2": _make_blank_frame()})
        if st.blind_hypotheses:
            for hyps in st.blind_hypotheses.values():
                if hyps:
                    saw_blind = True
                    prob_sum = sum(h.probability for h in hyps)
                    assert prob_sum <= 1.0 + 1e-6
                    assert all(h.probability > 0.0 for h in hyps)
        if st.blockade_points:
            saw_blockade = True
            assert all(b.eta_seconds > 0.0 for b in st.blockade_points)

    assert saw_blind
    assert saw_blockade


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area8
def test_a8_t1_04_cross_camera_reacquisition_and_forensic_path(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cam2_cx = ox + 250.0
    cfg = _build_pipeline_config(
        tmp_path,
        synthetic_reid_onnx_path,
        synthetic_graphml_path,
        utm_origin,
        cam2_center_x=cam2_cx,
    )
    step = {"i": 0}
    dt = 0.2
    v_mps = 12.0  # 2.4 m per 0.2s step

    def det_cam1(_frame: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 15:
            x_utm = ox + 40.0 + i * v_mps * dt  # 40 -> 76 (inside east exit zone)
            bbox = _utm_to_bbox(x_utm, oy, ox + 40.0, oy)
            return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0.93, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    def det_cam2(_frame: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i >= 75:
            # 40 + 75*2.4 = 220.0 -> cleanly inside cam_2 [210, 290) centered at 250
            x_utm = ox + 40.0 + i * v_mps * dt
            if x_utm <= ox + 285.0:
                bbox = _utm_to_bbox(x_utm, oy, cam2_cx, oy)
                return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0.93, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    cam1_gids: set[int] = set()
    cam2_gids: set[int] = set()
    forensic_count = 0

    for i in range(95):
        step["i"] = i
        x_utm = ox + 40.0 + i * v_mps * dt
        f1 = _make_blank_frame()
        f2 = _make_blank_frame()
        if i <= 15:
            f1 = _draw_vehicle(f1, _utm_to_bbox(x_utm, oy, ox + 40.0, oy), (30, 30, 220))
        if i >= 75 and x_utm <= ox + 285.0:
            f2 = _draw_vehicle(f2, _utm_to_bbox(x_utm, oy, cam2_cx, oy), (30, 30, 220))

        st = pipeline.step(timestamp_s=i * dt, frames_bgr={"cam_1": f1, "cam_2": f2})
        for tr in st.camera_tracks.get("cam_1", []):
            if "global_id" in tr and tr["global_id"] is not None:
                cam1_gids.add(int(tr["global_id"]))
        for tr in st.camera_tracks.get("cam_2", []):
            if "global_id" in tr and tr["global_id"] is not None:
                cam2_gids.add(int(tr["global_id"]))
        forensic_count = max(forensic_count, len(st.forensic_paths))

    assert len(cam1_gids) == 1
    assert cam2_gids == cam1_gids
    assert forensic_count >= 1


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area8
def test_a8_t1_05_pipeline_run_and_cli_entrypoint(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    import cv2

    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )
    # Write short synthetic mp4s for cam_1 and cam_2 so pipeline.run() can read real files if needed
    for cam in cfg.cameras:
        writer = cv2.VideoWriter(
            cam.source,
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (640, 480),
        )
        for i in range(6):
            frame = _make_blank_frame()
            if cam.cam_id == "cam_1":
                frame = _draw_vehicle(frame, _utm_to_bbox(ox + 20.0 + i * 2.0, oy, ox + 40.0, oy))
            writer.write(frame)
        writer.release()

    step = {"i": 0}

    def det_cam1(_frame: np.ndarray) -> np.ndarray:
        i = step["i"]
        step["i"] += 1
        bbox = _utm_to_bbox(ox + 20.0 + min(i, 5) * 2.0, oy, ox + 40.0, oy)
        return np.array([[bbox[0], bbox[1], bbox[2], bbox[3], 0.90, 2.0]], dtype=np.float32)

    def det_cam2(_frame: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    run_sig = inspect.signature(pipeline.run)
    if "max_frames" in run_sig.parameters:
        result = pipeline.run(max_frames=6)
    else:
        result = pipeline.run()
    assert isinstance(result, dict)

    proc = subprocess.run(
        [str(venv_python), "-m", "tracker.pipeline", "--help"],
        cwd=str(repo_root),
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"CLI --help failed: {proc.stderr}"


# =============================================================================
# Area 8: Causal Pipeline — Tier 2 Boundary & Corner Cases (5 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area8
def test_a8_t2_01_strict_causality_no_lookahead_and_monotonic_timestamps(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin

    def make_pipe() -> tuple[Any, dict[str, int]]:
        cfg = _build_pipeline_config(
            tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
        )
        step = {"i": 0}

        def d1(_f: np.ndarray) -> np.ndarray:
            x_utm = ox + 20.0 + step["i"] * 2.0
            b = _utm_to_bbox(x_utm, oy, ox + 40.0, oy)
            return np.array([[b[0], b[1], b[2], b[3], 0.91, 2.0]], dtype=np.float32)

        def d2(_f: np.ndarray) -> np.ndarray:
            return np.empty((0, 6), dtype=np.float32)

        percs = {
            "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, d1),
            "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, d2),
        }
        return _instantiate_pipeline(cfg, synthetic_road_network, synthetic_road_engine, percs, {"cam_1": d1, "cam_2": d2}), step

    pipe_a, step_a = make_pipe()
    pipe_b, step_b = make_pipe()

    frames_prefix = [
        (i * 0.2, _draw_vehicle(_make_blank_frame(), _utm_to_bbox(ox + 20.0 + i * 2.0, oy, ox + 40.0, oy)))
        for i in range(6)
    ]

    states_a = []
    for i, (t, f1) in enumerate(frames_prefix):
        step_a["i"] = i
        states_a.append(pipe_a.step(t, {"cam_1": f1, "cam_2": _make_blank_frame()}))

    # Non-monotonic timestamp must raise ValueError
    with pytest.raises(ValueError):
        pipe_a.step(0.5, {"cam_1": _make_blank_frame(), "cam_2": _make_blank_frame()})

    # Full stream on pipe_b must produce bit-identical prefix states for t <= 1.0
    states_b = []
    for i in range(15):
        step_b["i"] = i
        t = i * 0.2
        f1 = _draw_vehicle(_make_blank_frame(), _utm_to_bbox(ox + 20.0 + i * 2.0, oy, ox + 40.0, oy))
        states_b.append(pipe_b.step(t, {"cam_1": f1, "cam_2": _make_blank_frame()}))

    for sa, sb in zip(states_a, states_b[: len(states_a)]):
        assert sa.global_states.keys() == sb.global_states.keys()
        for gid in sa.global_states:
            assert np.allclose(sa.global_states[gid].x, sb.global_states[gid].x, atol=0.0, rtol=0.0)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area8
def test_a8_t2_02_empty_camera_dict_and_black_frames(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )
    step = {"i": 0}

    def det_cam1(frame: np.ndarray) -> np.ndarray:
        if frame.sum() == 0:
            return np.empty((0, 6), dtype=np.float32)
        b = _utm_to_bbox(ox + 50.0 + step["i"] * 5.0, oy, ox + 40.0, oy)
        return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)

    def det_cam2(_f: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    # Empty frames dict before any target
    st0 = pipeline.step(0.0, {})
    assert st0.camera_tracks == {} or all(len(v) == 0 for v in st0.camera_tracks.values())

    # Drive vehicle into exit zone then pass black frames and empty dict
    for i in range(1, 7):
        step["i"] = i
        f1 = _draw_vehicle(_make_blank_frame(), _utm_to_bbox(ox + 50.0 + i * 5.0, oy, ox + 40.0, oy))
        pipeline.step(i * 0.2, {"cam_1": f1, "cam_2": _make_blank_frame()})

    black = np.zeros((480, 640, 3), dtype=np.uint8)
    st_black = pipeline.step(1.4, {"cam_1": black, "cam_2": black})
    assert all(len(v) == 0 for v in st_black.camera_tracks.values())

    st_empty = pipeline.step(1.6, {})
    assert isinstance(st_empty.blind_hypotheses, dict)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area8
def test_a8_t2_03_single_active_camera_dropped_feed(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )
    step = {"i": 0}

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        b = _utm_to_bbox(ox + 20.0 + step["i"] * 2.0, oy, ox + 40.0, oy)
        return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)

    def det_cam2(_f: np.ndarray) -> np.ndarray:
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    for i in range(6):
        step["i"] = i
        f1 = _draw_vehicle(_make_blank_frame(), _utm_to_bbox(ox + 20.0 + i * 2.0, oy, ox + 40.0, oy))
        st = pipeline.step(i * 0.2, {"cam_1": f1})
        assert "cam_1" in st.camera_tracks
        if i >= 3:
            assert len(st.camera_tracks["cam_1"]) >= 1
            assert len(st.global_states) >= 1


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area8
def test_a8_t2_04_concurrent_multi_vehicle_blind_crossings(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cam2_cx = ox + 250.0
    cfg = _build_pipeline_config(
        tmp_path,
        synthetic_reid_onnx_path,
        synthetic_graphml_path,
        utm_origin,
        cam2_center_x=cam2_cx,
    )
    step = {"i": 0}
    dt = 0.2
    v = 12.0

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        dets = []
        if i <= 15:
            xa = ox + 40.0 + i * v * dt
            ba = _utm_to_bbox(xa, oy + 2.0, ox + 40.0, oy)
            dets.append([ba[0], ba[1], ba[2], ba[3], 0.92, 2.0])
        if 2 <= i <= 17:
            xb = ox + 40.0 + (i - 2) * v * dt
            bb = _utm_to_bbox(xb, oy - 2.0, ox + 40.0, oy)
            dets.append([bb[0], bb[1], bb[2], bb[3], 0.91, 2.0])
        return np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

    def det_cam2(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        dets = []
        if 75 <= i <= 90:
            xa = ox + 40.0 + i * v * dt
            ba = _utm_to_bbox(xa, oy + 2.0, cam2_cx, oy)
            dets.append([ba[0], ba[1], ba[2], ba[3], 0.92, 2.0])
        if 77 <= i <= 92:
            xb = ox + 40.0 + (i - 2) * v * dt
            bb = _utm_to_bbox(xb, oy - 2.0, cam2_cx, oy)
            dets.append([bb[0], bb[1], bb[2], bb[3], 0.91, 2.0])
        return np.array(dets, dtype=np.float32) if dets else np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    cam1_ids_top: set[int] = set()
    cam1_ids_bot: set[int] = set()
    cam2_ids_top: set[int] = set()
    cam2_ids_bot: set[int] = set()

    for i in range(95):
        step["i"] = i
        f1 = _make_blank_frame()
        f2 = _make_blank_frame()
        if i <= 15:
            f1 = _draw_vehicle(f1, _utm_to_bbox(ox + 40.0 + i * v * dt, oy + 2.0, ox + 40.0, oy), (20, 20, 230))
        if 2 <= i <= 17:
            f1 = _draw_vehicle(f1, _utm_to_bbox(ox + 40.0 + (i - 2) * v * dt, oy - 2.0, ox + 40.0, oy), (230, 60, 20))
        if 75 <= i <= 90:
            f2 = _draw_vehicle(f2, _utm_to_bbox(ox + 40.0 + i * v * dt, oy + 2.0, cam2_cx, oy), (20, 20, 230))
        if 77 <= i <= 92:
            f2 = _draw_vehicle(f2, _utm_to_bbox(ox + 40.0 + (i - 2) * v * dt, oy - 2.0, cam2_cx, oy), (230, 60, 20))

        st = pipeline.step(i * dt, {"cam_1": f1, "cam_2": f2})
        for tr in st.camera_tracks.get("cam_1", []):
            gid = tr.get("global_id")
            bbox = tr.get("bbox_xyxy")
            if gid is not None and bbox is not None:
                cy = 0.5 * (bbox[1] + bbox[3])
                if cy < 240.0:
                    cam1_ids_top.add(int(gid))
                else:
                    cam1_ids_bot.add(int(gid))
        for tr in st.camera_tracks.get("cam_2", []):
            gid = tr.get("global_id")
            bbox = tr.get("bbox_xyxy")
            if gid is not None and bbox is not None:
                cy = 0.5 * (bbox[1] + bbox[3])
                if cy < 240.0:
                    cam2_ids_top.add(int(gid))
                else:
                    cam2_ids_bot.add(int(gid))

    assert len(cam1_ids_top) == 1 and len(cam1_ids_bot) == 1
    assert cam1_ids_top != cam1_ids_bot
    assert cam2_ids_top == cam1_ids_top
    assert cam2_ids_bot == cam1_ids_bot


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area8
def test_a8_t2_05_local_occlusion_vs_exit_zone_lifecycle(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    ox, oy, _ = utm_origin
    cfg = _build_pipeline_config(
        tmp_path, synthetic_reid_onnx_path, synthetic_graphml_path, utm_origin
    )
    step = {"i": 0}
    dt = 0.2
    v = 10.0

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 4:
            x = ox + 10.0 + i * v * dt
        elif 5 <= i <= 8:
            # 0.8 s center occlusion
            return np.empty((0, 6), dtype=np.float32)
        elif 9 <= i <= 25:
            x = ox + 10.0 + i * v * dt  # moves into exit zone
        else:
            return np.empty((0, 6), dtype=np.float32)
        b = _utm_to_bbox(x, oy, ox + 40.0, oy)
        return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)

    def det_cam2(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        if 80 <= i <= 90:
            x = ox + 10.0 + i * v * dt
            b = _utm_to_bbox(x, oy, ox + 260.0, oy)
            return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1),
        "cam_2": _build_local_perception(cfg.detector, cfg.cameras[1].exit_zones, det_cam2),
    }
    pipeline = _instantiate_pipeline(
        cfg,
        synthetic_road_network,
        synthetic_road_engine,
        perceptions,
        {"cam_1": det_cam1, "cam_2": det_cam2},
    )

    saw_blind_during_gap = False
    for i in range(91):
        step["i"] = i
        f1 = _make_blank_frame()
        f2 = _make_blank_frame()
        if i <= 4 or (9 <= i <= 25):
            f1 = _draw_vehicle(f1, _utm_to_bbox(ox + 10.0 + i * v * dt, oy, ox + 40.0, oy))
        if 80 <= i <= 90:
            f2 = _draw_vehicle(f2, _utm_to_bbox(ox + 10.0 + i * v * dt, oy, ox + 260.0, oy))

        st = pipeline.step(i * dt, {"cam_1": f1, "cam_2": f2})
        if 5 <= i <= 8:
            # Center occlusion must NOT spawn blind_hypotheses
            assert all(len(hyps) == 0 for hyps in st.blind_hypotheses.values())
        elif 30 <= i <= 75:
            if any(len(hyps) > 0 for hyps in st.blind_hypotheses.values()):
                saw_blind_during_gap = True
        elif i >= 85:
            # Post-reacquisition clears stale blind hypotheses
            assert all(len(hyps) == 0 for hyps in st.blind_hypotheses.values())

    assert saw_blind_during_gap
