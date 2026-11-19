from __future__ import annotations

import inspect
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


def _make_blank_frame(width: int = 640, height: int = 480) -> np.ndarray:
    frame = np.full((height, width, 3), 38, dtype=np.uint8)
    frame[140:360, :] = (72, 72, 72)
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


def _make_homography(x0_utm: float, y0_utm: float, scale_m_per_px: float = 0.125) -> list[list[float]]:
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


def _make_tilted_projector(utm_origin: tuple[float, float, int]) -> Any:
    from tracker.geometry import CameraProjector

    ox, oy, _ = utm_origin
    K = np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    pitch = math.radians(25.0)
    # Camera pitched down 25 deg looking North (+Y_utm)
    R_cam = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -math.sin(pitch), -math.cos(pitch)],
            [0.0, math.cos(pitch), -math.sin(pitch)],
        ],
        dtype=np.float64,
    )
    cam_pos = np.array([ox + 10.0, oy - 25.0, 8.0], dtype=np.float64)
    t = -R_cam @ cam_pos
    return CameraProjector.from_3d_params(K, R_cam, t, epsg_utm=32723)


# =============================================================================
# Tier 3: Cross-Feature Pairwise Integration Tests (T3-01 .. T3-12)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_01_config_and_prepare_datasets_manifest(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    from tracker.config import (
        CameraConfig,
        PipelineConfig,
        ScenarioConfig,
        load_config,
        load_scenarios,
        save_config,
    )

    out_dir = tmp_path / "prepared_data"
    script = repo_root / "scripts" / "prepare_datasets.py"
    proc = subprocess.run(
        [
            str(venv_python),
            str(script),
            "--datasets",
            "synthehicle",
            "--data-dir",
            str(out_dir),
            "--offline",
        ],
        cwd=str(repo_root),
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"prepare_datasets failed: {proc.stderr}"
    assert (out_dir / "MANIFEST.json").is_file()
    manifest = json.loads((out_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "entries" in manifest and any("synthehicle" in k for k in manifest["entries"])
    assert (out_dir / "synthehicle" / "Town01-N-day" / "calibration.json").is_file()

    # Validate configs/scenarios.yaml via load_scenarios
    scenarios_path = repo_root / "configs" / "scenarios.yaml"
    scenarios = load_scenarios(scenarios_path)
    assert isinstance(scenarios, dict) and len(scenarios) >= 1
    assert "demo_cicc_pursuit" in scenarios
    assert all(isinstance(sc, ScenarioConfig) for sc in scenarios.values())

    # Validate PipelineConfig via default/cameras_demo YAML if present, plus YAML roundtrip
    for yaml_name in ("default.yaml", "cameras_demo.yaml"):
        yaml_path = repo_root / "configs" / yaml_name
        if yaml_path.is_file():
            loaded_cfg = load_config(yaml_path)
            assert isinstance(loaded_cfg, PipelineConfig)

    sample_cfg = PipelineConfig(
        cameras=[
            CameraConfig(
                cam_id="cam_synth",
                source="synthetic://cam_synth",
                fps=15.0,
                resolution=(640, 480),
            )
        ]
    )
    roundtrip_yaml = tmp_path / "pipeline_roundtrip.yaml"
    save_config(sample_cfg, roundtrip_yaml)
    reloaded = load_config(roundtrip_yaml)
    assert isinstance(reloaded, PipelineConfig)
    assert len(reloaded.cameras) == 1
    assert reloaded.cameras[0].cam_id == "cam_synth"


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_02_camera_config_to_projector_roundtrip(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.config import CameraConfig
    from tracker.geometry import CameraProjector

    ox, oy, _ = utm_origin
    H = _make_homography(ox + 50.0, oy + 20.0)
    cam_h = CameraConfig(
        cam_id="cam_h",
        source="cam_h.mp4",
        fps=15.0,
        resolution=(640, 480),
        calib_path=None,
        homography=H,
        K=None,
        R=None,
        t=None,
        exit_zones=[],
    )
    proj_h = CameraProjector.from_homography(np.array(cam_h.homography, dtype=np.float64), epsg_utm=32723)
    gx, gy = proj_h.pixel_to_ground(340.0, 260.0)
    u2, v2 = proj_h.ground_to_pixel(gx, gy)
    assert math.hypot(u2 - 340.0, v2 - 260.0) < 1.0

    # Also test 3D K, R, t config path
    K = [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    pitch = math.radians(25.0)
    R = [
        [1.0, 0.0, 0.0],
        [0.0, -math.sin(pitch), -math.cos(pitch)],
        [0.0, math.cos(pitch), -math.sin(pitch)],
    ]
    t_vec = (-np.array(R) @ np.array([ox, oy - 25.0, 8.0])).tolist()
    cam_3d = CameraConfig(
        cam_id="cam_3d",
        source="cam_3d.mp4",
        fps=15.0,
        resolution=(640, 480),
        calib_path=None,
        homography=None,
        K=K,
        R=R,
        t=t_vec,
        exit_zones=[],
    )
    proj_3d = CameraProjector.from_3d_params(
        np.array(cam_3d.K, dtype=np.float64),
        np.array(cam_3d.R, dtype=np.float64),
        np.array(cam_3d.t, dtype=np.float64),
        epsg_utm=32723,
    )
    gx3, gy3 = proj_3d.pixel_to_ground(320.0, 280.0)
    u3, v3 = proj_3d.ground_to_pixel(gx3, gy3)
    assert math.hypot(u3 - 320.0, v3 - 280.0) < 1.0


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_03_geometry_bbox_to_ground_into_adaptive_ekf(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.ekf import AdaptiveEKF

    ox, oy, _ = utm_origin
    proj = _make_tilted_projector(utm_origin)

    rng = np.random.default_rng(17)
    dt = 0.1
    v_true = 10.0
    xs_gt = []
    ys_gt = []
    bboxes = []
    for k in range(30):
        xg = ox + 10.0
        yg = oy + 5.0 + k * v_true * dt
        xs_gt.append(xg)
        ys_gt.append(yg)
        uc, vc = proj.ground_to_pixel(xg, yg)
        # Simulate 3D bbox height (~60px) where contact point is 0.90*h from top
        h_box = 60.0
        w_box = 44.0
        nu, nv = rng.normal(0.0, 1.5, size=2)
        x1 = uc + nu - w_box * 0.5
        y1 = vc + nv - 0.90 * h_box
        bboxes.append((x1, y1, x1 + w_box, y1 + h_box))

    ekf_corr = AdaptiveEKF()
    ekf_corr.initialize(xs_gt[0], ys_gt[0], v0=v_true, theta0=math.pi / 2.0)

    ekf_naive = AdaptiveEKF()
    ekf_naive.initialize(xs_gt[0], ys_gt[0], v0=v_true, theta0=math.pi / 2.0)

    errs_corr = []
    errs_naive = []
    for k in range(1, 30):
        ekf_corr.predict(dt)
        xc, yc, Rc = proj.bbox_to_ground(bboxes[k], sigma_px=(4.0, 4.0), use_perspective_correction=True)
        stc = ekf_corr.update(np.array([xc, yc]), Rc, timestamp=k * dt)
        assert np.all(np.linalg.eigvalsh(stc.P_pos_2x2) > 0)
        assert stc.lambda_k == pytest.approx(1.0, abs=0.2)
        errs_corr.append(math.hypot(stc.x_utm - xs_gt[k], stc.y_utm - ys_gt[k]))

        ekf_naive.predict(dt)
        xn, yn, Rn = proj.bbox_to_ground(bboxes[k], sigma_px=(4.0, 4.0), use_perspective_correction=False)
        stn = ekf_naive.update(np.array([xn, yn]), Rn, timestamp=k * dt)
        errs_naive.append(math.hypot(stn.x_utm - xs_gt[k], stn.y_utm - ys_gt[k]))

    rmse_corr = math.sqrt(sum(e * e for e in errs_corr) / len(errs_corr))
    rmse_naive = math.sqrt(sum(e * e for e in errs_naive) / len(errs_naive))
    assert rmse_corr < rmse_naive


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_04_adaptive_ekf_exit_state_into_roadgraph_tree(
    utm_origin: tuple[float, float, int],
    synthetic_road_engine: Any,
) -> None:
    from tracker.ekf import AdaptiveEKF

    ox, oy, _ = utm_origin
    # High-speed straight vehicle approaching N1 (ox+100, oy) before 90-deg R=8m corner
    ekf_fast = AdaptiveEKF()
    ekf_fast.initialize(ox + 90.0, oy, v0=28.0, theta0=0.0)
    st_fast = ekf_fast.update(np.array([ox + 98.0, oy]), np.diag([0.5, 0.5]), timestamp=0.3)

    hyps_fast = synthetic_road_engine.propagate_route_tree(st_fast, elapsed_s=4.0, dt_step=0.5)
    assert len(hyps_fast) >= 1

    # Low-speed vehicle approaching same junction can turn onto N1->N4
    ekf_slow = AdaptiveEKF()
    ekf_slow.initialize(ox + 90.0, oy, v0=6.0, theta0=0.0)
    st_slow = ekf_slow.update(np.array([ox + 98.0, oy]), np.diag([0.5, 0.5]), timestamp=1.3)

    hyps_slow = synthetic_road_engine.propagate_route_tree(st_slow, elapsed_s=4.0, dt_step=0.5)
    assert len(hyps_slow) >= 1

    # At 28 m/s (>> v_crit ~ 7.41 m/s), straight eastbound branches dominate over sharp south turns
    straight_fast = sum(h.probability for h in hyps_fast if not h.pruned_by_vcrit and abs(h.theta_rad) < 0.4)
    assert straight_fast >= 0.85


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_05_roadgraph_into_forensic_backtracking(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig
    from tracker.ekf import FilterState
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    eid_01, hdg = synthetic_road_engine.snap_initial_heading(ox + 20.0, oy)
    eid_12, _ = synthetic_road_engine.snap_initial_heading(ox + 150.0, oy)
    eid_23, _ = synthetic_road_engine.snap_initial_heading(ox + 250.0, oy)

    P5 = np.eye(5, dtype=np.float64)
    exit_state = FilterState(
        x=np.array([ox + 50.0, oy, 12.0, hdg, 0.0]),
        P=P5,
        x_utm=ox + 50.0,
        y_utm=oy,
        v_mps=12.0,
        theta_rad=hdg,
        P_pos_2x2=P5[:2, :2].copy(),
    )
    hyp = BlindHypothesis(
        hypothesis_id="h_corridor",
        edge_sequence=[eid_01, eid_12, eid_23],
        current_edge=eid_23,
        s_m=200.0,
        x_utm=ox + 250.0,
        y_utm=oy,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([6.0, 6.0]).astype(np.float64),
        probability=1.0,
        eta_s=16.0,
    )
    emb = np.ones(512, dtype=np.float32) / math.sqrt(512.0)
    rec = BlindTrackRecord(
        global_id=3,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=exit_state,
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=None,
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=8,
        entry_time=16.0,
        x_utm=ox + 250.5,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=(1.8, 4.3),
        reid_emb=emb,
        plate=None,
    )

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid"), road_engine=synthetic_road_engine)
    matches = assoc.associate([rec], [app])
    assert len(matches) == 1
    fp = matches[0].forensic_path
    assert fp is not None and len(fp.edge_ids) >= 2
    for e_a, e_b in zip(fp.edge_ids, fp.edge_ids[1:]):
        assert synthetic_road_network.edges[e_a].v == synthetic_road_network.edges[e_b].u


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_06_detector_exit_zone_separation_into_pipeline_roadgraph(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
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
    cfg = PipelineConfig(
        cameras=[
            CameraConfig(
                cam_id="cam_1",
                source="cam1.mp4",
                fps=5.0,
                resolution=(640, 480),
                calib_path=None,
                homography=_make_homography(ox + 40.0, oy),
                K=None,
                R=None,
                t=None,
                exit_zones=[[(560.0, 140.0), (640.0, 140.0), (640.0, 340.0), (560.0, 340.0)]],
            )
        ],
        detector=DetectorConfig(model_path=tmp_path / "unused.pt", tracker_type="bytetrack", device="cpu"),
        reid=ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False),
        alpr=ALPRConfig(enabled=False),
        ekf=EKFConfig(model_type="adaptive_bicycle"),
        road_graph=RoadGraphConfig(place_name=None, bbox_wgs84=None, graphml_cache=synthetic_graphml_path),
        association=AssociationConfig(mode="hybrid"),
        visualizer=VisualizerConfig(output_video=None),
    )

    step = {"i": 0}
    dt = 0.2

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 3:
            x = ox + 15.0 + i * 2.0
        elif 4 <= i <= 6:
            # 0.6s center occlusion -> OCCLUDED_LOCAL (no blind_hypotheses)
            return np.empty((0, 6), dtype=np.float32)
        elif 7 <= i <= 18:
            x = ox + 30.0 + (i - 7) * 4.5  # drives into east Exit Zone
        else:
            return np.empty((0, 6), dtype=np.float32)
        b = _utm_to_bbox(x, oy, ox + 40.0, oy)
        return np.array([[b[0], b[1], b[2], b[3], 0.93, 2.0]], dtype=np.float32)

    p1 = _build_local_perception(cfg.detector, cfg.cameras[0].exit_zones, det_cam1)
    pipe = _instantiate_pipeline(cfg, synthetic_road_network, synthetic_road_engine, {"cam_1": p1}, {"cam_1": det_cam1})

    blind_after_exit = False
    for i in range(25):
        step["i"] = i
        f = _make_blank_frame()
        if i <= 3:
            f = _draw_vehicle(f, _utm_to_bbox(ox + 15.0 + i * 2.0, oy, ox + 40.0, oy))
        elif 7 <= i <= 18:
            f = _draw_vehicle(f, _utm_to_bbox(ox + 30.0 + (i - 7) * 4.5, oy, ox + 40.0, oy))

        st = pipe.step(i * dt, {"cam_1": f})
        if 4 <= i <= 6:
            assert all(len(v) == 0 for v in st.blind_hypotheses.values())
        if i >= 20 and any(len(v) > 0 for v in st.blind_hypotheses.values()):
            blind_after_exit = True

    assert blind_after_exit


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_07_geometry_dims_and_reid_alpr_into_association(
    utm_origin: tuple[float, float, int],
    synthetic_reid_onnx_path: Path,
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig, ReIDConfig
    from tracker.ekf import FilterState
    from tracker.geometry import CameraProjector
    from tracker.reid import ReIDExtractor
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    proj1 = CameraProjector.from_homography(np.array(_make_homography(ox, oy, 0.10)), epsg_utm=32723)
    proj2 = CameraProjector.from_homography(np.array(_make_homography(ox + 100.0, oy, 0.14)), epsg_utm=32723)

    bbox = (280.0, 210.0, 320.0, 250.0)
    wl1 = proj1.bbox_dimensions_ground(bbox)
    wl2 = proj2.bbox_dimensions_ground(bbox)

    reid = ReIDExtractor(ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False))
    emb1 = reid.extract(synthetic_bgr_frames_and_crops["crop_red_a"])
    emb2 = reid.extract(synthetic_bgr_frames_and_crops["crop_red_a_shifted"])

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid", auto_calibrate_scale=True, min_scale_samples=5))
    for _ in range(5):
        assoc.update_scale_calibration("cam_1", wl1, "cam_2", wl2)
    cw, cl = assoc.get_scale_factors("cam_1", "cam_2")
    assert cw == pytest.approx(1.4, abs=0.1)
    assert cl == pytest.approx(1.4, abs=0.1)

    P5 = np.eye(5, dtype=np.float64)
    hyp = BlindHypothesis(
        hypothesis_id="h_g",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=100.0,
        x_utm=ox + 100.0,
        y_utm=oy,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([4.0, 4.0]).astype(np.float64),
        probability=1.0,
        eta_s=8.0,
    )
    rec = BlindTrackRecord(
        global_id=1,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=FilterState(np.array([ox, oy, 12.0, 0.0, 0.0]), P5, ox, oy, 12.0, 0.0, P5[:2, :2]),
        dims_ground_wl=wl1,
        reid_emb=emb1,
        plate=None,
        hypotheses=[hyp],
    )
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=2,
        entry_time=8.0,
        x_utm=ox + 100.4,
        y_utm=oy,
        R_k=np.diag([0.5, 0.5]).astype(np.float64),
        dims_ground_wl=wl2,
        reid_emb=emb2,
        plate=None,
    )
    matches = assoc.associate([rec], [app])
    assert len(matches) == 1
    assert matches[0].global_id == 1


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_08_alpr_override_resets_ekf_covariance(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.alpr import ALPREngine
    from tracker.config import ALPRConfig
    from tracker.ekf import AdaptiveEKF

    ox, oy, _ = utm_origin
    alpr = ALPREngine(ALPRConfig(enabled=True, min_conf=0.5))
    norm, valid = alpr.normalize_plate("BRA-2E19")
    assert norm == "BRA2E19" and valid is True

    ekf = AdaptiveEKF()
    ekf.initialize(ox, oy, v0=12.0, theta0=0.0)
    for _ in range(60):  # 6.0 s blind propagation
        st_blind = ekf.predict(0.1)

    pre_trace = float(np.trace(st_blind.P_pos_2x2))
    assert pre_trace > 50.0

    R_tight = np.diag([0.25, 0.25]).astype(np.float64)
    st_reset = ekf.reset_position_covariance(np.array([ox + 72.0, oy]), R_tight)
    post_trace = float(np.trace(st_reset.P_pos_2x2))
    assert post_trace < pre_trace * 0.1
    assert post_trace == pytest.approx(float(np.trace(R_tight)), rel=0.5)


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_09_causal_pipeline_into_cicc_visualizer_mp4(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
    import cv2
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
    from tracker.visualizer import CICCVisualizer

    ox, oy, _ = utm_origin
    out_mp4 = tmp_path / "t3_09_cicc.mp4"
    vis_cfg = VisualizerConfig(output_video=out_mp4, fps=5.0, split_screen=True, width=1280, height=480)
    cam1 = CameraConfig(
        cam_id="cam_1",
        source="cam1.mp4",
        fps=5.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography(ox + 40.0, oy),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(560.0, 140.0), (640.0, 140.0), (640.0, 340.0), (560.0, 340.0)]],
    )
    cfg = PipelineConfig(
        cameras=[cam1],
        detector=DetectorConfig(model_path=tmp_path / "unused.pt", tracker_type="bytetrack", device="cpu"),
        reid=ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False),
        alpr=ALPRConfig(enabled=False),
        ekf=EKFConfig(model_type="adaptive_bicycle"),
        road_graph=RoadGraphConfig(place_name=None, bbox_wgs84=None, graphml_cache=synthetic_graphml_path),
        association=AssociationConfig(mode="hybrid"),
        visualizer=vis_cfg,
    )

    step = {"i": 0}

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 5:
            b = _utm_to_bbox(ox + 45.0 + i * 6.0, oy, ox + 40.0, oy)
            return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    p1 = _build_local_perception(cfg.detector, cam1.exit_zones, det_cam1)
    pipe = _instantiate_pipeline(cfg, synthetic_road_network, synthetic_road_engine, {"cam_1": p1}, {"cam_1": det_cam1})
    vis = CICCVisualizer(vis_cfg, synthetic_road_network, [cam1])

    vis.open()
    for i in range(10):
        step["i"] = i
        f = _make_blank_frame()
        if i <= 5:
            f = _draw_vehicle(f, _utm_to_bbox(ox + 45.0 + i * 6.0, oy, ox + 40.0, oy))
        st = pipe.step(i * 0.2, {"cam_1": f})
        vis.write(st)
    vis.close()

    assert out_mp4.exists() and out_mp4.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_mp4))
    try:
        count = 0
        while cap.read()[0]:
            count += 1
        assert count == 10
    finally:
        cap.release()


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_10_roadgraph_blockades_into_pipeline_and_visualizer(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_graphml_path: Path,
    synthetic_road_network: Any,
    synthetic_road_engine: Any,
    synthetic_reid_onnx_path: Path,
) -> None:
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
    from tracker.visualizer import CICCVisualizer

    ox, oy, _ = utm_origin
    cam1 = CameraConfig(
        cam_id="cam_1",
        source="cam1.mp4",
        fps=5.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography(ox + 40.0, oy),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(560.0, 140.0), (640.0, 140.0), (640.0, 340.0), (560.0, 340.0)]],
    )
    vis_cfg = VisualizerConfig(output_video=None, width=1280, height=480, show_blockades=True, show_escape_tree=True)
    cfg = PipelineConfig(
        cameras=[cam1],
        detector=DetectorConfig(model_path=tmp_path / "unused.pt", tracker_type="bytetrack", device="cpu"),
        reid=ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False),
        alpr=ALPRConfig(enabled=False),
        ekf=EKFConfig(model_type="adaptive_bicycle"),
        road_graph=RoadGraphConfig(place_name=None, bbox_wgs84=None, graphml_cache=synthetic_graphml_path),
        association=AssociationConfig(mode="hybrid"),
        visualizer=vis_cfg,
    )

    step = {"i": 0}

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        i = step["i"]
        if i <= 6:
            b = _utm_to_bbox(ox + 45.0 + i * 5.0, oy, ox + 40.0, oy)
            return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    p1 = _build_local_perception(cfg.detector, cam1.exit_zones, det_cam1)
    pipe = _instantiate_pipeline(cfg, synthetic_road_network, synthetic_road_engine, {"cam_1": p1}, {"cam_1": det_cam1})
    vis = CICCVisualizer(vis_cfg, synthetic_road_network, [cam1])

    blind_state = None
    for i in range(15):
        step["i"] = i
        f = _make_blank_frame()
        if i <= 6:
            f = _draw_vehicle(f, _utm_to_bbox(ox + 45.0 + i * 5.0, oy, ox + 40.0, oy))
        st = pipe.step(i * 0.2, {"cam_1": f})
        if st.blockade_points:
            blind_state = st

    assert blind_state is not None
    probs = [b.interception_prob for b in blind_state.blockade_points]
    assert probs[0] == max(probs)
    img = vis.render_frame(blind_state)
    assert img.shape == (480, 1280, 3) and img.dtype == np.uint8


@pytest.mark.e2e
@pytest.mark.tier3
def test_t3_11_uadetract_blind_mask_into_causal_pipeline(
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

    out_dir = tmp_path / "uadetrac_masked"
    script = repo_root / "scripts" / "prepare_datasets.py"
    proc = subprocess.run(
        [
            str(venv_python),
            str(script),
            "--datasets",
            "ua_detrac",
            "--data-dir",
            str(out_dir),
            "--blind-start-s",
            "2.0",
            "--blind-duration-s",
            "4.0",
            "--offline",
        ],
        cwd=str(repo_root),
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, f"prepare_datasets ua_detrac failed: {proc.stderr}"

    seq_dir = out_dir / "ua_detrac" / "detrac_100_MVI_40905"
    mask_path = seq_dir / "blind_mask.json"
    gt_metric_path = seq_dir / "gt_metric.json"
    assert mask_path.is_file(), f"Missing blind_mask.json at {mask_path}"
    assert gt_metric_path.is_file(), f"Missing gt_metric.json at {gt_metric_path}"

    mask_meta = json.loads(mask_path.read_text(encoding="utf-8"))
    assert mask_meta["blind_start_s"] == pytest.approx(2.0)
    assert mask_meta["blind_duration_s"] == pytest.approx(4.0)
    assert len(mask_meta["masked_frame_indices"]) > 0

    gt_metric = json.loads(gt_metric_path.read_text(encoding="utf-8"))
    assert len(gt_metric["records"]) > 0
    blind_gt_records = [r for r in gt_metric["records"] if r.get("is_blind")]
    assert len(blind_gt_records) > 0
    assert all(
        math.isfinite(r["x_utm"]) and math.isfinite(r["y_utm"]) for r in blind_gt_records
    )

    ox, oy, _ = utm_origin
    cam1 = CameraConfig(
        cam_id="cam_1",
        source="cam1.mp4",
        fps=5.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography(ox + 80.0, oy, scale_m_per_px=0.25),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(360.0, 140.0), (640.0, 140.0), (640.0, 340.0), (360.0, 340.0)]],
    )
    cfg = PipelineConfig(
        cameras=[cam1],
        detector=DetectorConfig(model_path=tmp_path / "unused.pt", tracker_type="bytetrack", device="cpu"),
        reid=ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False),
        alpr=ALPRConfig(enabled=False),
        ekf=EKFConfig(model_type="adaptive_bicycle"),
        road_graph=RoadGraphConfig(place_name=None, bbox_wgs84=None, graphml_cache=synthetic_graphml_path),
        association=AssociationConfig(mode="hybrid"),
        visualizer=VisualizerConfig(output_video=None),
    )

    # Simulate 8 s trajectory with [2.0, 6.0] s masked out
    step = {"t": 0.0}
    dt = 0.2
    v = 10.0

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        t = step["t"]
        if 2.0 <= t <= 6.0:
            return np.empty((0, 6), dtype=np.float32)
        x_gt = ox + 10.0 + t * v
        b = _utm_to_bbox(x_gt, oy, ox + 80.0, oy, scale_m_per_px=0.25)
        return np.array([[b[0], b[1], b[2], b[3], 0.92, 2.0]], dtype=np.float32)

    p1 = _build_local_perception(cfg.detector, cam1.exit_zones, det_cam1)
    pipe = _instantiate_pipeline(cfg, synthetic_road_network, synthetic_road_engine, {"cam_1": p1}, {"cam_1": det_cam1})

    blind_errors: list[float] = []
    for k in range(40):
        t = k * dt
        step["t"] = t
        x_gt = ox + 10.0 + t * v
        f = _make_blank_frame()
        if not (2.0 <= t <= 6.0):
            f = _draw_vehicle(f, _utm_to_bbox(x_gt, oy, ox + 80.0, oy, scale_m_per_px=0.25))
        st = pipe.step(t, {"cam_1": f})
        if 2.0 <= t <= 6.0 and st.blind_hypotheses:
            hyps = next(iter(st.blind_hypotheses.values()))
            if hyps:
                top = max(hyps, key=lambda h: h.probability)
                err = math.hypot(top.x_utm - x_gt, top.y_utm - oy)
                assert math.isfinite(err)
                blind_errors.append(err)

    assert len(blind_errors) >= 1
    rmse = math.sqrt(sum(e * e for e in blind_errors) / len(blind_errors))
    assert rmse < 35.0


@pytest.mark.e2e
@pytest.mark.tier3
@pytest.mark.slow
def test_t3_12_core_modules_consistent_with_benchmarks(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.ekf import AdaptiveEKF, LinearCVFilter

    out_dir = tmp_path / "t3_bench"
    proc = subprocess.run(
        [
            str(venv_python),
            str(repo_root / "scripts" / "run_benchmarks.py"),
            "--out",
            str(out_dir),
            "--experiments",
            "A1",
            "A3",
        ],
        cwd=str(repo_root),
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
    assert data["A1"]["results"]["adaptive_bicycle"]["rmse_maneuver_m"] < data["A1"]["results"]["linear_cv"]["rmse_maneuver_m"]
    assert data["A3"]["results"]["corrected_contact_point_mean_error_m"] < data["A3"]["results"]["naive_bottom_center_mean_error_m"]

    # Direct standalone check on curved turn confirms same AdaptiveEKF < LinearCV ranking
    ox, oy, _ = utm_origin
    aekf = AdaptiveEKF()
    lcv = LinearCVFilter()
    aekf.initialize(ox, oy, v0=12.0, theta0=0.0)
    lcv.initialize(ox, oy, v0=12.0, theta0=0.0)
    R = np.diag([0.25, 0.25])
    dt = 0.1
    omega = 0.8
    errs_a, errs_l = [], []
    x_t, y_t, th = ox, oy, 0.0
    for k in range(1, 25):
        th += omega * dt
        x_t += 12.0 * math.cos(th) * dt
        y_t += 12.0 * math.sin(th) * dt
        z = np.array([x_t, y_t])
        sa = aekf.predict(dt)
        sa = aekf.update(z, R, timestamp=k * dt)
        sl = lcv.predict(dt)
        sl = lcv.update(z, R, timestamp=k * dt)
        errs_a.append(math.hypot(sa.x_utm - x_t, sa.y_utm - y_t))
        errs_l.append(math.hypot(sl.x_utm - x_t, sl.y_utm - y_t))
    assert sum(errs_a) / len(errs_a) < sum(errs_l) / len(errs_l)
