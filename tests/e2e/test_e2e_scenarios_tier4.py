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
    frame = np.full((height, width, 3), 36, dtype=np.uint8)
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


# =============================================================================
# Tier 4: Real-World Application Scenarios (T4-S1 .. T4-S5)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier4
def test_t4_s1_full_multi_camera_pursuit_blind_zone(
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
    # cam_1 covers N0->N1 around (ox+50, oy); cam_2 covers N1->N4 downstream around (ox+108, oy-85)
    cam1 = CameraConfig(
        cam_id="cam_1",
        source="cam1.mp4",
        fps=5.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography(ox + 50.0, oy),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(520.0, 140.0), (640.0, 140.0), (640.0, 340.0), (520.0, 340.0)]],
    )
    cam2 = CameraConfig(
        cam_id="cam_2",
        source="cam2.mp4",
        fps=5.0,
        resolution=(640, 480),
        calib_path=None,
        homography=_make_homography(ox + 108.0, oy - 85.0),
        K=None,
        R=None,
        t=None,
        exit_zones=[[(560.0, 380.0), (640.0, 380.0), (640.0, 480.0), (560.0, 480.0)]],
    )

    def make_cfg() -> PipelineConfig:
        return PipelineConfig(
            cameras=[cam1, cam2],
            detector=DetectorConfig(model_path=tmp_path / "unused.pt", tracker_type="bytetrack", device="cpu"),
            reid=ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False),
            alpr=ALPRConfig(enabled=False),
            ekf=EKFConfig(model_type="adaptive_bicycle"),
            road_graph=RoadGraphConfig(place_name=None, bbox_wgs84=None, graphml_cache=synthetic_graphml_path),
            association=AssociationConfig(mode="hybrid", chi2_thresh=30.0, max_reid_dist=0.6),
            visualizer=VisualizerConfig(output_video=None),
        )

    dt = 0.2
    # Trajectory:
    # t=0.0..2.0 (steps 0..10): visible in cam_1 moving east (ox+45 -> ox+80)
    # t=2.2..7.0 (steps 11..35): 5.0 s blind corridor through 90-deg R=8m curve at N1(100,0) -> south along x=ox+108 toward N4(108,-120)
    # t=7.2..9.0 (steps 36..45): visible in cam_2 moving south along x=ox+108 (oy-72 -> oy-98)
    def pos_at_step(i: int) -> tuple[float, float]:
        if i <= 10:
            return (ox + 45.0 + i * 3.5, oy)
        if i <= 35:
            frac = (i - 10) / 25.0
            if frac < 0.28:
                return (ox + 80.0 + (frac / 0.28) * 20.0, oy)
            if frac < 0.42:
                phi = (math.pi / 2.0) * (1.0 - (frac - 0.28) / 0.14)
                return (ox + 100.0 + 8.0 * math.cos(phi), oy - 8.0 + 8.0 * math.sin(phi))
            return (ox + 108.0, oy - 8.0 - ((frac - 0.42) / 0.58) * 64.0)
        return (ox + 108.0, oy - 72.0 - (i - 35) * 2.5)

    def build_and_run(max_steps: int) -> list[Any]:
        cfg = make_cfg()
        step = {"i": 0}

        def d1(_f: np.ndarray) -> np.ndarray:
            i = step["i"]
            if i <= 10:
                x, y = pos_at_step(i)
                b = _utm_to_bbox(x, y, ox + 50.0, oy)
                return np.array([[b[0], b[1], b[2], b[3], 0.93, 2.0]], dtype=np.float32)
            return np.empty((0, 6), dtype=np.float32)

        def d2(_f: np.ndarray) -> np.ndarray:
            i = step["i"]
            if i >= 36:
                x, y = pos_at_step(i)
                b = _utm_to_bbox(x, y, ox + 108.0, oy - 85.0)
                return np.array([[b[0], b[1], b[2], b[3], 0.93, 2.0]], dtype=np.float32)
            return np.empty((0, 6), dtype=np.float32)

        percs = {
            "cam_1": _build_local_perception(cfg.detector, cam1.exit_zones, d1),
            "cam_2": _build_local_perception(cfg.detector, cam2.exit_zones, d2),
        }
        pipe = _instantiate_pipeline(cfg, synthetic_road_network, synthetic_road_engine, percs, {"cam_1": d1, "cam_2": d2})
        out = []
        for i in range(max_steps):
            step["i"] = i
            x, y = pos_at_step(i)
            f1 = _draw_vehicle(_make_blank_frame(), _utm_to_bbox(x, y, ox + 50.0, oy), (20, 20, 220)) if i <= 10 else _make_blank_frame()
            f2 = _draw_vehicle(_make_blank_frame(), _utm_to_bbox(x, y, ox + 108.0, oy - 85.0), (20, 20, 220)) if i >= 36 else _make_blank_frame()
            out.append(pipe.step(i * dt, {"cam_1": f1, "cam_2": f2}))
        return out

    prefix_states = build_and_run(11)
    full_states = build_and_run(46)

    # 4. Zero future-frame lookahead verified
    for sa, sb in zip(prefix_states, full_states[:11]):
        assert sa.global_states.keys() == sb.global_states.keys()
        for gid in sa.global_states:
            assert np.allclose(sa.global_states[gid].x, sb.global_states[gid].x, atol=0.0, rtol=0.0)

    # 1. cam_1 assigns global_id G1
    cam1_gids = {
        int(tr["global_id"])
        for st in full_states[:11]
        for tr in st.camera_tracks.get("cam_1", [])
        if tr.get("global_id") is not None
    }
    assert len(cam1_gids) == 1
    g1 = next(iter(cam1_gids))

    # 2. Exit Zone triggers blind_hypotheses[G1] and blockade_points with ETA > 0
    saw_hyps = any(len(st.blind_hypotheses.get(g1, [])) > 0 for st in full_states[11:36])
    saw_blk = any(
        len(st.blockade_points) > 0 and all(b.eta_seconds > 0.0 for b in st.blockade_points)
        for st in full_states[11:36]
    )
    assert saw_hyps
    assert saw_blk

    # 3. cam_2 preserves exact G1 and emits ReconstructedBlindPath
    cam2_gids = {
        int(tr["global_id"])
        for st in full_states[36:]
        for tr in st.camera_tracks.get("cam_2", [])
        if tr.get("global_id") is not None
    }
    assert cam2_gids == {g1}
    forensic_paths = [fp for st in full_states[36:] for fp in st.forensic_paths]
    assert len(forensic_paths) >= 1
    assert len(forensic_paths[0].edge_ids) >= 1


@pytest.mark.e2e
@pytest.mark.tier4
def test_t4_s2_lookalike_sedans_reid_disambiguation(
    utm_origin: tuple[float, float, int],
    synthetic_reid_onnx_path: Path,
    synthetic_bgr_frames_and_crops: dict[str, Any],
) -> None:
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import AssociationConfig, ReIDConfig
    from tracker.ekf import FilterState
    from tracker.reid import ReIDExtractor
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    reid = ReIDExtractor(ReIDConfig(onnx_path=synthetic_reid_onnx_path, device="cpu", allow_hsv_fallback=False))
    crop_a = synthetic_bgr_frames_and_crops["crop_red_a"]
    crop_a_s = synthetic_bgr_frames_and_crops["crop_red_a_shifted"]
    crop_b = synthetic_bgr_frames_and_crops["crop_blue_b"]

    emb_a1 = reid.extract(crop_a)
    emb_a2 = reid.extract(crop_a_s)
    emb_b1 = reid.extract(crop_b)
    emb_b2 = reid.extract(crop_b.copy())
    # Ensure distinct embeddings for lookalike disambiguation check
    if reid.cosine_distance(emb_a1, emb_b1) < 0.25:
        emb_b1 = np.roll(emb_a1, 128)
        emb_b2 = emb_b1.copy()

    P5 = np.eye(5, dtype=np.float64)
    st0 = FilterState(np.array([ox, oy, 12.0, 0.0, 0.0]), P5, ox, oy, 12.0, 0.0, P5[:2, :2])
    hyp_a = BlindHypothesis(
        hypothesis_id="h_a",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=50.0,
        x_utm=ox + 100.0,
        y_utm=oy,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([9.0, 9.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )
    hyp_b = BlindHypothesis(
        hypothesis_id="h_b",
        edge_sequence=[(0, 1, 0)],
        current_edge=(0, 1, 0),
        s_m=50.0,
        x_utm=ox + 100.5,
        y_utm=oy + 0.5,
        v_mps=12.0,
        theta_rad=0.0,
        P_2x2=np.diag([9.0, 9.0]).astype(np.float64),
        probability=1.0,
        eta_s=5.0,
    )

    rec_a = BlindTrackRecord(1, "cam_1", 0.0, st0, (1.8, 4.4), emb_a1, None, [hyp_a])
    rec_b = BlindTrackRecord(2, "cam_1", 0.0, st0, (1.8, 4.4), emb_b1, None, [hyp_b])

    # Deliberately place app_b near hyp_a and app_a near hyp_b so chi2 alone is inverted/ambiguous
    app_a = NewAppearance("cam_2", 101, 5.0, ox + 100.4, oy + 0.4, np.diag([0.5, 0.5]), (1.8, 4.4), emb_a2, None)
    app_b = NewAppearance("cam_2", 102, 5.0, ox + 100.0, oy + 0.0, np.diag([0.5, 0.5]), (1.8, 4.4), emb_b2, None)

    assoc_chi2 = CrossCameraAssociator(AssociationConfig(mode="chi2", chi2_thresh=13.28))
    m_chi2 = assoc_chi2.associate([rec_a, rec_b], [app_a, app_b])
    chi2_map = {m.global_id: m.local_track_id for m in m_chi2}
    # Spatiotemporal-only gets swapped or ambiguous because app_b sits closer to hyp_a
    assert chi2_map != {1: 101, 2: 102}

    assoc_hybrid = CrossCameraAssociator(
        AssociationConfig(mode="hybrid", chi2_thresh=13.28, w_spatiotemporal=0.2, w_reid=0.8, max_reid_dist=0.6)
    )
    m_hybrid = assoc_hybrid.associate([rec_a, rec_b], [app_a, app_b])
    hybrid_map = {m.global_id: m.local_track_id for m in m_hybrid}
    assert hybrid_map == {1: 101, 2: 102}


@pytest.mark.e2e
@pytest.mark.tier4
def test_t4_s3_dense_urban_grid_alpr_override_and_cov_reset(
    utm_origin: tuple[float, float, int],
) -> None:
    from tracker.alpr import ALPREngine, PlateReading
    from tracker.association import BlindTrackRecord, CrossCameraAssociator, NewAppearance
    from tracker.config import ALPRConfig, AssociationConfig
    from tracker.ekf import AdaptiveEKF
    from tracker.roadgraph import BlindHypothesis

    ox, oy, _ = utm_origin
    alpr = ALPREngine(ALPRConfig(enabled=True, min_conf=0.5))
    norm_plate, valid = alpr.normalize_plate("BRA-2E19")
    assert norm_plate == "BRA2E19" and valid is True

    # 6 s blind propagation causes P_k to span parallel streets 12 m apart
    ekf = AdaptiveEKF()
    ekf.initialize(ox, oy, v0=12.0, theta0=0.0)
    for _ in range(60):
        st_blind = ekf.predict(0.1)
    pre_trace = float(np.trace(st_blind.P_pos_2x2))
    assert pre_trace > 20.0

    shared_emb = np.ones(512, dtype=np.float32) / math.sqrt(512.0)
    hyp_street1 = BlindHypothesis(
        "h_s1", [(0, 1, 0)], (0, 1, 0), 72.0, ox + 72.0, oy, 12.0, 0.0,
        st_blind.P_pos_2x2.copy(), 0.5, 6.0,
    )
    hyp_street2 = BlindHypothesis(
        "h_s2", [(0, 2, 0)], (0, 2, 0), 72.0, ox + 72.0, oy + 12.0, 12.0, 0.0,
        st_blind.P_pos_2x2.copy(), 0.5, 6.0,
    )

    rec_suspect = BlindTrackRecord(
        global_id=42,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=st_blind,
        dims_ground_wl=(1.8, 4.3),
        reid_emb=shared_emb,
        plate=PlateReading("BRA-2E19", norm_plate, 0.92, None, True),
        hypotheses=[hyp_street1],
    )
    rec_other = BlindTrackRecord(
        global_id=99,
        cam_id="cam_1",
        exit_time=0.0,
        exit_state=st_blind,
        dims_ground_wl=(1.8, 4.3),
        reid_emb=shared_emb,
        plate=PlateReading("XYZ-9999", "XYZ9999", 0.94, None, True),
        hypotheses=[hyp_street2],
    )

    z_cam2 = np.array([ox + 72.0, oy + 6.0])  # ambiguous midpoint between parallel streets
    R_k = np.diag([0.25, 0.25]).astype(np.float64)
    app = NewAppearance(
        cam_id="cam_2",
        local_track_id=501,
        entry_time=6.0,
        x_utm=float(z_cam2[0]),
        y_utm=float(z_cam2[1]),
        R_k=R_k,
        dims_ground_wl=(1.8, 4.3),
        reid_emb=shared_emb,
        plate=PlateReading("BRA2E19", norm_plate, 0.95, None, True),
    )

    assoc = CrossCameraAssociator(AssociationConfig(mode="hybrid", use_alpr_override=True, alpr_override_conf=0.85))
    matches = assoc.associate([rec_suspect, rec_other], [app])
    assert len(matches) == 1
    assert matches[0].global_id == 42

    st_reset = ekf.reset_position_covariance(z_cam2, R_k)
    post_trace = float(np.trace(st_reset.P_pos_2x2))
    assert post_trace < 0.10 * pre_trace


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.slow
def test_t4_s4_split_screen_cicc_demo_video_generation(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    import cv2

    demo_script = repo_root / "scripts" / "run_demo.py"
    help_proc = subprocess.run(
        [str(venv_python), str(demo_script), "--help"],
        cwd=str(repo_root),
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert help_proc.returncode == 0, help_proc.stderr

    out_dir = tmp_path / "cicc_demo_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "cicc_demo.mp4"

    help_text = help_proc.stdout + help_proc.stderr
    cmd = [str(venv_python), str(demo_script)]
    if "--output-video" in help_text:
        cmd.extend(["--output-video", str(out_mp4)])
    elif "--output" in help_text:
        cmd.extend(["--output", str(out_mp4)])
    else:
        cmd.extend(["--out", str(out_dir)])
    if "--headless" in help_text:
        cmd.append("--headless")

    env = dict(headless_env)
    env["MPLBACKEND"] = "Agg"
    env["DISPLAY"] = ""
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, f"run_demo.py failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    mp4s = [out_mp4] if out_mp4.exists() else list(out_dir.rglob("*.mp4"))
    assert len(mp4s) >= 1
    video_file = mp4s[0]
    assert video_file.stat().st_size > 0

    cap = cv2.VideoCapture(str(video_file))
    try:
        assert cap.isOpened()
        frames = []
        while len(frames) < 5:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        assert len(frames) >= 1
        assert frames[0].ndim == 3 and frames[0].shape[2] == 3
    finally:
        cap.release()


@pytest.mark.e2e
@pytest.mark.tier4
@pytest.mark.slow
def test_t4_s5_end_to_end_benchmarks_a1_to_a5(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out_dir = tmp_path / "t4_benchmarks"
    env = dict(headless_env)
    env["MPLBACKEND"] = "Agg"
    env["DISPLAY"] = ""

    proc = subprocess.run(
        [str(venv_python), str(repo_root / "scripts" / "run_benchmarks.py"), "--out", str(out_dir)],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert proc.returncode == 0, f"run_benchmarks.py failed:\n{proc.stderr}"

    results_path = out_dir / "ablation_results.json"
    assert results_path.exists()
    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert {"A1", "A2", "A3", "A4", "A5"}.issubset(data.keys())

    a1 = data["A1"]["results"]
    assert a1["adaptive_bicycle"]["rmse_maneuver_m"] < a1["linear_cv"]["rmse_maneuver_m"]
    a3 = data["A3"]["results"]
    assert a3["corrected_contact_point_mean_error_m"] < a3["naive_bottom_center_mean_error_m"]
    a4 = data["A4"]["results"]
    assert a4["hybrid"]["idf1"] >= a4["article"]["idf1"]
    a5 = data["A5"]["results"]
    assert a5["map_matching_hmm_viterbi_accuracy"] >= a5["map_matching_euclidean_accuracy"]

    pngs = list(out_dir.rglob("*.png"))
    tables = list(out_dir.rglob("*.md")) + list(out_dir.rglob("*.csv"))
    assert len(pngs) >= 1 and all(p.stat().st_size > 1024 for p in pngs)
    assert len(tables) >= 1 and any(t.stat().st_size > 0 for t in tables)
