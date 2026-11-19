from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest


def _make_camera_configs(num_cams: int = 2) -> list[Any]:
    from tracker.config import CameraConfig

    cams = []
    for idx in range(num_cams):
        cams.append(
            CameraConfig(
                cam_id=f"cam_{idx + 1}",
                source=f"cam_{idx + 1}.mp4",
                fps=15.0,
                resolution=(640, 480),
                calib_path=None,
                homography=[[0.1, 0.0, 330000.0 + idx * 100.0], [0.0, -0.1, 7390050.0], [0.0, 0.0, 1.0]],
                K=None,
                R=None,
                t=None,
                exit_zones=[],
            )
        )
    return cams


def _make_filter_state(
    x_utm: float,
    y_utm: float,
    v_mps: float = 11.5,
    theta_rad: float = 0.0,
    P_pos_2x2: np.ndarray | None = None,
) -> Any:
    from tracker.ekf import FilterState

    P_pos = (
        P_pos_2x2.astype(np.float64)
        if P_pos_2x2 is not None
        else np.diag([1.0, 1.0]).astype(np.float64)
    )
    P = np.eye(5, dtype=np.float64)
    P[:2, :2] = P_pos
    x = np.array([x_utm, y_utm, v_mps, theta_rad, 0.0], dtype=np.float64)
    return FilterState(
        x=x,
        P=P,
        x_utm=x_utm,
        y_utm=y_utm,
        v_mps=v_mps,
        theta_rad=theta_rad,
        P_pos_2x2=P_pos,
    )


def _make_frame_state(
    utm_origin: tuple[float, float, int],
    road_network: Any,
    *,
    visible: bool = True,
    P_pos_2x2: np.ndarray | None = None,
    with_tree_and_blockades: bool = False,
    empty: bool = False,
    num_cams: int = 2,
    frame_shapes: list[tuple[int, int]] | None = None,
) -> Any:
    from tracker.pipeline import PipelineFrameState
    from tracker.roadgraph import BlindHypothesis, BlockadePoint

    ox, oy, _ = utm_origin
    shapes = frame_shapes or [(480, 640)] * num_cams
    frames_bgr = {
        f"cam_{i + 1}": np.full((h, w, 3), 45 + i * 15, dtype=np.uint8)
        for i, (h, w) in enumerate(shapes)
    }

    if empty:
        return PipelineFrameState(
            timestamp_s=1.0,
            frames_bgr=frames_bgr,
            camera_tracks={},
            global_states={},
            blind_hypotheses={},
            blockade_points=[],
            forensic_paths=[],
            events=[],
        )

    st = _make_filter_state(ox + 120.0, oy, P_pos_2x2=P_pos_2x2)
    camera_tracks: dict[str, list[dict]] = {f"cam_{i + 1}": [] for i in range(num_cams)}
    if visible:
        camera_tracks["cam_1"] = [
            {
                "global_id": 1,
                "local_track_id": 1,
                "bbox_xyxy": [180.0, 200.0, 290.0, 280.0],
                "conf": 0.92,
                "class_id": 2,
                "status": "ACTIVE",
                "v_mps": 11.5,
                "plate": "ABC1D23",
            }
        ]

    hyps: dict[int, list[Any]] = {}
    blockades: list[Any] = []
    if with_tree_and_blockades or not visible:
        eids = list(road_network.edges.keys())
        e0 = eids[0]
        e1 = eids[min(1, len(eids) - 1)]
        nids = list(road_network.nodes_utm.keys())
        n0 = nids[min(2, len(nids) - 1)]
        n1 = nids[min(3, len(nids) - 1)]
        nx0, ny0 = road_network.nodes_utm[n0]
        nx1, ny1 = road_network.nodes_utm[n1]
        hyps[1] = [
            BlindHypothesis(
                hypothesis_id="h_1",
                edge_sequence=[e0],
                current_edge=e0,
                s_m=50.0,
                x_utm=ox + 140.0,
                y_utm=oy,
                v_mps=11.5,
                theta_rad=0.0,
                P_2x2=np.diag([25.0, 9.0]).astype(np.float64),
                probability=0.70,
                eta_s=6.7,
            ),
            BlindHypothesis(
                hypothesis_id="h_2",
                edge_sequence=[e1],
                current_edge=e1,
                s_m=45.0,
                x_utm=ox + 100.0,
                y_utm=oy - 50.0,
                v_mps=8.0,
                theta_rad=-math.pi / 2.0,
                P_2x2=np.diag([16.0, 16.0]).astype(np.float64),
                probability=0.30,
                eta_s=9.2,
            ),
        ]
        blockades = [
            BlockadePoint(
                node_id=n0,
                x_utm=nx0,
                y_utm=ny0,
                interception_prob=0.85,
                eta_seconds=6.7,
                converging_routes=["h_1"],
            ),
            BlockadePoint(
                node_id=n1,
                x_utm=nx1,
                y_utm=ny1,
                interception_prob=0.55,
                eta_seconds=9.2,
                converging_routes=["h_2"],
            ),
        ]

    events = ["ZONA CEGA" if not visible else "VISIVEL"]
    return PipelineFrameState(
        timestamp_s=2.5,
        frames_bgr=frames_bgr,
        camera_tracks=camera_tracks,
        global_states={1: st},
        blind_hypotheses=hyps,
        blockade_points=blockades,
        forensic_paths=[],
        events=events,
    )


def _assert_all_finite_numbers(obj: Any, path: str = "root") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_all_finite_numbers(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_all_finite_numbers(v, f"{path}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        assert math.isfinite(float(obj)), f"Non-finite number at {path}: {obj}"


def _run_benchmarks_cli(
    venv_python: Path,
    repo_root: Path,
    headless_env: dict[str, str],
    args: list[str],
) -> subprocess.CompletedProcess[str]:
    script = repo_root / "scripts" / "run_benchmarks.py"
    return subprocess.run(
        [str(venv_python), str(script), *args],
        cwd=str(repo_root),
        env=headless_env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


# =============================================================================
# Area 9: CICC Visualizer (tracker.visualizer) — Tier 1 (5 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area9
def test_a9_t1_01_split_screen_composite_dimensions_and_format(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cfg = VisualizerConfig(output_video=None, split_screen=True, width=1920, height=720)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))
    state = _make_frame_state(utm_origin, synthetic_road_network, visible=True)

    img = vis.render_frame(state)
    assert isinstance(img, np.ndarray)
    assert img.shape == (720, 1920, 3)
    assert img.dtype == np.uint8
    assert float(img[:, :960].std()) > 0.0
    assert float(img[:, 960:].std()) > 0.0


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area9
def test_a9_t1_02_visible_vs_zona_cega_status_rendering(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cfg = VisualizerConfig(output_video=None, split_screen=True, width=1280, height=480)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))

    state_visible = _make_frame_state(utm_origin, synthetic_road_network, visible=True)
    state_blind = _make_frame_state(
        utm_origin, synthetic_road_network, visible=False, with_tree_and_blockades=True
    )

    img_a = vis.render_frame(state_visible)
    img_b = vis.render_frame(state_blind)
    assert img_a.shape == (480, 1280, 3) and img_a.dtype == np.uint8
    assert img_b.shape == (480, 1280, 3) and img_b.dtype == np.uint8
    assert not np.array_equal(img_a, img_b)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area9
def test_a9_t1_03_covariance_ellipse_pk_rendering(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cfg = VisualizerConfig(
        output_video=None,
        split_screen=True,
        width=1280,
        height=480,
        show_cov_ellipse=True,
    )
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))

    state_small = _make_frame_state(
        utm_origin,
        synthetic_road_network,
        visible=True,
        P_pos_2x2=np.diag([0.5, 0.5]),
    )
    state_large = _make_frame_state(
        utm_origin,
        synthetic_road_network,
        visible=True,
        P_pos_2x2=np.diag([400.0, 200.0]),
    )

    img_small = vis.render_frame(state_small)
    img_large = vis.render_frame(state_large)
    right_diff = np.abs(
        img_large[:, 640:].astype(np.int32) - img_small[:, 640:].astype(np.int32)
    ).sum()
    assert right_diff > 0


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area9
def test_a9_t1_04_escape_tree_and_blockade_eta_rendering(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cams = _make_camera_configs(2)
    state = _make_frame_state(
        utm_origin,
        synthetic_road_network,
        visible=False,
        with_tree_and_blockades=True,
    )

    vis_on = CICCVisualizer(
        VisualizerConfig(
            output_video=None,
            split_screen=True,
            width=1280,
            height=480,
            show_escape_tree=True,
            show_blockades=True,
        ),
        synthetic_road_network,
        cams,
    )
    vis_off = CICCVisualizer(
        VisualizerConfig(
            output_video=None,
            split_screen=True,
            width=1280,
            height=480,
            show_escape_tree=False,
            show_blockades=False,
        ),
        synthetic_road_network,
        cams,
    )

    img_on = vis_on.render_frame(state)
    img_off = vis_off.render_frame(state)
    right_diff = np.abs(
        img_on[:, 640:].astype(np.int32) - img_off[:, 640:].astype(np.int32)
    ).sum()
    assert right_diff > 0


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area9
def test_a9_t1_05_video_writer_lifecycle_produces_playable_mp4(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    import cv2
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    out_mp4 = tmp_path / "cicc_test.mp4"
    cfg = VisualizerConfig(output_video=out_mp4, fps=10.0, width=1280, height=480)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))

    vis.open()
    for i in range(5):
        state = _make_frame_state(
            utm_origin,
            synthetic_road_network,
            visible=(i < 2),
            with_tree_and_blockades=(i >= 2),
        )
        vis.write(state)
    vis.close()

    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0

    cap = cv2.VideoCapture(str(out_mp4))
    try:
        assert cap.isOpened()
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        assert len(frames) == 5
        assert all(f.shape == (480, 1280, 3) for f in frames)
    finally:
        cap.release()


# =============================================================================
# Area 9: CICC Visualizer — Tier 2 Boundary & Corner Cases (5 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area9
def test_a9_t2_01_headless_agg_rendering_without_display(
    monkeypatch: pytest.MonkeyPatch,
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("DISPLAY", "")

    cfg = VisualizerConfig(output_video=None, split_screen=True, width=1280, height=480)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))
    img = vis.render_frame(_make_frame_state(utm_origin, synthetic_road_network))
    assert img.shape == (480, 1280, 3)
    assert img.dtype == np.uint8


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area9
def test_a9_t2_02_zero_target_empty_hud(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cfg = VisualizerConfig(output_video=None, split_screen=True, width=1280, height=480)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))
    empty_state = _make_frame_state(utm_origin, synthetic_road_network, empty=True)
    img = vis.render_frame(empty_state)
    assert img.shape == (480, 1280, 3)
    assert img.dtype == np.uint8


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area9
@pytest.mark.parametrize("num_cameras", [1, 2, 4])
def test_a9_t2_03_multi_camera_grid_scaling_1_2_4_cameras(
    num_cameras: int,
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    shapes = [(480, 640), (720, 1280), (360, 480), (540, 960)][:num_cameras]
    cfg = VisualizerConfig(output_video=None, split_screen=True, width=1920, height=720)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(num_cameras))
    state = _make_frame_state(
        utm_origin,
        synthetic_road_network,
        num_cams=num_cameras,
        frame_shapes=shapes,
    )
    img = vis.render_frame(state)
    assert img.shape == (720, 1920, 3)
    assert img.dtype == np.uint8


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area9
def test_a9_t2_04_degenerate_and_huge_covariance_ellipse_safe(
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cfg = VisualizerConfig(output_video=None, split_screen=True, width=1280, height=480)
    vis = CICCVisualizer(cfg, synthetic_road_network, _make_camera_configs(2))

    matrices = [
        1e-12 * np.eye(2),
        np.array([[4.0, 4.0], [4.0, 4.0]], dtype=np.float64),
        1e7 * np.eye(2),
    ]
    for P_mat in matrices:
        state = _make_frame_state(utm_origin, synthetic_road_network, P_pos_2x2=P_mat)
        img = vis.render_frame(state)
        assert img.shape == (480, 1280, 3)
        assert img.dtype == np.uint8


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area9
def test_a9_t2_05_writer_error_and_double_close_safety(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> None:
    from tracker.config import VisualizerConfig
    from tracker.visualizer import CICCVisualizer

    cams = _make_camera_configs(2)
    state = _make_frame_state(utm_origin, synthetic_road_network)

    # output_video=None: close before open & double close must not crash, and render_frame stays valid
    vis_none = CICCVisualizer(VisualizerConfig(output_video=None), synthetic_road_network, cams)
    vis_none.close()
    vis_none.close()
    rendered = vis_none.render_frame(state)
    assert isinstance(rendered, np.ndarray)
    assert rendered.ndim == 3 and rendered.shape[2] == 3 and rendered.dtype == np.uint8

    # output_video set: write() without open() should raise RuntimeError/ValueError or auto-open cleanly
    out_mp4 = tmp_path / "pre_open.mp4"
    vis_mp4 = CICCVisualizer(
        VisualizerConfig(output_video=out_mp4, width=1280, height=480),
        synthetic_road_network,
        cams,
    )
    vis_mp4.close()  # close before open safe
    raised = False
    try:
        vis_mp4.write(state)
    except (RuntimeError, ValueError):
        raised = True
    vis_mp4.close()
    vis_mp4.close()
    if raised:
        assert not out_mp4.exists() or out_mp4.stat().st_size == 0
    else:
        assert out_mp4.exists() and out_mp4.stat().st_size > 0


# =============================================================================
# Area 10: Benchmarks A1-A5 (scripts/run_benchmarks.py) — Tier 1 (5 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t1_01_run_benchmarks_cli_executes_a1_to_a5(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out_dir = tmp_path / "bench_out"
    proc = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(out_dir)])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    json_path = out_dir / "ablation_results.json"
    assert json_path.exists() and json_path.stat().st_size > 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t1_02_ablation_results_json_schema_complete(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out_dir = tmp_path / "bench_schema"
    proc = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(out_dir)])
    assert proc.returncode == 0, proc.stderr

    data = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
    assert {"A1", "A2", "A3", "A4", "A5"}.issubset(data.keys())
    for key in ("A1", "A2", "A3", "A4", "A5"):
        assert {"experiment", "description", "results"}.issubset(data[key].keys())

    a1 = data["A1"]["results"]
    for model_key in ("linear_cv", "canonical_bicycle", "adaptive_bicycle", "ctrv", "imm"):
        assert model_key in a1
        assert {
            "rmse_overall_m",
            "rmse_maneuver_m",
            "rmse_blind_outage_m",
            "max_error_m",
            "lambda_mean",
            "lambda_max",
        }.issubset(a1[model_key].keys())

    a2 = data["A2"]["results"]
    assert len(a2) >= 3
    a2_entries = list(a2.values()) if isinstance(a2, dict) else a2
    for entry in a2_entries:
        assert {
            "adaptive_maneuver_rmse_m",
            "canonical_maneuver_rmse_m",
            "error_reduction_pct",
            "peak_lambda",
        }.issubset(entry.keys())

    a3 = data["A3"]["results"]
    assert {
        "naive_bottom_center_mean_error_m",
        "corrected_contact_point_mean_error_m",
        "bias_reduction_pct",
    }.issubset(a3.keys())

    a4 = data["A4"]["results"]
    for mode_key in ("article", "chi2", "reid_only", "hybrid"):
        assert mode_key in a4
        assert {
            "idf1",
            "cross_camera_accuracy",
            "route_top1_accuracy",
            "route_topk_accuracy",
            "blind_transitions",
            "throughput_fps",
        }.issubset(a4[mode_key].keys())

    a5 = data["A5"]["results"]
    assert {
        "map_matching_euclidean_accuracy",
        "map_matching_hmm_viterbi_accuracy",
        "high_speed_straight_top1_prob",
        "low_speed_turning_top1_prob",
        "recommended_police_blockades",
    }.issubset(a5.keys())


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t1_03_ablation_scientific_invariants_hold(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out_dir = tmp_path / "bench_invariants"
    proc = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(out_dir)])
    assert proc.returncode == 0, proc.stderr

    data = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
    a1 = data["A1"]["results"]
    assert a1["adaptive_bicycle"]["rmse_maneuver_m"] < a1["linear_cv"]["rmse_maneuver_m"]
    assert a1["ctrv"]["rmse_maneuver_m"] < a1["linear_cv"]["rmse_maneuver_m"]

    a3 = data["A3"]["results"]
    assert a3["corrected_contact_point_mean_error_m"] < a3["naive_bottom_center_mean_error_m"]
    assert a3["bias_reduction_pct"] > 0.0

    a4 = data["A4"]["results"]
    assert a4["hybrid"]["idf1"] >= a4["article"]["idf1"]
    for mode_key in ("article", "chi2", "reid_only", "hybrid"):
        assert 0.0 <= a4[mode_key]["idf1"] <= 1.0

    a5 = data["A5"]["results"]
    assert a5["map_matching_hmm_viterbi_accuracy"] >= a5["map_matching_euclidean_accuracy"]
    blockades = a5["recommended_police_blockades"]
    assert len(blockades) >= 1
    for blk in blockades:
        assert "node_id" in blk
        assert "eta_seconds" in blk and blk["eta_seconds"] >= 0.0
        assert "interception_prob" in blk and 0.0 <= blk["interception_prob"] <= 1.0
        assert "converging_routes" in blk


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t1_04_benchmark_plots_and_tables_generated(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out_dir = tmp_path / "bench_artifacts"
    proc = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(out_dir)])
    assert proc.returncode == 0, proc.stderr

    pngs = list(out_dir.rglob("*.png"))
    assert len(pngs) >= 1
    for p in pngs:
        raw = p.read_bytes()
        assert len(raw) > 1024
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"

    tables = list(out_dir.rglob("*.md")) + list(out_dir.rglob("*.csv"))
    assert len(tables) >= 1
    assert any(t.stat().st_size > 0 for t in tables)


@pytest.mark.e2e
@pytest.mark.tier1
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t1_05_benchmark_outputs_finite_and_deterministic(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out1 = tmp_path / "run_1"
    out2 = tmp_path / "run_2"
    p1 = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(out1)])
    p2 = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(out2)])
    assert p1.returncode == 0, p1.stderr
    assert p2.returncode == 0, p2.stderr

    d1 = json.loads((out1 / "ablation_results.json").read_text(encoding="utf-8"))
    d2 = json.loads((out2 / "ablation_results.json").read_text(encoding="utf-8"))
    _assert_all_finite_numbers(d1)
    _assert_all_finite_numbers(d2)

    assert d1["A1"] == d2["A1"]
    assert d1["A3"] == d2["A3"]
    assert d1["A5"] == d2["A5"]


# =============================================================================
# Area 10: Benchmarks A1-A5 — Tier 2 Boundary & Corner Cases (5 Tests)
# =============================================================================


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area10
def test_a10_t2_01_invalid_ablation_flag_exits_nonzero(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    proc = _run_benchmarks_cli(
        venv_python,
        repo_root,
        headless_env,
        ["--out", str(tmp_path / "bad_out"), "--experiments", "INVALID_A99"],
    )
    assert proc.returncode != 0
    assert (proc.stderr + proc.stdout).strip() != ""


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t2_02_missing_nested_output_dir_auto_created(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    nested = tmp_path / "nested" / "deep" / "bench_out"
    assert not nested.exists()
    proc = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(nested)])
    assert proc.returncode == 0, proc.stderr
    assert (nested / "ablation_results.json").exists()


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area10
def test_a10_t2_03_subset_experiment_selection(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    out_dir = tmp_path / "subset_out"
    proc = _run_benchmarks_cli(
        venv_python,
        repo_root,
        headless_env,
        ["--out", str(out_dir), "--experiments", "A1", "A3"],
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
    assert "A1" in data and "A3" in data
    _assert_all_finite_numbers(data)


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area10
@pytest.mark.slow
def test_a10_t2_04_headless_agg_plot_export_no_display(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    env = dict(headless_env)
    env["MPLBACKEND"] = "Agg"
    env["DISPLAY"] = ""
    out_dir = tmp_path / "headless_bench"
    proc = _run_benchmarks_cli(venv_python, repo_root, env, ["--out", str(out_dir)])
    assert proc.returncode == 0, proc.stderr
    pngs = list(out_dir.rglob("*.png"))
    assert len(pngs) >= 1


@pytest.mark.e2e
@pytest.mark.tier2
@pytest.mark.area10
def test_a10_t2_05_unwritable_output_path_fails_cleanly(
    tmp_path: Path,
    repo_root: Path,
    venv_python: Path,
    headless_env: dict[str, str],
) -> None:
    regular_file = tmp_path / "not_a_directory.txt"
    regular_file.write_text("occupied", encoding="utf-8")
    proc = _run_benchmarks_cli(venv_python, repo_root, headless_env, ["--out", str(regular_file)])
    assert proc.returncode != 0
