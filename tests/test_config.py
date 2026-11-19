"""Unit tests for typed Pydantic v2 configuration schemas and YAML configs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import tracker
from tracker.config import (
    ALPRConfig,
    AssociationConfig,
    BlindMaskConfig,
    CameraConfig,
    DetectorConfig,
    EKFConfig,
    PipelineConfig,
    RoadGraphConfig,
    VisualizerConfig,
    load_config,
    load_scenarios,
    save_config,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_schemas() -> None:
    cfg = PipelineConfig()
    assert cfg.detector.model_path == Path("weights/yolo11n.pt")
    assert cfg.detector.classes == [2, 3, 5, 7]
    assert cfg.reid.onnx_path == Path("weights/reid/osnet_x0_25_msmt17.onnx")
    assert cfg.reid.allow_hsv_fallback is False
    assert cfg.ekf.theta_window == 4
    assert cfg.ekf.nis_alpha == 0.05
    assert cfg.ekf.nis_persistence_frames == 2
    assert cfg.association.w_spatiotemporal + cfg.association.w_reid == pytest.approx(1.0)
    assert cfg.road_graph.overpass_urls[0] == "https://overpass.osm.ch/api/interpreter"
    assert "https://api.openstreetmap.org/api/0.6/map" in cfg.road_graph.overpass_urls


def test_load_default_and_demo_yaml() -> None:
    default_cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    demo_cfg = load_config(REPO_ROOT / "configs" / "cameras_demo.yaml")

    assert isinstance(default_cfg, PipelineConfig)
    assert isinstance(demo_cfg, PipelineConfig)
    assert len(default_cfg.cameras) >= 1
    assert len(demo_cfg.cameras) == 3

    assert default_cfg.road_graph.graphml_cache == Path("data/osm/sao_paulo_paulista.graphml")
    assert demo_cfg.road_graph.graphml_cache == Path("data/osm/sao_paulo_paulista.graphml")

    for cam in [*default_cfg.cameras, *demo_cfg.cameras]:
        assert cam.homography is not None
        assert len(cam.exit_zones) >= 1
        assert all(len(poly) >= 3 for poly in cam.exit_zones)
        H = np.array(cam.homography, dtype=np.float64)
        assert abs(float(np.linalg.det(H))) > 1e-12
        # Project image center (640, 360) and confirm Sao Paulo UTM 23S bounds
        pt = H @ np.array([640.0, 360.0, 1.0])
        x_utm = float(pt[0] / pt[2])
        y_utm = float(pt[1] / pt[2])
        assert 330000.0 <= x_utm <= 333000.0
        assert 7392500.0 <= y_utm <= 7395000.0

    cam_02 = next(c for c in demo_cfg.cameras if c.cam_id == "cam_02")
    assert cam_02.K is not None
    assert cam_02.R is not None
    assert cam_02.t is not None


def test_load_scenarios_yaml() -> None:
    scenarios = load_scenarios(REPO_ROOT / "configs" / "scenarios.yaml")
    expected_ids = {
        "demo_cicc_pursuit",
        "synthehicle_town05",
        "cityflow_s01",
        "ua_detrac_blind_mask",
        "A1_kinematic_models",
        "A2_adaptive_fading",
        "A3_projection_bias",
        "A4_association_ablation",
        "A5_map_matching_cicc",
    }
    assert expected_ids.issubset(scenarios.keys())
    ua = scenarios["ua_detrac_blind_mask"]
    assert ua.blind_mask is not None
    assert ua.blind_mask.enabled is True
    assert 3.0 <= ua.blind_mask.duration_s <= 8.0


def test_yaml_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(REPO_ROOT / "configs" / "cameras_demo.yaml")
    out_file = tmp_path / "roundtrip.yaml"
    returned_path = save_config(cfg, out_file)
    assert returned_path == out_file
    reloaded = load_config(out_file)
    assert reloaded == cfg


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cam_id": "c1", "source": "a.mp4", "fps": 0.0},
        {"cam_id": "c1", "source": "a.mp4", "fps": -5.0},
        {"cam_id": "c1", "source": "a.mp4", "resolution": (0, 720)},
        {"cam_id": "c1", "source": "a.mp4", "homography": [[1.0, 0.0], [0.0, 1.0]]},
        {
            "cam_id": "c1",
            "source": "a.mp4",
            "homography": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        },
        {
            "cam_id": "c1",
            "source": "a.mp4",
            "K": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        {
            "cam_id": "c1",
            "source": "a.mp4",
            "K": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            "R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "t": [0.0, 1.0],
        },
        {"cam_id": "c1", "source": "a.mp4", "exit_zones": [[(0.0, 0.0), (10.0, 10.0)]]},
    ],
)
def test_invalid_camera_config(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        CameraConfig(**kwargs)


def test_duplicate_cam_ids_rejected() -> None:
    c1 = CameraConfig(cam_id="cam_dup", source="a.mp4")
    c2 = CameraConfig(cam_id="cam_dup", source="b.mp4")
    with pytest.raises(ValidationError):
        PipelineConfig(cameras=[c1, c2])


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [
        (EKFConfig, {"theta_window": 0}),
        (EKFConfig, {"nis_alpha": 0.0}),
        (EKFConfig, {"nis_alpha": 1.0}),
        (EKFConfig, {"wheelbase_m": -1.0}),
        (EKFConfig, {"sigma_px": (0.0, 4.0)}),
        (EKFConfig, {"model_type": "unknown_model"}),
        (RoadGraphConfig, {"bbox_wgs84": (-46.65, -23.56, -46.66, -23.55)}),
        (RoadGraphConfig, {"bbox_wgs84": (-46.66, -23.56, -46.65, 95.0)}),
        (RoadGraphConfig, {"overpass_urls": []}),
        (AssociationConfig, {"w_spatiotemporal": 0.7, "w_reid": 0.7}),
        (AssociationConfig, {"min_scale_samples": 0}),
        (AssociationConfig, {"mode": "invalid_mode"}),
        (PipelineConfig, {"unknown_extra_field": 123}),
    ],
)
def test_invalid_ekf_roadgraph_association_configs(cls: type, kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        cls(**kwargs)


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")

    bad_yaml = tmp_path / "list.yaml"
    bad_yaml.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_yaml)


def test_package_init_is_lightweight() -> None:
    import importlib

    submods = (
        "tracker.geometry",
        "tracker.kinematics",
        "tracker.ekf",
        "tracker.osm_loader",
        "tracker.roadgraph",
        "tracker.detector",
        "tracker.reid",
        "tracker.alpr",
        "tracker.association",
        "tracker.pipeline",
        "tracker.visualizer",
    )
    saved = {m: sys.modules.pop(m) for m in submods if m in sys.modules}
    try:
        importlib.reload(tracker)
        assert isinstance(tracker.__version__, str)
        assert hasattr(tracker, "PipelineConfig")
        assert hasattr(tracker, "load_config")
        for mod_name in submods:
            assert mod_name not in sys.modules
    finally:
        sys.modules.update(saved)


@pytest.mark.parametrize("bad_val", [float("inf"), float("-inf"), float("nan")])
@pytest.mark.parametrize(
    ("cls", "field"),
    [
        (EKFConfig, "wheelbase_m"),
        (EKFConfig, "lambda_max"),
        (EKFConfig, "min_speed_for_heading_mps"),
        (RoadGraphConfig, "gravity"),
        (RoadGraphConfig, "horizon_s"),
        (AssociationConfig, "chi2_thresh"),
        (VisualizerConfig, "fps"),
        (BlindMaskConfig, "start_s"),
        (BlindMaskConfig, "duration_s"),
        (DetectorConfig, "conf_thresh"),
        (ALPRConfig, "min_conf"),
    ],
)
def test_non_finite_floats_rejected_by_strict_model(
    cls: type, field: str, bad_val: float
) -> None:
    with pytest.raises(ValidationError):
        cls(**{field: bad_val})


def test_cam02_3d_projection_aligns_with_homography_and_utm_bounds() -> None:
    import osmnx as ox

    demo_cfg = load_config(REPO_ROOT / "configs" / "cameras_demo.yaml")
    cam_02 = next(c for c in demo_cfg.cameras if c.cam_id == "cam_02")
    assert cam_02.homography is not None
    assert cam_02.K is not None and cam_02.R is not None and cam_02.t is not None

    H = np.array(cam_02.homography, dtype=np.float64)
    assert H[2, 2] == pytest.approx(1.0, abs=1e-12)

    K = np.array(cam_02.K, dtype=np.float64)
    R = np.array(cam_02.R, dtype=np.float64)
    t = np.array(cam_02.t, dtype=np.float64)

    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12)
    C_utm = -R.T @ t
    assert np.allclose(C_utm, np.array([331120.0, 7393817.24, 15.0]), atol=1e-6)

    graphml_path = REPO_ROOT / "data" / "osm" / "sao_paulo_paulista.graphml"
    assert graphml_path.is_file()
    G = ox.load_graphml(graphml_path)
    xs = [float(d["x"]) for _, d in G.nodes(data=True)]
    ys = [float(d["y"]) for _, d in G.nodes(data=True)]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    K_inv = np.linalg.inv(K)
    test_pixels: list[tuple[float, float]] = [(0.0, 0.0), (640.0, 360.0), (1280.0, 720.0)]
    for poly in cam_02.exit_zones:
        test_pixels.extend(poly)

    for u, v in test_pixels:
        uv1 = np.array([u, v, 1.0], dtype=np.float64)
        p_h = H @ uv1
        xh = float(p_h[0] / p_h[2])
        yh = float(p_h[1] / p_h[2])

        ray_world = R.T @ (K_inv @ uv1)
        assert ray_world[2] < 0.0
        scale = -float(C_utm[2]) / float(ray_world[2])
        xr = float(C_utm[0] + scale * ray_world[0])
        yr = float(C_utm[1] + scale * ray_world[1])

        assert float(np.hypot(xh - xr, yh - yr)) < 1e-3
        assert xmin <= xh <= xmax
        assert ymin <= yh <= ymax
