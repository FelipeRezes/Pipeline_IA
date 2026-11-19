from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register E2E test suite markers so --strict-markers passes cleanly."""
    markers = [
        "e2e: Opaque-box end-to-end test suite.",
        "tier1: Tier 1 feature coverage (happy-path) E2E tests.",
        "tier2: Tier 2 boundary and corner-case E2E tests.",
        "tier3: Tier 3 cross-feature pairwise integration E2E tests.",
        "tier4: Tier 4 real-world application scenario E2E tests.",
        "area1: Feature Area 1 (Config & Datasets).",
        "area2: Feature Area 2 (Geometry).",
        "area3: Feature Area 3 (Kinematics & EKF).",
        "area4: Feature Area 4 (OSM, RoadGraph & Calibration).",
        "area5: Feature Area 5 (Detection & ExitZones).",
        "area6: Feature Area 6 (Re-ID & ALPR).",
        "area7: Feature Area 7 (Association & Forensic Backtracking).",
        "area8: Feature Area 8 (Causal Multi-Cam Pipeline).",
        "area9: Feature Area 9 (CICC Split-Screen Visualizer).",
        "area10: Feature Area 10 (Ablation Benchmarks A1-A5).",
        "slow: Multi-second subprocess or full demo/benchmark tests.",
        "online: Tests requiring live internet connectivity.",
    ]
    for marker in markers:
        config.addinivalue_line("markers", marker)


def _compute_normalized_dlt_homography(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> np.ndarray:
    """Pure-NumPy normalized 4+ point DLT homography for fixture generation."""
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)

    def _norm_matrix(pts: np.ndarray) -> np.ndarray:
        mean = pts.mean(axis=0)
        diffs = pts - mean
        rms = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))
        scale = math.sqrt(2.0) / max(rms, 1e-9)
        return np.array(
            [
                [scale, 0.0, -scale * mean[0]],
                [0.0, scale, -scale * mean[1]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    t_src = _norm_matrix(src)
    t_dst = _norm_matrix(dst)
    src_h = np.column_stack([src, np.ones(len(src), dtype=np.float64)])
    dst_h = np.column_stack([dst, np.ones(len(dst), dtype=np.float64)])
    src_n = (t_src @ src_h.T).T
    dst_n = (t_dst @ dst_h.T).T

    rows = []
    for (u, v, _), (x, y, _) in zip(src_n, dst_n):
        rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
        rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v, -x])
    a_mat = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a_mat)
    h_norm = vt[-1].reshape(3, 3)
    h_mat = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(h_mat[2, 2]) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]
    return h_mat.astype(np.float64)


def _apply_homography(H: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts_uv, dtype=np.float64)
    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def venv_python(repo_root: Path) -> Path:
    if os.name == "nt":
        return repo_root / ".venv" / "Scripts" / "python.exe"
    return repo_root / ".venv" / "bin" / "python"


@pytest.fixture
def headless_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    overrides = {
        "MPLBACKEND": "Agg",
        "DISPLAY": "",
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    for key, val in overrides.items():
        monkeypatch.setenv(key, val)
    env = os.environ.copy()
    env.update(overrides)
    return env


@pytest.fixture
def run_cli(
    venv_python: Path,
    repo_root: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    def _runner(
        args: list[str],
        env: dict[str, str] | None = None,
        timeout: int = 120,
    ) -> subprocess.CompletedProcess[str]:
        if not venv_python.exists():
            raise FileNotFoundError(f"Expected project virtualenv Python at {venv_python}")
        merged_env = os.environ.copy()
        merged_env.update(
            {
                "MPLBACKEND": "Agg",
                "DISPLAY": "",
                "QT_QPA_PLATFORM": "offscreen",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            }
        )
        if env is not None:
            merged_env.update(env)
        cmd = [str(venv_python), *args]
        return subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=merged_env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )

    return _runner


@pytest.fixture(scope="session")
def utm_origin() -> tuple[float, float, int]:
    return (330000.0, 7390000.0, 32723)


@pytest.fixture(scope="session")
def synthetic_graphml_path(
    tmp_path_factory: pytest.TempPathFactory,
    utm_origin: tuple[float, float, int],
) -> Path:
    import networkx as nx

    ox, oy, epsg = utm_origin
    graph = nx.MultiDiGraph(crs=f"EPSG:{epsg}", simplified="True")

    nodes: dict[int, tuple[float, float]] = {
        100: (ox + 0.0, oy + 0.0),       # N0
        101: (ox + 100.0, oy + 0.0),     # N1
        102: (ox + 200.0, oy + 0.0),     # N2
        103: (ox + 300.0, oy + 0.0),     # N3
        104: (ox + 100.0, oy + 15.0),    # N_cross (15m north of N1)
        105: (ox + 50.0, oy + 12.0),     # N_side1 (12m north of corridor)
        106: (ox + 150.0, oy + 12.0),    # N_side2 (12m north of corridor)
        107: (ox + 108.0, oy - 120.0),   # N4 (after R=8m 90-deg curve)
        108: (ox + 200.0, oy + 100.0),   # N_in_only (north of N2)
        109: (ox + 200.0, oy - 120.0),   # N5 (south of N2)
        110: (ox + 200.0, oy - 200.0),   # N6 (convergence blockade node)
        201: (ox + 500.0, oy + 500.0),   # N_iso1 (disconnected island)
        202: (ox + 550.0, oy + 500.0),   # N_iso2
    }
    for nid, (x, y) in nodes.items():
        graph.add_node(nid, x=float(x), y=float(y), street_count=2)

    def _wkt(coords: list[tuple[float, float]]) -> str:
        pts_str = ", ".join(f"{px:.6f} {py:.6f}" for px, py in coords)
        return f"LINESTRING ({pts_str})"

    def _poly_len(coords: list[tuple[float, float]]) -> float:
        return float(
            sum(
                math.hypot(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1])
                for i in range(len(coords) - 1)
            )
        )

    def _add_edge(
        u: int,
        v: int,
        coords: list[tuple[float, float]],
        oneway: bool,
        maxspeed_kmh: str,
        lanes: str,
        osmid: int,
    ) -> None:
        graph.add_edge(
            u,
            v,
            key=0,
            osmid=osmid,
            length=_poly_len(coords),
            oneway=oneway,
            maxspeed=maxspeed_kmh,
            lanes=lanes,
            highway="secondary" if lanes == "2" else "residential",
            geometry=_wkt(coords),
        )
        if not oneway:
            rev_coords = list(reversed(coords))
            graph.add_edge(
                v,
                u,
                key=0,
                osmid=osmid,
                length=_poly_len(rev_coords),
                oneway=False,
                maxspeed=maxspeed_kmh,
                lanes=lanes,
                highway="secondary" if lanes == "2" else "residential",
                geometry=_wkt(rev_coords),
            )

    # Main eastbound arterial N0 -> N1 -> N2 -> N3
    _add_edge(100, 101, [nodes[100], nodes[101]], True, "50", "2", 1001)
    _add_edge(101, 102, [nodes[101], nodes[102]], True, "50", "2", 1002)
    _add_edge(102, 103, [nodes[102], nodes[103]], True, "50", "2", 1003)

    # Parallel / cross street traps for HMM vs Euclidean map-matching
    _add_edge(101, 104, [nodes[101], nodes[104]], False, "30", "1", 1004)
    _add_edge(105, 106, [nodes[105], nodes[106]], False, "30", "1", 1005)

    # N1 -> N4: 90-degree curve of radius R=8.0m centered at (ox+100, oy-8)
    # tangent to eastbound at N1 (phi=pi/2) curving southbound to (ox+108, oy-8) (phi=0),
    # then straight south to N4=(ox+108, oy-120).
    arc_pts: list[tuple[float, float]] = []
    for k in range(17):
        phi = (math.pi / 2.0) * (1.0 - k / 16.0)
        arc_pts.append((ox + 100.0 + 8.0 * math.cos(phi), oy - 8.0 + 8.0 * math.sin(phi)))
    arc_pts.append((ox + 108.0, oy - 64.0))
    arc_pts.append(nodes[107])
    _add_edge(101, 107, arc_pts, True, "50", "2", 1006)

    # One-way entry N_in_only -> N2, one-way exit N2 -> N5, convergence N4 -> N6 and N5 -> N6
    _add_edge(108, 102, [nodes[108], nodes[102]], True, "40", "1", 1007)
    _add_edge(102, 109, [nodes[102], nodes[109]], True, "50", "2", 1008)
    _add_edge(107, 110, [nodes[107], nodes[110]], True, "50", "2", 1009)
    _add_edge(109, 110, [nodes[109], nodes[110]], True, "50", "2", 1010)

    # Disconnected island N_iso1 -> N_iso2
    _add_edge(201, 202, [nodes[201], nodes[202]], True, "40", "1", 1011)

    out_dir = tmp_path_factory.mktemp("osm_cache")
    out_path = out_dir / "synthetic_utm.graphml"
    nx.write_graphml(graph, out_path)
    return out_path


@pytest.fixture(scope="session")
def synthetic_road_network(synthetic_graphml_path: Path) -> Any:
    osm_loader = importlib.import_module("tracker.osm_loader")
    return osm_loader.load_utm_road_graph(
        graphml_cache=synthetic_graphml_path,
        overpass_urls=[],
    )


@pytest.fixture(scope="session")
def synthetic_road_engine(
    synthetic_road_network: Any,
    synthetic_graphml_path: Path,
) -> Any:
    cfg_mod = importlib.import_module("tracker.config")
    rg_mod = importlib.import_module("tracker.roadgraph")
    rg_cfg = cfg_mod.RoadGraphConfig(
        place_name=None,
        bbox_wgs84=None,
        graphml_cache=synthetic_graphml_path,
        mu_friction=0.7,
        gravity=9.81,
        corner_angle_deg=30.0,
        horizon_s=15.0,
        min_branch_prob=0.02,
    )
    return rg_mod.RoadGraphEngine(synthetic_road_network, rg_cfg)


@pytest.fixture(scope="session")
def synthetic_calibration_bundle(
    tmp_path_factory: pytest.TempPathFactory,
    utm_origin: tuple[float, float, int],
) -> dict[str, Any]:
    import cv2
    import yaml

    ox, oy, epsg = utm_origin
    out_dir = tmp_path_factory.mktemp("calib_bundle")

    # 4 non-collinear corners defining a realistic perspective camera trapezoid -> 60m x 40m UTM box
    src_corners = np.array(
        [[60.0, 420.0], [580.0, 420.0], [490.0, 140.0], [150.0, 140.0]],
        dtype=np.float64,
    )
    dst_corners = np.array(
        [
            [ox + 50.0, oy - 20.0],
            [ox + 110.0, oy - 20.0],
            [ox + 110.0, oy + 20.0],
            [ox + 50.0, oy + 20.0],
        ],
        dtype=np.float64,
    )
    h_gt = _compute_normalized_dlt_homography(src_corners, dst_corners)

    # 6 non-collinear correspondences derived from H_gt
    pts_uv_6 = np.array(
        [
            [80.0, 400.0],
            [560.0, 400.0],
            [480.0, 160.0],
            [160.0, 160.0],
            [320.0, 280.0],
            [240.0, 340.0],
        ],
        dtype=np.float64,
    )
    pts_utm_6 = _apply_homography(h_gt, pts_uv_6)

    def _pts_payload(uv_arr: np.ndarray, utm_arr: np.ndarray) -> dict[str, Any]:
        return {
            "epsg_utm": epsg,
            "points": [
                {
                    "u": float(uv[0]),
                    "v": float(uv[1]),
                    "x_utm": float(xy[0]),
                    "y_utm": float(xy[1]),
                }
                for uv, xy in zip(uv_arr, utm_arr)
            ],
        }

    points_6_json = out_dir / "points_6.json"
    points_6_json.write_text(
        json.dumps(_pts_payload(pts_uv_6, pts_utm_6), indent=2),
        encoding="utf-8",
    )

    points_3_json = out_dir / "points_3.json"
    points_3_json.write_text(
        json.dumps(_pts_payload(pts_uv_6[:3], pts_utm_6[:3]), indent=2),
        encoding="utf-8",
    )

    collinear_uv = np.array(
        [[100.0, 240.0], [220.0, 240.0], [340.0, 240.0], [460.0, 240.0]],
        dtype=np.float64,
    )
    collinear_utm = _apply_homography(h_gt, collinear_uv)
    points_4_collinear_json = out_dir / "points_4_collinear.json"
    points_4_collinear_json.write_text(
        json.dumps(_pts_payload(collinear_uv, collinear_utm), indent=2),
        encoding="utf-8",
    )

    # Synthetic 640x480 BGR road reference image
    img = np.full((480, 640, 3), 65, dtype=np.uint8)
    cv2.line(img, (0, 240), (640, 240), (200, 200, 200), 2)
    for uv in pts_uv_6:
        cv2.circle(img, (int(round(uv[0])), int(round(uv[1]))), 4, (0, 220, 220), -1)
    ref_image_path = out_dir / "ref_road_640x480.png"
    cv2.imwrite(str(ref_image_path), img)

    cam_yaml_path = out_dir / "cam_synth.yaml"
    cam_yaml_payload = {
        "cam_id": "cam_synth",
        "epsg_utm": epsg,
        "resolution": [640, 480],
        "fps": 10.0,
        "homography": h_gt.tolist(),
    }
    cam_yaml_path.write_text(yaml.safe_dump(cam_yaml_payload, sort_keys=False), encoding="utf-8")

    return {
        "ref_image_path": ref_image_path,
        "H_gt": h_gt,
        "points_6_json": points_6_json,
        "points_3_json": points_3_json,
        "points_4_collinear_json": points_4_collinear_json,
        "cam_yaml_path": cam_yaml_path,
    }


@pytest.fixture(scope="session")
def synthetic_bgr_frames_and_crops() -> dict[str, Any]:
    def _make_crop(body_bgr: tuple[int, int, int], roof_bgr: tuple[int, int, int]) -> np.ndarray:
        crop = np.zeros((128, 64, 3), dtype=np.uint8)
        crop[:, :] = body_bgr
        crop[20:48, 8:56] = (40, 40, 40)  # dark windshield
        crop[52:92, 10:54] = roof_bgr     # roof contrast
        return crop

    crop_red_a = _make_crop((30, 30, 220), (40, 40, 230))
    crop_red_a_shifted = np.roll(crop_red_a, shift=2, axis=0)
    crop_red_a_shifted = np.clip(crop_red_a_shifted.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    crop_blue_b = _make_crop((220, 60, 30), (235, 235, 235))

    exit_zones_east = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def _draw_vehicle(
        frame: np.ndarray,
        bbox: tuple[float, float, float, float],
        color_bgr: tuple[int, int, int] = (30, 30, 220),
    ) -> None:
        x1 = max(0, min(640, int(round(bbox[0]))))
        y1 = max(0, min(480, int(round(bbox[1]))))
        x2 = max(0, min(640, int(round(bbox[2]))))
        y2 = max(0, min(480, int(round(bbox[3]))))
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = color_bgr

    def make_sequence_active(n_frames: int = 10) -> list[tuple[np.ndarray, np.ndarray]]:
        seq: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_frames):
            frame = np.full((480, 640, 3), 55, dtype=np.uint8)
            x1 = 100.0 + i * 20.0
            bbox = (x1, 200.0, x1 + 80.0, 260.0)
            _draw_vehicle(frame, bbox, (30, 30, 220))
            dets = np.array([[*bbox, 0.90, 2.0]], dtype=np.float32)
            seq.append((frame, dets))
        return seq

    def make_sequence_center_occlusion() -> list[tuple[np.ndarray, np.ndarray]]:
        seq: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(16):
            frame = np.full((480, 640, 3), 55, dtype=np.uint8)
            if i <= 5:
                x1 = 220.0 + i * 10.0
                bbox = (x1, 200.0, x1 + 60.0, 260.0)
                _draw_vehicle(frame, bbox, (30, 30, 220))
                dets = np.array([[*bbox, 0.90, 2.0]], dtype=np.float32)
            elif i <= 10:
                dets = np.empty((0, 6), dtype=np.float32)
            else:
                x1 = 310.0 + (i - 11) * 10.0
                bbox = (x1, 200.0, x1 + 60.0, 260.0)
                _draw_vehicle(frame, bbox, (30, 30, 220))
                dets = np.array([[*bbox, 0.90, 2.0]], dtype=np.float32)
            seq.append((frame, dets))
        return seq

    def make_sequence_exit_zone() -> list[tuple[np.ndarray, np.ndarray]]:
        seq: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(13):
            frame = np.full((480, 640, 3), 55, dtype=np.uint8)
            if i <= 8:
                x1 = 430.0 + i * 18.0  # reaches x1=574 (inside East ExitZone [560, 640]) by frame 8
                bbox = (x1, 210.0, min(635.0, x1 + 65.0), 280.0)
                _draw_vehicle(frame, bbox, (30, 30, 220))
                dets = np.array([[*bbox, 0.91, 2.0]], dtype=np.float32)
            else:
                dets = np.empty((0, 6), dtype=np.float32)
            seq.append((frame, dets))
        return seq

    def make_sequence_two_vehicles(n_frames: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
        seq: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(n_frames):
            frame = np.full((480, 640, 3), 55, dtype=np.uint8)
            ax1 = 80.0 + i * 25.0
            bx1 = 500.0 - i * 25.0
            bbox_a = (ax1, 140.0, ax1 + 70.0, 200.0)
            bbox_b = (bx1, 300.0, bx1 + 80.0, 370.0)
            _draw_vehicle(frame, bbox_a, (30, 30, 220))
            _draw_vehicle(frame, bbox_b, (220, 60, 30))
            dets = np.array(
                [
                    [*bbox_a, 0.92, 2.0],
                    [*bbox_b, 0.89, 7.0],
                ],
                dtype=np.float32,
            )
            seq.append((frame, dets))
        return seq

    return {
        "crop_red_a": crop_red_a,
        "crop_red_a_shifted": crop_red_a_shifted,
        "crop_blue_b": crop_blue_b,
        "exit_zones_east": exit_zones_east,
        "blank_frame": blank_frame,
        "make_sequence_active": make_sequence_active,
        "make_sequence_center_occlusion": make_sequence_center_occlusion,
        "make_sequence_exit_zone": make_sequence_exit_zone,
        "make_sequence_two_vehicles": make_sequence_two_vehicles,
    }


@pytest.fixture(scope="session")
def synthetic_reid_onnx_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out_dir = tmp_path_factory.mktemp("reid_onnx")
    onnx_path = out_dir / "tiny_reid_512.onnx"

    rng = np.random.default_rng(12345)
    raw = rng.standard_normal((512, 24), dtype=np.float64)
    q_mat, _ = np.linalg.qr(raw)
    proj_w = q_mat.T.astype(np.float32)  # (24, 512)

    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 3, 256, 128])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 512])
        w_init = numpy_helper.from_array(proj_w, name="proj_w")

        n1 = helper.make_node(
            "AveragePool",
            inputs=["input"],
            outputs=["pooled"],
            kernel_shape=[64, 64],
            strides=[64, 64],
        )
        n2 = helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1)
        n3 = helper.make_node("MatMul", inputs=["flat", "proj_w"], outputs=["logits"])
        n4 = helper.make_node("LpNormalization", inputs=["logits"], outputs=["output"], axis=1, p=2)

        graph = helper.make_graph(
            [n1, n2, n3, n4],
            "TinyReID512",
            [inp],
            [out],
            initializer=[w_init],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        onnx.checker.check_model(model)
        onnx.save(model, str(onnx_path))
        return onnx_path
    except ImportError:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class _TinyReID(nn.Module):
            def __init__(self, weight_24x512: np.ndarray) -> None:
                super().__init__()
                self.pool = nn.AvgPool2d(kernel_size=64, stride=64)
                self.register_buffer("w", torch.from_numpy(weight_24x512))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                p = self.pool(x).flatten(1)
                z = torch.matmul(p, self.w)
                return F.normalize(z, p=2, dim=1)

        mod = _TinyReID(proj_w).eval()
        dummy = torch.zeros((1, 3, 256, 128), dtype=torch.float32)
        torch.onnx.export(
            mod,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=13,
        )
        return onnx_path


@pytest.fixture(scope="session")
def synthetic_plate_crops() -> dict[str, np.ndarray]:
    import cv2

    def _draw_plate(text: str, mercosul: bool) -> np.ndarray:
        plate = np.full((80, 240, 3), 245, dtype=np.uint8)
        cv2.rectangle(plate, (2, 2), (237, 77), (25, 25, 25), 2)
        if mercosul:
            cv2.rectangle(plate, (4, 4), (235, 22), (180, 60, 20), -1)
        else:
            cv2.rectangle(plate, (4, 4), (235, 18), (190, 190, 190), -1)
        cv2.putText(
            plate,
            text,
            (18, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        return plate

    rng = np.random.default_rng(42)
    return {
        "plate_mercosul_abc1d23": _draw_plate("ABC1D23", mercosul=True),
        "plate_legacy_xyz9876": _draw_plate("XYZ9876", mercosul=False),
        "plate_blank_black": np.zeros((80, 240, 3), dtype=np.uint8),
        "plate_gaussian_noise": rng.integers(0, 256, (80, 240, 3), dtype=np.uint8),
    }


@pytest.fixture(scope="session")
def calibrated_two_cam_setup(
    tmp_path_factory: pytest.TempPathFactory,
    utm_origin: tuple[float, float, int],
    synthetic_road_network: Any,
) -> dict[str, Any]:
    cfg_mod = importlib.import_module("tracker.config")
    geom_mod = importlib.import_module("tracker.geometry")

    ox, oy, epsg = utm_origin
    src_corners = np.array(
        [[60.0, 420.0], [580.0, 420.0], [490.0, 140.0], [150.0, 140.0]],
        dtype=np.float64,
    )
    # cam_1 covers [ox+20..ox+80] along N0->N1; cam_2 covers [ox+160..ox+220] across N2
    # leaving an 80m blind corridor (~5s at 16 m/s) between cam_1 exit and cam_2 entry.
    dst_cam1 = np.array(
        [
            [ox + 20.0, oy - 15.0],
            [ox + 80.0, oy - 15.0],
            [ox + 80.0, oy + 15.0],
            [ox + 20.0, oy + 15.0],
        ],
        dtype=np.float64,
    )
    dst_cam2 = np.array(
        [
            [ox + 160.0, oy - 15.0],
            [ox + 220.0, oy - 15.0],
            [ox + 220.0, oy + 15.0],
            [ox + 160.0, oy + 15.0],
        ],
        dtype=np.float64,
    )
    h_cam1 = _compute_normalized_dlt_homography(src_corners, dst_cam1)
    h_cam2 = _compute_normalized_dlt_homography(src_corners, dst_cam2)

    exit_east = [[(560.0, 150.0), (640.0, 150.0), (640.0, 350.0), (560.0, 350.0)]]
    cam_1 = cfg_mod.CameraConfig(
        cam_id="cam_1",
        source="synthetic://cam_1",
        fps=10.0,
        resolution=(640, 480),
        calib_path=None,
        homography=h_cam1.tolist(),
        K=None,
        R=None,
        t=None,
        exit_zones=exit_east,
    )
    cam_2 = cfg_mod.CameraConfig(
        cam_id="cam_2",
        source="synthetic://cam_2",
        fps=10.0,
        resolution=(640, 480),
        calib_path=None,
        homography=h_cam2.tolist(),
        K=None,
        R=None,
        t=None,
        exit_zones=exit_east,
    )

    proj_1 = geom_mod.CameraProjector.from_homography(h_cam1, epsg_utm=epsg)
    proj_2 = geom_mod.CameraProjector.from_homography(h_cam2, epsg_utm=epsg)

    out_dir = tmp_path_factory.mktemp("two_cam_setup")
    dummy_yolo = out_dir / "yolo_placeholder.pt"
    dummy_onnx = out_dir / "reid_placeholder.onnx"

    # Build PipelineConfig using default sub-configs if available or explicit field values
    graphml_file = out_dir / "network.graphml"
    import networkx as nx

    nx.write_graphml(synthetic_road_network.graph, graphml_file)

    pipeline_config = cfg_mod.PipelineConfig(
        cameras=[cam_1, cam_2],
        detector=cfg_mod.DetectorConfig(
            model_path=dummy_yolo,
            tracker_type="bytetrack",
        ),
        reid=cfg_mod.ReIDConfig(
            onnx_path=dummy_onnx,
            allow_hsv_fallback=True,
        ),
        alpr=cfg_mod.ALPRConfig(enabled=False),
        ekf=cfg_mod.EKFConfig(),
        road_graph=cfg_mod.RoadGraphConfig(
            place_name=None,
            bbox_wgs84=None,
            graphml_cache=graphml_file,
        ),
        association=cfg_mod.AssociationConfig(),
        visualizer=cfg_mod.VisualizerConfig(output_video=None, width=1280, height=480),
    )

    return {
        "cam_1": cam_1,
        "cam_2": cam_2,
        "projectors": {"cam_1": proj_1, "cam_2": proj_2},
        "pipeline_config": pipeline_config,
    }
