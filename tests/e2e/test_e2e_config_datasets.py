from __future__ import annotations

import importlib
import json
import re
import tomllib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.area1]


# ---------------------------------------------------------------------------
# Tier 1 — Happy Path (7 tests)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_a1_01_packaging_and_venv(
    repo_root: Path,
    venv_python: Path,
    run_cli: Callable[..., Any],
) -> None:
    pyproject_path = repo_root / "pyproject.toml"
    requirements_path = repo_root / "requirements.txt"

    assert pyproject_path.is_file(), f"Missing {pyproject_path}"
    assert requirements_path.is_file(), f"Missing {requirements_path}"
    assert requirements_path.read_text(encoding="utf-8").strip(), "requirements.txt is empty"
    assert venv_python.is_file(), f"Dedicated virtualenv python missing at {venv_python}"

    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    assert "project" in pyproject or "tool" in pyproject
    assert "src" in pyproject_text

    res = run_cli(
        [
            "-c",
            (
                "import sys, tracker, pytest; "
                "from tracker.config import load_config, PipelineConfig; "
                "assert sys.version_info >= (3, 11)"
            ),
        ]
    )
    assert res.returncode == 0, f"venv smoke import failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


@pytest.mark.tier1
def test_t1_a1_02_typed_config_and_yamls(repo_root: Path, tmp_path: Path) -> None:
    import yaml

    cfg_mod = importlib.import_module("tracker.config")
    load_config = cfg_mod.load_config
    load_scenarios = cfg_mod.load_scenarios
    PipelineConfig = cfg_mod.PipelineConfig
    CameraConfig = cfg_mod.CameraConfig
    ScenarioConfig = cfg_mod.ScenarioConfig

    default_yaml = repo_root / "configs" / "default.yaml"
    cameras_yaml = repo_root / "configs" / "cameras_demo.yaml"
    scenarios_yaml = repo_root / "configs" / "scenarios.yaml"

    for path in (default_yaml, cameras_yaml, scenarios_yaml):
        assert path.is_file(), f"Expected config file at {path}"

    cfg_default = load_config(default_yaml)
    assert isinstance(cfg_default, PipelineConfig)
    assert cfg_default.ekf.theta_window == 4
    assert cfg_default.ekf.nis_persistence_frames == 2
    assert cfg_default.road_graph.gravity == pytest.approx(9.81)

    cfg_cams = load_config(cameras_yaml)
    assert isinstance(cfg_cams, PipelineConfig)
    assert len(cfg_cams.cameras) >= 2
    for cam in cfg_cams.cameras:
        assert isinstance(cam, CameraConfig)
        assert cam.cam_id.strip()
        has_h = cam.homography is not None and np.asarray(cam.homography).shape == (3, 3)
        has_calib = cam.calib_path is not None
        has_3d = cam.K is not None and cam.R is not None and cam.t is not None
        assert has_h or has_calib or has_3d

    scenarios = load_scenarios(scenarios_yaml)
    assert isinstance(scenarios, dict) and len(scenarios) >= 1
    for sc_id, sc in scenarios.items():
        assert isinstance(sc, ScenarioConfig)
        assert sc.scenario_id == sc_id

    # Custom override YAML in tmp_path
    raw_default = yaml.safe_load(default_yaml.read_text(encoding="utf-8")) or {}
    raw_default.setdefault("ekf", {})["wheelbase_m"] = 3.1
    raw_default.setdefault("association", {})["chi2_thresh"] = 16.5
    override_path = tmp_path / "custom_override.yaml"
    override_path.write_text(yaml.safe_dump(raw_default, sort_keys=False), encoding="utf-8")

    cfg_override = load_config(override_path)
    assert isinstance(cfg_override, PipelineConfig)
    assert cfg_override.ekf.wheelbase_m == pytest.approx(3.1)
    assert cfg_override.association.chi2_thresh == pytest.approx(16.5)


@pytest.mark.tier1
def test_t1_a1_03_prepare_synthehicle_sample(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    script = repo_root / "scripts" / "prepare_datasets.py"
    assert script.is_file(), f"Missing {script}"

    out_dir = tmp_path / "data_synthehicle"
    weights_dir = tmp_path / "weights_unused"
    res = run_cli(
        [
            str(script),
            "--datasets",
            "synthehicle",
            "--data-dir",
            str(out_dir),
            "--weights-dir",
            str(weights_dir),
            "--offline",
        ],
        timeout=180,
    )
    assert res.returncode == 0, f"prepare_datasets synthehicle failed:\n{res.stdout}\n{res.stderr}"

    calib_files = list(out_dir.rglob("calibration.json"))
    gt_files = list(out_dir.rglob("gt.txt"))
    assert calib_files, f"No calibration.json produced under {out_dir}"
    assert gt_files, f"No gt.txt produced under {out_dir}"

    calib_data = json.loads(calib_files[0].read_text(encoding="utf-8"))
    assert isinstance(calib_data, dict) and len(calib_data) >= 1

    gt_lines = [ln.strip() for ln in gt_files[0].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(gt_lines) >= 1
    first_fields = re.split(r"[,\s]+", gt_lines[0])
    assert len(first_fields) >= 6
    parsed_nums = [float(tok) for tok in first_fields[:6]]
    assert all(np.isfinite(parsed_nums))


@pytest.mark.tier1
def test_t1_a1_04_prepare_cityflow_sample(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    script = repo_root / "scripts" / "prepare_datasets.py"
    out_dir = tmp_path / "data_cityflow"
    weights_dir = tmp_path / "weights_unused"

    res = run_cli(
        [
            str(script),
            "--datasets",
            "cityflow",
            "--data-dir",
            str(out_dir),
            "--weights-dir",
            str(weights_dir),
            "--offline",
        ],
        timeout=180,
    )
    assert res.returncode == 0, f"prepare_datasets cityflow failed:\n{res.stdout}\n{res.stderr}"

    calib_txts = list(out_dir.rglob("calibration.txt"))
    assert len(calib_txts) >= 2, f"Expected >=2 camera calibration.txt files under {out_dir}"

    for cpath in calib_txts:
        txt = cpath.read_text(encoding="utf-8")
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)]
        assert len(nums) >= 9, f"Expected 3x3 calibration matrix in {cpath}, got {len(nums)} floats"
        mat = np.asarray(nums[:9], dtype=np.float64).reshape(3, 3)
        assert np.all(np.isfinite(mat))

    gt_candidates = list(out_dir.rglob("gt*.txt")) + list(out_dir.rglob("gt*.csv")) + list(out_dir.rglob("gt*.json"))
    assert gt_candidates, f"Expected multi-cam GT file under {out_dir}"
    assert any(p.stat().st_size > 0 for p in gt_candidates)


@pytest.mark.tier1
def test_t1_a1_05_prepare_uadetract_with_blind_mask(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    import cv2

    script = repo_root / "scripts" / "prepare_datasets.py"
    out_dir = tmp_path / "data_uadetract"
    weights_dir = tmp_path / "weights_unused"

    res = run_cli(
        [
            str(script),
            "--datasets",
            "ua_detrac",
            "--data-dir",
            str(out_dir),
            "--weights-dir",
            str(weights_dir),
            "--blind-start-s",
            "2.0",
            "--blind-duration-s",
            "5.0",
            "--offline",
        ],
        timeout=180,
    )
    assert res.returncode == 0, f"prepare_datasets ua_detrac failed:\n{res.stdout}\n{res.stderr}"

    gt_metric_files = list(out_dir.rglob("gt_metric.json"))
    blind_mask_files = list(out_dir.rglob("blind_mask.json"))
    assert gt_metric_files, f"No gt_metric.json produced under {out_dir}"
    assert blind_mask_files, f"No blind_mask.json produced under {out_dir}"

    gt_payload = json.loads(gt_metric_files[0].read_text(encoding="utf-8"))
    mask_payload = json.loads(blind_mask_files[0].read_text(encoding="utf-8"))

    assert mask_payload["blind_start_s"] == pytest.approx(2.0)
    assert mask_payload["blind_duration_s"] == pytest.approx(5.0)
    masked_indices = mask_payload["masked_frame_indices"]
    assert isinstance(masked_indices, list) and len(masked_indices) > 0
    start_frame = int(mask_payload["start_frame"])
    end_frame = int(mask_payload["end_frame"])
    assert masked_indices == list(range(start_frame, end_frame + 1))

    records = gt_payload["records"]
    assert isinstance(records, list) and len(records) == int(gt_payload["total_frames"])
    blind_records = [r for r in records if r["is_blind"] is True]
    unblind_records = [r for r in records if r["is_blind"] is False]
    assert len(blind_records) == len(masked_indices)
    assert len(unblind_records) > 0
    assert any(int(r["frame_id"]) < start_frame for r in unblind_records)
    assert any(int(r["frame_id"]) > end_frame for r in unblind_records)
    assert all(np.isfinite(r["x_utm"]) and np.isfinite(r["y_utm"]) for r in records)

    # Verify masked frames on disk are zeroed inside blind window and non-zero outside
    seq_dir = blind_mask_files[0].parent
    masked_dir = seq_dir / "masked_frames"
    assert masked_dir.is_dir()
    first_blind_png = masked_dir / f"img{start_frame:05d}.png"
    assert first_blind_png.is_file()
    blind_img = cv2.imread(str(first_blind_png))
    assert blind_img is not None and int(blind_img.sum()) == 0
    outside_img = cv2.imread(str(masked_dir / "img00001.jpg"))
    assert outside_img is not None and int(outside_img.sum()) > 0


@pytest.mark.tier1
def test_t1_a1_06_prepare_osm_cache_and_aux_datasets(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    import networkx as nx

    script = repo_root / "scripts" / "prepare_datasets.py"
    out_osm = tmp_path / "data_osm"
    weights_dir = tmp_path / "weights_unused"
    res_osm = run_cli(
        [
            str(script),
            "--osm-only",
            "--data-dir",
            str(out_osm),
            "--weights-dir",
            str(weights_dir),
            "--offline",
        ],
        timeout=180,
    )
    assert res_osm.returncode == 0, f"prepare_datasets osm failed:\n{res_osm.stdout}\n{res_osm.stderr}"

    graphml_files = list(out_osm.rglob("*.graphml"))
    assert graphml_files, f"No .graphml cache produced under {out_osm}"
    g = nx.read_graphml(graphml_files[0])
    assert g.number_of_nodes() > 0
    assert g.number_of_edges() > 0

    for aux in ("veri776", "rodosol"):
        out_aux = tmp_path / f"data_{aux}"
        res_aux = run_cli(
            [
                str(script),
                "--datasets",
                aux,
                "--data-dir",
                str(out_aux),
                "--weights-dir",
                str(weights_dir),
                "--offline",
            ],
            timeout=180,
        )
        assert res_aux.returncode == 0, f"prepare_datasets {aux} failed:\n{res_aux.stdout}\n{res_aux.stderr}"
        imgs = [p for p in out_aux.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        labels = [p for p in out_aux.rglob("*") if p.suffix.lower() in {".txt", ".json", ".csv", ".xml"}]
        assert len(imgs) >= 1, f"No sample image created for {aux}"
        assert len(labels) >= 1, f"No sample label file created for {aux}"


@pytest.mark.tier1
@pytest.mark.slow
def test_t1_a1_07_prepare_weights_downloader(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    import onnxruntime as ort

    script = repo_root / "scripts" / "prepare_datasets.py"
    data_dir = tmp_path / "data_weights"
    weights_dir = tmp_path / "weights"

    res = run_cli(
        [
            str(script),
            "--weights-only",
            "--data-dir",
            str(data_dir),
            "--weights-dir",
            str(weights_dir),
            "--offline",
        ],
        timeout=300,
    )
    assert res.returncode == 0, f"prepare_datasets weights failed:\n{res.stdout}\n{res.stderr}"

    yolo_pt = weights_dir / "yolo11n.pt"
    reid_onnx = weights_dir / "reid" / "osnet_x0_25_msmt17.onnx"
    alpr_det = weights_dir / "alpr" / "yolo-v9-t-384-license-plates-end2end.onnx"
    alpr_ocr = weights_dir / "alpr" / "cct_xs_v1_global.onnx"
    alpr_cfg = weights_dir / "alpr" / "cct_xs_v1_global_plate_config.yaml"

    for p in (yolo_pt, reid_onnx, alpr_det, alpr_ocr, alpr_cfg):
        assert p.is_file() and p.stat().st_size > 0, f"Expected non-empty weight artifact at {p}"

    assert reid_onnx.stat().st_size > 1024, f"Synthetic Re-ID ONNX {reid_onnx} suspiciously small"
    sess = ort.InferenceSession(str(reid_onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out = sess.run(None, {in_name: np.zeros((2, 3, 256, 128), dtype=np.float32)})[0]
    assert out.shape == (2, 512)

    manifest_path = data_dir / "MANIFEST.json"
    assert manifest_path.is_file(), f"Expected MANIFEST.json at {manifest_path}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "weights/reid/osnet_x0_25_msmt17.onnx" in manifest.get("entries", {})


# ---------------------------------------------------------------------------
# Tier 2 — Boundary & Corner Cases (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_t2_a1_01_invalid_yaml_and_schema_rejection(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    import yaml

    cfg_mod = importlib.import_module("tracker.config")
    load_config = cfg_mod.load_config

    base_data = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text(encoding="utf-8")) or {}

    # (a) Negative wheelbase_m
    bad_a = json.loads(json.dumps(base_data))
    bad_a.setdefault("ekf", {})["wheelbase_m"] = -2.0
    p_a = tmp_path / "bad_wheelbase.yaml"
    p_a.write_text(yaml.safe_dump(bad_a), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p_a)

    # (b) Unknown EKF model_type
    bad_b = json.loads(json.dumps(base_data))
    bad_b.setdefault("ekf", {})["model_type"] = "invalid_model"
    p_b = tmp_path / "bad_model.yaml"
    p_b.write_text(yaml.safe_dump(bad_b), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p_b)

    # (c) Wrong 2x3 homography shape
    bad_c = json.loads(json.dumps(base_data))
    bad_c["cameras"] = [
        {
            "cam_id": "cam_bad",
            "source": "dummy.mp4",
            "fps": 10.0,
            "resolution": [640, 480],
            "homography": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "exit_zones": [],
        }
    ]
    p_c = tmp_path / "bad_homography.yaml"
    p_c.write_text(yaml.safe_dump(bad_c), encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(p_c)

    # (d) Missing file path
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nonexistent_config.yaml")


@pytest.mark.tier2
def test_t2_a1_02_degenerate_vs_empty_exit_zones() -> None:
    cfg_mod = importlib.import_module("tracker.config")
    CameraConfig = cfg_mod.CameraConfig

    cam_empty = CameraConfig(
        cam_id="cam_empty_ez",
        source="synthetic://0",
        fps=10.0,
        resolution=(640, 480),
        calib_path=None,
        homography=np.eye(3).tolist(),
        K=None,
        R=None,
        t=None,
        exit_zones=[],
    )
    assert len(cam_empty.exit_zones) == 0

    with pytest.raises(ValueError):
        CameraConfig(
            cam_id="cam_degen_ez",
            source="synthetic://0",
            fps=10.0,
            resolution=(640, 480),
            calib_path=None,
            homography=np.eye(3).tolist(),
            K=None,
            R=None,
            t=None,
            exit_zones=[[(0.0, 0.0), (10.0, 0.0)]],
        )


@pytest.mark.tier2
def test_t2_a1_03_blind_mask_duration_boundaries(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    script = repo_root / "scripts" / "prepare_datasets.py"
    weights_dir = tmp_path / "weights_unused"

    for valid_dur in (3.0, 5.0, 8.0):
        out_dir = tmp_path / f"dur_{valid_dur}"
        res = run_cli(
            [
                str(script),
                "--datasets",
                "ua_detrac",
                "--data-dir",
                str(out_dir),
                "--weights-dir",
                str(weights_dir),
                "--blind-start-s",
                "1.0",
                "--blind-duration-s",
                str(valid_dur),
                "--offline",
            ],
            timeout=120,
        )
        assert res.returncode == 0, f"Expected duration {valid_dur} to succeed:\n{res.stdout}\n{res.stderr}"
        assert list(out_dir.rglob("blind_mask.json")), f"Expected blind_mask.json for duration {valid_dur}"
        assert list(out_dir.rglob("gt_metric.json")), f"Expected gt_metric.json for duration {valid_dur}"

    for invalid_dur in (0.0, -2.0, 2.5, 8.5, 9999.0):
        out_dir = tmp_path / f"dur_bad_{invalid_dur}"
        res = run_cli(
            [
                str(script),
                "--datasets",
                "ua_detrac",
                "--data-dir",
                str(out_dir),
                "--weights-dir",
                str(weights_dir),
                "--blind-start-s",
                "1.0",
                "--blind-duration-s",
                str(invalid_dur),
                "--offline",
            ],
            timeout=120,
        )
        assert res.returncode != 0, f"Invalid blind duration {invalid_dur} should fail non-zero"


@pytest.mark.tier2
def test_t2_a1_04_cityflow_offline_fallback_determinism(
    repo_root: Path,
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    script = repo_root / "scripts" / "prepare_datasets.py"
    out_dir = tmp_path / "cityflow_offline"
    weights_dir = tmp_path / "weights_unused"

    offline_env = {
        "HTTP_PROXY": "http://127.0.0.1:1",
        "HTTPS_PROXY": "http://127.0.0.1:1",
    }
    res = run_cli(
        [
            str(script),
            "--datasets",
            "cityflow",
            "--data-dir",
            str(out_dir),
            "--weights-dir",
            str(weights_dir),
            "--offline",
        ],
        env=offline_env,
        timeout=120,
    )
    assert res.returncode == 0, f"Offline cityflow fallback failed:\n{res.stdout}\n{res.stderr}"
    calib_txts = list(out_dir.rglob("calibration.txt"))
    assert len(calib_txts) >= 2
    for cpath in calib_txts:
        nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cpath.read_text(encoding="utf-8"))]
        assert len(nums) >= 9


@pytest.mark.tier2
def test_t2_a1_05_utf8_and_spaces_in_config_paths(
    repo_root: Path,
    tmp_path: Path,
) -> None:
    import yaml

    cfg_mod = importlib.import_module("tracker.config")
    load_config = cfg_mod.load_config

    base_data = yaml.safe_load((repo_root / "configs" / "default.yaml").read_text(encoding="utf-8")) or {}
    base_data["cameras"] = [
        {
            "cam_id": "Câm_01",
            "source": "vídeo teste/câmera 1.mp4",
            "fps": 15.0,
            "resolution": [1280, 720],
            "homography": np.eye(3).tolist(),
            "exit_zones": [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]],
        }
    ]

    nested_dir = tmp_path / "cenário CICC (teste)"
    nested_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = nested_dir / "cfg_espaço.yaml"
    cfg_path.write_text(yaml.safe_dump(base_data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    loaded = load_config(cfg_path)
    assert loaded.cameras[0].cam_id == "Câm_01"
    assert "câmera 1.mp4" in loaded.cameras[0].source


@pytest.mark.tier2
def test_t2_a1_06_corrupt_zero_byte_weight_guard(tmp_path: Path) -> None:
    cfg_mod = importlib.import_module("tracker.config")
    reid_mod = importlib.import_module("tracker.reid")

    bad_onnx = tmp_path / "bad.onnx"
    bad_onnx.write_bytes(b"")

    cfg = cfg_mod.ReIDConfig(onnx_path=bad_onnx, allow_hsv_fallback=False)
    with pytest.raises(Exception):
        reid_mod.ReIDExtractor(cfg)
