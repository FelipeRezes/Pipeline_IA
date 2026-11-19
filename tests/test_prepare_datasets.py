"""Offline deterministic unit tests for scripts/prepare_datasets.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prepare_datasets as pd_mod


def _make_synthetic_detrac_seq(seq_dir: Path, n_frames: int = 120, fps: float = 10.0) -> None:
    img_dir = seq_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = ["frame,id,xmin,ymin,xmax,ymax"]
    for f in range(1, n_frames + 1):
        img = np.full((120, 160, 3), 50, dtype=np.uint8)
        cv2.rectangle(img, (10 + f, 40), (40 + f, 70), (20, 180, 220), -1)
        cv2.imwrite(str(img_dir / f"img{f:05d}.jpg"), img)
        rows.append(f"{f},7,{10 + f},40,{40 + f},70")
    (seq_dir / "annotations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


@pytest.mark.parametrize("duration_s", [3.0, 5.0, 8.0])
def test_blind_zone_mask_duration_and_gt_preservation(tmp_path: Path, duration_s: float) -> None:
    seq_dir = tmp_path / "detrac_seq"
    fps = 10.0
    n_frames = 120
    _make_synthetic_detrac_seq(seq_dir, n_frames=n_frames, fps=fps)

    meta = pd_mod.generate_ua_detrac_blind_mask(
        seq_dir=seq_dir,
        fps=fps,
        blind_start_s=2.0,
        blind_duration_s=duration_s,
    )
    start_frame = meta["start_frame"]
    end_frame = meta["end_frame"]
    assert start_frame == 21
    assert end_frame == round((2.0 + duration_s) * fps)

    # Masked frames must have zero pixel sum; unmasked frames must be non-zero
    masked_dir = seq_dir / "masked_frames"
    for f in range(start_frame, end_frame + 1):
        p = masked_dir / f"img{f:05d}.png"
        assert p.is_file()
        frame = cv2.imread(str(p))
        assert frame is not None
        assert int(frame.sum()) == 0

    outside_frame = cv2.imread(str(masked_dir / "img00005.jpg"))
    assert outside_frame is not None
    assert int(outside_frame.sum()) > 0

    # Metric GT must retain 100% of records including inside blind window
    gt_data = json.loads((seq_dir / "gt_metric.json").read_text(encoding="utf-8"))
    records = gt_data["records"]
    assert len(records) == n_frames
    blind_records = [r for r in records if r["is_blind"]]
    assert len(blind_records) == (end_frame - start_frame + 1)
    assert all(np.isfinite(r["x_utm"]) and np.isfinite(r["y_utm"]) for r in records)


@pytest.mark.parametrize("bad_duration", [0.0, -1.0, 2.5, 8.5])
def test_blind_zone_mask_rejects_invalid_duration(tmp_path: Path, bad_duration: float) -> None:
    seq_dir = tmp_path / "detrac_bad"
    _make_synthetic_detrac_seq(seq_dir, n_frames=120, fps=10.0)
    with pytest.raises(ValueError):
        pd_mod.generate_ua_detrac_blind_mask(
            seq_dir=seq_dir,
            fps=10.0,
            blind_start_s=2.0,
            blind_duration_s=bad_duration,
        )


def test_parse_detrac_real_csv_format(tmp_path: Path) -> None:
    seq_dir = tmp_path / "detrac_real_fmt"
    seq_dir.mkdir(parents=True)
    (seq_dir / "annotations.csv").write_text(
        ",filename,width,height,class,xmin,ymin,xmax,ymax,confidence,target_id\n"
        "0,img00001.jpg,960,540,vehicle,10.0,20.0,50.0,60.0,1.0,42\n"
        "1,img00002.jpg,960,540,vehicle,14.0,22.0,54.0,62.0,1.0,42\n",
        encoding="utf-8",
    )
    parsed = pd_mod._parse_detrac_annotations(seq_dir)
    assert parsed[1] == [(42, 10.0, 20.0, 50.0, 60.0)]
    assert parsed[2] == [(42, 14.0, 22.0, 54.0, 62.0)]


def test_strip_git_merge_conflicts() -> None:
    dirty = (
        "<<<<<<< HEAD\n"
        "version https://git-lfs.github.com/spec/v1\n"
        "=======\n"
        '{"intrinsic_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}\n'
        ">>>>>>> abcdef123\n"
    )
    cleaned = pd_mod.strip_git_merge_conflicts(dirty)
    parsed = json.loads(cleaned)
    assert parsed["intrinsic_matrix"][0][0] == 1


def test_synthehicle_and_cityflow_fallback_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    syn_entries = pd_mod.prepare_synthehicle(
        data_dir, scene="Town01-N-day", cams=["C01", "C02"], offline=True, base_dir=tmp_path
    )
    syn_dir = data_dir / "synthehicle" / "Town01-N-day"
    assert (syn_dir / "calibration.json").is_file()
    assert (syn_dir / "C01" / "gt" / "gt.txt").is_file()
    assert (syn_dir / "C01" / "seqinfo.ini").is_file()
    assert (syn_dir / "C01" / "video.mp4").is_file()
    assert syn_entries["data/synthehicle/Town01-N-day"]["source_type"] == "synthetic_fallback"

    cf_entries = pd_mod.prepare_cityflow(data_dir, base_dir=tmp_path)
    cf_dir = data_dir / "cityflow" / "S01_synthetic"
    for cam in ("c001", "c002", "c003"):
        calib_text = (cf_dir / cam / "calibration.txt").read_text(encoding="utf-8")
        assert "Homography matrix:" in calib_text
        assert (cf_dir / cam / "gt" / "gt.txt").is_file()
        assert len(list((cf_dir / cam / "frames").glob("*.jpg"))) > 0
    assert cf_entries["data/cityflow/S01_synthetic"]["source_type"] == "synthetic_fallback"


def test_veri776_and_rodosol_fallback_layout(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    veri_entries = pd_mod.prepare_veri776(data_dir, base_dir=tmp_path)
    assert (data_dir / "veri776_sample" / "gt_index.json").is_file()
    assert len(list((data_dir / "veri776_sample" / "image_query").glob("*.jpg"))) > 0
    assert veri_entries["data/veri776_sample"]["source_type"] == "synthetic_fallback"

    rodo_entries = pd_mod.prepare_rodosol_alpr(data_dir, base_dir=tmp_path)
    ann_files = sorted((data_dir / "rodosol_alpr_sample" / "annotations").glob("*.txt"))
    assert len(ann_files) == 4
    assert any("Mercosur" in p.read_text(encoding="utf-8") for p in ann_files)
    assert rodo_entries["data/rodosol_alpr_sample"]["source_type"] == "synthetic_fallback"


def test_manifest_provenance_schema(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    entries = {}
    entries.update(pd_mod.prepare_cityflow(data_dir, base_dir=tmp_path))
    entries.update(pd_mod.prepare_veri776(data_dir, base_dir=tmp_path))
    manifest_path = pd_mod.write_manifest(data_dir, entries)

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert "generated_at_utc" in payload
    for entry in payload["entries"].values():
        assert entry["source_type"] in {"real", "synthetic_fallback"}
        assert len(entry["sha256"]) == 64
        assert entry["size_bytes"] > 0
        assert entry["url"]
        assert entry["reason"]
        assert len(entry["files"]) > 0


def test_osm_cache_mirror_and_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import osmnx as ox

    osm_dir = tmp_path / "data" / "osm"
    # First: offline mode creates valid GraphML files loadable by osmnx
    entries_offline = pd_mod.prepare_osm_cache(osm_dir, offline=True, base_dir=tmp_path)
    sp_path = osm_dir / "sao_paulo_paulista.graphml"
    zh_path = osm_dir / "zurich_centro.graphml"
    assert sp_path.is_file() and zh_path.is_file()
    G_loaded = ox.load_graphml(sp_path)
    assert G_loaded.number_of_nodes() > 0
    assert entries_offline["data/osm/sao_paulo_paulista.graphml"]["source_type"] == "synthetic_fallback"

    # Second: force=False reuses cache without hitting download_file or graph_from_bbox
    mock_dl = MagicMock()
    mock_bbox = MagicMock()
    monkeypatch.setattr(pd_mod, "download_file", mock_dl)
    monkeypatch.setattr(ox, "graph_from_bbox", mock_bbox)
    pd_mod.prepare_osm_cache(osm_dir, offline=False, force=False, base_dir=tmp_path)
    mock_dl.assert_not_called()
    mock_bbox.assert_not_called()


def test_weight_downloader_mocked_and_sha_check(tmp_path: Path) -> None:
    payload = b"mock-onnx-weights-bytes-1234567890"
    expected_sha = pd_mod.hashlib.sha256(payload).hexdigest()

    fake_resp = MagicMock()
    fake_resp.__enter__ = MagicMock(return_value=fake_resp)
    fake_resp.__exit__ = MagicMock(return_value=False)
    fake_resp.raise_for_status = MagicMock()
    fake_resp.iter_content = MagicMock(return_value=[payload[:10], payload[10:]])

    fake_session = MagicMock()
    fake_session.get = MagicMock(return_value=fake_resp)

    dest = tmp_path / "weights" / "model.onnx"
    out = pd_mod.download_file(
        "https://example.com/model.onnx",
        dest,
        expected_sha256=expected_sha,
        session=fake_session,
    )
    assert out == dest
    assert dest.read_bytes() == payload
    assert fake_session.get.call_count == 1

    # Second call with valid file & matching SHA skips network
    pd_mod.download_file(
        "https://example.com/model.onnx",
        dest,
        expected_sha256=expected_sha,
        session=fake_session,
    )
    assert fake_session.get.call_count == 1

    # Hash mismatch raises ValueError and does not overwrite/leave corrupt file
    bad_dest = tmp_path / "weights" / "bad.onnx"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        pd_mod.download_file(
            "https://example.com/bad.onnx",
            bad_dest,
            expected_sha256="0" * 64,
            session=fake_session,
        )
    assert not bad_dest.exists()
    assert not bad_dest.with_suffix(".onnx.part").exists()


def test_offline_weights_create_valid_dynamic_osnet_onnx(tmp_path: Path) -> None:
    import onnxruntime as ort

    weights_dir = tmp_path / "weights"
    entries = pd_mod.prepare_weights(weights_dir, offline=True, base_dir=tmp_path)
    onnx_path = weights_dir / "reid" / "osnet_x0_25_msmt17.onnx"
    assert onnx_path.is_file()
    assert entries["weights/reid/osnet_x0_25_msmt17.onnx"]["source_type"] == "synthetic_fallback"

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out1 = sess.run(None, {"images": np.zeros((1, 3, 256, 128), dtype=np.float32)})[0]
    out4 = sess.run(None, {"images": np.zeros((4, 3, 256, 128), dtype=np.float32)})[0]
    assert out1.shape == (1, 512)
    assert out4.shape == (4, 512)


def test_blind_zone_remask_clears_stale_files(tmp_path: Path) -> None:
    seq_dir = tmp_path / "detrac_remask"
    fps = 10.0
    n_frames = 120
    _make_synthetic_detrac_seq(seq_dir, n_frames=n_frames, fps=fps)

    # First run: blind window frames 21..50 -> img00021..00050 written as .png
    pd_mod.generate_ua_detrac_blind_mask(
        seq_dir=seq_dir, fps=fps, blind_start_s=2.0, blind_duration_s=3.0
    )
    # Second run: shifted blind window frames 51..80 -> frames 21..50 must only exist as .jpg
    pd_mod.generate_ua_detrac_blind_mask(
        seq_dir=seq_dir, fps=fps, blind_start_s=5.0, blind_duration_s=3.0
    )

    masked_dir = seq_dir / "masked_frames"
    img_files = sorted(
        p for p in masked_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    stems = [p.stem for p in img_files]
    assert len(img_files) == n_frames
    assert len(set(stems)) == n_frames

    for f in range(21, 51):
        assert (masked_dir / f"img{f:05d}.jpg").is_file()
        assert not (masked_dir / f"img{f:05d}.png").exists()
    for f in range(51, 81):
        assert (masked_dir / f"img{f:05d}.png").is_file()
        assert not (masked_dir / f"img{f:05d}.jpg").exists()


def test_cached_offline_provenance_preserved_on_reuse(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    osm_dir = data_dir / "osm"

    entries = {}
    entries.update(pd_mod.prepare_osm_cache(osm_dir, offline=True, base_dir=tmp_path))
    entries.update(
        pd_mod.prepare_ua_detrac(
            data_dir, seq_name="detrac_test", max_frames=120, fps=10.0,
            blind_start_s=2.0, blind_duration_s=3.0, offline=True, base_dir=tmp_path,
        )
    )
    pd_mod.write_manifest(data_dir, entries)

    # Re-run with force=False and offline=False: must preserve synthetic_fallback from MANIFEST.json
    osm_reused = pd_mod.prepare_osm_cache(osm_dir, offline=False, force=False, base_dir=tmp_path)
    detrac_reused = pd_mod.prepare_ua_detrac(
        data_dir, seq_name="detrac_test", max_frames=120, fps=10.0,
        blind_start_s=2.0, blind_duration_s=3.0, offline=False, force=False, base_dir=tmp_path,
    )
    assert osm_reused["data/osm/sao_paulo_paulista.graphml"]["source_type"] == "synthetic_fallback"
    assert osm_reused["data/osm/zurich_centro.graphml"]["source_type"] == "synthetic_fallback"
    assert detrac_reused["data/ua_detrac/detrac_test"]["source_type"] == "synthetic_fallback"
