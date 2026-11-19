"""Dataset, OSM Road-Graph Cache, Pretrained Weights & Blind-Zone Mask Preparer.

Downloads public pretrained weights (YOLO11n, OSNet Deep Re-ID ONNX, fast-alpr ONNX),
caches projected UTM OpenStreetMap road networks (Sao Paulo Av. Paulista + Zurich Centro),
fetches real samples from Synthehicle (HuggingFace) and UA-DETRAC (HTTP Range ZIP extract),
applies a deterministic 3-8 s blind-zone blackout mask while retaining 100% metric GT,
and generates spec-compliant synthetic fallbacks for portal-restricted datasets
(CityFlow, VeRi-776, RodoSol-ALPR) with full SHA-256 provenance in data/MANIFEST.json.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import ssl
import struct
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import certifi
import cv2
import networkx as nx
import numpy as np
import requests

# Canonical URLs
YOLO11N_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
OSNET_PT_URL = (
    "https://huggingface.co/paulosantiago/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.pt"
)
OSNET_ONNX_FALLBACK_URL = (
    "https://huggingface.co/anriha/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx"
)
ALPR_DET_URL = (
    "https://github.com/ankandrew/open-image-models/releases/download/assets/"
    "yolo-v9-t-384-license-plates-end2end.onnx"
)
ALPR_OCR_URL = (
    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/cct_xs_v1_global.onnx"
)
ALPR_CFG_URL = (
    "https://github.com/ankandrew/cnn-ocr-lp/releases/download/arg-plates/"
    "cct_xs_v1_global_plate_config.yaml"
)

SAO_PAULO_BBOX = (-46.6590, -23.5620, -46.6500, -23.5535)  # (left, bottom, right, top)
ZURICH_BBOX = (8.535, 47.368, 8.545, 47.376)  # (left, bottom, right, top)
OSM_API_URL = "https://api.openstreetmap.org/api/0.6/map"
OVERPASS_CH_URL = "https://overpass.osm.ch/api/interpreter"

SYNTHEHICLE_BASE_URL = "https://huggingface.co/datasets/fubel/synthehicle/resolve/main"
UA_DETRAC_ZIP_URL = (
    "https://huggingface.co/datasets/abhineet123/ua_detrac/resolve/main/ua_detrac_test_set.zip"
)


def _make_session() -> requests.Session:
    session = requests.Session()
    ca_path = certifi.where()
    if ca_path and Path(ca_path).is_file():
        session.verify = ca_path
    session.headers.update({"User-Agent": "Pipeline_IA-DatasetPreparer/0.1"})
    return session


def _ssl_context() -> ssl.SSLContext:
    ca_path = certifi.where()
    if ca_path and Path(ca_path).is_file():
        return ssl.create_default_context(cafile=ca_path)
    return ssl.create_default_context()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _rel_path(path: Path, base_dir: Path) -> str:
    """Return a normalized POSIX relative path when path is inside base_dir."""
    rp = path.resolve()
    rb = base_dir.resolve()
    return rp.relative_to(rb).as_posix() if rp.is_relative_to(rb) else path.as_posix()


def sha256_tree(files: list[Path], base_dir: Path) -> str:
    """Compute deterministic aggregate SHA-256 over relative paths and file hashes."""
    h = hashlib.sha256()
    for p in sorted(files, key=lambda x: x.as_posix()):
        if p.is_file():
            rel = _rel_path(p, base_dir)
            h.update(f"{rel}:{sha256_file(p)}\n".encode())
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    expected_sha256: str | None = None,
    force: bool = False,
    retries: int = 3,
    timeout: int = 120,
    session: requests.Session | None = None,
) -> Path:
    """Stream-download a URL to dest with certifi TLS, retries, SHA-256 verification, and atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force and dest.stat().st_size > 0:
        if expected_sha256 is None or sha256_file(dest).lower() == expected_sha256.lower():
            return dest

    sess = session or _make_session()
    last_exc: Exception | None = None
    tmp_path = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with sess.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
                resp.raise_for_status()
                h = hashlib.sha256()
                with tmp_path.open("wb") as fh:
                    for chunk in resp.iter_content(chunk_size=262144):
                        if chunk:
                            fh.write(chunk)
                            h.update(chunk)
            digest = h.hexdigest()
            if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
                tmp_path.unlink(missing_ok=True)
                raise ValueError(
                    f"SHA-256 mismatch for {url}: expected {expected_sha256}, got {digest}"
                )
            os.replace(tmp_path, dest)
            return dest
        except Exception as exc:
            last_exc = exc
            tmp_path.unlink(missing_ok=True)
            if isinstance(exc, ValueError):
                raise
            if attempt < retries:
                time.sleep(min(2.0**attempt, 8.0))

    raise RuntimeError(f"Failed to download {url} after {retries} attempts: {last_exc}") from last_exc


@dataclass
class ZipEntryInfo:
    filename: str
    compress_type: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


class RemoteZipReader:
    """Pure-Python HTTP 206 Range ZIP reader for extracting specific members without full download."""

    def __init__(
        self,
        url: str,
        session: requests.Session | None = None,
        timeout: int = 60,
        retries: int = 3,
    ) -> None:
        self.url = url
        self.session = session or _make_session()
        self.timeout = timeout
        self.retries = retries
        self._entries: list[ZipEntryInfo] | None = None
        self._total_size: int | None = None

    def _range_get(self, start: int, end: int) -> bytes:
        headers = {"Range": f"bytes={start}-{end}"}
        last_exc: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(
                    self.url, headers=headers, timeout=self.timeout, allow_redirects=True
                )
                if resp.status_code not in (200, 206):
                    resp.raise_for_status()
                return resp.content
            except Exception as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(min(2.0**attempt, 6.0))
        raise RuntimeError(f"HTTP Range GET failed ({start}-{end}): {last_exc}") from last_exc

    def _get_total_size(self) -> int:
        if self._total_size is not None:
            return self._total_size
        resp = self.session.head(self.url, timeout=self.timeout, allow_redirects=True)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        if cl:
            self._total_size = int(cl)
            return self._total_size
        # Fallback: 1-byte range request to inspect Content-Range: bytes 0-0/TOTAL
        r = self.session.get(
            self.url, headers={"Range": "bytes=0-0"}, timeout=self.timeout, allow_redirects=True
        )
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            self._total_size = int(cr.split("/")[-1])
            return self._total_size
        raise RuntimeError(f"Could not determine Content-Length for {self.url}")

    def _load_central_directory(self) -> list[ZipEntryInfo]:
        if self._entries is not None:
            return self._entries
        total = self._get_total_size()
        tail_len = min(65557, total)
        tail_start = total - tail_len
        tail = self._range_get(tail_start, total - 1)
        eocd_idx = tail.rfind(b"PK\x05\x06")
        if eocd_idx < 0:
            raise RuntimeError(f"EOCD signature not found in {self.url}")
        eocd = tail[eocd_idx : eocd_idx + 22]
        if len(eocd) < 22:
            raise RuntimeError("Truncated EOCD record")
        _, _, _, _, _, cd_size, cd_offset, _ = struct.unpack("<IHHHHIIH", eocd)
        if cd_offset == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:
            loc_idx = tail.rfind(b"PK\x06\x07")
            if loc_idx >= 0 and loc_idx + 20 <= len(tail):
                _, _, zip64_eocd_off, _ = struct.unpack("<IIQI", tail[loc_idx : loc_idx + 20])
                z64 = self._range_get(zip64_eocd_off, zip64_eocd_off + 55)
                if z64[:4] == b"PK\x06\x06":
                    _, _, _, _, _, _, _, _, cd_size, cd_offset = struct.unpack("<IQHHIIQQQQ", z64[:56])

        cd_bytes = self._range_get(cd_offset, cd_offset + cd_size - 1)
        entries: list[ZipEntryInfo] = []
        pos = 0
        while pos + 46 <= len(cd_bytes):
            if cd_bytes[pos : pos + 4] != b"PK\x01\x02":
                break
            fields = struct.unpack("<IHHHHHHIIIHHHHHII", cd_bytes[pos : pos + 46])
            compress_type = fields[4]
            comp_size = fields[8]
            uncomp_size = fields[9]
            fname_len = fields[10]
            extra_len = fields[11]
            comment_len = fields[12]
            local_offset = fields[16]
            fname_start = pos + 46
            fname_bytes = cd_bytes[fname_start : fname_start + fname_len]
            fname = fname_bytes.decode("utf-8", errors="replace")

            # Parse ZIP64 extra field (0x0001) if any 32-bit field is 0xFFFFFFFF
            extra_bytes = cd_bytes[fname_start + fname_len : fname_start + fname_len + extra_len]
            epos = 0
            while epos + 4 <= len(extra_bytes):
                hid, hsz = struct.unpack("<HH", extra_bytes[epos : epos + 4])
                if hid == 0x0001:
                    vals = extra_bytes[epos + 4 : epos + 4 + hsz]
                    vpos = 0
                    if uncomp_size == 0xFFFFFFFF and vpos + 8 <= len(vals):
                        uncomp_size = struct.unpack("<Q", vals[vpos : vpos + 8])[0]
                        vpos += 8
                    if comp_size == 0xFFFFFFFF and vpos + 8 <= len(vals):
                        comp_size = struct.unpack("<Q", vals[vpos : vpos + 8])[0]
                        vpos += 8
                    if local_offset == 0xFFFFFFFF and vpos + 8 <= len(vals):
                        local_offset = struct.unpack("<Q", vals[vpos : vpos + 8])[0]
                    break
                epos += 4 + hsz

            entries.append(
                ZipEntryInfo(
                    filename=fname,
                    compress_type=compress_type,
                    compressed_size=comp_size,
                    uncompressed_size=uncomp_size,
                    local_header_offset=local_offset,
                )
            )
            pos += 46 + fname_len + extra_len + comment_len
        self._entries = entries
        return entries

    def list_entries(self, prefix: str = "") -> list[ZipEntryInfo]:
        entries = self._load_central_directory()
        if not prefix:
            return list(entries)
        return [e for e in entries if e.filename.startswith(prefix)]

    def extract_member(self, entry: ZipEntryInfo, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Read local file header (30 bytes) + likely filename/extra overhead
        probe_end = entry.local_header_offset + 30 + len(entry.filename.encode("utf-8")) + 256
        probe = self._range_get(entry.local_header_offset, probe_end + entry.compressed_size - 1)
        if probe[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid local file header for {entry.filename}")
        _, _, _, _, _, _, _, _, _, fname_len, extra_len = struct.unpack("<IHHHHHIIIHH", probe[:30])
        data_rel_start = 30 + fname_len + extra_len
        data_rel_end = data_rel_start + entry.compressed_size
        if data_rel_end <= len(probe):
            payload = probe[data_rel_start:data_rel_end]
        else:
            abs_start = entry.local_header_offset + data_rel_start
            payload = self._range_get(abs_start, abs_start + entry.compressed_size - 1)

        if entry.compress_type == 0:
            raw = payload
        elif entry.compress_type == 8:
            raw = zlib.decompress(payload, -15)
        else:
            raise NotImplementedError(f"Unsupported ZIP compression method: {entry.compress_type}")

        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(raw)
        os.replace(tmp, dest)
        return dest


def strip_git_merge_conflicts(text: str) -> str:
    """Resolve git merge conflict markers by keeping the incoming (======= ... >>>>>>>) block."""
    if "<<<<<<<" not in text:
        return text
    out_lines: list[str] = []
    state: Literal["normal", "ours", "theirs"] = "normal"
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("<<<<<<<"):
            state = "ours"
            continue
        if stripped.startswith("=======") and state == "ours":
            state = "theirs"
            continue
        if stripped.startswith(">>>>>>>") and state in ("ours", "theirs"):
            state = "normal"
            continue
        if state != "ours":
            out_lines.append(line)
    return "".join(out_lines)


def _export_osnet_dynamic_onnx(pt_path: Path, onnx_path: Path) -> bool:
    """Export OSNet PyTorch checkpoint to dynamic-batch [B,3,256,128]->[B,512] ONNX and verify."""
    try:
        import onnxruntime as ort
        import torch
        from boxmot.reid.backbones.families.osnet.variants import osnet_x0_25

        model = osnet_x0_25(num_classes=1, pretrained=False, loss="softmax")
        ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        cleaned = {}
        for k, v in state_dict.items():
            name = k.removeprefix("module.")
            if "classifier" in name:
                continue
            cleaned[name] = v
        model.load_state_dict(cleaned, strict=False)
        model.eval()

        dummy = torch.randn(1, 3, 256, 128, dtype=torch.float32)
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_onnx = onnx_path.with_name(f".{onnx_path.stem}.tmp.onnx")
        export_kwargs: dict[str, Any] = {
            "input_names": ["images"],
            "output_names": ["output"],
            "dynamic_axes": {"images": {0: "batch"}, "output": {0: "batch"}},
            "opset_version": 17,
            "do_constant_folding": True,
        }
        import inspect

        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        torch.onnx.export(model, dummy, str(tmp_onnx), **export_kwargs)

        sess = ort.InferenceSession(str(tmp_onnx), providers=["CPUExecutionProvider"])
        out1 = sess.run(None, {"images": np.zeros((1, 3, 256, 128), dtype=np.float32)})[0]
        out3 = sess.run(None, {"images": np.zeros((3, 3, 256, 128), dtype=np.float32)})[0]
        if out1.shape != (1, 512) or out3.shape != (3, 512):
            tmp_onnx.unlink(missing_ok=True)
            return False
        os.replace(tmp_onnx, onnx_path)
        return True
    except Exception:
        return False


def _create_synthetic_osnet_onnx(onnx_path: Path) -> None:
    """Create a tiny valid dynamic-batch ONNX graph [B,3,256,128]->[B,512] for offline testing."""
    import onnx
    from onnx import TensorProto, helper

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    # GlobalAveragePool: [B, 3, 256, 128] -> [B, 3, 1, 1] -> Flatten -> [B, 3]
    # MatMul with [3, 512] constant weights -> [B, 512]
    rng = np.random.default_rng(42)
    proj = (rng.standard_normal((3, 512)).astype(np.float32) * 0.1).flatten().tolist()
    proj_init = helper.make_tensor("proj_w", TensorProto.FLOAT, [3, 512], proj)

    gap = helper.make_node("GlobalAveragePool", ["images"], ["pooled"])
    flat = helper.make_node("Flatten", ["pooled"], ["flat"], axis=1)
    mm = helper.make_node("MatMul", ["flat", "proj_w"], ["output"])

    graph = helper.make_graph(
        [gap, flat, mm],
        "synthetic_osnet",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, ["batch", 3, 256, 128])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 512])],
        initializer=[proj_init],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(onnx_path))


def prepare_weights(
    weights_dir: Path,
    offline: bool = False,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Download/export pretrained YOLO, Deep Re-ID ONNX, and ALPR ONNX weights."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    root = base_dir or weights_dir.parent
    entries: dict[str, dict[str, Any]] = {}

    # 1. YOLO11n
    yolo_path = weights_dir / "yolo11n.pt"
    if offline:
        if not yolo_path.exists() or force:
            yolo_path.write_bytes(b"OFFLINE_YOLO11N_PLACEHOLDER")
        src_type = "synthetic_fallback"
        reason = "Offline mode: generated placeholder weight file"
    else:
        download_file(YOLO11N_URL, yolo_path, force=force)
        src_type = "real"
        reason = "Official Ultralytics v8.3.0 YOLO11n weights"
    entries["weights/yolo11n.pt"] = {
        "category": "weight",
        "source_type": src_type,
        "url": YOLO11N_URL,
        "reason": reason,
        "sha256": sha256_file(yolo_path),
        "size_bytes": yolo_path.stat().st_size,
        "files": [_rel_path(yolo_path, root)],
    }

    # 2. Deep Re-ID ONNX (dynamic batch export via BoxMOT or pre-exported fallback)
    reid_dir = weights_dir / "reid"
    reid_dir.mkdir(parents=True, exist_ok=True)
    reid_onnx = reid_dir / "osnet_x0_25_msmt17.onnx"
    if offline:
        if not reid_onnx.exists() or force:
            _create_synthetic_osnet_onnx(reid_onnx)
        reid_src_type = "synthetic_fallback"
        reid_url = OSNET_PT_URL
        reid_reason = "Offline mode: generated synthetic dynamic-batch 512-D ONNX graph"
    else:
        if not reid_onnx.exists() or force or reid_onnx.stat().st_size < 10000:
            pt_tmp = reid_dir / "osnet_x0_25_msmt17.pt"
            exported = False
            try:
                download_file(OSNET_PT_URL, pt_tmp, force=force)
                exported = _export_osnet_dynamic_onnx(pt_tmp, reid_onnx)
            finally:
                pt_tmp.unlink(missing_ok=True)
            if exported:
                reid_url = OSNET_PT_URL
                reid_reason = "Exported dynamic-batch [B,3,256,128]->[B,512] ONNX from paulosantiago/osnet_x0_25_msmt17.pt"
            else:
                download_file(OSNET_ONNX_FALLBACK_URL, reid_onnx, force=True)
                reid_url = OSNET_ONNX_FALLBACK_URL
                reid_reason = "Pre-exported static-batch ONNX fallback from anriha/osnet_x0_25_msmt17"
        else:
            reid_url = OSNET_PT_URL
            reid_reason = "Cached OSNet Deep Re-ID ONNX weights"
        reid_src_type = "real"

    entries["weights/reid/osnet_x0_25_msmt17.onnx"] = {
        "category": "weight",
        "source_type": reid_src_type,
        "url": reid_url,
        "reason": reid_reason,
        "sha256": sha256_file(reid_onnx),
        "size_bytes": reid_onnx.stat().st_size,
        "files": [_rel_path(reid_onnx, root)],
    }

    # 3. ALPR ONNX weights
    alpr_dir = weights_dir / "alpr"
    alpr_dir.mkdir(parents=True, exist_ok=True)
    alpr_specs = [
        ("yolo-v9-t-384-license-plates-end2end.onnx", ALPR_DET_URL, "open-image-models plate detector ONNX"),
        ("cct_xs_v1_global.onnx", ALPR_OCR_URL, "fast-plate-ocr CCT-XS global ONNX"),
        ("cct_xs_v1_global_plate_config.yaml", ALPR_CFG_URL, "fast-plate-ocr CCT-XS plate config YAML"),
    ]
    alpr_files: list[Path] = []
    for fname, url, desc in alpr_specs:
        dest = alpr_dir / fname
        if offline:
            if not dest.exists() or force:
                if fname.endswith(".yaml"):
                    dest.write_text("max_plate_slots: 9\nalphabet: '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'\n", encoding="utf-8")
                else:
                    dest.write_bytes(b"OFFLINE_ALPR_PLACEHOLDER")
        else:
            download_file(url, dest, force=force)
        alpr_files.append(dest)

    entries["weights/alpr"] = {
        "category": "weight",
        "source_type": "synthetic_fallback" if offline else "real",
        "url": ALPR_DET_URL,
        "reason": "Offline mode: generated ALPR placeholders" if offline else "Official fast-alpr ONNX detector & OCR weights",
        "sha256": sha256_tree(alpr_files, root),
        "size_bytes": sum(p.stat().st_size for p in alpr_files),
        "files": [_rel_path(p, root) for p in alpr_files],
    }
    return entries


def _build_synthetic_utm_grid_graph(epsg_utm: int, origin_x: float, origin_y: float) -> nx.MultiDiGraph:
    """Create a synthetic urban UTM MultiDiGraph fallback with OSMnx-compatible node/edge tags."""
    G = nx.MultiDiGraph(crs=f"EPSG:{epsg_utm}")
    node_id = 1
    grid_ids: dict[tuple[int, int], int] = {}
    for r in range(4):
        for c in range(4):
            nid = node_id
            node_id += 1
            x = origin_x + c * 150.0
            y = origin_y + r * 150.0
            G.add_node(nid, x=x, y=y, street_count=4)
            grid_ids[(r, c)] = nid

    for r in range(4):
        for c in range(4):
            u = grid_ids[(r, c)]
            if c + 1 < 4:
                v = grid_ids[(r, c + 1)]
                G.add_edge(u, v, key=0, length=150.0, oneway=False, maxspeed="50", highway="secondary", lanes="2")
                G.add_edge(v, u, key=0, length=150.0, oneway=False, maxspeed="50", highway="secondary", lanes="2")
            if r + 1 < 4:
                v = grid_ids[(r + 1, c)]
                G.add_edge(u, v, key=0, length=150.0, oneway=True, maxspeed="40", highway="residential", lanes="2")
    return G


def _lookup_manifest_source_type(
    manifest_paths: list[Path], entry_key: str
) -> Literal["real", "synthetic_fallback"] | None:
    """Return recorded source_type from an existing MANIFEST.json if valid."""
    seen: set[Path] = set()
    for mpath in manifest_paths:
        resolved = mpath.resolve()
        if resolved in seen or not mpath.is_file():
            continue
        seen.add(resolved)
        try:
            payload = json.loads(mpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            continue
        entries = payload.get("entries")
        if not isinstance(entries, dict):
            continue
        entry = entries.get(entry_key)
        if isinstance(entry, dict):
            src_type = entry.get("source_type")
            if src_type in ("real", "synthetic_fallback"):
                return src_type
    return None


def prepare_osm_cache(
    osm_dir: Path,
    offline: bool = False,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Download and cache UTM-projected GraphML road networks for Sao Paulo Paulista & Zurich Centro."""
    import osmnx as ox

    osm_dir.mkdir(parents=True, exist_ok=True)
    root = base_dir or osm_dir.parent
    entries: dict[str, dict[str, Any]] = {}
    manifest_candidates = [osm_dir.parent / "MANIFEST.json", root / "data" / "MANIFEST.json"]

    # 1. Sao Paulo — Av. Paulista / Bela Vista (UTM Zone 23S EPSG:32723)
    sp_path = osm_dir / "sao_paulo_paulista.graphml"
    sp_url = (
        f"{OSM_API_URL}?bbox={SAO_PAULO_BBOX[0]},{SAO_PAULO_BBOX[1]},"
        f"{SAO_PAULO_BBOX[2]},{SAO_PAULO_BBOX[3]}"
    )
    if sp_path.exists() and not force and sp_path.stat().st_size > 0:
        manifest_src = _lookup_manifest_source_type(
            manifest_candidates, "data/osm/sao_paulo_paulista.graphml"
        )
        sp_src = manifest_src or ("synthetic_fallback" if offline else "real")
        sp_reason = (
            "Reused cached Sao Paulo Av. Paulista UTM GraphML"
            if sp_src == "real"
            else "Reused cached synthetic Sao Paulo Av. Paulista UTM GraphML"
        )
    elif offline:
        G_syn = _build_synthetic_utm_grid_graph(32723, 330750.0, 7393400.0)
        ox.save_graphml(G_syn, sp_path)
        sp_src = "synthetic_fallback"
        sp_reason = "Offline mode: generated synthetic UTM 23S grid GraphML"
    else:
        try:
            with tempfile.NamedTemporaryFile(suffix=".osm", delete=False) as tf:
                tmp_osm = Path(tf.name)
            try:
                download_file(sp_url, tmp_osm, force=True)
                G_sp = ox.graph_from_xml(tmp_osm, simplify=True)
                G_sp_utm = ox.project_graph(G_sp, to_crs="EPSG:32723")
                ox.save_graphml(G_sp_utm, sp_path)
            finally:
                tmp_osm.unlink(missing_ok=True)
            sp_src = "real"
            sp_reason = "Downloaded real OSM XML via api.openstreetmap.org/api/0.6/map and projected to EPSG:32723"
        except Exception as exc:
            G_syn = _build_synthetic_utm_grid_graph(32723, 330750.0, 7393400.0)
            ox.save_graphml(G_syn, sp_path)
            sp_src = "synthetic_fallback"
            sp_reason = f"OSM download failed ({exc}); generated synthetic UTM 23S grid GraphML"

    entries["data/osm/sao_paulo_paulista.graphml"] = {
        "category": "osm",
        "source_type": sp_src,
        "url": sp_url,
        "reason": sp_reason,
        "sha256": sha256_file(sp_path),
        "size_bytes": sp_path.stat().st_size,
        "files": [_rel_path(sp_path, root)],
    }

    # 2. Zurich Centro (Swiss Overpass mirror test map, UTM Zone 32N EPSG:32632)
    zh_path = osm_dir / "zurich_centro.graphml"
    if zh_path.exists() and not force and zh_path.stat().st_size > 0:
        manifest_src = _lookup_manifest_source_type(
            manifest_candidates, "data/osm/zurich_centro.graphml"
        )
        zh_src = manifest_src or ("synthetic_fallback" if offline else "real")
        zh_reason = (
            "Reused cached Zurich Centro UTM GraphML"
            if zh_src == "real"
            else "Reused cached synthetic Zurich Centro UTM GraphML"
        )
    elif offline:
        G_syn = _build_synthetic_utm_grid_graph(32632, 464900.0, 5246300.0)
        ox.save_graphml(G_syn, zh_path)
        zh_src = "synthetic_fallback"
        zh_reason = "Offline mode: generated synthetic UTM 32N grid GraphML"
    else:
        prev_url = getattr(ox.settings, "overpass_url", None)
        prev_cache = getattr(ox.settings, "use_cache", True)
        try:
            ox.settings.overpass_url = OVERPASS_CH_URL
            ox.settings.use_cache = False
            G_zh = ox.graph_from_bbox(bbox=ZURICH_BBOX, network_type="drive", simplify=True)
            G_zh_utm = ox.project_graph(G_zh, to_crs="EPSG:32632")
            ox.save_graphml(G_zh_utm, zh_path)
            zh_src = "real"
            zh_reason = "Downloaded real Zurich Centro road graph via overpass.osm.ch and projected to EPSG:32632"
        except Exception as exc:
            G_syn = _build_synthetic_utm_grid_graph(32632, 464900.0, 5246300.0)
            ox.save_graphml(G_syn, zh_path)
            zh_src = "synthetic_fallback"
            zh_reason = f"Overpass mirror failed ({exc}); generated synthetic UTM 32N grid GraphML"
        finally:
            if prev_url is not None:
                ox.settings.overpass_url = prev_url
            ox.settings.use_cache = prev_cache

    entries["data/osm/zurich_centro.graphml"] = {
        "category": "osm",
        "source_type": zh_src,
        "url": OVERPASS_CH_URL,
        "reason": zh_reason,
        "sha256": sha256_file(zh_path),
        "size_bytes": zh_path.stat().st_size,
        "files": [_rel_path(zh_path, root)],
    }

    return entries


def _write_synthetic_video(path: Path, width: int = 640, height: int = 360, fps: float = 10.0, n_frames: int = 40, seed: int = 0) -> None:
    """Render a compact synthetic traffic video (MP4) with moving colored vehicle rectangles."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    try:
        for i in range(n_frames):
            frame = np.full((height, width, 3), 40, dtype=np.uint8)
            cv2.rectangle(frame, (0, height // 2 - 60), (width, height // 2 + 60), (65, 65, 65), -1)
            x = int(40 + i * ((width - 140) / max(1, n_frames - 1)))
            y = height // 2 - 20 + (seed * 15) % 20
            cv2.rectangle(frame, (x, y), (x + 70, y + 36), (30, 140, 230), -1)
            writer.write(frame)
    finally:
        writer.release()


def prepare_synthehicle(
    data_dir: Path,
    scene: str = "Town01-N-day",
    cams: list[str] | None = None,
    offline: bool = False,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Download real Synthehicle cameras from HuggingFace or create offline synthetic fallback."""
    cams = cams or ["C01", "C02"]
    scene_dir = data_dir / "synthehicle" / scene
    scene_dir.mkdir(parents=True, exist_ok=True)
    root = base_dir or data_dir.parent
    town = scene.split("-")[0]
    created_files: list[Path] = []
    calibs: dict[str, Any] = {}

    if offline:
        for cam in cams:
            cam_dir = scene_dir / cam
            gt_dir = cam_dir / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)
            video_path = cam_dir / "video.mp4"
            if not video_path.exists() or force:
                _write_synthetic_video(video_path, fps=10.0, n_frames=40, seed=int(cam[1:]))
            gt_path = gt_dir / "gt.txt"
            lines = [
                f"{f},1,{40 + f * 12},160,70,36,1,3,-1,-1" for f in range(1, 41)
            ]
            gt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            ini_path = cam_dir / "seqinfo.ini"
            ini_path.write_text(
                f"[Sequence]\nname = {scene}-{cam}\nframerate = 10\nseqlength = 40\nimwidth = 640\nimheight = 360\n",
                encoding="utf-8",
            )
            calibs[cam] = {
                "intrinsic_matrix": [[500.0, 0.0, 320.0], [0.0, 500.0, 180.0], [0.0, 0.0, 1.0]],
                "extrinsic_matrix": [[1.0, 0.0, 0.0, float(cams.index(cam) * 80)], [0.0, -0.5, -0.866, 0.0], [0.0, 0.866, -0.5, 12.0], [0.0, 0.0, 0.0, 1.0]],
            }
            created_files.extend([video_path, gt_path, ini_path])
        calib_path = scene_dir / "calibration.json"
        calib_path.write_text(json.dumps(calibs, indent=2), encoding="utf-8")
        created_files.append(calib_path)
        src_type = "synthetic_fallback"
        reason = "Offline mode: generated synthetic Synthehicle-layout multi-camera fixture"
    else:
        sess = _make_session()
        split = "test" if town in ("Town06", "Town07") else "train"
        for cam in cams:
            cam_num = int(re.sub(r"\D", "", cam) or "1")
            cam_dir = scene_dir / cam
            gt_dir = cam_dir / "gt"
            gt_dir.mkdir(parents=True, exist_ok=True)

            video_url = f"{SYNTHEHICLE_BASE_URL}/{split}/{scene}/{cam}/out_rgb/video.mp4"
            gt_url = f"{SYNTHEHICLE_BASE_URL}/{split}/{scene}/{cam}/gt/gt.txt"
            ini_url = f"{SYNTHEHICLE_BASE_URL}/seqinfos/{scene}/{scene}-{cam}/seqinfo.ini"
            cal_url = f"{SYNTHEHICLE_BASE_URL}/calibration/non_overlapping/{town}/camera_info/camera_{cam_num}.txt"

            video_path = download_file(video_url, cam_dir / "video.mp4", force=force, session=sess)
            gt_path = download_file(gt_url, gt_dir / "gt.txt", force=force, session=sess)

            ini_raw = cam_dir / "seqinfo.ini.raw"
            download_file(ini_url, ini_raw, force=force, session=sess)
            ini_clean = strip_git_merge_conflicts(ini_raw.read_text(encoding="utf-8", errors="replace"))
            ini_path = cam_dir / "seqinfo.ini"
            ini_path.write_text(ini_clean, encoding="utf-8")
            ini_raw.unlink(missing_ok=True)

            cal_raw = cam_dir / "camera_info.raw"
            download_file(cal_url, cal_raw, force=force, session=sess)
            cal_clean = strip_git_merge_conflicts(cal_raw.read_text(encoding="utf-8", errors="replace"))
            calibs[cam] = json.loads(cal_clean)
            cal_raw.unlink(missing_ok=True)

            created_files.extend([video_path, gt_path, ini_path])

        calib_path = scene_dir / "calibration.json"
        calib_path.write_text(json.dumps(calibs, indent=2), encoding="utf-8")
        created_files.append(calib_path)
        src_type = "real"
        reason = f"Downloaded real CARLA Synthehicle {scene} ({','.join(cams)}) from HuggingFace fubel/synthehicle with conflict markers resolved"

    key = f"data/synthehicle/{scene}"
    return {
        key: {
            "category": "dataset",
            "source_type": src_type,
            "url": "https://huggingface.co/datasets/fubel/synthehicle",
            "reason": reason,
            "sha256": sha256_tree(created_files, root),
            "size_bytes": sum(p.stat().st_size for p in created_files if p.exists()),
            "files": [_rel_path(p, root) for p in created_files],
        }
    }


def _default_detrac_homography() -> np.ndarray:
    """Canonical street-plane homography mapping 960x540 CCTV pixels to Sao Paulo UTM 23S meters."""
    H_local = np.array(
        [
            [0.05, 0.0, -24.0],
            [0.0, -0.06, 20.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    T_offset = np.array(
        [
            [1.0, 0.0, 331100.0],
            [0.0, 1.0, 7393800.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return T_offset @ H_local


def _parse_detrac_annotations(seq_dir: Path) -> dict[int, list[tuple[int, float, float, float, float]]]:
    """Parse annotations.csv (UA-DETRAC) or gt/gt.txt (MOT) into {frame_id: [(tid, x1, y1, x2, y2)]}."""
    records: dict[int, list[tuple[int, float, float, float, float]]] = {}
    csv_path = seq_dir / "annotations.csv"
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames:
                fields = {f.strip().lower(): f for f in reader.fieldnames if f is not None}
                f_key = fields.get("frame") or fields.get("frame_id") or fields.get("image_id")
                fn_key = fields.get("filename") or fields.get("file_name")
                t_key = (
                    fields.get("id")
                    or fields.get("track_id")
                    or fields.get("object_id")
                    or fields.get("target_id")
                )
                if (f_key or fn_key) and t_key:
                    for row in reader:
                        if f_key:
                            fid = int(float(row[f_key]))
                        else:
                            m = re.search(r"(\d+)", str(row[fn_key]))  # type: ignore[index]
                            if not m:
                                continue
                            fid = int(m.group(1))
                        tid = int(float(row[t_key]))
                        if "xmin" in fields and "ymin" in fields and "xmax" in fields and "ymax" in fields:
                            x1 = float(row[fields["xmin"]])
                            y1 = float(row[fields["ymin"]])
                            x2 = float(row[fields["xmax"]])
                            y2 = float(row[fields["ymax"]])
                        elif "x" in fields and "y" in fields and "w" in fields and "h" in fields:
                            x1 = float(row[fields["x"]])
                            y1 = float(row[fields["y"]])
                            x2 = x1 + float(row[fields["w"]])
                            y2 = y1 + float(row[fields["h"]])
                        elif "x1" in fields and "y1" in fields and "x2" in fields and "y2" in fields:
                            x1 = float(row[fields["x1"]])
                            y1 = float(row[fields["y1"]])
                            x2 = float(row[fields["x2"]])
                            y2 = float(row[fields["y2"]])
                        else:
                            continue
                        records.setdefault(fid, []).append((tid, x1, y1, x2, y2))
        if records:
            return records

        # Fallback: headerless CSV (frame, id, x, y, w, h, ...)
        with csv_path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 6 and parts[0].lstrip("-").isdigit():
                    fid = int(parts[0])
                    tid = int(parts[1])
                    x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    records.setdefault(fid, []).append((tid, x1, y1, x1 + w, y1 + h))
        return records

    mot_path = seq_dir / "gt" / "gt.txt"
    if mot_path.is_file():
        with mot_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = [p.strip() for p in line.strip().split(",")]
                if len(parts) >= 6:
                    fid = int(parts[0])
                    tid = int(parts[1])
                    x1, y1, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    records.setdefault(fid, []).append((tid, x1, y1, x1 + w, y1 + h))
    return records


def generate_ua_detrac_blind_mask(
    seq_dir: Path,
    fps: float = 25.0,
    blind_start_s: float = 4.0,
    blind_duration_s: float = 5.0,
    homography: np.ndarray | None = None,
    mask_mode: Literal["zero_frame", "drop_detections"] = "zero_frame",
    strict_blind_range: bool = True,
) -> dict[str, Any]:
    """Apply a deterministic 3-8 s blind-zone blackout mask while retaining 100% unmasked metric GT."""
    if fps <= 0.0:
        raise ValueError(f"fps must be > 0, got {fps}")
    if blind_start_s < 0.0:
        raise ValueError(f"blind_start_s must be >= 0, got {blind_start_s}")
    if blind_duration_s <= 0.0:
        raise ValueError(f"blind_duration_s must be > 0, got {blind_duration_s}")
    if strict_blind_range and not (3.0 <= blind_duration_s <= 8.0):
        raise ValueError(
            f"blind_duration_s must satisfy 3.0 <= duration <= 8.0 s, got {blind_duration_s}"
        )

    img_dir = seq_dir / "images"
    if not img_dir.is_dir():
        img_dir = seq_dir / "frames"
    image_paths = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    total_frames = len(image_paths)
    start_frame = int(round(blind_start_s * fps)) + 1
    end_frame = int(round((blind_start_s + blind_duration_s) * fps))

    if total_frames < end_frame + int(round(fps)):
        raise ValueError(
            f"Sequence in {seq_dir} has only {total_frames} frames; requires at least "
            f"{end_frame + int(round(fps))} frames for pre-blind + blind ({start_frame}..{end_frame}) + post-blind"
        )

    H = homography if homography is not None else _default_detrac_homography()
    parsed = _parse_detrac_annotations(seq_dir)

    gt_records: list[dict[str, Any]] = []
    masked_dir = seq_dir / "masked_frames"
    masked_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in masked_dir.iterdir():
        if stale_file.is_file() and stale_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            stale_file.unlink()
    masked_frame_indices: list[int] = []

    for idx, img_path in enumerate(image_paths, start=1):
        is_blind = start_frame <= idx <= end_frame
        if is_blind:
            masked_frame_indices.append(idx)

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((540, 960, 3), dtype=np.uint8)
        out_img = np.zeros_like(img) if (is_blind and mask_mode == "zero_frame") else img
        # Save lossless PNG in masked_frames/ when blacked out if needed or JPG/PNG matching source
        out_path = masked_dir / img_path.name
        if is_blind and mask_mode == "zero_frame" and out_path.suffix.lower() in {".jpg", ".jpeg"}:
            # PNG guarantees exact zeros without JPEG artifacts while preserving stem
            out_path = masked_dir / f"{img_path.stem}.png"
        cv2.imwrite(str(out_path), out_img)

        t_s = (idx - 1) / fps
        for tid, x1, y1, x2, y2 in parsed.get(idx, []):
            cx = 0.5 * (x1 + x2)
            cy_contact = y1 + 0.90 * (y2 - y1)
            vec = H @ np.array([cx, cy_contact, 1.0], dtype=np.float64)
            w_val = float(vec[2]) if abs(vec[2]) > 1e-9 else 1e-9
            x_utm = float(vec[0] / w_val)
            y_utm = float(vec[1] / w_val)
            gt_records.append(
                {
                    "frame_id": idx,
                    "timestamp_s": round(t_s, 6),
                    "track_id": tid,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "contact_px": [float(cx), float(cy_contact)],
                    "x_utm": x_utm,
                    "y_utm": y_utm,
                    "is_blind": is_blind,
                }
            )

    gt_metric_path = seq_dir / "gt_metric.json"
    gt_metric_path.write_text(
        json.dumps(
            {
                "fps": fps,
                "total_frames": total_frames,
                "homography": H.tolist(),
                "records": gt_records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mask_meta = {
        "fps": fps,
        "blind_start_s": blind_start_s,
        "blind_duration_s": blind_duration_s,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "total_frames": total_frames,
        "mask_mode": mask_mode,
        "masked_frame_indices": masked_frame_indices,
        "unmasked_gt_path": "gt_metric.json",
    }
    (seq_dir / "blind_mask.json").write_text(json.dumps(mask_meta, indent=2), encoding="utf-8")
    return mask_meta


def _generate_synthetic_detrac_seq(seq_dir: Path, n_frames: int = 300, fps: float = 25.0) -> None:
    img_dir = seq_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = ["frame,id,xmin,ymin,xmax,ymax"]
    for f in range(1, n_frames + 1):
        img = np.full((540, 960, 3), 35, dtype=np.uint8)
        cv2.rectangle(img, (0, 200), (960, 360), (60, 60, 60), -1)
        x1 = 60.0 + (f - 1) * 2.5
        y1 = 240.0
        x2 = x1 + 80.0
        y2 = y1 + 48.0
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (40, 180, 240), -1)
        cv2.imwrite(str(img_dir / f"img{f:05d}.jpg"), img)
        rows.append(f"{f},1,{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}")
    (seq_dir / "annotations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def prepare_ua_detrac(
    data_dir: Path,
    seq_name: str = "detrac_100_MVI_40905",
    max_frames: int = 300,
    fps: float = 25.0,
    blind_start_s: float = 4.0,
    blind_duration_s: float = 5.0,
    offline: bool = False,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Extract a real UA-DETRAC sequence via HTTP Range ZIP read and apply 3-8 s blind mask."""
    seq_dir = data_dir / "ua_detrac" / seq_name
    img_dir = seq_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    root = base_dir or data_dir.parent

    existing_imgs = sorted(img_dir.glob("img*.jpg"))
    has_cached = len(existing_imgs) >= max_frames and (seq_dir / "annotations.csv").is_file()
    entry_key = f"data/ua_detrac/{seq_name}"

    if has_cached and not force:
        manifest_src = _lookup_manifest_source_type(
            [data_dir / "MANIFEST.json", root / "data" / "MANIFEST.json"],
            entry_key,
        )
        src_type = manifest_src or ("synthetic_fallback" if offline else "real")
        if src_type == "real":
            reason = f"Reused cached real UA-DETRAC sequence {seq_name} ({len(existing_imgs)} frames)"
        else:
            reason = f"Reused cached synthetic UA-DETRAC sequence {seq_name} ({len(existing_imgs)} frames)"
    elif offline:
        _generate_synthetic_detrac_seq(seq_dir, n_frames=max_frames, fps=fps)
        src_type = "synthetic_fallback"
        reason = "Offline mode: generated synthetic UA-DETRAC sequence with metric GT"
    else:
        reader = RemoteZipReader(UA_DETRAC_ZIP_URL)
        entries = reader.list_entries(prefix=f"{seq_name}/")
        ann_entries = [e for e in entries if e.filename.endswith("annotations.csv")]
        img_entries = sorted(
            [e for e in entries if e.filename.lower().endswith(".jpg") and "/img" in e.filename],
            key=lambda e: e.filename,
        )[:max_frames]
        if not ann_entries or not img_entries:
            raise RuntimeError(f"Sequence {seq_name} not found inside {UA_DETRAC_ZIP_URL}")
        reader.extract_member(ann_entries[0], seq_dir / "annotations.csv")
        for ent in img_entries:
            fname = Path(ent.filename).name
            dest_img = img_dir / fname
            if not dest_img.exists() or force:
                reader.extract_member(ent, dest_img)
        src_type = "real"
        reason = f"Extracted {len(img_entries)} frames + annotations.csv of {seq_name} via HTTP 206 Range from abhineet123/ua_detrac"

    generate_ua_detrac_blind_mask(
        seq_dir=seq_dir,
        fps=fps,
        blind_start_s=blind_start_s,
        blind_duration_s=blind_duration_s,
    )

    key_files = [
        p
        for p in [
            seq_dir / "annotations.csv",
            seq_dir / "gt_metric.json",
            seq_dir / "blind_mask.json",
        ]
        if p.is_file()
    ]
    all_files = sorted(img_dir.glob("*")) + sorted((seq_dir / "masked_frames").glob("*")) + key_files
    return {
        f"data/ua_detrac/{seq_name}": {
            "category": "dataset",
            "source_type": src_type,
            "url": UA_DETRAC_ZIP_URL,
            "reason": reason,
            "sha256": sha256_tree(all_files, root),
            "size_bytes": sum(p.stat().st_size for p in all_files if p.is_file()),
            "files": [_rel_path(p, root) for p in key_files]
            + [f"data/ua_detrac/{seq_name}/images ({len(list(img_dir.glob('*')))} files)"]
            + [f"data/ua_detrac/{seq_name}/masked_frames ({len(list((seq_dir / 'masked_frames').glob('*')))} files)"],
        }
    }


def prepare_cityflow(
    data_dir: Path,
    local_dir: Path | None = None,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Ingest local CityFlow dataset or generate spec-compliant multi-camera CityFlow fixture."""
    root = base_dir or data_dir.parent
    out_dir = data_dir / "cityflow" / "S01_synthetic"
    if local_dir and local_dir.is_dir():
        target = data_dir / "cityflow" / local_dir.name
        if force or not target.exists():
            shutil.copytree(local_dir, target, dirs_exist_ok=True)
        files = [p for p in target.rglob("*") if p.is_file()]
        return {
            f"data/cityflow/{local_dir.name}": {
                "category": "dataset",
                "source_type": "real",
                "url": str(local_dir),
                "reason": "Ingested user-supplied CityFlow archive",
                "sha256": sha256_tree(files, root),
                "size_bytes": sum(p.stat().st_size for p in files),
                "files": [_rel_path(p, root) for p in files[:25]],
            }
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    # 3 cameras with CityFlow calibration.txt and MOT gt.txt
    base_offsets = [(331050.0, 7393820.0), (331280.0, 7393910.0), (331510.0, 7393790.0)]
    for idx, (ox, oy) in enumerate(base_offsets, start=1):
        cam_dir = out_dir / f"c00{idx}"
        frames_dir = cam_dir / "frames"
        gt_dir = cam_dir / "gt"
        frames_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

        # Invertible 3x3 homography in CityFlow calibration.txt format
        H = np.array(
            [
                [0.04, 0.002, ox - 20.0],
                [-0.001, -0.04, oy + 15.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        calib_txt = cam_dir / "calibration.txt"
        mat_str = "; ".join(" ".join(f"{val:.8f}" for val in row) for row in H)
        calib_txt.write_text(f"Homography matrix: {mat_str}\nReprojection error: 0.12\n", encoding="utf-8")

        gt_lines: list[str] = []
        for f in range(1, 21):
            img = np.full((360, 640, 3), 38, dtype=np.uint8)
            x = 50 + (f - 1) * 24
            y = 160
            cv2.rectangle(img, (x, y), (x + 68, y + 36), (35, 160, 220), -1)
            fpath = frames_dir / f"frame_{f:04d}.jpg"
            cv2.imwrite(str(fpath), img)
            created.append(fpath)
            gt_lines.append(f"{f},101,{x},{y},68,36,1,3,-1,-1")

        gt_txt = gt_dir / "gt.txt"
        gt_txt.write_text("\n".join(gt_lines) + "\n", encoding="utf-8")
        created.extend([calib_txt, gt_txt])

    return {
        "data/cityflow/S01_synthetic": {
            "category": "dataset",
            "source_type": "synthetic_fallback",
            "url": "https://www.aicitychallenge.org/",
            "reason": "Official CityFlow/CityFlowV2 portal requires registration approval; generated multi-cam synthetic fixture with CityFlow calibration.txt and MOT gt.txt layout",
            "sha256": sha256_tree(created, root),
            "size_bytes": sum(p.stat().st_size for p in created),
            "files": [
                "data/cityflow/S01_synthetic/c001/calibration.txt",
                "data/cityflow/S01_synthetic/c001/gt/gt.txt",
                "data/cityflow/S01_synthetic/c002/calibration.txt",
                "data/cityflow/S01_synthetic/c003/calibration.txt",
            ],
        }
    }


def prepare_veri776(
    data_dir: Path,
    local_dir: Path | None = None,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Ingest local VeRi-776 dataset or generate synthetic multi-camera Re-ID crop fixture."""
    root = base_dir or data_dir.parent
    out_dir = data_dir / "veri776_sample"
    if local_dir and local_dir.is_dir():
        if force or not out_dir.exists():
            shutil.copytree(local_dir, out_dir, dirs_exist_ok=True)
        files = [p for p in out_dir.rglob("*") if p.is_file()]
        return {
            "data/veri776_sample": {
                "category": "dataset",
                "source_type": "real",
                "url": str(local_dir),
                "reason": "Ingested user-supplied VeRi-776 archive",
                "sha256": sha256_tree(files, root),
                "size_bytes": sum(p.stat().st_size for p in files),
                "files": [_rel_path(p, root) for p in files[:25]],
            }
        }

    created: list[Path] = []
    index: dict[str, list[dict[str, Any]]] = {"train": [], "query": [], "gallery": []}
    colors = [(30, 60, 210), (200, 180, 40), (45, 175, 70), (190, 50, 160)]
    for split, dir_name in [("train", "image_train"), ("query", "image_query"), ("gallery", "image_gallery")]:
        sdir = out_dir / dir_name
        sdir.mkdir(parents=True, exist_ok=True)
        for vid in range(1, 5):
            for cam in range(1, 3):
                crop = np.full((128, 256, 3), 30, dtype=np.uint8)
                cv2.rectangle(crop, (24, 24), (232, 104), colors[vid - 1], -1)
                cv2.putText(crop, f"V{vid:04d}_C{cam}", (40, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)
                fname = f"{vid:04d}_c{cam:03d}_0001_{split}.jpg"
                fpath = sdir / fname
                cv2.imwrite(str(fpath), crop)
                created.append(fpath)
                index[split].append({"file": f"{dir_name}/{fname}", "vehicle_id": vid, "camera_id": cam})

    idx_path = out_dir / "gt_index.json"
    idx_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    created.append(idx_path)

    return {
        "data/veri776_sample": {
            "category": "dataset",
            "source_type": "synthetic_fallback",
            "url": "https://github.com/VehicleReId/VeRi",
            "reason": "VeRi-776 dataset requires email/license request; generated synthetic multi-camera Re-ID image_train/query/gallery fixture",
            "sha256": sha256_tree(created, root),
            "size_bytes": sum(p.stat().st_size for p in created),
            "files": ["data/veri776_sample/gt_index.json"] + [_rel_path(p, root) for p in created[:6]],
        }
    }


def prepare_rodosol_alpr(
    data_dir: Path,
    local_dir: Path | None = None,
    force: bool = False,
    base_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Ingest local RodoSol-ALPR dataset or generate synthetic BR/Mercosul plate fixture."""
    root = base_dir or data_dir.parent
    out_dir = data_dir / "rodosol_alpr_sample"
    if local_dir and local_dir.is_dir():
        if force or not out_dir.exists():
            shutil.copytree(local_dir, out_dir, dirs_exist_ok=True)
        files = [p for p in out_dir.rglob("*") if p.is_file()]
        return {
            "data/rodosol_alpr_sample": {
                "category": "dataset",
                "source_type": "real",
                "url": str(local_dir),
                "reason": "Ingested user-supplied RodoSol-ALPR archive",
                "sha256": sha256_tree(files, root),
                "size_bytes": sum(p.stat().st_size for p in files),
                "files": [_rel_path(p, root) for p in files[:25]],
            }
        }

    img_dir = out_dir / "images"
    ann_dir = out_dir / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    samples = [
        ("car", "Brazilian", "ABC1234"),
        ("car", "Mercosur", "BRA2E19"),
        ("motorcycle", "Brazilian", "XYZ9876"),
        ("car", "Mercosur", "MER4C05"),
    ]
    created: list[Path] = []
    for idx, (vtype, layout, plate) in enumerate(samples, start=1):
        canvas = np.full((240, 400, 3), 210, dtype=np.uint8)
        if layout == "Mercosur":
            cv2.rectangle(canvas, (40, 60), (360, 180), (245, 245, 245), -1)
            cv2.rectangle(canvas, (40, 60), (360, 88), (180, 50, 20), -1)
        else:
            cv2.rectangle(canvas, (40, 60), (360, 180), (195, 195, 195), -1)
        cv2.rectangle(canvas, (40, 60), (360, 180), (20, 20, 20), 3)
        cv2.putText(canvas, plate, (58, 148), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (15, 15, 15), 4)

        img_path = img_dir / f"plate_{idx:03d}.jpg"
        ann_path = ann_dir / f"plate_{idx:03d}.txt"
        cv2.imwrite(str(img_path), canvas)
        ann_path.write_text(
            f"type: {vtype}\nlayout: {layout}\nplate: {plate}\ncorners: 40,60 360,60 360,180 40,180\n",
            encoding="utf-8",
        )
        created.extend([img_path, ann_path])

    return {
        "data/rodosol_alpr_sample": {
            "category": "dataset",
            "source_type": "synthetic_fallback",
            "url": "https://github.com/raysonlaroca/rodosol-alpr-dataset",
            "reason": "RodoSol-ALPR requires academic license agreement email; generated synthetic Brazilian legacy & Mercosul plate fixture",
            "sha256": sha256_tree(created, root),
            "size_bytes": sum(p.stat().st_size for p in created),
            "files": [_rel_path(p, root) for p in created],
        }
    }


def write_manifest(data_dir: Path, entries: dict[str, dict[str, Any]]) -> Path:
    """Write or update data/MANIFEST.json with provenance metadata."""
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "MANIFEST.json"
    existing_entries: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and isinstance(old.get("entries"), dict):
                existing_entries = old["entries"]
        except Exception:
            existing_entries = {}
    existing_entries.update(entries)
    payload = {
        "generated_at_utc": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "schema_version": "1.0",
        "entries": existing_entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare weights, OSM caches, and datasets for Pipeline_IA.")
    parser.add_argument("--all", action="store_true", help="Prepare all weights, OSM caches, and datasets (default).")
    parser.add_argument("--weights-only", action="store_true", help="Only prepare pretrained weights.")
    parser.add_argument("--osm-only", action="store_true", help="Only prepare OSM GraphML caches.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["synthehicle", "ua_detrac", "cityflow", "veri776", "rodosol"],
        help="Specific datasets to prepare.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--synthehicle-scene", default="Town01-N-day")
    parser.add_argument("--synthehicle-cams", default="C01,C02")
    parser.add_argument("--detrac-seq", default="detrac_100_MVI_40905")
    parser.add_argument("--detrac-max-frames", type=int, default=300)
    parser.add_argument("--detrac-fps", type=float, default=25.0)
    parser.add_argument("--blind-start-s", type=float, default=4.0)
    parser.add_argument("--blind-duration-s", type=float, default=5.0)
    parser.add_argument("--cityflow-dir", type=Path, default=None)
    parser.add_argument("--veri-dir", type=Path, default=None)
    parser.add_argument("--rodosol-dir", type=Path, default=None)
    parser.add_argument("--offline", action="store_true", help="Skip all network calls and generate synthetic fixtures.")
    parser.add_argument("--force", action="store_true", help="Re-download or regenerate existing artifacts.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_all = args.all or (not args.weights_only and not args.osm_only and not args.datasets)
    base_dir = args.data_dir.parent.resolve()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.weights_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: dict[str, dict[str, Any]] = {}

    if run_all or args.weights_only:
        manifest_entries.update(
            prepare_weights(args.weights_dir, offline=args.offline, force=args.force, base_dir=base_dir)
        )

    if run_all or args.osm_only:
        manifest_entries.update(
            prepare_osm_cache(args.data_dir / "osm", offline=args.offline, force=args.force, base_dir=base_dir)
        )

    selected_datasets = (
        ["synthehicle", "ua_detrac", "cityflow", "veri776", "rodosol"]
        if run_all
        else (args.datasets or [])
    )
    if "synthehicle" in selected_datasets:
        cams = [c.strip() for c in args.synthehicle_cams.split(",") if c.strip()]
        manifest_entries.update(
            prepare_synthehicle(
                args.data_dir,
                scene=args.synthehicle_scene,
                cams=cams,
                offline=args.offline,
                force=args.force,
                base_dir=base_dir,
            )
        )
    if "ua_detrac" in selected_datasets:
        manifest_entries.update(
            prepare_ua_detrac(
                args.data_dir,
                seq_name=args.detrac_seq,
                max_frames=args.detrac_max_frames,
                fps=args.detrac_fps,
                blind_start_s=args.blind_start_s,
                blind_duration_s=args.blind_duration_s,
                offline=args.offline,
                force=args.force,
                base_dir=base_dir,
            )
        )
    if "cityflow" in selected_datasets:
        manifest_entries.update(
            prepare_cityflow(args.data_dir, local_dir=args.cityflow_dir, force=args.force, base_dir=base_dir)
        )
    if "veri776" in selected_datasets:
        manifest_entries.update(
            prepare_veri776(args.data_dir, local_dir=args.veri_dir, force=args.force, base_dir=base_dir)
        )
    if "rodosol" in selected_datasets:
        manifest_entries.update(
            prepare_rodosol_alpr(args.data_dir, local_dir=args.rodosol_dir, force=args.force, base_dir=base_dir)
        )

    manifest_path = write_manifest(args.data_dir, manifest_entries)
    print(f"Prepared {len(manifest_entries)} entries -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
