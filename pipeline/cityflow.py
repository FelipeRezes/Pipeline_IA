"""Adaptador para CityFlow / CityFlowV2 (AI City Challenge) e formato sintético.

Formato de linha aceito (compatível com gt/gt.txt e track1.txt do challenge):

    camera_id  obj_id  frame_id  x  y  width  height  [x_world y_world score]

As detecções baseline de câmera única fornecidas pelo challenge são tratadas como
tracklets locais (obj_id por câmera); a pipeline é responsável por uni-las em IDs
globais, exatamente como descrito nas Secs. 3 e 5.3 do artigo.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .geometry import bbox_bottom_center

Array = np.ndarray


@dataclass(frozen=True)
class Detection:
    camera_id: str
    track_id: int
    frame_id: int
    x: float
    y: float
    w: float
    h: float
    score: float = 1.0
    x_world: Optional[float] = None
    y_world: Optional[float] = None
    embedding: Optional[Array] = None
    vehicle_class: int = 2

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    @property
    def bottom_center(self) -> Array:
        return bbox_bottom_center(self.box)


@dataclass
class Calibration:
    camera_id: str
    H: Array                      # 3x3: pixels -> metros no plano de solo
    offset_s: float = 0.0
    width: int = 1280
    height: int = 960

    def timestamp(self, frame_id: int, fps: float) -> float:
        return self.offset_s + (float(frame_id) - 1.0) / max(fps, 1e-6)


@dataclass
class CityFlowDataset:
    detections: Dict[str, List[Detection]] = field(default_factory=dict)
    gt: Dict[str, Dict[int, List[Detection]]] = field(default_factory=dict)
    calibrations: Dict[str, Calibration] = field(default_factory=dict)
    fps: float = 10.0
    root: Optional[str] = None
    graph_path: Optional[str] = None

    # ------------------------------------------------------------------ utilidades
    def camera_ids(self) -> List[str]:
        return sorted(set(self.detections) | set(self.gt))

    def frames(self) -> List[int]:
        return sorted({d.frame_id for dets in self.detections.values() for d in dets})

    def gt_pairs(self) -> List[tuple[str, int, float, float, float, float]]:
        """(camera, obj_id, frame, x, y, w, h) achatado a partir do GT."""
        return [(cam, oid, d.frame_id, d.x, d.y, d.w, d.h)
                for cam, objs in self.gt.items()
                for oid, dets in objs.items() for d in dets]

    def summary(self) -> str:
        ndet = sum(len(v) for v in self.detections.values())
        ngt = sum(len(d) for objs in self.gt.values() for d in objs.values())
        nobj = sum(len(v) for v in self.gt.values())
        return (f"CityFlowDataset(cameras={len(self.camera_ids())}, "
                f"detections={ndet}, gt_boxes={ngt}, gt_objects={nobj}, "
                f"fps={self.fps})")


_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_rows(path: Path) -> List[List[str]]:
    rows: List[List[str]] = []
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nums = _NUM_RE.findall(line.replace(",", " "))
            if len(nums) >= 7:
                rows.append(nums)
    return rows


def _find_file(root: Path, names: Sequence[str], explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    for name in names:
        found = sorted(root.rglob(name))
        if found:
            return found[0]
    return None


def _read_calibration_file(path: Path, cam_hint: Optional[str] = None) -> tuple[str, Array]:
    nums = [float(n) for n in _NUM_RE.findall(path.read_text(encoding="utf-8", errors="ignore"))]
    if len(nums) < 9:
        raise ValueError(f"calibração inválida: {path}")
    H = np.asarray(nums[:9], dtype=float).reshape(3, 3)
    cam = cam_hint or path.stem
    return cam, H / H[2, 2]


def _default_calibration() -> Array:
    """Fallback: 1 px ~= 0,05 m, origem no centro da imagem (demo/sem calibração)."""
    return np.array([[0.05, 0.0, -32.0],
                     [0.0, -0.05, 24.0],
                     [0.0, 0.0, 1.0]])


def load_cityflow(root: str,
                  detections_file: Optional[str] = None,
                  gt_file: Optional[str] = None,
                  fps: float = 10.0) -> CityFlowDataset:
    """Carrega um diretório no layout CityFlow/CityFlowV2.

    A detecção de arquivos é flexível para aceitar tanto o layout oficial quanto
    o diretório gerado por ``pipeline.synthetic``.
    """
    rootp = Path(root).expanduser()
    if not rootp.is_dir():
        raise FileNotFoundError(f"diretório não encontrado: {rootp}")

    ds = CityFlowDataset(root=str(rootp), fps=fps)
    det_path = _find_file(rootp, ["detections.txt", "det.txt", "track1.txt",
                                  "baseline.txt", "single_camera.txt"], detections_file)
    gt_path = _find_file(rootp, ["gt.txt", "test.txt", "gt/gt.txt", "gt/test.txt"], gt_file)

    if det_path is None and gt_path is None:
        raise ValueError(f"nenhum arquivo de detecções/GT encontrado em {rootp}")

    def ingest(path: Optional[Path], target: str) -> None:
        if path is None:
            return
        for row in _parse_rows(path):
            cam = str(int(float(row[0])))
            oid = int(float(row[1]))
            frame = int(float(row[2]))
            x, y, w, h = map(float, row[3:7])
            score = float(row[9]) if len(row) >= 10 else 1.0
            wx = float(row[7]) if len(row) >= 8 else None
            wy = float(row[8]) if len(row) >= 8 else None
            det = Detection(cam, oid, frame, x, y, w, h, score, wx, wy)
            if target == "det":
                ds.detections.setdefault(cam, []).append(det)
            else:
                ds.gt.setdefault(cam, {}).setdefault(oid, []).append(det)

    ingest(det_path, "det")
    ingest(gt_path, "gt")

    if not ds.detections and ds.gt:  # usa GT como detecções (modo diagnóstico)
        for cam, objs in ds.gt.items():
            for dets in objs.values():
                ds.detections.setdefault(cam, []).extend(dets)
        warnings.warn("Nenhum arquivo de detecções encontrado; usando GT como entrada.",
                      RuntimeWarning)

    for cam in ds.camera_ids():
        ds.detections.setdefault(cam, [])
        ds.detections[cam].sort(key=lambda d: (d.frame_id, d.track_id))
        for dets in ds.gt.get(cam, {}).values():
            dets.sort(key=lambda d: d.frame_id)

    # Se houver GT com identidade global, hidrata os embeddings Re-ID por identidade real
    if ds.gt:
        from .reid import synthetic_vehicle_embedding
        gt_lookup: dict[tuple[str, int, int], int] = {}
        for cam, objs in ds.gt.items():
            for oid, g_dets in objs.items():
                for gd in g_dets:
                    gt_lookup[(cam, gd.frame_id, int(round(gd.x)))] = oid
        for cam in list(ds.detections):
            new_list: List[Detection] = []
            for d in ds.detections[cam]:
                if d.embedding is None:
                    gid = gt_lookup.get((cam, d.frame_id, int(round(d.x))), d.track_id)
                    emb = synthetic_vehicle_embedding(gid, cam, cross_camera_noise=0.12)
                    d = Detection(
                        d.camera_id, d.track_id, d.frame_id,
                        d.x, d.y, d.w, d.h, d.score, d.x_world, d.y_world,
                        embedding=emb, vehicle_class=d.vehicle_class
                    )
                new_list.append(d)
            ds.detections[cam] = new_list

    # --- calibrações ------------------------------------------------------------
    calib_dir = rootp / "calib"
    calib_files = (sorted(calib_dir.glob("*.txt")) if calib_dir.is_dir()
                   else sorted(rootp.glob("calib_*.txt")))
    for cf in calib_files:
        cam, H = _read_calibration_file(cf)
        ds.calibrations[cam] = Calibration(cam, H)

    offsets = _find_file(rootp, ["offsets.txt", "offset.txt", "time_offsets.txt"])
    offsets_map: dict[str, float] = {}
    if offsets is not None:
        for line in offsets.read_text(encoding="utf-8").splitlines():
            nums = _NUM_RE.findall(line)
            if len(nums) >= 2:
                offsets_map[str(int(float(nums[0])))] = float(nums[1])

    for cam in ds.camera_ids():
        if cam not in ds.calibrations:
            ds.calibrations[cam] = Calibration(cam, _default_calibration())
        ds.calibrations[cam].offset_s = offsets_map.get(cam, 0.0)

    meta_path = rootp / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ds.fps = float(meta.get("fps", fps))
        ds.graph_path = meta.get("graph_path")
    return ds


def write_cityflow(ds: CityFlowDataset, root: str) -> None:
    """Escreve o dataset no formato aceito por ``load_cityflow``."""
    rootp = Path(root).expanduser()
    (rootp / "gt").mkdir(parents=True, exist_ok=True)
    (rootp / "calib").mkdir(parents=True, exist_ok=True)

    def line(d: Detection) -> str:
        wx = d.x_world if d.x_world is not None else 0.0
        wy = d.y_world if d.y_world is not None else 0.0
        return (f"{int(d.camera_id)} {d.track_id} {d.frame_id} "
                f"{d.x:.2f} {d.y:.2f} {d.w:.2f} {d.h:.2f} {wx:.3f} {wy:.3f} {d.score:.2f}")

    with (rootp / "detections.txt").open("w", encoding="utf-8") as fh:
        for cam in sorted(ds.detections):
            for d in sorted(ds.detections[cam], key=lambda z: (z.frame_id, z.track_id)):
                fh.write(line(d) + "\n")

    with (rootp / "gt" / "gt.txt").open("w", encoding="utf-8") as fh:
        for cam in sorted(ds.gt):
            for oid in sorted(ds.gt[cam]):
                for d in sorted(ds.gt[cam][oid], key=lambda z: z.frame_id):
                    fh.write(line(d) + "\n")

    for cam, cal in ds.calibrations.items():
        np.savetxt(rootp / "calib" / f"{cam}.txt", cal.H, fmt="%.10f")
    with (rootp / "offsets.txt").open("w", encoding="utf-8") as fh:
        for cam in sorted(ds.calibrations):
            fh.write(f"{cam} {ds.calibrations[cam].offset_s:.3f}\n")

    meta = {"fps": ds.fps, "graph_path": ds.graph_path,
            "cameras": sorted(ds.camera_ids())}
    (rootp / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
