"""Camada de vídeo: ingestão de arquivos/streams públicos + detecção + SCT.

Fecha o pedido do professor: a arquitetura não consome apenas anotações — ela
recebe vídeos de múltiplas câmeras (CityFlowV2, RoundaboutHD, webcams/RTSP/HLS),
detecta veículos quadro a quadro e alimenta o EKF adaptativo com as caixas 2D
projetadas no plano de solo.

Componentes:
- VideoSource       : cv2.VideoCapture com fallback FFmpeg (HLS/RTSP/URLs públicas);
- YOLODetector      : detector de veículos (classes COCO 2,3,5,7) via Ultralytics;
- AnnotationDetector: replay de anotações (validação sem detector neural);
- IoUTracker        : tracklets locais por câmera (IDs necessários ao EKF);
- VideoAnalytics    : orquestra leitura -> detecção -> tracklets -> pipeline EKF.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .cityflow import Calibration, CityFlowDataset, Detection
from .config import PipelineConfig
from .optimize import linear_sum_assignment

try:
    import cv2  # opcional em modo somente-anotação
except Exception:  # pragma: no cover
    cv2 = None


# --------------------------------------------------------------------------- dados
@dataclass
class RawBox:
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x + 0.5 * self.w, self.y + 0.5 * self.h])


@dataclass
class Frame:
    camera_id: str
    index: int
    timestamp: float
    image: np.ndarray


# ---------------------------------------------------------------------- video source
@dataclass
class VideoSource:
    """Uma câmera: arquivo local, URL HTTP(S), RTSP ou playlist HLS (.m3u8)."""
    camera_id: str
    path: str
    offset_s: float = 0.0
    fps: Optional[float] = None
    max_frames: int = 0
    width: Optional[int] = None
    height: Optional[int] = None

    def __post_init__(self) -> None:
        self._cap = None
        self._proc = None
        self._idx = 0
        self._opened = False

    # ------------------------------------------------------------------ abertura
    def _probe(self) -> None:
        if self.fps and self.width and self.height:
            return
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height,r_frame_rate",
                 "-of", "csv=p=0", self.path],
                capture_output=True, text=True, timeout=20)
            parts = out.stdout.strip().split(",")
            if len(parts) >= 3:
                self.width = self.width or int(parts[0])
                self.height = self.height or int(parts[1])
                num, den = parts[2].split("/")
                self.fps = self.fps or float(num) / max(float(den), 1e-6)
        except Exception:
            pass
        self.fps = self.fps or 10.0
        self.width = self.width or 1280
        self.height = self.height or 720

    def open(self) -> "VideoSource":
        self._probe()
        if cv2 is not None:
            cap = cv2.VideoCapture(self.path)
            if cap.isOpened():
                self._cap = cap
                self._opened = True
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
                self.fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
                return self
            cap.release()
        # fallback FFmpeg: decodifica HLS/RTSP para BGR24 em pipe
        cmd = ["ffmpeg", "-loglevel", "error", "-i", self.path,
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-vf", f"fps={self.fps}", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE)
        self._opened = True
        return self

    # -------------------------------------------------------------------- leitura
    def read(self) -> Optional[Frame]:
        if not self._opened:
            self.open()
        if self.max_frames and self._idx >= self.max_frames:
            return None
        img = None
        if self._cap is not None:
            ok, img = self._cap.read()
            if not ok:
                return None
        elif self._proc is not None:
            size = int(self.width) * int(self.height) * 3
            raw = self._proc.stdout.read(size)
            if len(raw) != size:
                return None
            img = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(self.height), int(self.width), 3)
        if img is None:
            return None
        self._idx += 1
        return Frame(self.camera_id, self._idx,
                     self.offset_s + (self._idx - 1) / max(self.fps or 10.0, 1e-6),
                     img)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._proc is not None:
            try:
                self._proc.stdout.close()
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            self._proc = None
        self._opened = False

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()


# -------------------------------------------------------------------------- detectores
class YOLODetector:
    """Detector de veículos com Ultralytics YOLO (COCO: carro, moto, ônibus, caminhão)."""
    VEHICLE_CLASSES = (2, 3, 5, 7)

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.35,
                 imgsz: int = 640, device: str = "cpu",
                 classes: Sequence[int] = VEHICLE_CLASSES):
        from ultralytics import YOLO  # import tardio: só necessário neste modo
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.classes = tuple(classes)

    def detect(self, image: np.ndarray) -> List[RawBox]:
        results = self.model.predict(image, imgsz=self.imgsz, conf=self.conf,
                                     device=self.device, verbose=False)[0]
        boxes: List[RawBox] = []
        if results.boxes is None:
            return boxes
        for b in results.boxes:
            cls = int(b.cls[0].item())
            if cls not in self.classes:
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
            boxes.append(RawBox(x1, y1, x2 - x1, y2 - y1,
                                float(b.conf[0].item()), cls))
        return boxes


class AnnotationDetector:
    """Replay de caixas anotadas — valida o caminho de vídeo sem modelo neural."""

    def __init__(self, dataset: CityFlowDataset):
        self.ds = dataset
        self._cache: Dict[Tuple[str, int], List[RawBox]] = {}
        for cam, dets in dataset.detections.items():
            for d in dets:
                self._cache.setdefault((cam, d.frame_id), []).append(
                    RawBox(d.x, d.y, d.w, d.h, 1.0, 2))

    def detect(self, image: np.ndarray, camera_id: str = "",
               frame_index: int = 0) -> List[RawBox]:
        return self._cache.get((camera_id, frame_index), [])


# ------------------------------------------------------------------ tracklet local
@dataclass
class _LocalTrack:
    track_id: int
    box: np.ndarray          # [x, y, w, h]
    misses: int = 0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class IoUTracker:
    """Associa caixas a tracklets locais dentro de uma câmera (simples e rápido)."""
    def __init__(self, max_miss: int = 8, min_iou: float = 0.12):
        self.max_miss = max_miss
        self.min_iou = min_iou
        self.tracks: Dict[int, _LocalTrack] = {}
        self._next_id = 1

    def update(self, boxes: Sequence[RawBox]) -> List[Tuple[int, RawBox]]:
        ids = list(self.tracks)
        if not boxes:
            for tid in ids:
                self.tracks[tid].misses += 1
            self.tracks = {k: v for k, v in self.tracks.items()
                           if v.misses <= self.max_miss}
            return []
        if not ids:
            out = []
            for b in boxes:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = _LocalTrack(tid, np.array([b.x, b.y, b.w, b.h]))
                out.append((tid, b))
            return out

        B = np.array([[b.x, b.y, b.w, b.h] for b in boxes])
        cost = np.array([[1.0 - _iou(self.tracks[t].box, b) for b in B]
                         for t in ids])
        rows, cols = linear_sum_assignment(cost)
        assigned_rows, assigned_cols = set(), set()
        out: List[Tuple[int, RawBox]] = []
        for i, j in zip(rows, cols):
            if cost[i, j] <= 1.0 - self.min_iou:
                tid = ids[int(i)]
                self.tracks[tid].box = B[int(j)]
                self.tracks[tid].misses = 0
                out.append((tid, boxes[int(j)]))
                assigned_rows.add(int(i))
                assigned_cols.add(int(j))
        for i, tid in enumerate(ids):
            if i not in assigned_rows:
                self.tracks[tid].misses += 1
        self.tracks = {k: v for k, v in self.tracks.items()
                       if v.misses <= self.max_miss}
        for j, b in enumerate(boxes):
            if j in assigned_cols:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = _LocalTrack(tid, np.array([b.x, b.y, b.w, b.h]))
            out.append((tid, b))
        return out


# --------------------------------------------------------------- geração de vídeo
def render_videos(ds: CityFlowDataset, out_dir: str,
                  draw_ids: bool = True) -> Dict[str, str]:
    """Renderiza as detecções como vídeos MP4 por câmera (útil para testes/demo)."""
    if cv2 is None:
        raise RuntimeError("OpenCV é necessário para renderizar vídeos")
    root = Path(out_dir)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    last_frame = max((d.frame_id for dets in ds.detections.values() for d in dets),
                     default=0)
    for cam, dets in ds.detections.items():
        path = root / "videos" / f"{cam}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 ds.fps, (1280, 960))
        by_frame: Dict[int, List[Detection]] = {}
        for d in dets:
            by_frame.setdefault(d.frame_id, []).append(d)
        for f in range(1, last_frame + 1):
            img = np.full((960, 1280, 3), 32, dtype=np.uint8)
            for d in by_frame.get(f, []):
                p1 = (int(d.x), int(d.y))
                p2 = (int(d.x + d.w), int(d.y + d.h))
                cv2.rectangle(img, p1, p2, (40, 200, 90), 2)
                if draw_ids:
                    cv2.putText(img, f"ID {d.track_id}", (p1[0], max(12, p1[1] - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            writer.write(img)
        writer.release()
        paths[cam] = str(path)
    return paths


# --------------------------------------------------------------------- orquestração
@dataclass
class VideoStats:
    frames_read: int = 0
    boxes_detected: int = 0
    tracklets: int = 0
    elapsed_s: float = 0.0
    source_errors: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"frames_read": self.frames_read,
                "boxes_detected": self.boxes_detected,
                "tracklets": self.tracklets,
                "elapsed_s": round(self.elapsed_s, 3),
                "source_errors": self.source_errors}


class VideoAnalytics:
    """Vídeo -> detecções -> tracklets -> EKF/grafo -> rotas prováveis."""

    def __init__(self, sources: Sequence[VideoSource],
                 calibrations: Dict[str, Calibration],
                 detector,
                 cfg: Optional[PipelineConfig] = None,
                 graph=None):
        self.sources = list(sources)
        self.calibrations = calibrations
        self.detector = detector
        self.cfg = cfg or PipelineConfig()
        self.graph = graph
        self.stats = VideoStats()

    def collect(self) -> CityFlowDataset:
        """Lê todos os vídeos e produz um CityFlowDataset em memória."""
        t0 = time.perf_counter()
        ds = CityFlowDataset(fps=self.cfg.fps)
        ds.calibrations = dict(self.calibrations)
        trackers = {s.camera_id: IoUTracker() for s in self.sources}
        readers = []
        for src in self.sources:
            try:
                readers.append(src.open())
            except Exception as exc:
                self.stats.source_errors[src.camera_id] = str(exc)

        active = True
        while active and readers:
            active = False
            for src in readers:
                frame = src.read()
                if frame is None:
                    continue
                active = True
                self.stats.frames_read += 1
                try:
                    if isinstance(self.detector, AnnotationDetector):
                        boxes = self.detector.detect(frame.image, src.camera_id,
                                                     frame.index)
                    else:
                        boxes = self.detector.detect(frame.image)
                except Exception as exc:
                    self.stats.source_errors.setdefault(
                        src.camera_id, f"detector: {exc}")
                    boxes = []
                self.stats.boxes_detected += len(boxes)
                cal = self.calibrations[src.camera_id]
                for tid, box in trackers[src.camera_id].update(boxes):
                    det = Detection(src.camera_id, tid, frame.index,
                                    float(box.x), float(box.y),
                                    float(box.w), float(box.h),
                                    score=float(box.conf))
                    ds.detections.setdefault(src.camera_id, []).append(det)
        for src in readers:
            src.close()
        for cam, tracker in trackers.items():
            self.stats.tracklets += sum(1 for t in tracker.tracks.values()
                                        if t.misses == 0)
        ds.fps = float(np.median([s.fps or self.cfg.fps for s in readers])) \
            if readers else self.cfg.fps
        self.cfg.fps = ds.fps
        self.cfg.dt = 1.0 / ds.fps
        self.stats.elapsed_s = time.perf_counter() - t0
        self.stats.tracklets = sum(len({d.track_id for d in v})
                                   for v in ds.detections.values())
        return ds

    def run(self):
        """Executa coleta + pipeline EKF; devolve (dataset, resultado)."""
        from .pipeline import SurveillancePipeline
        ds = self.collect()
        if self.graph is None:
            return ds, None
        result = SurveillancePipeline(ds, self.graph, self.cfg).run()
        return ds, result
