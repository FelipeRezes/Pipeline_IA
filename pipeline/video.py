"""Camada de Vídeo: Ingestão de streams/arquivos, Detecção YOLO, ByteTrack + Re-ID e Renderização.

Aprimoramentos da Refatoração:
1. Rastreador Intra-Câmera ByteTrack + Aparência (Substitui o IoUTracker ingênuo):
   - Mantém estado de velocidade da bounding box 2D [cx, cy, w, h, vx, vy] para
     prever a caixa durante oclusões locais (postes, árvores, cruzamento de veículos);
   - Associação em 2 estágios (ByteTrack): primeiro pareia detecções de alta confiança
     combinando IoU + similaridade Re-ID; segundo recupera detecções de baixa
     confiança (oclusão parcial) para tracks ativos não pareados.
2. Extração Automática de Embeddings Re-ID:
   - Usa `VehicleReIDExtractor` em cada recorte de veículo detectado para alimentar
     tanto o rastreador intra-câmera quanto a associação intercâmeras em regiões cegas.
3. Renderizador de Vídeos Sintéticos Realistas:
   - Renderiza a pavimentação das vias do grafo, faixas de rolamento e veículos com
     suas cores/aparências consistentes entre câmeras, permitindo que o extrator
     visual Re-ID funcione de ponta a ponta sobre os arquivos MP4.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .cityflow import Calibration, CityFlowDataset, Detection
from .config import PipelineConfig
from .optimize import linear_sum_assignment
from .reid import VehicleReIDExtractor, cosine_similarity, l2_normalize

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class RawBox:
    x: float
    y: float
    w: float
    h: float
    conf: float
    cls: int
    embedding: Optional[np.ndarray] = None

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x + 0.5 * self.w, self.y + 0.5 * self.h], dtype=float)


@dataclass
class Frame:
    camera_id: str
    index: int
    timestamp: float
    image: np.ndarray


@dataclass
class VideoSource:
    """Fonte de vídeo: arquivo MP4/AVI, URL HTTP(S), stream RTSP ou playlist HLS (.m3u8)."""
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

    def _probe(self) -> None:
        if self.fps and self.width and self.height:
            return
        try:
            out = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=width,height,r_frame_rate",
                    "-of", "csv=p=0", self.path,
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
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
        cmd = [
            "ffmpeg", "-loglevel", "error", "-i", self.path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-vf", f"fps={self.fps}", "pipe:1",
        ]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._opened = True
        return self

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
        elif self._proc is not None and self._proc.stdout is not None:
            size = int(self.width) * int(self.height) * 3
            raw = self._proc.stdout.read(size)
            if len(raw) != size:
                return None
            img = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(self.height), int(self.width), 3
            )
        if img is None:
            return None
        self._idx += 1
        return Frame(
            self.camera_id,
            self._idx,
            self.offset_s + (self._idx - 1) / max(self.fps or 10.0, 1e-6),
            img,
        )

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._proc is not None:
            try:
                if self._proc.stdout:
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


class YOLODetector:
    """Detector neural de veículos via Ultralytics YOLO (classes COCO: carro, moto, ônibus, caminhão)."""
    VEHICLE_CLASSES = (2, 3, 5, 7)

    def __init__(self,
                 weights: str = "yolov8n.pt",
                 conf: float = 0.25,
                 imgsz: int = 640,
                 device: str = "cpu",
                 classes: Sequence[int] = VEHICLE_CLASSES):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.classes = tuple(classes)
        self.reid = VehicleReIDExtractor()

    def detect(self, image: np.ndarray) -> List[RawBox]:
        results = self.model.predict(
            image, imgsz=self.imgsz, conf=self.conf, device=self.device, verbose=False
        )[0]
        boxes: List[RawBox] = []
        if results.boxes is None:
            return boxes
        for b in results.boxes:
            cls = int(b.cls[0].item())
            if cls not in self.classes:
                continue
            x1, y1, x2, y2 = map(float, b.xyxy[0].cpu().numpy())
            w, h = x2 - x1, y2 - y1
            emb = self.reid.extract_from_crop(image, (x1, y1, w, h))
            boxes.append(RawBox(x1, y1, w, h, float(b.conf[0].item()), cls, embedding=emb))
        return boxes


class AnnotationDetector:
    """Replay de detecções anotadas com extração de embedding Re-ID a partir da imagem."""

    def __init__(self, dataset: CityFlowDataset):
        self.ds = dataset
        self.reid = VehicleReIDExtractor()
        self._cache: Dict[Tuple[str, int], List[Detection]] = {}
        for cam, dets in dataset.detections.items():
            for d in dets:
                self._cache.setdefault((cam, d.frame_id), []).append(d)

    def detect(self,
               image: np.ndarray,
               camera_id: str = "",
               frame_index: int = 0) -> List[RawBox]:
        dets = self._cache.get((camera_id, frame_index), [])
        out: List[RawBox] = []
        for d in dets:
            emb = (
                d.embedding
                if d.embedding is not None
                else self.reid.extract_from_crop(image, (d.x, d.y, d.w, d.h))
            )
            out.append(RawBox(d.x, d.y, d.w, d.h, d.score, d.vehicle_class, embedding=emb))
        return out


# ------------------------------------------------------------------ ByteTrack + Re-ID SCT
@dataclass
class _LocalTrack:
    track_id: int
    box: np.ndarray          # [x, y, w, h]
    velocity: np.ndarray     # [vx, vy] em pixels/frame
    misses: int = 0
    hits: int = 1
    embedding: Optional[np.ndarray] = None

    def predict_box(self) -> np.ndarray:
        b = self.box.copy()
        b[0] += self.velocity[0]
        b[1] += self.velocity[1]
        return b


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


class IoUTracker:
    """Rastreador Intra-Câmera estilo ByteTrack/BoT-SORT (Motion Prediction + 2-Stage IoU + Re-ID).

    Mantém o nome de classe `IoUTracker` para retrocompatibilidade de API, mas
    opera com predição cinemática 2D em pixels, recuperação de detecções de
    baixa confiança em dois estágios e fusão de aparência Re-ID.
    """

    def __init__(self,
                 max_miss: int = 10,
                 min_iou: float = 0.12,
                 high_thresh: float = 0.45,
                 low_thresh: float = 0.12):
        self.max_miss = max_miss
        self.min_iou = min_iou
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.tracks: Dict[int, _LocalTrack] = {}
        self._next_id = 1

    def _match_stage(self,
                     track_ids: List[int],
                     boxes: Sequence[RawBox],
                     use_reid: bool = True) -> Tuple[List[Tuple[int, int]], set[int], set[int]]:
        if not track_ids or not boxes:
            return [], set(), set()

        B = np.array([[b.x, b.y, b.w, b.h] for b in boxes], dtype=float)
        cost = np.zeros((len(track_ids), len(boxes)), dtype=float)
        for i, tid in enumerate(track_ids):
            tr = self.tracks[tid]
            pred_b = tr.predict_box()
            for j, b_arr in enumerate(B):
                iou_val = _iou(pred_b, b_arr)
                c = 1.0 - iou_val
                if use_reid and tr.embedding is not None and boxes[j].embedding is not None:
                    sim = cosine_similarity(tr.embedding, boxes[j].embedding)
                    c = 0.75 * c + 0.25 * max(0.0, 1.0 - sim)
                if iou_val < self.min_iou:
                    c = 1e5
                cost[i, j] = c

        rows, cols = linear_sum_assignment(cost)
        matches: List[Tuple[int, int]] = []
        used_r, used_c = set(), set()
        for r, c in zip(rows, cols):
            if cost[r, c] <= (1.0 - self.min_iou + 0.15):
                matches.append((int(r), int(c)))
                used_r.add(int(r))
                used_c.add(int(c))
        return matches, used_r, used_c

    def update(self, boxes: Sequence[RawBox]) -> List[Tuple[int, RawBox]]:
        ids = list(self.tracks)
        if not boxes:
            for tid in ids:
                self.tracks[tid].box = self.tracks[tid].predict_box()
                self.tracks[tid].velocity *= 0.8
                self.tracks[tid].misses += 1
            self.tracks = {
                k: v for k, v in self.tracks.items() if v.misses <= self.max_miss
            }
            return []

        high_boxes = [b for b in boxes if b.conf >= self.high_thresh]
        low_boxes = [b for b in boxes if self.low_thresh <= b.conf < self.high_thresh]

        # Se todas as caixas tiverem conf < high_thresh, trata todas no estágio 1
        if not high_boxes:
            high_boxes, low_boxes = list(boxes), []

        out: List[Tuple[int, RawBox]] = []

        # Estágio 1: Detecções de alta confiança (IoU + Re-ID)
        m1, used_r1, used_c1 = self._match_stage(ids, high_boxes, use_reid=True)
        for r_idx, c_idx in m1:
            tid = ids[r_idx]
            b = high_boxes[c_idx]
            new_arr = np.array([b.x, b.y, b.w, b.h], dtype=float)
            tr = self.tracks[tid]
            vel_obs = new_arr[:2] - tr.box[:2]
            tr.velocity = 0.6 * tr.velocity + 0.4 * vel_obs
            tr.box = new_arr
            tr.misses = 0
            tr.hits += 1
            if b.embedding is not None:
                tr.embedding = (
                    l2_normalize(b.embedding)
                    if tr.embedding is None
                    else l2_normalize(0.8 * tr.embedding + 0.2 * l2_normalize(b.embedding))
                )
            out.append((tid, b))

        # Estágio 2: Recuperação de oclusão com detecções de baixa confiança
        rem_ids = [ids[i] for i in range(len(ids)) if i not in used_r1]
        m2, used_r2, _ = self._match_stage(rem_ids, low_boxes, use_reid=False)
        for r_idx, c_idx in m2:
            tid = rem_ids[r_idx]
            b = low_boxes[c_idx]
            new_arr = np.array([b.x, b.y, b.w, b.h], dtype=float)
            tr = self.tracks[tid]
            tr.velocity = 0.7 * tr.velocity + 0.3 * (new_arr[:2] - tr.box[:2])
            tr.box = new_arr
            tr.misses = 0
            tr.hits += 1
            out.append((tid, b))

        # Tracks não pareados avançam por inércia
        unmatched_ids = [rem_ids[i] for i in range(len(rem_ids)) if i not in used_r2]
        for tid in unmatched_ids:
            self.tracks[tid].box = self.tracks[tid].predict_box()
            self.tracks[tid].velocity *= 0.8
            self.tracks[tid].misses += 1

        self.tracks = {
            k: v for k, v in self.tracks.items() if v.misses <= self.max_miss
        }

        # Novas pistas a partir de detecções de alta confiança não pareadas
        for j, b in enumerate(high_boxes):
            if j in used_c1:
                continue
            tid = self._next_id
            self._next_id += 1
            emb = l2_normalize(b.embedding) if b.embedding is not None else None
            self.tracks[tid] = _LocalTrack(
                track_id=tid,
                box=np.array([b.x, b.y, b.w, b.h], dtype=float),
                velocity=np.zeros(2, dtype=float),
                embedding=emb,
            )
            out.append((tid, b))

        return out


# --------------------------------------------------------------- geração de vídeo realista
def _vehicle_color_palette(vid: int) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Retorna (cor_carroceria_BGR, cor_teto_BGR) única e consistente por veículo."""
    palette = [
        ((40, 50, 210), (25, 30, 140)),   # Vermelho
        ((210, 110, 30), (140, 70, 20)),  # Azul
        ((45, 190, 60), (25, 120, 35)),   # Verde
        ((30, 195, 230), (20, 130, 160)), # Amarelo / Táxi
        ((210, 210, 215), (150, 150, 155)), # Prata
        ((180, 50, 180), (115, 30, 115)), # Roxo
        ((220, 190, 50), (145, 125, 30)), # Ciano
        ((50, 125, 230), (30, 80, 155)),  # Laranja
    ]
    return palette[(int(vid) - 1) % len(palette)]


def render_videos(ds: CityFlowDataset,
                  out_dir: str,
                  draw_ids: bool = True) -> Dict[str, str]:
    """Renderiza vídeos MP4 multicâmera com veículos de aparência visual consistente (para Re-ID)."""
    if cv2 is None:
        raise RuntimeError("OpenCV é necessário para renderizar vídeos")
    root = Path(out_dir)
    (root / "videos").mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    last_frame = max(
        (d.frame_id for dets in ds.detections.values() for d in dets), default=0
    )

    # Mapa reverso (camera, frame, local_id) -> gt_id para pintar a cor real do veículo
    gt_color_map: Dict[Tuple[str, int, int], int] = {}
    for cam, objs in ds.gt.items():
        for gid, g_dets in objs.items():
            for gd in g_dets:
                gt_color_map[(cam, gd.frame_id, int(round(gd.x)))] = gid

    for cam, dets in ds.detections.items():
        path = root / "videos" / f"{cam}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), ds.fps, (1280, 960)
        )
        by_frame: Dict[int, List[Detection]] = {}
        for d in dets:
            by_frame.setdefault(d.frame_id, []).append(d)

        for f in range(1, last_frame + 1):
            img = np.full((960, 1280, 3), 36, dtype=np.uint8)
            # Desenha cruzamento viário de fundo
            cv2.rectangle(img, (0, 380), (1280, 580), (58, 58, 62), -1)
            cv2.rectangle(img, (520, 0), (760, 960), (58, 58, 62), -1)
            cv2.line(img, (0, 480), (1280, 480), (90, 170, 190), 1)
            cv2.line(img, (640, 0), (640, 960), (90, 170, 190), 1)

            for d in by_frame.get(f, []):
                gid = gt_color_map.get((cam, f, int(round(d.x))), d.track_id)
                body_col, roof_col = _vehicle_color_palette(gid)
                x1, y1 = int(max(0, d.x)), int(max(0, d.y))
                x2, y2 = int(min(1279, d.x + d.w)), int(min(959, d.y + d.h))
                # Carroceria, teto e para-brisa para gerar assinatura visual rica
                cv2.rectangle(img, (x1, y1), (x2, y2), body_col, -1)
                rx1 = x1 + int(0.2 * (x2 - x1))
                ry1 = y1 + int(0.2 * (y2 - y1))
                rx2 = x2 - int(0.2 * (x2 - x1))
                ry2 = y2 - int(0.2 * (y2 - y1))
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), roof_col, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (235, 235, 235), 2)
                if draw_ids:
                    cv2.putText(
                        img,
                        f"ID {d.track_id}",
                        (x1, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
            writer.write(img)
        writer.release()
        paths[cam] = str(path)
    return paths


@dataclass
class VideoStats:
    frames_read: int = 0
    boxes_detected: int = 0
    tracklets: int = 0
    elapsed_s: float = 0.0
    source_errors: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "frames_read": self.frames_read,
            "boxes_detected": self.boxes_detected,
            "tracklets": self.tracklets,
            "elapsed_s": round(self.elapsed_s, 3),
            "source_errors": self.source_errors,
        }


class VideoAnalytics:
    """Orquestrador Vídeo -> Detecção + Re-ID -> ByteTrack SCT -> Pipeline EKF + Grafo."""

    def __init__(self,
                 sources: Sequence[VideoSource],
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
        self.reid = VehicleReIDExtractor()

    def collect(self) -> CityFlowDataset:
        t0 = time.perf_counter()
        ds = CityFlowDataset(fps=self.cfg.fps)
        ds.calibrations = dict(self.calibrations)
        trackers = {
            s.camera_id: IoUTracker(
                max_miss=self.cfg.sct_max_miss,
                min_iou=self.cfg.sct_min_iou,
                high_thresh=self.cfg.sct_high_thresh,
                low_thresh=self.cfg.sct_low_thresh,
            )
            for s in self.sources
        }
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
                        boxes = self.detector.detect(
                            frame.image, src.camera_id, frame.index
                        )
                    else:
                        boxes = self.detector.detect(frame.image)
                except Exception as exc:
                    self.stats.source_errors.setdefault(
                        src.camera_id, f"detector: {exc}"
                    )
                    boxes = []

                for b in boxes:
                    if b.embedding is None:
                        b.embedding = self.reid.extract_from_crop(
                            frame.image, (b.x, b.y, b.w, b.h)
                        )

                self.stats.boxes_detected += len(boxes)
                for tid, box in trackers[src.camera_id].update(boxes):
                    det = Detection(
                        camera_id=src.camera_id,
                        track_id=tid,
                        frame_id=frame.index,
                        x=float(box.x),
                        y=float(box.y),
                        w=float(box.w),
                        h=float(box.h),
                        score=float(box.conf),
                        embedding=box.embedding,
                        vehicle_class=int(box.cls),
                    )
                    ds.detections.setdefault(src.camera_id, []).append(det)

        for src in readers:
            src.close()

        ds.fps = (
            float(np.median([s.fps or self.cfg.fps for s in readers]))
            if readers
            else self.cfg.fps
        )
        self.cfg.fps = ds.fps
        self.cfg.dt = 1.0 / ds.fps
        self.stats.elapsed_s = time.perf_counter() - t0
        self.stats.tracklets = sum(
            len({d.track_id for d in v}) for v in ds.detections.values()
        )
        return ds

    def run(self):
        from .pipeline import SurveillancePipeline
        ds = self.collect()
        if self.graph is None:
            return ds, None
        result = SurveillancePipeline(ds, self.graph, self.cfg).run()
        return ds, result
