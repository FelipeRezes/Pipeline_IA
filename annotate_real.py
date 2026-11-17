"""Gera um MP4 anotado com caixas/IDs do YOLO + IoUTracker sobre vídeo real."""
import sys
from pathlib import Path

import cv2

from pipeline.video import IoUTracker, YOLODetector

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/real_video/kitti_inference.mp4"
DST = sys.argv[2] if len(sys.argv) > 2 else str(
    Path(SRC).with_name(Path(SRC).stem + "_yolo_tracks.mp4"))
LABEL = Path(SRC).stem

det = YOLODetector(weights="weights/yolov8n.pt", conf=0.35, imgsz=640)
cap = cv2.VideoCapture(SRC)
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(DST, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
tracker = IoUTracker()
frames = boxes = 0
while True:
    ok, img = cap.read()
    if not ok:
        break
    frames += 1
    dets = det.detect(img)
    boxes += len(dets)
    for tid, b in tracker.update(dets):
        x1, y1 = int(b.x), int(b.y)
        x2, y2 = int(b.x + b.w), int(b.y + b.h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 220, 90), 2)
        cv2.putText(img, f"ID {tid}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, f"{LABEL} | frame {frames} | {len(dets)} boxes",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)
    writer.write(img)
cap.release()
writer.release()
print(f"{DST}: {frames} quadros, {boxes} deteccoes")
