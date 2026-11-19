"""End-to-End CICC Split-Screen Multi-Camera Pursuit & Blind-Zone Demonstration Runner."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_benchmarks import _build_benchmark_road_network
from tracker.config import (
    ALPRConfig,
    AssociationConfig,
    CameraConfig,
    DetectorConfig,
    EKFConfig,
    PipelineConfig,
    ReIDConfig,
    RoadGraphConfig,
    VisualizerConfig,
)
from tracker.detector import LocalPerception
from tracker.pipeline import CausalMultiCamPipeline
from tracker.visualizer import CICCVisualizer


def _make_homography(x0_utm: float, y0_utm: float, scale_m_per_px: float = 0.125) -> list[list[float]]:
    return [
        [scale_m_per_px, 0.0, x0_utm - 320.0 * scale_m_per_px],
        [0.0, -scale_m_per_px, y0_utm + 240.0 * scale_m_per_px],
        [0.0, 0.0, 1.0],
    ]


def _utm_to_bbox(
    x_utm: float,
    y_utm: float,
    x0_utm: float,
    y0_utm: float,
    scale_m_per_px: float = 0.125,
    box_w_px: float = 52.0,
    box_h_px: float = 38.0,
) -> tuple[float, float, float, float]:
    u_c = 320.0 + (x_utm - x0_utm) / scale_m_per_px
    v_c = 240.0 - (y_utm - y0_utm) / scale_m_per_px
    x1 = u_c - box_w_px * 0.5
    x2 = u_c + box_w_px * 0.5
    y1 = v_c - 0.90 * box_h_px
    y2 = y1 + box_h_px
    return (x1, y1, x2, y2)


def _render_road_frame(
    bbox: tuple[float, float, float, float] | None = None,
    color_bgr: tuple[int, int, int] = (30, 40, 230),
) -> np.ndarray:
    frame = np.full((480, 640, 3), (38, 42, 48), dtype=np.uint8)
    frame[140:340, :] = (68, 72, 76)
    cv2.line(frame, (0, 240), (640, 240), (190, 195, 200), 2)
    if bbox is not None:
        x1 = max(0, min(640, int(round(bbox[0]))))
        y1 = max(0, min(480, int(round(bbox[1]))))
        x2 = max(0, min(640, int(round(bbox[2]))))
        y2 = max(0, min(480, int(round(bbox[3]))))
        if x2 > x1 and y2 > y1:
            frame[y1:y2, x1:x2] = color_bgr
            cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), 1)
    return frame


def run_cicc_demo(output_video_path: Path, num_frames: int = 24) -> Path:
    """Execute a complete 2-camera pursuit across a blind zone and render the CICC Split-Screen MP4."""
    road_net, road_eng = _build_benchmark_road_network()
    ox, oy = 330000.0, 7390000.0

    cam1 = CameraConfig(
        cam_id="cam_1",
        source="synthetic://cam_1",
        fps=10.0,
        resolution=(640, 480),
        homography=_make_homography(ox + 50.0, oy),
        exit_zones=[[(520.0, 130.0), (640.0, 130.0), (640.0, 350.0), (520.0, 350.0)]],
    )
    cam2 = CameraConfig(
        cam_id="cam_2",
        source="synthetic://cam_2",
        fps=10.0,
        resolution=(640, 480),
        homography=_make_homography(ox + 108.0, oy - 85.0),
        exit_zones=[[(540.0, 360.0), (640.0, 360.0), (640.0, 480.0), (540.0, 480.0)]],
    )

    vis_cfg = VisualizerConfig(
        output_video=output_video_path,
        fps=10.0,
        split_screen=True,
        width=1280,
        height=480,
        show_cov_ellipse=True,
        show_escape_tree=True,
        show_blockades=True,
    )
    reid_onnx = Path("weights/reid/osnet_x0_25_msmt17.onnx")
    cfg = PipelineConfig(
        cameras=[cam1, cam2],
        detector=DetectorConfig(model_path=Path("weights/yolo11n.pt"), tracker_type="bytetrack", device="cpu"),
        reid=ReIDConfig(
            onnx_path=reid_onnx if reid_onnx.is_file() else Path("hsv_hog_test"),
            device="cpu",
            allow_hsv_fallback=True,
        ),
        alpr=ALPRConfig(enabled=False),
        ekf=EKFConfig(model_type="adaptive_bicycle"),
        road_graph=RoadGraphConfig(),
        association=AssociationConfig(mode="hybrid", chi2_thresh=30.0, max_reid_dist=0.6),
        visualizer=vis_cfg,
    )

    step_ref = {"i": 0}
    total_steps = max(12, int(num_frames))

    def pos_at(i: int) -> tuple[float, float]:
        if i <= 6:
            return (ox + 45.0 + i * 5.8, oy)
        if i <= 16:
            frac = (i - 6) / 10.0
            if frac < 0.3:
                return (ox + 80.0 + (frac / 0.3) * 20.0, oy)
            if frac < 0.5:
                phi = (math.pi / 2.0) * (1.0 - (frac - 0.3) / 0.2)
                return (ox + 100.0 + 8.0 * math.cos(phi), oy - 8.0 + 8.0 * math.sin(phi))
            return (ox + 108.0, oy - 8.0 - ((frac - 0.5) / 0.5) * 64.0)
        return (ox + 108.0, oy - 72.0 - (i - 16) * 3.5)

    def det_cam1(_f: np.ndarray) -> np.ndarray:
        i = step_ref["i"]
        if i <= 6:
            x, y = pos_at(i)
            b = _utm_to_bbox(x, y, ox + 50.0, oy)
            return np.array([[b[0], b[1], b[2], b[3], 0.94, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    def det_cam2(_f: np.ndarray) -> np.ndarray:
        i = step_ref["i"]
        if i >= 17:
            x, y = pos_at(i)
            b = _utm_to_bbox(x, y, ox + 108.0, oy - 85.0)
            return np.array([[b[0], b[1], b[2], b[3], 0.93, 2.0]], dtype=np.float32)
        return np.empty((0, 6), dtype=np.float32)

    perceptions = {
        "cam_1": LocalPerception(cfg.detector, cam1.exit_zones, detector_fn=det_cam1),
        "cam_2": LocalPerception(cfg.detector, cam2.exit_zones, detector_fn=det_cam2),
    }
    pipeline = CausalMultiCamPipeline(
        cfg,
        road_network=road_net,
        road_engine=road_eng,
        perceptions=perceptions,
        detectors={"cam_1": det_cam1, "cam_2": det_cam2},
    )
    visualizer = CICCVisualizer(vis_cfg, road_net, [cam1, cam2])

    visualizer.open()
    try:
        for i in range(total_steps):
            step_ref["i"] = i
            x, y = pos_at(i)
            f1 = _render_road_frame(_utm_to_bbox(x, y, ox + 50.0, oy) if i <= 6 else None)
            f2 = _render_road_frame(_utm_to_bbox(x, y, ox + 108.0, oy - 85.0) if i >= 17 else None)
            state = pipeline.step(i * 0.2, {"cam_1": f1, "cam_2": f2})
            visualizer.write(state)
    finally:
        visualizer.close()

    return output_video_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render the end-to-end Split-Screen CICC Tactical Dashboard demo video."
    )
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Output MP4 video file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Alias for --output-video.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/demo"),
        help="Output directory if --output-video is not specified.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=24,
        help="Number of synchronous frames to render.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run headless without opening GUI windows.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if args.output_video is not None:
            out_mp4 = Path(args.output_video)
        elif args.output is not None:
            out_mp4 = Path(args.output)
        else:
            out_target = Path(args.out)
            out_mp4 = out_target if out_target.suffix.lower() == ".mp4" else (out_target / "cicc_demo.mp4")

        out_mp4.parent.mkdir(parents=True, exist_ok=True)
        run_cicc_demo(out_mp4, num_frames=args.max_frames)
        print(f"Rendered CICC Split-Screen demo video -> {out_mp4}")
        return 0
    except Exception as exc:
        print(f"Demo error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
