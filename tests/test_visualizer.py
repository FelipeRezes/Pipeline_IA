"""Unit tests for src/tracker/visualizer.py."""

from __future__ import annotations

import numpy as np
from scripts.run_benchmarks import _build_benchmark_road_network
from tracker.config import CameraConfig, VisualizerConfig
from tracker.pipeline import PipelineFrameState
from tracker.visualizer import CICCVisualizer


def test_cicc_visualizer_render_frame() -> None:
    net, _ = _build_benchmark_road_network()
    cam = CameraConfig(cam_id="cam_1", source="cam1.mp4", resolution=(640, 480))
    vis = CICCVisualizer(VisualizerConfig(width=1280, height=480), net, [cam])
    st = PipelineFrameState(
        timestamp_s=1.0,
        frames_bgr={"cam_1": np.full((480, 640, 3), 50, dtype=np.uint8)},
        camera_tracks={"cam_1": []},
        global_states={},
        blind_hypotheses={},
        blockade_points=[],
        forensic_paths=[],
        events=[],
    )
    img = vis.render_frame(st)
    assert img.shape == (480, 1280, 3) and img.dtype == np.uint8
