"""Unit tests for src/tracker/pipeline.py."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from scripts.run_benchmarks import _build_benchmark_road_network
from tracker.config import ALPRConfig, CameraConfig, DetectorConfig, PipelineConfig, ReIDConfig
from tracker.detector import LocalPerception
from tracker.pipeline import CausalMultiCamPipeline


def test_causal_pipeline_step_and_monotonic_check(tmp_path: Path) -> None:
    net, eng = _build_benchmark_road_network()
    cam = CameraConfig(
        cam_id="cam_1",
        source="synthetic://cam_1",
        fps=10.0,
        resolution=(640, 480),
        homography=[[0.1, 0.0, 330000.0], [0.0, -0.1, 7390050.0], [0.0, 0.0, 1.0]],
    )
    cfg = PipelineConfig(
        cameras=[cam],
        detector=DetectorConfig(model_path=tmp_path / "unused.pt"),
        reid=ReIDConfig(onnx_path=Path("hsv_hog_test"), allow_hsv_fallback=True),
        alpr=ALPRConfig(enabled=False),
    )

    def det_fn(_f: np.ndarray) -> np.ndarray:
        return np.array([[200.0, 200.0, 260.0, 250.0, 0.9, 2.0]], dtype=np.float32)

    p = LocalPerception(cfg.detector, [], detector_fn=det_fn)
    pipe = CausalMultiCamPipeline(cfg, road_network=net, road_engine=eng, perceptions={"cam_1": p})
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    st = pipe.step(0.0, {"cam_1": frame})
    assert len(st.global_states) == 1
    with pytest.raises(ValueError):
        pipe.step(-0.1, {"cam_1": frame})
