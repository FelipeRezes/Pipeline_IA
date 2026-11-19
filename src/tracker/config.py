"""Typed Pydantic v2 configuration schemas and YAML I/O for Pipeline_IA."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_matrix_3x3(mat: list[list[float]], name: str) -> list[list[float]]:
    if len(mat) != 3 or any(len(row) != 3 for row in mat):
        raise ValueError(f"{name} must be a 3x3 matrix")
    out: list[list[float]] = []
    for row in mat:
        clean_row = [float(v) for v in row]
        if any(not math.isfinite(v) for v in clean_row):
            raise ValueError(f"{name} must contain only finite numeric entries")
        out.append(clean_row)
    return out


def _det3x3(m: list[list[float]]) -> float:
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class CameraConfig(_StrictModel):
    """Per-camera calibration, video source, and Exit Zone configuration."""

    cam_id: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1)
    fps: float = Field(default=15.0, gt=0.0)
    resolution: tuple[int, int] = (1280, 720)
    calib_path: Path | None = None
    homography: list[list[float]] | None = None
    K: list[list[float]] | None = None
    R: list[list[float]] | None = None
    t: list[float] | None = None
    exit_zones: list[list[tuple[float, float]]] = Field(default_factory=list)
    epsg_utm: int | None = Field(default=None, gt=0)

    @field_validator("cam_id", "source")
    @classmethod
    def _strip_non_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("must be a non-empty string")
        return s

    @field_validator("fps")
    @classmethod
    def _finite_fps(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0.0:
            raise ValueError("fps must be finite and > 0")
        return float(v)

    @field_validator("resolution")
    @classmethod
    def _valid_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError("resolution width and height must both be > 0")
        return (int(w), int(h))

    @field_validator("homography")
    @classmethod
    def _valid_homography(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return None
        mat = _validate_matrix_3x3(v, "homography")
        if abs(_det3x3(mat)) <= 1e-12:
            raise ValueError("homography matrix must be non-singular (|det| > 1e-12)")
        return mat

    @field_validator("K")
    @classmethod
    def _valid_k(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return None
        mat = _validate_matrix_3x3(v, "K")
        if mat[0][0] <= 0.0 or mat[1][1] <= 0.0:
            raise ValueError("K focal lengths K[0][0] and K[1][1] must be > 0")
        return mat

    @field_validator("R")
    @classmethod
    def _valid_r(cls, v: list[list[float]] | None) -> list[list[float]] | None:
        if v is None:
            return None
        mat = _validate_matrix_3x3(v, "R")
        if abs(_det3x3(mat)) <= 1e-6:
            raise ValueError("rotation matrix R must be non-singular")
        return mat

    @field_validator("t")
    @classmethod
    def _valid_t(cls, v: list[float] | None) -> list[float] | None:
        if v is None:
            return None
        if len(v) != 3:
            raise ValueError("translation vector t must have exactly 3 elements")
        out = [float(x) for x in v]
        if any(not math.isfinite(x) for x in out):
            raise ValueError("translation vector t entries must be finite")
        return out

    @field_validator("exit_zones")
    @classmethod
    def _valid_exit_zones(
        cls, v: list[list[tuple[float, float]]]
    ) -> list[list[tuple[float, float]]]:
        cleaned: list[list[tuple[float, float]]] = []
        for idx, poly in enumerate(v):
            if len(poly) < 3:
                raise ValueError(f"exit_zones[{idx}] must have at least 3 vertices")
            pts: list[tuple[float, float]] = []
            for x, y in poly:
                fx, fy = float(x), float(y)
                if not (math.isfinite(fx) and math.isfinite(fy)):
                    raise ValueError(f"exit_zones[{idx}] contains non-finite coordinates")
                pts.append((fx, fy))
            cleaned.append(pts)
        return cleaned

    @model_validator(mode="after")
    def _check_3d_calib_group(self) -> CameraConfig:
        parts = (self.K, self.R, self.t)
        provided = sum(p is not None for p in parts)
        if 0 < provided < 3:
            raise ValueError("K, R, and t must all be provided together when using 3D calibration")
        return self


class DetectorConfig(_StrictModel):
    """Ultralytics YOLO + BoxMOT single-camera detection & tracking settings."""

    model_path: Path = Path("weights/yolo11n.pt")
    tracker_type: Literal["bytetrack", "botsort", "deepocsort"] = "bytetrack"
    conf_thresh: float = Field(default=0.35, gt=0.0, le=1.0)
    iou_thresh: float = Field(default=0.5, gt=0.0, le=1.0)
    classes: list[int] = Field(default_factory=lambda: [2, 3, 5, 7])
    device: str = Field(default="auto", min_length=1)

    @field_validator("classes")
    @classmethod
    def _valid_classes(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("classes must be a non-empty list of COCO class indices")
        if any(c < 0 for c in v):
            raise ValueError("all class indices must be >= 0")
        return [int(c) for c in v]


class ReIDConfig(_StrictModel):
    """Deep Re-ID ONNX feature extractor configuration."""

    onnx_path: Path = Path("weights/reid/osnet_x0_25_msmt17.onnx")
    device: str = Field(default="auto", min_length=1)
    input_size: tuple[int, int] = (256, 128)
    allow_hsv_fallback: bool = False
    embedding_dim: int = Field(default=512, ge=1)

    @field_validator("input_size")
    @classmethod
    def _valid_input_size(cls, v: tuple[int, int]) -> tuple[int, int]:
        h, w = v
        if h <= 0 or w <= 0:
            raise ValueError("input_size (height, width) dimensions must be > 0")
        return (int(h), int(w))


class ALPRConfig(_StrictModel):
    """Automatic License Plate Recognition configuration."""

    enabled: bool = True
    backend: Literal["fast_alpr", "paddleocr", "auto"] = "auto"
    min_conf: float = Field(default=0.5, ge=0.0, le=1.0)


class EKFConfig(_StrictModel):
    """Kinematic filter and Adaptive EKF stabilization configuration."""

    model_type: Literal[
        "adaptive_bicycle", "canonical_bicycle", "ctrv", "linear_cv", "imm"
    ] = "adaptive_bicycle"
    wheelbase_m: float = Field(default=2.5, gt=0.0)
    adaptive_fading: bool = True
    theta_window: int = Field(default=4, ge=1)
    heading_gate_rad: float = Field(default=0.35, ge=0.0, le=math.pi)
    nis_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    nis_persistence_frames: int = Field(default=2, ge=1)
    sigma_px: tuple[float, float] = (4.0, 4.0)
    lambda_max: float = Field(default=25.0, ge=1.0)
    min_speed_for_heading_mps: float = Field(default=1.0, ge=0.0)

    @field_validator("sigma_px")
    @classmethod
    def _valid_sigma_px(cls, v: tuple[float, float]) -> tuple[float, float]:
        sx, sy = float(v[0]), float(v[1])
        if not (math.isfinite(sx) and math.isfinite(sy)) or sx <= 0.0 or sy <= 0.0:
            raise ValueError("sigma_px entries must be finite and > 0")
        return (sx, sy)


class RoadGraphConfig(_StrictModel):
    """OSMnx UTM road network and blind-zone physics configuration."""

    place_name: str | None = None
    bbox_wgs84: tuple[float, float, float, float] | None = None
    graphml_cache: Path | None = None
    overpass_urls: list[str] = Field(
        default_factory=lambda: [
            "https://overpass.osm.ch/api/interpreter",
            "https://api.openstreetmap.org/api/0.6/map",
            "https://overpass-api.de/api/interpreter",
        ]
    )
    mu_friction: float = Field(default=0.7, gt=0.0, le=2.0)
    gravity: float = Field(default=9.81, gt=0.0)
    corner_angle_deg: float = Field(default=30.0, gt=0.0, le=180.0)
    horizon_s: float = Field(default=15.0, gt=0.0)
    min_branch_prob: float = Field(default=0.02, ge=0.0, le=1.0)

    @field_validator("bbox_wgs84")
    @classmethod
    def _valid_bbox(
        cls, v: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if v is None:
            return None
        left, bottom, right, top = (float(x) for x in v)
        if not all(math.isfinite(x) for x in (left, bottom, right, top)):
            raise ValueError("bbox_wgs84 coordinates must be finite")
        if not (-180.0 <= left < right <= 180.0):
            raise ValueError("bbox_wgs84 must satisfy -180 <= left < right <= 180")
        if not (-90.0 <= bottom < top <= 90.0):
            raise ValueError("bbox_wgs84 must satisfy -90 <= bottom < top <= 90")
        return (left, bottom, right, top)

    @field_validator("overpass_urls")
    @classmethod
    def _valid_urls(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("overpass_urls must be a non-empty list")
        cleaned: list[str] = []
        for url in v:
            s = url.strip()
            if not s.startswith(("http://", "https://")):
                raise ValueError(f"invalid Overpass/OSM URL: {url!r}")
            cleaned.append(s)
        return cleaned


class AssociationConfig(_StrictModel):
    """Inter-camera hybrid association and scale self-calibration configuration."""

    mode: Literal["hybrid", "chi2", "reid_only", "article"] = "hybrid"
    chi2_thresh: float = Field(default=13.28, gt=0.0)
    w_spatiotemporal: float = Field(default=0.5, ge=0.0, le=1.0)
    w_reid: float = Field(default=0.5, ge=0.0, le=1.0)
    max_reid_dist: float = Field(default=0.45, gt=0.0, le=2.0)
    use_alpr_override: bool = True
    alpr_override_conf: float = Field(default=0.85, ge=0.0, le=1.0)
    auto_calibrate_scale: bool = True
    min_scale_samples: int = Field(default=5, ge=1)

    @model_validator(mode="after")
    def _check_weights_sum(self) -> AssociationConfig:
        if not math.isclose(self.w_spatiotemporal + self.w_reid, 1.0, abs_tol=1e-6):
            raise ValueError(
                "w_spatiotemporal + w_reid must sum to 1.0 "
                f"(got {self.w_spatiotemporal + self.w_reid})"
            )
        return self


class VisualizerConfig(_StrictModel):
    """CICC Split-Screen tactical dashboard rendering configuration."""

    output_video: Path | None = None
    fps: float = Field(default=15.0, gt=0.0)
    split_screen: bool = True
    width: int = Field(default=1920, gt=0)
    height: int = Field(default=720, gt=0)
    show_cov_ellipse: bool = True
    show_escape_tree: bool = True
    show_blockades: bool = True
    export_html: Path | None = None


class PipelineConfig(_StrictModel):
    """Top-level configuration for the causal multi-camera tracking pipeline."""

    cameras: list[CameraConfig] = Field(default_factory=list)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    reid: ReIDConfig = Field(default_factory=ReIDConfig)
    alpr: ALPRConfig = Field(default_factory=ALPRConfig)
    ekf: EKFConfig = Field(default_factory=EKFConfig)
    road_graph: RoadGraphConfig = Field(default_factory=RoadGraphConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)
    data_dir: Path = Path("data")
    weights_dir: Path = Path("weights")
    strictly_causal: bool = True
    random_seed: int = 42

    @field_validator("cameras")
    @classmethod
    def _unique_cam_ids(cls, v: list[CameraConfig]) -> list[CameraConfig]:
        seen: set[str] = set()
        for cam in v:
            if cam.cam_id in seen:
                raise ValueError(f"duplicate cam_id detected: {cam.cam_id!r}")
            seen.add(cam.cam_id)
        return v


class BlindMaskConfig(_StrictModel):
    """Artificial blind-zone blackout configuration for benchmark scenarios."""

    enabled: bool = False
    start_s: float = Field(default=3.0, ge=0.0)
    duration_s: float = Field(default=5.0, gt=0.0, le=30.0)


class ScenarioConfig(_StrictModel):
    """Named scenario / ablation experiment entry in configs/scenarios.yaml."""

    scenario_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    description: str = ""
    dataset: Literal[
        "synthehicle", "cityflow", "ua_detrac", "veri776", "rodosol_alpr", "synthetic"
    ] = "synthetic"
    ablation_id: Literal["A1", "A2", "A3", "A4", "A5", "DEMO"] | None = None
    config_path: Path | None = None
    blind_mask: BlindMaskConfig | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class ScenariosFileConfig(_StrictModel):
    """Root container for configs/scenarios.yaml."""

    scenarios: dict[str, ScenarioConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_keys_match_ids(self) -> ScenariosFileConfig:
        for key, scenario in self.scenarios.items():
            if key != scenario.scenario_id:
                raise ValueError(
                    f"scenario key {key!r} does not match scenario_id {scenario.scenario_id!r}"
                )
        return self


def load_config(path: Path | str) -> PipelineConfig:
    """Load and validate a PipelineConfig from a YAML file."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {cfg_path}, got {type(raw).__name__}")
    return PipelineConfig.model_validate(raw)


def save_config(config: PipelineConfig, path: Path | str) -> Path:
    """Serialize a PipelineConfig to YAML and return the output path."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = config.model_dump(mode="json")
    with out_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
    return out_path


def load_scenarios(path: Path | str) -> dict[str, ScenarioConfig]:
    """Load and validate named benchmark/demo scenarios from YAML."""
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Scenarios file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected YAML mapping in {cfg_path}, got {type(raw).__name__}")
    parsed = ScenariosFileConfig.model_validate(raw)
    return parsed.scenarios
