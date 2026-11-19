"""Pipeline_IA multi-camera vehicle tracking package."""

from tracker.config import (
    ALPRConfig,
    AssociationConfig,
    BlindMaskConfig,
    CameraConfig,
    DetectorConfig,
    EKFConfig,
    PipelineConfig,
    ReIDConfig,
    RoadGraphConfig,
    ScenarioConfig,
    ScenariosFileConfig,
    VisualizerConfig,
    load_config,
    load_scenarios,
    save_config,
)

__version__ = "0.1.0"

__all__ = [
    "ALPRConfig",
    "AssociationConfig",
    "BlindMaskConfig",
    "CameraConfig",
    "DetectorConfig",
    "EKFConfig",
    "PipelineConfig",
    "ReIDConfig",
    "RoadGraphConfig",
    "ScenarioConfig",
    "ScenariosFileConfig",
    "VisualizerConfig",
    "__version__",
    "load_config",
    "load_scenarios",
    "save_config",
]
