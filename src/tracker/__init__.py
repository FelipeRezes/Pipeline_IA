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

from tracker.sota_baselines import (
    GRUTrajectoryPredictor,
    LSTMTrajectoryPredictor,
    ParticleFilter,
    TemporalGNNPredictor,
)

__version__ = "0.2.0"

__all__ = [
    "ALPRConfig",
    "AssociationConfig",
    "BlindMaskConfig",
    "CameraConfig",
    "DetectorConfig",
    "EKFConfig",
    "GRUTrajectoryPredictor",
    "LSTMTrajectoryPredictor",
    "ParticleFilter",
    "PipelineConfig",
    "ReIDConfig",
    "RoadGraphConfig",
    "ScenarioConfig",
    "ScenariosFileConfig",
    "TemporalGNNPredictor",
    "VisualizerConfig",
    "__version__",
    "load_config",
    "load_scenarios",
    "save_config",
]
