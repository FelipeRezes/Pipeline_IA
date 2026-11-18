"""Pipeline de rastreamento multicâmera, filtragem não linear e predição de rotas de fuga.

Refatoração completa inspirada no artigo:
Modelagem Cinemática Não Linear com Filtro de Kalman Estendido e Restrições
Topológicas para Predição de Rotas de Fuga em Redes Urbanas de Videovigilância
Policial (Medeiros et al.).

Componentes principais:
- Laboratório de filtros cinemáticos: AdaptiveEKF (Two-Point-Bicycle + desvanecimento),
  CanonicalBicycleEKF, CoordinatedTurnEKF (CTRV), LinearKalmanCV e IMMFilter;
- Projeção homográfica planar e calibração volumétrica 3D (K, R, t) com correção de perspectiva;
- Re-identificação visual de veículos (VehicleReIDExtractor) e rastreamento ByteTrack SCT;
- Map-matching contínuo e HMM/Viterbi, poda física por velocidade crítica de curva,
  propagação multi-hipótese em zonas cegas e recomendação de cerco policial (CICC);
- Suite de 5 Experimentos Científicos de Ablação (A1 a A5).
"""

from .config import PipelineConfig
from .ekf import AdaptiveEKF, CoordinatedTurnEKF, IMMFilter, LinearKalmanCV, create_filter
from .pipeline import PipelineResult, SurveillancePipeline

__all__ = [
    "PipelineConfig",
    "SurveillancePipeline",
    "PipelineResult",
    "AdaptiveEKF",
    "CoordinatedTurnEKF",
    "LinearKalmanCV",
    "IMMFilter",
    "create_filter",
]
__version__ = "2.0.0"
