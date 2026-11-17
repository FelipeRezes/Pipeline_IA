"""Pipeline de rastreamento e predição de rotas de fuga inspirada no artigo:
Modelagem Cinemática Não Linear com Filtro de Kalman Estendido e Restrições
Topológicas para Predição de Rotas de Fuga em Redes Urbanas de Videovigilância
Policial (Medeiros et al.).

Componentes principais:
- EKF adaptativo com modelo Two-Point-Bicycle e fator de desvanecimento;
- projeção homográfica para plano de solo comum;
- associação gaussiana intercâmeras em regiões cegas;
- map-matching em grafo viário direcionado e árvore de rotas prováveis;
- adaptador para CityFlow / CityFlowV2 (AI City Challenge) e avaliação.
"""

from .config import PipelineConfig
from .pipeline import SurveillancePipeline, PipelineResult

__all__ = ["PipelineConfig", "SurveillancePipeline", "PipelineResult"]
__version__ = "0.1.0"
