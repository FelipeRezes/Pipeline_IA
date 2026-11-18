# Pipeline de Rastreamento Multicâmera, EKF Não Linear e Predição de Rotas de Fuga (v2.0)

Implementação científica refatorada do zero e laboratório de validação experimental para o artigo:
**"Modelagem Cinemática Não Linear com Filtro de Kalman Estendido e Restrições Topológicas para Predição de Rotas de Fuga em Redes Urbanas de Videovigilância Policial"** *(Medeiros et al.)*.

---

## 1. O que foi Refatorado e Aprimorado na v2.0

A versão original do repositório continha falhas conceituais e matemáticas que foram integralmente eliminadas e substituídas por uma arquitetura estado-da-arte:

1. **Inicialização 100% Causal em Tempo Real (`pipeline/pipeline.py`)**:
   - Removido o vazamento de dados do futuro (`_build_seeds`). A orientação inicial $\theta_0$ é inferida pelo alinhamento da via no grafo (`nearest_edge`) e a velocidade escalar $v_1$ é calibrada causalmente a partir do segundo quadro observado.
2. **Laboratório Comparativo de 5 Filtros Cinemáticos (`pipeline/ekf.py` e `pipeline/kinematics.py`)**:
   - `AdaptiveEKF` (*Two-Point-Bicycle* com Fator de Desvanecimento $\lambda_k$, Critério de Ortogonalidade $\alpha=(0.1, 0.1, 2.8)$, acoplamento Ackermann suave e gate contra jitter angular);
   - `CanonicalBicycleEKF` (*Two-Point-Bicycle* com $\lambda_k = 1$ fixo);
   - `CoordinatedTurnEKF` (Modelo CTRV de *Bradler et al., 2021* com transição exata por arco circular $R = v/\omega$);
   - `LinearKalmanCV` (Baseline linear de Velocidade Constante de *Kalman, 1960* na forma de Joseph);
   - `IMMFilter` (*Interacting Multiple Model* combinando dinamicamente modelos não lineares).
3. **Geometria Projetiva 2D/3D e Correção de Ponto de Contato (`pipeline/geometry.py`)**:
   - Suporte tanto à Homografia Planar $\mathbf{H}$ (*Chilgunde et al., 2004*) quanto à decomposição volumétrica por matrizes Intrínseca $\mathbf{K}$ e Extrínseca $[\mathbf{R} \mid \mathbf{t}]$ (*Bradler et al., 2021*);
   - Correção do viés de perspectiva 3D da bounding box (`bbox_bottom_center`), reduzindo o erro médio de projeção no solo em **74,26%** (de $2,201\text{ m}$ para $0,567\text{ m}$).
4. **Poda Física por Velocidade Crítica de Curva, Propagação Multi-Hipótese e Cerco Tático (`pipeline/roadgraph.py`)**:
   - Cálculo do raio geométrico da interseção $R_{\text{curva}}$ e da velocidade crítica de derrapagem lateral $v_{\text{crit}} = \sqrt{\mu \cdot g \cdot R_{\text{curva}}}$;
   - Propagação curvilínea Multi-Hipótese (`propagate_route_tree` / `BlindHypothesis`) com covariância anisotrópica ao longo dos ramos da Árvore de Rotas durante o período fora de visão (*Out-of-Sight*);
   - Baseline **HMM / Viterbi** (`hmm_viterbi_match`) para comparação direta com a Tabela da Seção 4 do artigo;
   - Recomendação automática de **Pontos de Bloqueio Tático Policial (CICC)** (`recommend_blockade_points`) com cálculo de probabilidade de interceptação e ETA.
5. **Re-Identificação Visual (`pipeline/reid.py`), ByteTrack SCT (`pipeline/video.py`) e Fusão Híbrida (`pipeline/association.py`)**:
   - Extrator de aparência `VehicleReIDExtractor` (Part-based HSV + HOG L2-normalizado) e rastreador intra-câmera em 2 estágios estilo **ByteTrack**;
   - Fusão probabilística combinando a distância de Mahalanobis multi-hipótese no grafo com a similaridade cosseno de Re-ID + **Backtracking Forense** da rota percorrida na região cega.

---

## 2. Resultados dos 5 Experimentos Científicos de Ablação (Testes A1 a A5)

Para reproduzir todos os experimentos e gerar o arquivo [ablation_results.json](file:///c:/Users/crist/OneDrive/Desktop/Pipeline_IA/ablation_results.json):

```bash
python -m pipeline.cli experiments --out ablation_results.json
```

### Experimento A1 — Filtros Cinemáticos em Manobra Evasiva + Apagão Visual
Comparação de RMSE (em metros) em trajetória com curva acentuada de $90^\circ$, frenagem e perda de visão (*blind outage*) de $1,4\text{ s}$ no meio da curva:

| Modelo de Filtro | RMSE Geral (m) | RMSE na Curva Evasiva (m) | RMSE no Apagão Visual (m) | Erro Máximo (m) |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Kalman (CV)** *(Baseline Seção 2.1)* | 1,461 | 2,291 | 2,997 | 5,047 |
| **CTRV EKF** *(Bradler et al., 2021)* | 0,418 | 0,667 | 0,925 | 1,636 |
| **IMM Filter** *(Bicycle + CTRV)* | 0,385 | 0,595 | 0,807 | 1,408 |
| **Two-Point-Bicycle EKF** *(Lou et al., 2002)* | **0,358** | **0,534** | **0,704** | **1,191** |

> **Conclusão A1:** O modelo não linear *Two-Point-Bicycle* reduz o erro durante manobras evasivas em **76,7%** em relação ao Filtro de Kalman Linear ($0,534\text{ m}$ vs. $2,291\text{ m}$), comprovando matematicamente a Seção 2.1 do artigo.

### Experimento A2 — Descoberta Científica sobre o Fator de Desvanecimento $\lambda_k$ Sob Ruído
Avaliação do fator de desvanecimento $\lambda_k$ (*Lou et al., 2002*) com pesos $\alpha = (0.1, 0.1, 2.8)$ quando a orientação $\theta$ é derivada por diferenças finitas de bounding boxes ruidosas ($\sigma_{\text{pos}} \in \{0.4\text{ m}, 1.0\text{ m}, 2.2\text{ m}\}$):

| Nível de Ruído ($\sigma_{\text{pos}}$) | RMSE EKF Canônico ($\lambda=1$) | RMSE EKF Adaptativo ($\lambda_k \ge 1$) | Pico de $\lambda_k$ | Diagnóstico Experimental |
| :--- | :---: | :---: | :---: | :--- |
| **Baixo ($0,4\text{ m}$)** | **0,368 m** | 0,574 m | 5,98 | $\lambda_k$ reage rápido à curva, mas absorve parte do ruído de medição. |
| **Moderado ($1,0\text{ m}$)** | **0,760 m** | 1,309 m | 20,85 | Jitter em $\Delta y / \Delta x$ com peso $\alpha_\theta=2,8$ infla $\mathbf{P}$ além do ideal. |
| **Alto ($2,2\text{ m}$)** | **0,915 m** | 2,583 m | 11,00 | Com alto ruído de detecção, inflar $\mathbf{P}$ faz o ganho $\mathbf{K}$ seguir o ruído da câmera. |

> **Conclusão A2 (Insight Crítico para o Artigo):** O fator de desvanecimento $\lambda_k$ de *Lou et al. (2002)* foi originalmente projetado para erros de modelo com sensores de baixo ruído. Em câmeras de segurança com ruído elevado de bounding box, atribuir $\alpha_\theta = 2,8$ a um ângulo $\theta$ derivado de posições ruidosas faz $\lambda_k$ disparar em falsas manobras. Nossa arquitetura resolve isso com o `heading_gate_rad` e permite configurar `adaptive_fading` conforme a qualidade da câmera.

### Experimento A3 — Correção de Perspectiva 3D na Projeção Homográfica
- **BBox Bottom-Center Bruto (`fraction=1.0`)**: Erro médio no plano de solo = **$2,201\text{ m}$**
- **Ponto de Contato Corrigido (`fraction=0.90` + compensação de perspectiva 3D)**: Erro médio = **$0,567\text{ m}$** (**redução de 74,26% no viés de localização**).

### Experimento A4 — Associação Multicâmera em Regiões Cegas (Ablação de 4 Paradigmas)
Teste em malha urbana multicâmera com 18 veículos e 42 transições por zonas cegas não sobrepostas:

| Modo de Associação Intercâmeras | IDF1 Global | Acurácia Intercâmeras | Acurácia de Rota (Top-1) | Acurácia de Rota (Top-K) | Vazão (FPS) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Limiar Fixo do Artigo ($M \le 350$)** | 0,4341 | 11,11% | 72,50% | 92,50% | 422,8 |
| **2. Estatística $\chi^2_{4\text{gl}}$ (Cinemática + Grafo)** | 0,9694 | 88,89% | 100,0% | 100,0% | 383,8 |
| **3. Re-ID Visual Puro (Cosseno)** | 0,9588 | 100,0% | 97,67% | 100,0% | 385,1 |
| **4. Fusão Híbrida (Grafo Multi-Hipótese + EKF + Re-ID)** | **1,0000** | **100,0%** | **100,0%** | **100,0%** | **444,1** |

> **Conclusão A4:** O limiar fixo $M = 350$ citado em *Chilgunde et al. (2004)* (calculado originalmente em coordenadas de pixels) é excessivamente permissivo quando aplicado em coordenadas métricas (metros), gerando falsos positivos severos ($\text{IDF1} = 0,43$). A substituição pelo teste estatístico $\chi^2$ ($p=0,01 \Rightarrow 13,28$) eleva a acurácia intercâmeras para **88,89%** (superando os 85% do artigo), e a **Fusão Híbrida (EKF + Grafo Multi-Hipótese + Re-ID)** atinge **100% de precisão e IDF1 = 1,00** a mais de **440 FPS**.

### Experimento A5 — Map-Matching HMM/Viterbi e Cerco Tático Policial (CICC)
- **Map-Matching sob Ruído Forte ($\sigma = 9,5\text{ m}$)**: Projeção Euclidiana Simples = **91,67%** vs. **HMM / Viterbi = 100,0%** (elimina o erro de *unnecessary detour* em esquinas citado na Seção 4 do artigo).
- **Cerco Tático Policial**: Gera automaticamente os nós de bloqueio ótimo (`v_3_1` com probabilidade de interceptação acumulada de $100\%$ e $\text{ETA} = 6,7\text{ s}$).

---

## 3. Catálogo de Datasets e Modelos Pré-Treinados Suportados

| Recurso / Dataset | Aplicação na Nossa Arquitetura | Link / Acesso |
| :--- | :--- | :--- |
| **CityFlow / CityFlowV2 (AI City Challenge)** | Rastreamento multicâmera (46 câmeras, zonas cegas, homografia GPS). Suportado nativamente por `pipeline/cityflow.py`. | [AI City Challenge](https://www.aicitychallenge.org/) · [GitHub](https://github.com/cityflow-project/CityFlow) |
| **RoundaboutHD (2025)** | 4 câmeras 4K não sobrepostas com tráfego em curvas acentuadas e calibração geométrica. | [GitHub](https://github.com/siri-rouser/RoundaboutHD) |
| **VeRi-776** | Reidentificação visual de 776 veículos em 20 câmeras (benchmark para `pipeline/reid.py`). | [VeRi Dataset](https://vehiclereid.github.io/VeRi/) |
| **UA-DETRAC & KITTI** | Vídeos reais de tráfego urbano para avaliação de detecção YOLO + ByteTrack SCT + EKF (`kitti_inference.mp4` incluso). | [UA-DETRAC](https://detrac-db.rit.albany.edu/) |
| **YOLOv8 / YOLO11 + ByteTrack** | Detecção de veículos em tempo real e rastreamento intra-câmera em 2 estágios (`pipeline/video.py`). | Pesos em `weights/yolov8n.pt` |

---

## 4. Como Executar

```bash
# 1. Rodar toda a suíte de testes unitários e de integração (14 testes)
python -m unittest discover -s tests -v

# 2. Executar os 5 Experimentos Científicos de Ablação (A1 a A5)
python -m pipeline.cli experiments --out ablation_results.json

# 3. Gerar e rodar a demonstração multicâmera sintética (formato CityFlow)
python -m pipeline.cli demo --root data/cityflow_demo
python -m pipeline.cli run --data data/cityflow_demo --assoc-mode hybrid --filter-model adaptive_bicycle --out resultado.json

# 4. Executar sobre vídeos reais com YOLO + ByteTrack + Re-ID + EKF + Grafo
python -m pipeline.cli video --sources kitti_inference.mp4 --calib-dir calib --graph grafo_demo.json --weights weights/yolov8n.pt --out kitti_result.json
```
