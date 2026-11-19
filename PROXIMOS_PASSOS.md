# Resumo do Projeto e Próximos Passos (`PROXIMOS_PASSOS.md`)

---

## 1. Resumo de Tudo que Foi Feito Até Agora

O repositório foi reestruturado do zero em uma arquitetura modular (`src/tracker/`, `scripts/`, `configs/`, `tests/`) com ambiente virtual dedicado (`.venv` em Python 3.12) e **256 testes automatizados (`pytest`) passando**:

1. **Configuração e Dados (`src/tracker/config.py`, `scripts/prepare_datasets.py`)**:
   - Schemas Pydantic v2 validando câmeras, homografias, EKF, grafo viário, associação e visualizador;
   - Pesos pré-treinados reais baixados em `weights/`:
     - `weights/yolo11n.pt` (detecção veicular Ultralytics YOLO11);
     - `weights/reid/osnet_x0_25_msmt17.onnx` (Deep Re-ID 512-D em ONNX Runtime);
     - `weights/alpr/*.onnx` (detecção e OCR de placas via `fast-alpr`).
   - Imagens reais e mapas preparados em `data/`:
     - `data/synthehicle/Town01-N-day/` (frames RGB reais `C01` e `C02`, `calibration.json` e `gt.txt`);
     - `data/ua_detrac/detrac_100_MVI_40905/` (frames reais de CFTV urbano + máscara de zona cega `blind_mask.json`);
     - `data/osm/sao_paulo_paulista.graphml` (malha viária real do OpenStreetMap em UTM).
2. **Núcleo Matemático e Cinemático (`geometry.py`, `kinematics.py`, `ekf.py`)**:
   - Homografia DLT normalizado + RANSAC, projeção 3D ($\mathbf{K}[\mathbf{R}|\mathbf{t}]$), conversão WGS84 $\leftrightarrow$ UTM, correção do ponto de contato 3D da *bounding box* (`fraction=0.90`) e covariância métrica $\mathbf{R}_k = \mathbf{J}\mathbf{\Sigma}_{\text{px}}\mathbf{J}^T$;
   - 5 filtros cinemáticos (`AdaptiveEKF`, `CanonicalEKF`, `CTRVFilter`, `LinearCVFilter`, `IMMFilter`) com suavização circular de orientação (4 frames), peso $\alpha_\theta(v)$ adaptativo e gatilho NIS ($\chi^2_{95\%}$ por $\ge 2$ frames).
3. **Malha Viária, Percepção, Associação e Visualizador (`osm_loader.py`, `roadgraph.py`, `detector.py`, `reid.py`, `alpr.py`, `association.py`, `pipeline.py`, `visualizer.py`)**:
   - Map-Matching HMM/Viterbi, poda física por velocidade crítica de curva ($v_{\text{crit}} = \sqrt{\mu g R_{\text{curva}}}$), frenagem pré-esquina, árvore de rotas na zona cega e recomendação de pontos de bloqueio CICC com ETA;
   - Wrapper `LocalPerception` (YOLO + BoxMOT + polígonos de *Exit Zones*), extrator ONNX `ReIDExtractor`, leitor de placas brasileiras/Mercosul `ALPREngine`, associador híbrido `CrossCameraAssociator` com Backtracking Forense e renderizador `CICCVisualizer`.

---

## 2. Diagnóstico: Por Que a Pipeline Deu Erro na Execução Direta

Ao rodar a pipeline diretamente fora da suíte de testes (`python -m tracker.pipeline` ou nos configs padrão), ocorrem 3 gargalos práticos que impedem a visualização imediata com imagens reais:

1. **Caminhos de Vídeo Inexistentes nos YAMLs (`configs/default.yaml` e `configs/cameras_demo.yaml`)**:
   - Os arquivos YAML apontam para `source: data/demo/cam_01.mp4`, que não existe por padrão em disco.
2. **Leitura Restrita a `.mp4` e Falta de Gravação Visual em `CausalMultiCamPipeline.run()` (`src/tracker/pipeline.py`)**:
   - O método `pipeline.run()` atualmente só abre arquivos `.mp4` via `cv2.VideoCapture` (não lê diretórios de frames `.jpg`/`.png` reais como `data/synthehicle/Town01-N-day/C01/img1/` ou `data/ua_detrac/.../img1/`);
   - Além disso, `pipeline.run()` retorna apenas um JSON resumido e **não instancia o `CICCVisualizer`** para gravar o vídeo *Split-Screen* final.
3. **Demo Sintética em `scripts/run_demo.py`**:
   - O `run_demo.py` atual desenha caixas sintéticas em fundo cinza para teste rápido, em vez de alimentar os frames reais baixados pelas redes `YOLO11n` + `OSNet ONNX`.

---

## 3. Próximos Passos (Passos Curtos e Testáveis até o Resultado Visual com Imagens Reais)

Cada passo abaixo é curto, isolado e possui um **teste imediato** antes de passar ao seguinte, até chegarmos na pipeline rodando ponta a ponta com vídeo e imagens reais.

---

### Passo 1 — Testar Detecção Real (YOLO11n) + Rastreamento Local + Exit Zones em Frames Reais
* **O que fazer:**
  - Garantir que `LocalPerception` (`src/tracker/detector.py`) carregue `weights/yolo11n.pt` sem injeção de mock e processe uma sequência real de imagens de `data/ua_detrac/detrac_100_MVI_40905/img1/` (ou `data/synthehicle/Town01-N-day/C01/img1/`).
* **Como será testado:**
  - Rodar um teste curto de 15 frames reais verificando:
    1. Quantos veículos reais são detectados por frame (`conf >= 0.35`, classes `[2, 3, 5, 7]`);
    2. Persistência do `local_track_id` ao longo dos frames;
    3. Exportar 1 imagem real anotada (`data/outputs/step1_yolo_real_check.jpg`) com as *bounding boxes* e *Exit Zones* desenhadas.

---

### Passo 2 — Testar Deep Re-ID ONNX (`osnet_x0_25_msmt17.onnx`) nos Recortes Reais dos Veículos
* **O que fazer:**
  - Pegar os `crop_bgr` reais extraídos pelo YOLO no Passo 1 e passá-los pelo `ReIDExtractor` (`src/tracker/reid.py`) usando o modelo ONNX real.
* **Como será testado:**
  - Verificar que os embeddings de 512 dimensões do **mesmo veículo real** em frames/câmeras diferentes têm distância cosseno significativamente menor ($< 0.30$) do que entre **veículos reais diferentes** ($> 0.45$).

---

### Passo 3 — Suporte a Pastas de Imagens Reais (`.jpg`/`.png`) e Vídeos em `src/tracker/pipeline.py`
* **O que fazer:**
  - Criar um leitor universal de stream (`FrameSource`) em `src/tracker/pipeline.py` que aceite em `CameraConfig.source`:
    - Um arquivo de vídeo (`.mp4`, `.avi`); **ou**
    - Um diretório de frames reais ordenados (`data/ua_detrac/detrac_100_MVI_40905/img1` ou `data/synthehicle/Town01-N-day/C01/img1`).
* **Como será testado:**
  - Apontar `CameraConfig.source` para a pasta de imagens reais e verificar que `pipeline.run(max_frames=10)` lê os frames reais na resolução correta sem cair em fallback de tela vazia.

---

### Passo 4 — Alinhar Calibração Homográfica e Grafo Viário com a Cena Real
* **O que fazer:**
  - Ajustar a matriz de homografia $\mathbf{H}$ e o grafo viário (`RoadNetwork`) no arquivo de configuração real (`configs/real_demo.yaml`) para que os pontos de contato dos veículos reais detectados pelo YOLO caiam exatamente sobre as vias em coordenadas métricas UTM.
* **Como será testado:**
  - Projetar as *bounding boxes* reais do Passo 1 via `CameraProjector.bbox_to_ground()` e confirmar que a distância até a aresta mais próxima do grafo viário (`snap_initial_heading`) é $< 5\text{ m}$ e que a velocidade estimada pelo `AdaptiveEKF` fica na faixa urbana realista ($20\text{ a }80\text{ km/h}$).

---

### Passo 5 — Integrar `CICCVisualizer` Dentro de `CausalMultiCamPipeline.run()`
* **O que fazer:**
  - Atualizar `CausalMultiCamPipeline.run()` em `src/tracker/pipeline.py` para que, sempre que `config.visualizer.output_video` estiver definido, ele abra o `CICCVisualizer`, chame `visualizer.write(state)` a cada frame processado e salve também snapshots `.png` dos momentos-chave (1: veículo visível, 2: veículo na zona cega com elipse EKF + árvore de fuga + bloqueios CICC, 3: veículo re-identificado com rota forense reconstruída).
* **Como será testado:**
  - Executar `python -m tracker.pipeline --config configs/real_demo.yaml --max-frames 30` e confirmar que o arquivo `.mp4` e os snapshots `.png` são criados automaticamente em disco.

---

### Passo 6 — "Finalmentes": Execução Completa e Resultado Visual com Imagens Reais
* **O que fazer:**
  - Atualizar `scripts/run_demo.py` e `configs/default.yaml` para rodarem por padrão sobre as **imagens reais** (câmeras reais com transição por zona cega / máscara de zona cega no `UA-DETRAC` e multicâmera no `Synthehicle`), usando `YOLO11n` + `OSNet ONNX` + `AdaptiveEKF` + `RoadGraphEngine` + `CICCVisualizer`.
* **Como será testado (Critério Final de Aceitação Visual):**
  - Executar um único comando:
    ```powershell
    .\.venv\Scripts\python.exe scripts/run_demo.py --output-video data/outputs/cicc_real_demo.mp4
    ```
  - Inspecionar o vídeo `data/outputs/cicc_real_demo.mp4` e as imagens finais em `data/outputs/` comprovando:
    1. **Lado Esquerdo (Imagens Reais das Câmeras):** Veículos reais detectados pelo YOLO11n com bounding boxes precisas, ID global consistente antes e depois da zona cega, velocidade em km/h e status (`VISÍVEL` / `ZONA CEGA`);
    2. **Lado Direito (Mapa Tático 2D UTM):** Posição dos veículos reais projetada na malha viária, elipse de covariância $\mathbf{P}_k$ do EKF expandindo durante a zona cega, árvore de rotas de fuga colorida por probabilidade, pontos de cerco policial CICC com ETA e rota reconstruída por backtracking forense após a re-identificação.
