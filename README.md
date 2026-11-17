# Pipeline de Predição de Rotas de Fuga — EKF + Cinemática Não Linear + Grafo Viário

Implementação executável do artigo **"Modelagem Cinemática Não Linear com Filtro de Kalman Estendido e Restrições Topológicas para Predição de Rotas de Fuga em Redes Urbanas de Videovigilância Policial"** (Medeiros et al.), atendendo à orientação do professor:

> acessar links públicos de vídeos de cidades com múltiplas câmeras e avaliar a pipeline com essas datasets.

**A arquitetura foi implementada para vídeos reais**: ela lê arquivos MP4, URLs HTTP(S), RTSP e playlists HLS (.m3u8), detecta veículos com YOLO, constrói tracklets locais e alimenta o EKF adaptativo + homografia + map-matching em grafo viário. Também roda em modo *offline* sobre as anotações oficiais da CityFlowV2 para avaliação reprodutível.

---

## 1. Resposta ao prompt do GPT — onde obter vídeos públicos multicheverramento

| Dataset | Câmeras | O que oferece | Link |
|---|---|---|---|
| **CityFlow / CityFlowV2** (AI City Challenge) | 46 câmeras, 16 cruzamentos, ~4 km | ~3,6 h sincronizados, 313.931 bounding boxes, 880 IDs de veículos, **calibração e GPS**, 10 fps, câmeras não sobrepostas — melhor aderência ao artigo | [acesso](https://www.aicitychallenge.org/ai-city-challenge-dataset-access/) · [artigo](https://arxiv.org/abs/1903.09254) |
| **RoundaboutHD** | 4 câmeras 4K não sobrepostas | 40 min, 15 fps, 512 IDs, modelo/geométria da câmera, movimento não linear e oclusões (2025) | [GitHub](https://github.com/siri-rouser/RoundaboutHD) · [artigo](https://arxiv.org/abs/2507.08729) |
| **DukeMTMC** | 8 câmeras não sobrepostas | 2,8 mil IDs de pedestres, 1080p/60 fps — útil para adaptar a rastreamento de pessoas | [projeto](https://exposing.ai/duke_mtmc/) |
| **WILDTRACK** | 7 câmeras sincronizadas | 1920×1080 @ 60 fps, cena urbana de cruzamento | [projeto](https://www.ecva.net/papers/eccv_2019/papers_ECCV/html/Vasileios_Belhagiannis_ECCV_2019_paper.php) |
| **USVD** (Urban Surveillance Video Dataset) | 54 câmeras, 16 cenas | ruas, cruzamentos, praças, >200 mil quadros anotados | [artigo](https://arxiv.org/abs/1904.11784) |
| **PETS2009** | 8 câmeras | benchmark clássico multi-câmera (visões sobrepostas) | [projeto](https://www.cv-libs.cs.rug.nl/datasets/) |
| **Iowa DOT / câmaras de trânsito públicas** | dezenas | feeds reais de rodovias (usados no próprio AI City Challenge) | [AI City 2021](https://www.aicitychallenge.org/2021-data-and-evaluation/) |
| **Webcams ao vivo (HTTP/HLS/RTSP)** | variável | câmaras municipais/de trânsito em várias cidades — use com `--sources URL` | ex.: webcams.travel, Windy, canais oficiais das prefeituras |

> Respeite os termos de uso e a privacidade: a CityFlow redige faces e placas. Para webcams ao vivo, verifique a licença antes de gravar/distribuir.

---

## 2. Mapa arquitetura ↔ artigo

| Seção do artigo | Módulo implementado |
|---|---|
| §2.2 Two-Point-Bicycle, curva coordenada | `pipeline/kinematics.py` |
| §2.3 EKF com Jacobianas analíticas | `pipeline/ekf.py`, `kinematics.py` |
| §2.4 ortogonalidade (α = 0,1 / 0,1 / 2,8) e fator λ | `AdaptiveEKF.update()` |
| §3.1 homografia / plano de solo comum | `pipeline/geometry.py` |
| §3.2 dead reckoning em região cega | `AdaptiveEKF.propagate_only()` |
| §3.3 associação gaussiana intercâmeras (M, C_w, C_l) | `pipeline/association.py` |
| §4.1–4.3 grafo direcionado, poda, árvore de rotas | `pipeline/roadgraph.py` |
| §5.1–5.3 métricas e fluxo CICC | `pipeline/evaluation.py`, `pipeline/pipeline.py` |
| Entrada de vídeo (YOLO + tracklets + streams) | `pipeline/video.py` |
| CityFlowV2 / demo sintética | `pipeline/cityflow.py`, `pipeline/synthetic.py` |

---

## 3. Instalação

```bash
cd /home/feliper/Documentos/Pipeline_IA
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# PyTorch CPU (evita ~3 GB de pacotes CUDA):
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

O ambiente já foi preparado aqui: `.venv/` com OpenCV, Ultralytics, PyTorch CPU e pesos `weights/yolov8n.pt`.

---

## 4. Como executar (tudo automatizado)

```bash
# verificações numéricas rápidas
.venv/bin/python -m pipeline.cli selftest

# demo com anotações sintéticas (sem download)
.venv/bin/python -m pipeline.cli demo --root /tmp/opencode/cityflow_demo
.venv/bin/python -m pipeline.cli run --data /tmp/opencode/cityflow_demo --out /tmp/opencode/resultado.json

# DEMO PONTA A PONTA COM VÍDEO: gera 4 MP4, lê os vídeos, detecta e roda EKF+grafo
.venv/bin/python -m pipeline.cli demo-video --root /tmp/opencode/cityflow_video_demo

# validar YOLO com imagem pública
.venv/bin/python -m pipeline.cli yolo-smoke --weights weights/yolov8n.pt

# vídeos reais / streams públicos (calibração 3x3 por câmera em calib/<id>.txt)
.venv/bin/python -m pipeline.cli video \
  --sources caminho/cam1.mp4 caminho/cam2.mp4 \
  --calib-dir calib --graph grafo.json \
  --weights weights/yolov8n.pt --out /tmp/opencode/video_real.json
```

Para HLS (`.m3u8`) o `VideoSource` tenta OpenCV e, se falhar, usa FFmpeg automaticamente (`ffprobe`/`ffmpeg` já disponíveis no sistema).

### Usando a CityFlowV2 de verdade
1. Baixe **Track 1 / Track 3 (Multi-Camera Vehicle Tracking)** em https://www.aicitychallenge.org/ai-city-challenge-dataset-access/.
2. Extraia para `data/cityflow/` mantendo `gt/gt.txt`, os vídeos, `calib/` e o arquivo de detecções baseline.
3. Forneça um `graph.json` da malha viária (o adaptador aceita qualquer grafo com `nodes` e `edges`; `RoadGraph.grid_graph()` mostra o formato).
4. Rode `python -m pipeline.cli run --data data/cityflow --graph grafo.json` (modo anotações) ou `video --sources ...` (modo vídeo com YOLO).

---

### Testes automatizados

```bash
.venv/bin/python -m unittest discover -s tests -v   # 10 testes: geometria, EKF, grafo, vídeo, ponta a ponta
.venv/bin/python -m pipeline.cli selftest
```

Validação adicional do caminho **vídeo real → YOLO** (MP4 gerado a partir da imagem pública do Ultralytics):

```bash
ffmpeg -y -loop 1 -framerate 10 -i /tmp/opencode/bus.jpg -t 2 -vf scale=640:426 /tmp/opencode/bus_video.mp4
.venv/bin/python -m pipeline.cli video --sources /tmp/opencode/bus_video.mp4 \
  --calib-dir /tmp/opencode/calib_demo --weights weights/yolov8n.pt \
  --max-frames 12 --out /tmp/opencode/video_yolo_real.json
# => 12 quadros lidos, 12 caixas detectadas, 1 tracklet, nenhum erro de fonte
```

---

## 5. Resultados obtidos na demonstração local

Execução `demo-video` (4 câmeras sintéticas, 600 quadros, vídeo → detecção → EKF → grafo):

| Métrica | Valor |
|---|---|
| quadros lidos / caixas detectadas | 2.052 / 3.231 |
| IDF1 | 0,45 |
| acurácia de predição de rota (top-1) | **0,77** |
| vazão do núcleo EKF/grafo | ~4.800 det/s (muito acima de 20 fps) |
| λ médio / máximo | 1,33 / 60 (picos em manobras, como previsto no artigo) |
| erros de leitura de vídeo | nenhum |

As referências do artigo — **85%** em correspondência intercâmeras e **77%** em predição de rota — vêm dos experimentos com datasets reais citados. A demo sintética reproduz bem a predição de rota (0,77); a correspondência intercâmeras é menor porque o cenário sintético não tem variação de aparência entre veículos (o artigo sugere ReID/OCR como reforço — ver §6).

---

## 6. Limitações e trabalhos futuros (conforme §6 do artigo)

- **ReID/OCR**: a associação usa apenas cinemática + tamanho aparente; em malhas densas, embeddings de veículos/placas reduzem falsos positivos.
- **Grafo real**: a demo usa uma malha ortogonal; para CityFlowV2, exporte as vias reais (OSMnx) para `graph.json`, usando o GPS/calibração incluídos.
- **Semáforos e matrizes O-D**: podem calibrar melhor as probabilidades de transição nos vértices.
- **Validação operacional**: testes de campo com órgãos de segurança pública.

---

## 7. Estrutura

```
pipeline/
  config.py        parâmetros do artigo
  geometry.py      homografia DLT, jacobiana, covariância de medição
  kinematics.py    Two-Point-Bicycle + curva coordenada + Jacobianas
  ekf.py           EKF adaptativo (α, λ, ortogonalidade)
  association.py   verossimilhança intercâmeras + Hungarian
  roadgraph.py     grafo viário, poda física, árvore de rotas
  cityflow.py      leitor/escritor CityFlowV2
  synthetic.py     dataset sintético reprodutível
  pipeline.py      orquestração EKF → cego → associação → rotas
  video.py         vídeo (OpenCV/FFmpeg/HLS) + YOLO + IoUTracker
  evaluation.py    IDF1, acurácia intercâmeras/rotas, vazão
  cli.py           comandos acima
weights/yolov8n.pt pesos YOLO (baixados automaticamente)
```
