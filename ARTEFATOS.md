# Artefatos gerados (vídeos e resultados)

Todos os arquivos abaixo estão na raiz deste projeto.

## Vídeos reais (YOLOv8n + IoUTracker + EKF/grafo)

| Arquivo | Descrição |
|---|---|
| `kitti_inference.mp4` | clipe público de KITTI (cena urbana real, 16,6 s, 1226×370, 30 fps) |
| `kitti_yolo_tracks.mp4` | mesmo clipe com caixas e IDs dos veículos desenhados |
| `kitti_result.json` | estatísticas e rotas: 498 quadros, 2.897 caixas, 77 tracklets |
| `anpr_traffic.mp4` | clipe real de câmera de trânsito (5,6 s, 1080×1920, ~30 fps) |
| `anpr_yolo_tracks.mp4` | clipe ANPR anotado com caixas/IDs |
| `anpr_result.json` | estatísticas e rotas: 168 quadros, 572 caixas, 9 tracklets |

Fonte dos clipes: assets públicos Ultralytics (release `v0.0.0`).

Para reproduzir:

```bash
.venv/bin/python -m pipeline.cli video \
  --sources kitti_inference.mp4 \
  --calib-dir calib --graph grafo_demo.json \
  --weights weights/yolov8n.pt --out kitti_result.json
```

`annotate_real.py` gera as versões anotadas (use `PYTHONPATH=.` ao executar).

## Demo sintética ponta a ponta

| Arquivo | Descrição |
|---|---|
| `montagem_4cam.mp4` | 4 câmeras sintéticas lado a lado (1280×960, 67,6 s) |
| `11.mp4`, `12.mp4`, `21.mp4`, `22.mp4` | vídeos individuais das câmeras |
| `video_result.json` | métricas da demo (2.704 quadros, 3.231 caixas) |
| `grafo_demo.json` | grafo viário usado pela demo e pelos clipes reais |

Para reproduzir a demo:

```bash
.venv/bin/python -m pipeline.cli demo-video \
  --root /tmp/opencode/cityflow_video_demo \
  --out video_result.json
```

> A calibração e o grafo da demo são sintéticos; as rotas previstas nos clipes reais são ilustrativas, não uma avaliação com ground truth.
