"""Interface de linha de comando da pipeline.

Exemplos:
    python -m pipeline.cli demo --root /tmp/opencode/cityflow_demo
    python -m pipeline.cli run  --data /tmp/opencode/cityflow_demo --out /tmp/opencode/resultado.json
    python -m pipeline.cli selftest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

from .cityflow import load_cityflow
from .config import PipelineConfig
from .evaluation import evaluate
from .pipeline import SurveillancePipeline
from .roadgraph import RoadGraph
from .synthetic import generate_synthetic_cityflow


def _load_graph(data_root: str, graph_arg: Optional[str]) -> RoadGraph:
    root = Path(data_root).expanduser()
    path = Path(graph_arg).expanduser() if graph_arg else None
    if path is None:
        meta = root / "meta.json"
        if meta.is_file():
            gp = json.loads(meta.read_text(encoding="utf-8")).get("graph_path")
            if gp:
                path = root / gp
    if path is None or not path.is_file():
        found = sorted(root.glob("graph.json"))
        path = found[0] if found else None
    if path is None:
        raise FileNotFoundError("grafo viário não encontrado (use --graph)")
    return RoadGraph.load_json(str(path))


def cmd_demo(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    ds, graph = generate_synthetic_cityflow(
        str(root), n_vehicles=args.vehicles, n_frames=args.frames, seed=args.seed)
    print(f"Dataset sintético gravado em {root}")
    print(ds.summary())
    print(f"Grafo: {len(graph.nodes)} vértices, {len(graph.edges)} arestas "
          f"(cameras: {', '.join(sorted(ds.calibrations))})")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = PipelineConfig(fps=args.fps, dt=1.0 / args.fps,
                         association_mode=args.assoc_mode)
    cfg.validate()
    ds = load_cityflow(args.data, detections_file=args.detections,
                       gt_file=args.gt, fps=args.fps)
    graph = _load_graph(args.data, args.graph)
    if args.max_frames > 0:
        limit = args.max_frames
        for cam in list(ds.detections):
            ds.detections[cam] = [d for d in ds.detections[cam] if d.frame_id <= limit]
    print("Entrada:", ds.summary())
    result = SurveillancePipeline(ds, graph, cfg).run()
    report = evaluate(ds, result)
    output = {"metrics": report,
              "matches": result.match_report(),
              "routes": result.route_report()[:args.max_routes_report]}
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"Resultado salvo em {out}")
    print("\n=== Métricas ===")
    for k, v in report.items():
        print(f"{k:28s} {v}")
    return 0


def _load_calibrations(calib_dir: str, camera_ids: Iterable[str]) -> dict:
    """Lê matrizes 3x3 de ``calib/<cam>.txt`` para câmeras de vídeo arbitrárias."""
    from .cityflow import Calibration, _default_calibration, _read_calibration_file
    from pathlib import Path
    root = Path(calib_dir).expanduser()
    out = {}
    for cam in camera_ids:
        candidates = [root / f"{cam}.txt", root / f"cam_{cam}.txt",
                      root / f"{int(cam):03d}.txt"]
        H = _default_calibration()
        for c in candidates:
            if c.is_file():
                _, H = _read_calibration_file(c, cam)
                break
        out[cam] = Calibration(cam, H)
    return out


def cmd_demo_video(args: argparse.Namespace) -> int:
    """Gera vídeos MP4 a partir da demo sintética e executa a arquitetura completa."""
    from .config import PipelineConfig
    from .evaluation import evaluate
    from .synthetic import generate_synthetic_cityflow
    from .video import (AnnotationDetector, VideoAnalytics, VideoSource,
                        render_videos)

    root = Path(args.root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    ds, graph = generate_synthetic_cityflow(
        str(root), n_vehicles=args.vehicles, n_frames=args.frames, seed=args.seed)
    print("Renderizando vídeos por câmera...")
    paths = render_videos(ds, str(root))
    sources = [VideoSource(cam, p, fps=ds.fps, max_frames=args.max_frames)
               for cam, p in sorted(paths.items())]
    cfg = PipelineConfig(fps=ds.fps, dt=1.0 / ds.fps)
    analytics = VideoAnalytics(sources, dict(ds.calibrations),
                               AnnotationDetector(ds), cfg, graph)
    ds_video, result = analytics.run()
    ds_video.gt = ds.gt  # rótulos para avaliação (não usados na detecção)
    metrics = evaluate(ds_video, result)
    print("\n=== Estatísticas de vídeo ===")
    for k, v in analytics.stats.as_dict().items():
        print(f"{k:20s} {v}")
    print("\n=== Métricas (vídeo -> EKF -> grafo) ===")
    for k, v in metrics.items():
        print(f"{k:28s} {v}")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"video": analytics.stats.as_dict(),
                               "metrics": metrics,
                               "matches": result.match_report()},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado salvo em {out}")
    return 0


def cmd_video(args: argparse.Namespace) -> int:
    """Executa a arquitetura sobre vídeos reais/streams públicos com YOLO."""
    from .config import PipelineConfig
    from .video import YOLODetector, VideoAnalytics, VideoSource

    sources: list = []
    for i, src in enumerate(args.sources):
        sources.append(VideoSource(str(i + 1), src, fps=args.fps or None,
                                   max_frames=args.max_frames))
    cams = [s.camera_id for s in sources]
    calibs = _load_calibrations(args.calib_dir, cams)
    cfg = PipelineConfig(fps=args.fps or 10.0, dt=1.0 / (args.fps or 10.0))
    graph = RoadGraph.load_json(args.graph) if args.graph else None
    detector = YOLODetector(weights=args.weights, conf=args.conf,
                            imgsz=args.imgsz, device=args.device)
    analytics = VideoAnalytics(sources, calibs, detector, cfg, graph)
    ds, result = analytics.run()
    print("Vídeos processados:", ds.summary())
    print("Estatísticas:", analytics.stats.as_dict())
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"video": analytics.stats.as_dict(),
                   "detections": sum(len(v) for v in ds.detections.values()),
                   "result": ({"matches": result.match_report()[:50],
                               "routes": result.route_report()[:10]}
                              if result else None)}
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"Resultado salvo em {out}")
    return 0


def cmd_yolo_smoke(args: argparse.Namespace) -> int:
    """Baixa uma imagem pública e confirma que os pesos YOLO detectam veículos."""
    from urllib.request import urlretrieve
    from .video import YOLODetector
    dest = Path(args.image)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        print("Baixando imagem de validação...")
        urlretrieve("https://ultralytics.com/images/bus.jpg", str(dest))
    det = YOLODetector(weights=args.weights, conf=0.25)
    import cv2
    img = cv2.imread(str(dest))
    boxes = det.detect(img)
    print(f"YOLO: {len(boxes)} veículos detectados em {dest}")
    print([(round(b.x), round(b.y), round(b.w), round(b.h), round(b.conf, 2))
           for b in boxes[:10]])
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    import numpy as np

    from .ekf import AdaptiveEKF
    from .geometry import apply_homography, fit_homography
    from .kinematics import bicycle_step
    from .optimize import linear_sum_assignment, chi2_ppf

    src = np.array([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40], [20, 60]], float)
    H_true = np.array([[1.2, 0.1, 10.0], [0.05, 0.9, -5.0],
                       [0.001, 0.0005, 1.0]])
    dst = apply_homography(H_true, src)
    H = fit_homography(src, dst)
    err = np.linalg.norm(apply_homography(H, src) - dst, axis=1).max()
    assert err < 1e-6, f"homografia: erro {err}"

    x = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    xp = bicycle_step(x, 0.1, 2.5)
    assert abs(xp[0] - 1.0) < 1e-9 and abs(xp[3]) < 1e-9

    ekf = AdaptiveEKF(x, cfg=PipelineConfig())
    for _ in range(10):
        ekf.predict()
        ekf.update(ekf.position + np.array([xp[0], 0.0]), np.eye(2) * 0.1)
    assert np.all(np.isfinite(ekf.x)) and np.all(np.isfinite(ekf.P))

    r, c = linear_sum_assignment(np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]]))
    assert sorted(zip(r.tolist(), c.tolist())) == [(0, 1), (1, 0), (2, 2)]
    assert abs(chi2_ppf(0.99, 4) - 13.2767) < 0.02
    print("Selftest: OK (homografia, cinemática, EKF, Hungarian, qui-quadrado)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("demo", help="gera dataset sintético no formato CityFlow")
    d.add_argument("--root", default="/tmp/opencode/cityflow_demo")
    d.add_argument("--vehicles", type=int, default=14)
    d.add_argument("--frames", type=int, default=900)
    d.add_argument("--seed", type=int, default=20261110)
    d.set_defaults(func=cmd_demo)

    r = sub.add_parser("run", help="executa a pipeline e imprime as métricas")
    r.add_argument("--data", required=True, help="diretório no formato CityFlow")
    r.add_argument("--graph", default=None, help="grafo viário JSON")
    r.add_argument("--detections", default=None)
    r.add_argument("--gt", default=None)
    r.add_argument("--fps", type=float, default=10.0)
    r.add_argument("--assoc-mode", choices=["chi2", "article"], default="chi2")
    r.add_argument("--max-frames", type=int, default=0)
    r.add_argument("--out", default=None)
    r.add_argument("--max-routes-report", type=int, default=10)
    r.set_defaults(func=cmd_run)

    dv = sub.add_parser("demo-video", help="gera vídeos e valida arquitetura ponta a ponta")
    dv.add_argument("--root", default="/tmp/opencode/cityflow_video_demo")
    dv.add_argument("--vehicles", type=int, default=20)
    dv.add_argument("--frames", type=int, default=1200)
    dv.add_argument("--seed", type=int, default=20261110)
    dv.add_argument("--max-frames", type=int, default=0)
    dv.add_argument("--out", default="/tmp/opencode/video_result.json")
    dv.set_defaults(func=cmd_demo_video)

    v = sub.add_parser("video", help="processa vídeos/streams públicos com YOLO")
    v.add_argument("--sources", nargs="+", required=True,
                   help="arquivos MP4, URLs HTTP(S), RTSP ou HLS (.m3u8)")
    v.add_argument("--calib-dir", default="calib",
                   help="diretório com H 3x3 por câmera")
    v.add_argument("--graph", default=None, help="grafo viário JSON")
    v.add_argument("--weights", default="yolov8n.pt")
    v.add_argument("--conf", type=float, default=0.35)
    v.add_argument("--imgsz", type=int, default=640)
    v.add_argument("--device", default="cpu")
    v.add_argument("--fps", type=float, default=None)
    v.add_argument("--max-frames", type=int, default=0)
    v.add_argument("--out", default=None)
    v.set_defaults(func=cmd_video)

    ys = sub.add_parser("yolo-smoke", help="valida detector YOLO com imagem pública")
    ys.add_argument("--weights", default="yolov8n.pt")
    ys.add_argument("--image", default="/tmp/opencode/bus.jpg")
    ys.set_defaults(func=cmd_yolo_smoke)

    s = sub.add_parser("selftest", help="verificações numéricas rápidas")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
