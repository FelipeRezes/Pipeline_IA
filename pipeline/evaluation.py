"""Métricas de avaliação: IDF1, acurácia de correspondência intercâmeras,
acurácia de predição de rota e desempenho computacional (Sec. 5 do artigo).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cityflow import CityFlowDataset
from .optimize import linear_sum_assignment
from .pipeline import PipelineResult

Array = np.ndarray


def map_tracklets_to_gt(ds: CityFlowDataset, tol_px: float = 30.0
                        ) -> Dict[Tuple[str, int], int]:
    """Associa cada tracklet local (câmera, id) ao ID GT pelo centro da bbox."""
    gt_by_frame: Dict[Tuple[str, int], List[tuple[int, Array]]] = defaultdict(list)
    for cam, objs in ds.gt.items():
        for oid, dets in objs.items():
            for d in dets:
                gt_by_frame[(cam, d.frame_id)].append(
                    (oid, np.array([d.x + 0.5 * d.w, d.y + d.h])))
    votes: Dict[Tuple[str, int], Counter] = defaultdict(Counter)
    for cam, dets in ds.detections.items():
        for d in dets:
            center = np.array([d.x + 0.5 * d.w, d.y + d.h])
            candidates = gt_by_frame.get((cam, d.frame_id), [])
            if not candidates:
                continue
            oid, dist = min(((oid, float(np.linalg.norm(center - c)))
                             for oid, c in candidates), key=lambda x: x[1])
            if dist <= tol_px:
                votes[(cam, d.track_id)][oid] += 1
    return {key: counter.most_common(1)[0][0]
            for key, counter in votes.items() if counter}


def compute_idf1(ds: CityFlowDataset, result: PipelineResult,
                 id_map: Dict[Tuple[str, int], int]) -> dict:
    """IDF1 simplificado: pareamento ótimo 1:1 entre IDs GT e IDs previstos."""
    pair_counts: Counter = Counter()
    gt_total = pred_total = 0
    for cam, objs in ds.gt.items():
        for dets in objs.values():
            gt_total += len(dets)
    for key, pid in result.assignments.items():
        pred_total += 1
        cam, frame, tid = key
        gid = id_map.get((cam, tid))
        if gid is not None:
            pair_counts[(gid, pid)] += 1

    if not pair_counts:
        return {"idf1": 0.0, "idtp": 0, "idfp": pred_total, "idfn": gt_total}

    gids = sorted({g for g, _ in pair_counts})
    pids = sorted({p for _, p in pair_counts})
    C = np.zeros((len(gids), len(pids)))
    for (g, p), n in pair_counts.items():
        C[gids.index(g), pids.index(p)] = n
    rows, cols = linear_sum_assignment(-C)
    idtp = int(sum(C[i, j] for i, j in zip(rows, cols)))
    idfp, idfn = pred_total - idtp, gt_total - idtp
    denom = 2 * idtp + idfp + idfn
    return {"idf1": round(2 * idtp / denom, 4) if denom else 0.0,
            "idtp": idtp, "idfp": idfp, "idfn": idfn}


def compute_cross_camera_accuracy(ds: CityFlowDataset, result: PipelineResult,
                                  id_map: Dict[Tuple[str, int], int]) -> dict:
    """Acurácia de correspondência intercâmeras (cf. 85% do artigo).

    Para cada ID GT visto em >=2 câmeras, toma-se o ID previsto modal em cada
    câmera e verifica-se se todas as câmeras concordam. A modalidade por câmera
    isola a associação intercâmeras de eventuais reaquisições dentro da mesma
    câmera (que são tratadas pelo rastreador local / IDF1).
    """
    per_gt_cam: Dict[int, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for (cam, frame, tid), pid in result.assignments.items():
        gid = id_map.get((cam, tid))
        if gid is not None:
            per_gt_cam[gid][cam][pid] += 1
    multi = [g for g, cams in per_gt_cam.items() if len(cams) >= 2]
    correct = 0
    for g in multi:
        modal = {cam: counter.most_common(1)[0][0]
                 for cam, counter in per_gt_cam[g].items()}
        if len(set(modal.values())) == 1:
            correct += 1
    return {"cross_camera_accuracy": round(correct / len(multi), 4) if multi else 0.0,
            "objects_evaluated": len(multi), "objects_correct": correct}


def compute_route_accuracy(result: PipelineResult) -> dict:
    """Acerto de rota nas transições entre câmeras (cf. 77% do artigo).

    Reaquisições na mesma câmera não são transições cegas intercâmeras e por isso
    são excluídas do denominador.
    """
    events = [m for m in result.matches if m.cross_camera]
    n = len(events)
    top1 = sum(m.top1_correct for m in events)
    topk = sum(m.topk_correct for m in events)
    return {"route_top1_accuracy": round(top1 / n, 4) if n else 0.0,
            "route_topk_accuracy": round(topk / n, 4) if n else 0.0,
            "blind_transitions": n, "top1_hits": top1, "topk_hits": topk,
            "same_camera_reacquisitions": len(result.matches) - n}


def compute_lambda_stats(result: PipelineResult) -> dict:
    vals = [x for t in result.tracks.values() for x in t.ekf.lambda_history]
    if not vals:
        return {"lambda_mean": 1.0, "lambda_max": 1.0,
                "maneuver_fraction": 0.0, "n_updates": 0}
    a = np.asarray(vals)
    return {"lambda_mean": round(float(a.mean()), 3),
            "lambda_max": round(float(a.max()), 3),
            "maneuver_fraction": round(float((a > 1.05).mean()), 4),
            "n_updates": len(vals)}


def evaluate(ds: CityFlowDataset, result: PipelineResult) -> dict:
    """Relatório completo de desempenho da pipeline."""
    id_map = map_tracklets_to_gt(ds)
    report: dict = {}
    report.update(compute_idf1(ds, result, id_map))
    report.update(compute_cross_camera_accuracy(ds, result, id_map))
    report.update(compute_route_accuracy(result))
    report.update(compute_lambda_stats(result))
    report.update({
        "elapsed_s": round(result.elapsed_s, 3),
        "throughput_fps": round(result.fps_processed, 1),
        "n_predictions": result.n_predictions,
        "n_tracks": len(result.tracks),
    })
    return report
