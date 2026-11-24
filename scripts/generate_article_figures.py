"""Generate publication-grade 300-DPI figures for the revised scientific article from real benchmark results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib

matplotlib.use("Agg")
from matplotlib import patches
import matplotlib.pyplot as plt
import numpy as np

from tracker.city_benchmark import build_city_road_network, place_non_overlapping_cameras
from tracker.roadgraph import _interpolate_polyline

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def make_fig1_architecture(out_path: Path) -> None:
    """Clean 6-stage architecture diagram for urban ITS multi-camera tracking (zero police references)."""
    fig, ax = plt.subplots(figsize=(11.2, 4.6), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    ax.text(
        6.0,
        4.88,
        "Arquitetura Modular Causal do Pipeline Multicâmera de Monitoramento Viário",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1b365d",
    )

    boxes = [
        (
            0.35,
            2.65,
            3.4,
            1.75,
            "#e8f1fa",
            "#2c6fb7",
            "1. Ingestão e Percepção Local",
            "• Vídeos MP4 / RTSP / Sequências RGB\n• Detecção veicular YOLO11n (COCO)\n• Rastreamento local ByteTrack/BoT-SORT\n• Polígonos de zonas de saída (Exit Zones)",
        ),
        (
            4.30,
            2.65,
            3.4,
            1.75,
            "#eaf7f1",
            "#2a9d8f",
            "2. Geometria Projetiva 2D/3D",
            "• Homografia DLT/RANSAC ou $K[R\\mid t]$\n• Correção de ponto de contato 3D ($f=0{,}90$)\n• Conversão WGS84 $\\leftrightarrow$ UTM métrico\n• Propagação analítica $R_k = J\\,\\Sigma_{\\mathrm{px}}\\,J^\\top$",
        ),
        (
            8.25,
            2.65,
            3.4,
            1.75,
            "#fef4e8",
            "#e07a24",
            "3. Filtragem Cinemática Adaptativa",
            "• Modelo Two-Point-Bicycle 5D $[X,Y,v,\\theta,\\delta]$\n• Suavização circular de $\\theta$ (4 quadros)\n• Gatilho NIS ($\\chi^2$ 95%, $\\geq 2$ quadros)\n• Fator $\\lambda_k$ estabilizado + Forma de Joseph",
        ),
        (
            8.25,
            0.35,
            3.4,
            1.75,
            "#f3eefb",
            "#6f42c1",
            "4. Acoplamento Topológico (Grafo)",
            "• Malha viária OSM / UTM direcionada\n• Map-matching HMM / Viterbi\n• Poda física por $v_{\\mathrm{crit}} = \\sqrt{\\mu g R_{\\mathrm{curva}}}$\n• Árvore multi-hipótese em regiões cegas",
        ),
        (
            4.30,
            0.35,
            3.4,
            1.75,
            "#e6f4f8",
            "#1d788b",
            "5. Associação Híbrida Intercâmeras",
            "• Gating espaço-temporal χ² multi-hipótese\n• Descritor Deep Re-ID (OSNet 512-D)\n• Leitura seletiva ALPR + autocalibração\n• Atribuição global 1:1 (Algoritmo Húngaro)",
        ),
        (
            0.35,
            0.35,
            3.4,
            1.75,
            "#fdeef0",
            "#c0392b",
            "6. Gestão de Tráfego e Análise (ITS)",
            "• Unificação de ID global multicâmera\n• Reconstrução retrospectiva (backtracking)\n• Nós críticos de cobertura e ETA\n• Painel analítico split-screen e JSON",
        ),
    ]

    for x, y, w, h, fc, ec, title, body in boxes:
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.08,rounding_size=0.14",
            linewidth=1.5,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h - 0.24, title, ha="center", va="center", fontsize=9.3, fontweight="bold", color=ec)
        ax.text(x + 0.14, y + h - 0.50, body, ha="left", va="top", fontsize=7.8, color="#222222", linespacing=1.35)

    arrow_kw = {"arrowstyle": "-|>", "mutation_scale": 14, "lw": 1.8, "color": "#3b4d61"}
    ax.annotate("", xy=(4.26, 3.52), xytext=(3.80, 3.52), arrowprops=arrow_kw)
    ax.annotate("", xy=(8.21, 3.52), xytext=(7.75, 3.52), arrowprops=arrow_kw)
    ax.annotate("", xy=(9.95, 2.15), xytext=(9.95, 2.60), arrowprops=arrow_kw)
    ax.annotate("", xy=(7.75, 1.22), xytext=(8.21, 1.22), arrowprops=arrow_kw)
    ax.annotate("", xy=(3.80, 1.22), xytext=(4.26, 1.22), arrowprops=arrow_kw)

    fig.tight_layout(pad=0.3)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig2_sota_kinematics(data: dict, out_path: Path) -> None:
    """3-panel SOTA comparison figure: (a) Maneuver & Blind RMSE, (b) Pareto Latency vs RMSE, (c) A2 Noise Robustness."""
    kin = data["A6"]["results"]["kinematic_and_sequence_models"]
    a2 = data["A2"]["results"]

    order = [
        ("linear_cv", "Kalman CV"),
        ("canonical_bicycle", "EKF Canônico"),
        ("adaptive_bicycle", "EKF Adapt."),
        ("ctrv", "EKF CTRV"),
        ("imm", "IMM"),
        ("particle_filter", "Partículas (SIR)"),
        ("lstm", "LSTM"),
        ("gru", "GRU"),
    ]

    labels = [lbl for _, lbl in order]
    man_means = [kin[k]["rmse_maneuver_mean_m"] for k, _ in order]
    man_stds = [kin[k]["rmse_maneuver_std_m"] for k, _ in order]
    bld_means = [kin[k]["rmse_blind_outage_mean_m"] for k, _ in order]
    bld_stds = [kin[k]["rmse_blind_outage_std_m"] for k, _ in order]
    lats = [kin[k]["latency_ms_per_step"] for k, _ in order]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3), dpi=300)

    # Panel (a): Grouped bars (Maneuver RMSE and Blind-Outage RMSE)
    x = np.arange(len(labels))
    w = 0.37
    axes[0].bar(x - w / 2, man_means, width=w, yerr=man_stds, capsize=2.5, color="#2c6fb7", label="RMSE Manobra Brusca (m)")
    axes[0].bar(x + w / 2, bld_means, width=w, yerr=bld_stds, capsize=2.5, color="#e07a24", label="RMSE Região Cega (m)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=32, ha="right", fontsize=8.0)
    axes[0].set_ylabel("RMSE de Posição (m)")
    axes[0].set_title("(a) Erro em Manobra e Região Cega (5 seeds)")
    axes[0].legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # Panel (b): Pareto Frontier (Latency vs Blind RMSE) with non-overlapping offset labels
    colors_p = ["#7f8c8d", "#5b84b1", "#1e8449", "#2e86c1", "#117a65", "#8e44ad", "#c0392b", "#d35400"]
    markers_p = ["s", "o", "*", "D", "P", "^", "v", "X"]
    offsets_pts = {
        "linear_cv": (8, -2),
        "canonical_bicycle": (8, 4),
        "adaptive_bicycle": (8, 2),
        "ctrv": (8, -6),
        "imm": (-58, 14),
        "particle_filter": (-78, -4),
        "lstm": (-46, 10),
        "gru": (-42, -16),
    }
    for idx, (k, lbl) in enumerate(order):
        ms = 120 if k in ("adaptive_bicycle", "imm") else 68
        axes[1].scatter(
            lats[idx],
            bld_means[idx],
            s=ms,
            color=colors_p[idx],
            marker=markers_p[idx],
            edgecolor="black",
            linewidth=0.6,
            zorder=4,
        )
        ox_pt, oy_pt = offsets_pts[k]
        axes[1].annotate(
            lbl,
            xy=(lats[idx], bld_means[idx]),
            xytext=(ox_pt, oy_pt),
            textcoords="offset points",
            fontsize=7.6,
            fontweight="bold" if k in ("adaptive_bicycle", "imm") else "normal",
            color="#1f2933",
            arrowprops={"arrowstyle": "-", "color": "#888888", "lw": 0.6} if abs(ox_pt) > 20 else None,
        )
    axes[1].set_xscale("log")
    axes[1].set_xlim(min(lats) * 0.80, max(lats) * 1.45)
    axes[1].set_ylim(3.2, max(bld_means) * 1.12)
    axes[1].set_xlabel("Latência Média por Passo (ms, escala log)")
    axes[1].set_ylabel("RMSE em Região Cega (m)")
    axes[1].set_title("(b) Fronteira Acurácia vs. Latência (CPU)")

    # Panel (c): A2 Noise stabilization
    sigmas = [v["sigma_m"] for v in a2.values()]
    rmse_a = [v["adaptive_maneuver_rmse_m"] for v in a2.values()]
    rmse_c = [v["canonical_maneuver_rmse_m"] for v in a2.values()]
    axes[2].plot(sigmas, rmse_c, marker="o", lw=2.0, color="#c0392b", linestyle="--", label="EKF Canônico (λ=1)")
    axes[2].plot(sigmas, rmse_a, marker="s", lw=2.2, color="#1e8449", label="EKF Adaptativo (Proposto)")
    axes[2].fill_between(sigmas, rmse_a, rmse_c, color="#1e8449", alpha=0.12)
    axes[2].set_xlabel("Ruído de Medição no Solo σ (m)")
    axes[2].set_ylabel("RMSE na Manobra (m)")
    axes[2].set_title("(c) Estabilidade sob Ruído Crescente (A2)")
    axes[2].legend(loc="upper left", frameon=True, facecolor="white")

    fig.tight_layout(pad=1.0)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig3_route_tgnn(data: dict, out_path: Path) -> None:
    """2-panel figure comparing road-graph route prediction models including T-GNN."""
    rt = data["A6"]["results"]["road_graph_route_models"]
    order = [
        ("euclidean_snap", "Projeção\nEuclidiana"),
        ("hmm_viterbi", "HMM /\nViterbi"),
        ("t_gnn", "Rede T-GNN\n(GCN+GRU)"),
        ("proposed_physics_tree", "Árvore Física\n(Proposto)"),
    ]
    labels = [lbl for _, lbl in order]
    top1 = [rt[k]["top1_route_accuracy"] * 100.0 for k, _ in order]
    topk = [rt[k]["topk_route_accuracy"] * 100.0 for k, _ in order]
    rmse = [rt[k]["blind_position_rmse_m"] for k, _ in order]
    lat = [rt[k]["latency_ms_per_query"] for k, _ in order]

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), dpi=300)

    x = np.arange(len(labels))
    w = 0.35
    b1 = axes[0].bar(x - w / 2, top1, width=w, color="#2c6fb7", label="Acurácia de Rota Top-1 (%)")
    b2 = axes[0].bar(x + w / 2, topk, width=w, color="#2a9d8f", label="Acurácia de Rota Top-2 (%)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.8)
    axes[0].set_ylim(0, 114)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Acurácia de Predição de Rota (%)")
    axes[0].set_title("(a) Predição de Transição em Cruzamentos Urbanos (Malha 8×8)")
    axes[0].legend(loc="upper left", frameon=True, facecolor="white")

    # Panel (b): Blind-zone position RMSE + Latency annotation
    bar_colors = ["#7f8c8d", "#5b84b1", "#8e44ad", "#1e8449"]
    bars_r = axes[1].bar(x, rmse, width=0.48, color=bar_colors)
    for idx, bar in enumerate(bars_r):
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            h + max(rmse) * 0.03,
            f"{h:.1f} m\n({lat[idx]:.2f} ms)",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )
    axes[1].set_ylim(0, max(rmse) * 1.28)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("RMSE de Posição após 4,0 s em Região Cega (m)")
    axes[1].set_title("(b) Erro Espacial em Região Cega e Latência Média")

    fig.tight_layout(pad=1.0)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig4_city_scalability(data: dict, out_path: Path) -> None:
    """2-panel figure showing City-Scale scalability across 4-24 cameras and 20-150 vehicles (A7)."""
    a7 = data["A7"]["results"]
    scales = [
        ("S1_4cam_20veh", "4 câm.\n20 veíc."),
        ("S2_8cam_50veh", "8 câm.\n50 veíc."),
        ("S3_16cam_100veh", "16 câm.\n100 veíc."),
        ("S4_24cam_150veh", "24 câm.\n150 veíc."),
    ]
    x = np.arange(len(scales))
    x_labels = [lbl for _, lbl in scales]

    mode_specs = [
        ("article", "Limiar Legado (M ≤ 350)", "#7f8c8d", "--", "o"),
        ("chi2", "Gating χ² (Cinemática + Grafo)", "#2c6fb7", "-.", "s"),
        ("reid_only", "Re-ID Visual Puro (OSNet)", "#e07a24", ":", "^"),
        ("hybrid", "Fusão Híbrida (Proposto)", "#1e8449", "-", "D"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1), dpi=300)

    for m_key, m_lbl, col, ls, mk in mode_specs:
        vals = [a7[sk]["modes"][m_key]["idf1"] for sk, _ in scales]
        lw = 2.6 if m_key == "hybrid" else 1.8
        axes[0].plot(x, vals, label=m_lbl, color=col, linestyle=ls, marker=mk, linewidth=lw, markersize=6.5)
        if m_key in ("hybrid", "reid_only", "chi2"):
            for xi, yi in zip(x, vals):
                off = 0.032 if m_key == "hybrid" else (-0.068 if m_key == "chi2" else -0.058)
                axes[0].text(xi, min(1.06, max(0.04, yi + off)), f"{yi:.3f}", ha="center", fontsize=7.4, color=col, fontweight="bold")

    axes[0].set_ylim(0.0, 1.12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_ylabel("IDF1 Global Intercâmeras")
    axes[0].set_title("(a) Escalabilidade de Associação Multicâmera (4 a 24 Câmeras)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=True, facecolor="white", fontsize=7.8)

    # Panel (b): Hybrid Route Top-1/Top-k + Throughput (det/s)
    hyb_acc = [a7[sk]["modes"]["hybrid"]["cross_camera_accuracy"] * 100.0 for sk, _ in scales]
    hyb_top1 = [a7[sk]["modes"]["hybrid"]["route_top1_accuracy"] * 100.0 for sk, _ in scales]
    hyb_topk = [a7[sk]["modes"]["hybrid"]["route_topk_accuracy"] * 100.0 for sk, _ in scales]
    hyb_tput = [a7[sk]["modes"]["hybrid"]["throughput_detections_s"] for sk, _ in scales]

    w = 0.25
    axes[1].bar(x - w, hyb_acc, width=w, color="#1e8449", label="Acurácia Intercâmeras (%)")
    axes[1].bar(x, hyb_top1, width=w, color="#2c6fb7", label="Rota Top-1 (%)")
    axes[1].bar(x + w, hyb_topk, width=w, color="#2a9d8f", label="Rota Top-k (%)")
    axes[1].set_ylim(0, 118)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_ylabel("Desempenho (%)")
    axes[1].set_title("(b) Desempenho do Modo Híbrido e Vazão Computacional")

    ax2 = axes[1].twinx()
    ax2.plot(x, hyb_tput, color="#c0392b", marker="o", lw=2.0, linestyle="-", label="Vazão (detecções/s)")
    ax2.set_ylabel("Vazão do Núcleo (detecções/s)", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_ylim(0, max(hyb_tput) * 1.35)
    ax2.spines["top"].set_visible(False)

    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.8, frameon=True, facecolor="white")

    fig.tight_layout(pad=1.0)
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_fig5_city_topology(out_path: Path) -> None:
    """Plot the actual 8x8 urban UTM road network (64 nodes) and 24 non-overlapping cameras."""
    net, _ = build_city_road_network(grid_rows=8, grid_cols=8, block_m=120.0)
    cams = place_non_overlapping_cameras(net, num_cameras=24, seed=700 + 3 * 13)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    ox, oy = 330000.0, 7390000.0

    plotted_labels: set[str] = set()
    for edge in net.edges.values():
        pts = edge.polyline_utm
        xs = pts[:, 0] - ox
        ys = pts[:, 1] - oy
        if edge.lanes >= 3:
            col, lw, lbl = "#1b4f72", 2.4, "Via Arterial (60 km/h, 3 faixas)"
        elif edge.lanes == 2:
            col, lw, lbl = "#5d8aa8", 1.5, "Via Coletora (50 km/h, 2 faixas)"
        else:
            col, lw, lbl = "#aab7b8", 1.0, "Via Local Mão Única (40 km/h)"
        ax.plot(xs, ys, color=col, linewidth=lw, alpha=0.85, label=lbl if lbl not in plotted_labels else None, zorder=1)
        plotted_labels.add(lbl)

    # Intersections
    nx_arr = [xy[0] - ox for xy in net.nodes_utm.values()]
    ny_arr = [xy[1] - oy for xy in net.nodes_utm.values()]
    ax.scatter(nx_arr, ny_arr, s=16, color="#2c3e50", zorder=2, label="Intersecções (64 vértices)")

    # 24 non-overlapping cameras
    cam_xs, cam_ys = [], []
    alpr_xs, alpr_ys = [], []
    for cam in cams:
        edge = net.edges[cam.edge_id]
        cx, cy, _ = _interpolate_polyline(edge.polyline_utm, 0.5 * (cam.s_start_m + cam.s_end_m))
        if cam.alpr_capable:
            alpr_xs.append(cx - ox)
            alpr_ys.append(cy - oy)
        else:
            cam_xs.append(cx - ox)
            cam_ys.append(cy - oy)

    ax.scatter(
        cam_xs,
        cam_ys,
        s=68,
        marker="^",
        color="#e67e22",
        edgecolor="black",
        linewidth=0.7,
        zorder=4,
        label="Câmeras Estáticas (Re-ID + EKF)",
    )
    ax.scatter(
        alpr_xs,
        alpr_ys,
        s=85,
        marker="s",
        color="#c0392b",
        edgecolor="black",
        linewidth=0.7,
        zorder=5,
        label="Câmeras com Leitura ALPR (25%)",
    )

    ax.set_xlabel("Coordenada Relativa UTM Leste ΔX (m)")
    ax.set_ylabel("Coordenada Relativa UTM Norte ΔY (m)")
    ax.set_title("Malha Viária Urbana 8×8 e Disposição de 24 Câmeras Não Sobrepostas")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=True, facecolor="white", fontsize=8.0)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    bench_json = Path("data/benchmarks/ablation_results.json")
    data = json.loads(bench_json.read_text(encoding="utf-8"))
    fig_dir = Path("artigo/figuras")
    fig_dir.mkdir(parents=True, exist_ok=True)

    make_fig1_architecture(fig_dir / "fig1_arquitetura.png")
    make_fig2_sota_kinematics(data, fig_dir / "fig2_sota_cinematica.png")
    make_fig3_route_tgnn(data, fig_dir / "fig3_rotas_tgnn.png")
    make_fig4_city_scalability(data, fig_dir / "fig4_escala_urbana.png")
    make_fig5_city_topology(fig_dir / "fig5_mapa_urbano.png")
    print(f"Generated 5 publication figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
