"""Build the revised, publication-grade scientific article (HTML + WeasyPrint PDF).

Reads exact experimental metrics directly from data/benchmarks/ablation_results.json,
embeds the 300-DPI figures from artigo/figuras/, enforces strict quality gates
(zero encoding noise, zero police terminology, zero source code in Sections 1-7),
and compiles artigo/Artigo_Rastreamento_Multicamera_EKF_Adaptativo_Redes_Urbanas.pdf.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from weasyprint import HTML


def _fmt_pt(val: float, decimals: int = 4) -> str:
    """Format floating-point number using Portuguese decimal comma."""
    return f"{val:.{decimals}f}".replace(".", ",")


def _fmt_en(val: float, decimals: int = 4) -> str:
    """Format floating-point number using English decimal dot."""
    return f"{val:.{decimals}f}"


def _fmt_int_pt(val: int | float) -> str:
    """Format integer with Portuguese thousands dot separator."""
    return f"{int(round(val)):,}".replace(",", ".")


def _fmt_int_en(val: int | float) -> str:
    """Format integer with English thousands comma separator."""
    return f"{int(round(val)):,}"


def _audit_article_compliance(html_text: str) -> None:
    """Verify strict adherence to Professor Harold's guidelines before PDF compilation."""
    # 1. Zero UTF-8 replacement characters or raw LaTeX/ASCII noise
    if "\ufffd" in html_text:
        raise ValueError("Encoding noise detected: U+FFFD replacement character present in HTML!")

    # Strip HTML tags for text audit
    plain = re.sub(r"<style.*?</style>", " ", html_text, flags=re.DOTALL | re.IGNORECASE)
    plain = re.sub(r"<[^>]+>", " ", plain)
    plain_lower = plain.lower()

    # 2. Zero police / law-enforcement terminology anywhere in the article
    forbidden_terms = [
        "polícia",
        "policia",
        "policial",
        "policiais",
        "police",
        "viatura",
        "cerco",
        "cordon",
        "patrol",
        "perseguição",
        "perseguicao",
        "pursuit",
        "rotas de fuga",
        "rota de fuga",
        "escape route",
        "evasiva",
        "evasive",
        "ilícito",
        "ilicito",
        "crime",
        "violência urbana",
        "segurança pública",
        "public safety",
        "public security",
        "cicc",
        "suspeito",
        "suspect",
        "interceptação tática",
        "interception",
        "forense",
    ]
    found_forbidden: list[str] = []
    for term in forbidden_terms:
        if term in plain_lower:
            found_forbidden.append(term)
    if found_forbidden:
        raise ValueError(f"Forbidden law-enforcement terminology found in article: {found_forbidden}")

    # 3. Verify zero <pre>/<code> source blocks inside main body (before Referências)
    ref_split = html_text.split('id="sec-referencias"')
    if len(ref_split) != 2:
        raise ValueError("Could not locate Referências section boundary for source-code audit")
    body_before_refs = ref_split[0]
    if "<pre" in body_before_refs.lower():
        raise ValueError("Source-code <pre> block found in main body before Referências!")


def build_html(data: dict[str, Any], fig_dir: Path) -> str:
    a1 = data["A1"]["results"]
    a2 = data["A2"]["results"]
    a3 = data["A3"]["results"]
    a4 = data["A4"]["results"]
    a5 = data["A5"]["results"]
    a6_kin = data["A6"]["results"]["kinematic_and_sequence_models"]
    a6_rt = data["A6"]["results"]["road_graph_route_models"]
    a7 = data["A7"]["results"]

    s1 = a7["S1_4cam_20veh"]
    s2 = a7["S2_8cam_50veh"]
    s3 = a7["S3_16cam_100veh"]
    s4 = a7["S4_24cam_150veh"]

    f1 = (fig_dir / "fig1_arquitetura.png").resolve().as_uri()
    f2 = (fig_dir / "fig2_sota_cinematica.png").resolve().as_uri()
    f3 = (fig_dir / "fig3_rotas_tgnn.png").resolve().as_uri()
    f4 = (fig_dir / "fig4_escala_urbana.png").resolve().as_uri()
    f5 = (fig_dir / "fig5_mapa_urbano.png").resolve().as_uri()

    # Exact percentage reductions computed directly from JSON
    a1_red_lin = 100.0 * (a1["linear_cv"]["rmse_maneuver_m"] - a1["adaptive_bicycle"]["rmse_maneuver_m"]) / a1["linear_cv"]["rmse_maneuver_m"]
    a1_red_can = 100.0 * (a1["canonical_bicycle"]["rmse_maneuver_m"] - a1["adaptive_bicycle"]["rmse_maneuver_m"]) / a1["canonical_bicycle"]["rmse_maneuver_m"]
    a6_red_lin = 100.0 * (a6_kin["linear_cv"]["rmse_maneuver_mean_m"] - a6_kin["adaptive_bicycle"]["rmse_maneuver_mean_m"]) / a6_kin["linear_cv"]["rmse_maneuver_mean_m"]
    a6_red_can = 100.0 * (a6_kin["canonical_bicycle"]["rmse_maneuver_mean_m"] - a6_kin["adaptive_bicycle"]["rmse_maneuver_mean_m"]) / a6_kin["canonical_bicycle"]["rmse_maneuver_mean_m"]

    css = """
    @page {
      size: A4;
      margin: 2.2cm 2.0cm 2.2cm 2.0cm;
      @bottom-right {
        content: "Página " counter(page) " de " counter(pages);
        font-family: "DejaVu Sans", sans-serif;
        font-size: 8.5pt;
        color: #55606e;
      }
      @top-left {
        content: "Rastreamento Multicâmera e Predição de Trajetórias Veiculares com EKF Adaptativo";
        font-family: "DejaVu Sans", sans-serif;
        font-size: 7.5pt;
        color: #7b8794;
      }
    }
    @page :first {
      @top-left { content: none; }
    }
    body {
      font-family: "DejaVu Serif", "Liberation Serif", Georgia, serif;
      font-size: 10.2pt;
      line-height: 1.48;
      color: #1f2933;
      text-align: justify;
      hyphens: auto;
    }
    h1.title-pt {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 15.5pt;
      line-height: 1.28;
      text-align: center;
      color: #102a43;
      margin: 0 0 6pt 0;
      font-weight: 700;
    }
    p.title-en {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 10.6pt;
      line-height: 1.32;
      text-align: center;
      font-style: italic;
      color: #334e68;
      margin: 0 0 10pt 0;
      hyphens: none;
    }
    p.authors {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 9.8pt;
      font-weight: 600;
      text-align: center;
      color: #1f2933;
      margin: 0 0 14pt 0;
    }
    .abstract-box {
      background: #f8fafc;
      border-left: 3.5pt solid #1b4f72;
      padding: 9pt 12pt;
      margin: 0 0 10pt 0;
      font-size: 9.4pt;
      line-height: 1.42;
    }
    .abstract-box h2 {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 10pt;
      color: #1b4f72;
      margin: 0 0 4pt 0;
      border: none;
      padding: 0;
    }
    .keywords {
      margin-top: 5pt;
      font-size: 9pt;
    }
    h2 {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 11.8pt;
      color: #102a43;
      border-bottom: 1.2pt solid #1b4f72;
      padding-bottom: 2.5pt;
      margin: 16pt 0 7pt 0;
      page-break-after: avoid;
      text-align: left;
    }
    h3 {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 10.3pt;
      color: #243b53;
      margin: 11pt 0 4pt 0;
      page-break-after: avoid;
      text-align: left;
    }
    p {
      margin: 0 0 6.5pt 0;
    }
    ul, ol {
      margin: 4pt 0 8pt 16pt;
      padding: 0;
    }
    li {
      margin-bottom: 3.5pt;
    }
    .eq-block {
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: #f8fafc;
      border-left: 2.5pt solid #486581;
      padding: 6.5pt 10pt;
      margin: 7pt 0 8pt 0;
      page-break-inside: avoid;
    }
    .eq-content {
      flex: 1;
      text-align: center;
      font-family: "DejaVu Serif", serif;
      font-size: 9.8pt;
      line-height: 1.5;
    }
    .eq-num {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 9pt;
      color: #486581;
      min-width: 28pt;
      text-align: right;
    }
    .vec { font-weight: 700; font-style: normal; }
    .math-hat, .math-tilde {
      position: relative;
      display: inline-block;
      line-height: 1.0;
    }
    .math-hat::before {
      content: "^";
      position: absolute;
      top: -0.58em;
      left: 50%;
      transform: translateX(-50%);
      font-size: 0.72em;
      font-weight: 600;
      font-style: normal;
      line-height: 1;
    }
    .math-tilde::before {
      content: "~";
      position: absolute;
      top: -0.58em;
      left: 50%;
      transform: translateX(-50%);
      font-size: 0.78em;
      font-weight: 600;
      font-style: normal;
      line-height: 1;
    }
    .mat-inline {
      display: inline-table;
      vertical-align: middle;
      border-left: 1.2pt solid #1f2933;
      border-right: 1.2pt solid #1f2933;
      padding: 0 4pt;
      margin: 0 2pt;
      border-collapse: collapse;
    }
    .mat-inline td {
      padding: 1pt 4.5pt;
      text-align: center;
      font-size: 8.6pt;
      border: none;
    }
    .table-wrap {
      page-break-inside: avoid;
      break-inside: avoid;
      margin: 8pt 0 11pt 0;
    }
    .table-caption {
      text-align: left;
      font-family: "DejaVu Sans", sans-serif;
      font-size: 8.6pt;
      color: #243b53;
      margin: 0 0 4pt 0;
      line-height: 1.35;
      page-break-after: avoid;
      break-after: avoid;
      hyphens: none;
    }
    table.data-table {
      width: 100%;
      border-collapse: collapse;
      margin: 0;
      font-family: "DejaVu Sans", sans-serif;
      font-size: 8.0pt;
      line-height: 1.32;
      hyphens: none;
    }
    table.data-table thead th {
      background: #1b4f72;
      color: #ffffff;
      font-weight: 600;
      text-align: left;
      padding: 5pt 5.5pt;
      border: 0.5pt solid #1b4f72;
    }
    table.data-table thead th.num, table.data-table tbody td.num {
      text-align: center;
    }
    table.data-table tbody td {
      padding: 4.0pt 5.2pt;
      border-bottom: 0.5pt solid #cbd2d9;
      vertical-align: top;
      text-align: left;
    }
    table.compact-table {
      font-size: 7.45pt;
      line-height: 1.24;
    }
    table.compact-table thead th,
    table.compact-table tbody td {
      padding: 3.1pt 4.4pt;
    }
    table.data-table tbody tr:nth-child(even) {
      background: #f4f7fa;
    }
    table.data-table tbody tr.highlight {
      background: #e6f4ea;
      font-weight: 600;
    }
    caption.table-caption {
      caption-side: top;
      text-align: left;
      font-family: "DejaVu Sans", sans-serif;
      font-size: 8.7pt;
      color: #243b53;
      margin-bottom: 4pt;
      line-height: 1.35;
    }
    figure {
      margin: 10pt 0 12pt 0;
      text-align: center;
      page-break-inside: avoid;
    }
    figure img {
      max-width: 100%;
      height: auto;
      border: 0.5pt solid #d9e2ec;
      border-radius: 2pt;
    }
    figcaption {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 8.4pt;
      color: #334e68;
      margin-top: 4.5pt;
      text-align: justify;
      line-height: 1.35;
    }
    .ref-list {
      font-size: 8.7pt;
      line-height: 1.42;
      text-align: left;
      hyphens: none;
    }
    .ref-list p {
      padding-left: 18pt;
      text-indent: -18pt;
      margin-bottom: 5.5pt;
    }
    .appendix-break {
      page-break-before: always;
    }
    pre.code-box {
      background: #f5f7fa;
      border: 0.6pt solid #bcccdc;
      border-left: 3pt solid #1b4f72;
      padding: 7pt 9pt;
      font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
      font-size: 7.6pt;
      line-height: 1.35;
      white-space: pre-wrap;
      word-break: break-word;
      text-align: left;
      margin: 6pt 0 10pt 0;
      page-break-inside: avoid;
    }
    .listing-title {
      font-family: "DejaVu Sans", sans-serif;
      font-size: 8.6pt;
      font-weight: 600;
      color: #1b4f72;
      margin: 8pt 0 2pt 0;
      page-break-after: avoid;
    }
    .url-text {
      font-family: "DejaVu Sans Mono", monospace;
      font-size: 7.3pt;
      color: #1b4f72;
      word-break: break-all;
    }
    """

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Modelagem Cinemática Não Linear com Filtro de Kalman Estendido Adaptativo e Restrições Topológicas para Rastreamento Multicâmera e Predição de Trajetórias Veiculares em Redes Urbanas</title>
  <style>{css}</style>
</head>
<body>

<h1 class="title-pt">Modelagem Cinemática Não Linear com Filtro de Kalman Estendido Adaptativo e Restrições Topológicas para Rastreamento Multicâmera e Predição de Trajetórias Veiculares em Redes Urbanas</h1>
<p class="title-en">Nonlinear Kinematic Modeling with Adaptive Extended Kalman Filtering and Topological Constraints for Multi-Camera Vehicle Tracking and Trajectory Prediction in Urban Networks</p>
<p class="authors">Alex C. Medeiros, Eduardo E. Carvalho, Felipe R. Moura, Gabriel H. J. Ramalho, Kayky M. Oliveira, Roque C. Gabriel N.</p>

<div class="abstract-box">
  <h2>Resumo</h2>
  <p>O monitoramento contínuo de fluxos veiculares em Sistemas Inteligentes de Transporte (ITS) enfrenta um obstáculo físico recorrente: a descontinuidade espacial da cobertura óptica causada por câmeras estáticas com campos de visão não sobrepostos (<em>non-overlapping fields of view</em>) e extensas regiões cegas urbanas (<em>blind regions</em>). Durante o deslocamento por trechos não monitorados — especialmente sob manobras bruscas de conversão, desaceleração ou múltiplas bifurcações viárias —, a perda temporária de observação degrada a manutenção de identidade global e a estimativa de trajetórias futuras. Este artigo formaliza matematicamente, implementa em uma arquitetura estritamente causal em Python e avalia empiricamente um <em>pipeline</em> modular que integra o Filtro de Kalman Estendido Adaptativo (EKF) — baseado no modelo cinemático não linear <em>Two-Point-Bicycle</em> com suavização circular de orientação, peso direcional adaptativo à velocidade e gatilho persistente de Inovação Normalizada ao Quadrado (NIS) — à geometria projetiva 2D/3D com correção de ponto de contato, ao acoplamento topológico em malha viária direcionada (<em>map-matching</em> HMM/Viterbi e poda física por velocidade crítica de curva) e à associação híbrida intercâmeras (estatística &chi;<sup>2</sup> multi-hipótese, descritores visuais <em>Deep Re-ID</em> OSNet e leitura seletiva de placas ALPR). A proposta é confrontada diretamente com métodos consolidados do estado da arte, incluindo Filtro de Kalman Linear (CV), EKF CTRV, IMM, Filtro de Partículas (SIR), redes recorrentes LSTM e GRU e Rede Neural em Grafos Temporal (T-GNN), além de ser validada em cenários urbanos de larga escala de 4 a 24 câmeras não sobrepostas e 20 a 150 veículos simultâneos (11.388 detecções e 232 transições em regiões cegas). Nos experimentos controlados sobre 5 sementes aleatórias, o EKF adaptativo reduziu em {_fmt_pt(a6_red_lin, 1)}% o RMSE em manobras bruscas frente ao filtro linear ({_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} m vs. {_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 3)} m) e em {_fmt_pt(a6_red_can, 1)}% frente ao EKF canônico, enquanto o modelo IMM atingiu {_fmt_pt(a6_kin['imm']['rmse_maneuver_mean_m'], 3)} m em manobras e menor dispersão entre execuções sem necessidade de treinamento prévio. Na predição topológica de rotas em malha urbana 8&times;8, a árvore multi-hipótese com poda física superou a projeção euclidiana, o HMM/Viterbi e a rede T-GNN em erro espacial após 4,0 s de apagão visual ({_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m da T-GNN), mantendo {_fmt_pt(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 1)}% de acerto Top-1 e {_fmt_pt(a6_rt['proposed_physics_tree']['topk_route_accuracy']*100, 1)}% Top-2 sem retreinamento por cidade. Em escala urbana de 24 câmeras e 150 veículos, a fusão híbrida sustentou IDF1 de {_fmt_pt(s4['modes']['hybrid']['idf1'], 4)} — superando o limiar legado ({_fmt_pt(s4['modes']['article']['idf1'], 4)}), o gating puramente cinemático ({_fmt_pt(s4['modes']['chi2']['idf1'], 4)}) e o Re-ID visual isolado ({_fmt_pt(s4['modes']['reid_only']['idf1'], 4)}) — com vazão superior a {_fmt_int_pt(s4['modes']['hybrid']['throughput_detections_s'])} detecções/s em CPU.</p>
  <p class="keywords"><strong>Palavras-chave:</strong> Sistemas Inteligentes de Transporte (ITS). Filtro de Kalman Estendido Adaptativo (EKF). Rastreamento Veicular Multicâmera. Predição de Trajetórias. Malha Viária (<em>Map-Matching</em>). Filtro de Partículas. LSTM/GRU. Redes Neurais em Grafos Temporais (T-GNN).</p>
</div>

<div class="abstract-box" lang="en" style="hyphens: none;">
  <h2>Abstract</h2>
  <p>Continuous vehicle flow monitoring in Intelligent Transportation Systems (ITS) faces a fundamental structural bottleneck: spatial visual discontinuity caused by static roadside cameras with non-overlapping fields of view (FOVs) and extensive urban blind regions. When vehicles traverse unmonitored corridors — particularly under sharp turning maneuvers, braking, and multi-branch urban intersections — temporary observation loss degrades global identity maintenance and future trajectory forecasting. This paper mathematically formalizes, implements as a strictly causal Python framework, and empirically evaluates a modular multi-camera tracking pipeline combining an Adaptive Extended Kalman Filter (EKF) — built upon the nonlinear Two-Point-Bicycle model with circular heading smoothing, speed-adaptive directional weighting, and persistent Normalized Innovation Squared (NIS) gating — with 2D/3D projective geometry using 3D contact-point bias correction, directed road-graph topological constraints (HMM/Viterbi map-matching and critical cornering speed pruning), and hybrid cross-camera association (multi-hypothesis &chi;<sup>2</sup> gating, OSNet Deep Re-ID, and selective ALPR). The approach is systematically benchmarked against state-of-the-art methods including Linear Kalman (CV), CTRV EKF, Interacting Multiple Model (IMM), Sequential Importance Resampling (SIR) Particle Filter, recurrent LSTM and GRU networks, and a Temporal Graph Neural Network (T-GNN), alongside city-scale scalability tests scaling from 4 to 24 non-overlapping cameras and 20 to 150 concurrent vehicles (11,388 detections and 232 blind-zone transitions). Across 5 random seeds, the stabilized Adaptive EKF reduced sharp-maneuver position RMSE by {_fmt_en(a6_red_lin, 1)}% over Linear CV ({_fmt_en(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} m vs. {_fmt_en(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 3)} m) and by {_fmt_en(a6_red_can, 1)}% over Canonical EKF, while IMM achieved {_fmt_en(a6_kin['imm']['rmse_maneuver_mean_m'], 3)} m maneuver RMSE and lower cross-seed variance with zero training parameters. On out-of-sample 8&times;8 urban intersection transitions, the proposed physics-constrained multi-hypothesis route tree outperformed Euclidean snapping, HMM/Viterbi, and T-GNN in 4.0 s blind-zone spatial RMSE ({_fmt_en(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_en(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m for T-GNN), reaching {_fmt_en(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 1)}% Top-1 and {_fmt_en(a6_rt['proposed_physics_tree']['topk_route_accuracy']*100, 1)}% Top-2 accuracy without city-specific retraining. At city scale (24 cameras, 150 vehicles), hybrid fusion achieved {_fmt_en(s4['modes']['hybrid']['idf1'], 4)} IDF1 — outperforming the legacy threshold ({_fmt_en(s4['modes']['article']['idf1'], 4)}), pure kinematic gating ({_fmt_en(s4['modes']['chi2']['idf1'], 4)}), and standalone visual Re-ID ({_fmt_en(s4['modes']['reid_only']['idf1'], 4)}) — at over {_fmt_int_en(s4['modes']['hybrid']['throughput_detections_s'])} detections/s on CPU.</p>
  <p class="keywords"><strong>Keywords:</strong> Intelligent Transportation Systems (ITS). Adaptive Extended Kalman Filter (EKF). Multi-Camera Vehicle Tracking. Trajectory Prediction. Road Network Map-Matching. Particle Filter. LSTM/GRU. Temporal Graph Neural Networks (T-GNN).</p>
</div>

<h2>1. Introdução</h2>
<p>O crescimento da densidade veicular nas grandes metrópoles impulsiona a adoção de Sistemas Inteligentes de Transporte (ITS) voltados ao monitoramento contínuo de corredores viários, estimação dinâmica de matrizes origem-destino, detecção automática de incidentes e gestão adaptativa de semáforos em Centros de Gestão de Tráfego (CGT). Embora redes municipais de circuito fechado de televisão (CFTV) e sensores ópticos viários gerem grandes volumes de vídeo em tempo real, o acompanhamento contínuo de veículos em escala urbana enfrenta uma limitação física inevitável: a cobertura visual descontínua. Como destacaram Chilgunde <em>et al.</em> (2004), implantar câmeras com campos de visão continuamente sobrepostos em toda a extensão de uma malha viária urbana é inviável sob os pontos de vista técnico e econômico. Consequentemente, redes reais operam com câmeras estáticas espaçadas em intersecções estratégicas, deixando extensas regiões cegas (<em>blind regions</em>) entre pontos consecutivos de observação.</p>

<p>Quando um veículo deixa o campo visual de uma câmera e ingressa em uma região cega, sua dinâmica real frequentemente envolve desacelerações, retomadas de velocidade e conversões acentuadas em cruzamentos. Formulações clássicas baseadas no Filtro de Kalman linear (Kalman, 1960) assumem velocidade ou aceleração constante em espaço cartesiano desacoplado, o que provoca divergência rápida diante dos acoplamentos trigonométricos entre velocidade longitudinal, ângulo de esterçamento e orientação do chassi (Lou <em>et al.</em>, 2002; Bradler <em>et al.</em>, 2021). Para superar essas limitações, a literatura recente explora três grandes vertentes: (i) filtros estocásticos não lineares — como o Filtro de Kalman Estendido Adaptativo (EKF), o modelo de curva coordenada (CTRV), filtros de múltiplos modelos interagentes (IMM) e Filtros de Partículas (SIR) (Gordon <em>et al.</em>, 1993; Arulampalam <em>et al.</em>, 2002); (ii) preditores neurais sequenciais e baseados em grafos — tais como redes recorrentes LSTM e GRU (Hochreiter e Schmidhuber, 1997; Cho <em>et al.</em>, 2014; Altché e de La Fortelle, 2017) e Redes Neurais em Grafos Temporais (T-GNN) (Yu <em>et al.</em>, 2018; Zhao <em>et al.</em>, 2020); e (iii) acoplamento explícito à topologia da malha viária por <em>map-matching</em> probabilístico (Chao <em>et al.</em>, 2020; Kinderby e Ridderstolpe, 2024; Zhang <em>et al.</em>, 2025).</p>

<p>Ainda que cada paradigma apresente virtudes específicas, permanece uma lacuna metodológica na avaliação integrada e estritamente causal de um <em>pipeline</em> completo — da detecção visual e projeção métrica 3D até a associação multicâmera e a predição de rotas em regiões cegas — comparado de forma controlada contra filtros bayesianos não paramétricos (Filtro de Partículas) e arquiteturas neurais (LSTM, GRU e T-GNN), bem como testado sob escalabilidade realista de dezenas de câmeras e centenas de veículos simultâneos. Adicionalmente, ensaios prévios de rastreamento puramente cinemático sobre fluxos de vídeo sem descritores de aparência evidenciaram severa queda de acurácia intercâmeras (~5%) em tráfego denso devido à fragmentação de trajetórias locais e à ambiguidade entre veículos homogêneos, demonstrando a necessidade de fusão multimodal calibrada.</p>

<p>Nesse contexto, este trabalho formaliza matematicamente, implementa em arquitetura estritamente causal e valida experimentalmente um <em>pipeline</em> multicâmera para monitoramento de tráfego urbano e predição de trajetórias em regiões cegas. As principais contribuições são:</p>
<ul>
  <li><strong>Filtragem não linear estabilizada e causal:</strong> formalização e implementação de um EKF adaptativo <em>Two-Point-Bicycle</em> com inicialização 100% causal alinhada à via, suavização circular de orientação (janela de 4 quadros), ponderação direcional adaptativa à velocidade &alpha;<sub>&theta;</sub>(<em>v</em>) e gatilho persistente de Inovação Normalizada ao Quadrado (NIS &chi;<sup>2</sup><sub>95%</sub> por &ge; 2 quadros), eliminando oscilações espúrias do fator de desvanecimento &lambda;<sub><em>k</em></sub> sob ruído elevado de detecção.</li>
  <li><strong>Geometria projetiva 2D/3D e restrições físicas topológicas:</strong> compensação do viés de perspectiva 3D na projeção do ponto de contato da caixa delimitadora (redução de {_fmt_pt(a3['bias_reduction_pct'], 2)}% no erro métrico de solo), acoplamento a grafos viários OpenStreetMap/UTM via HMM/Viterbi e propagação multi-hipótese com poda física por velocidade crítica de derrapagem lateral <em>v</em><sub>crit</sub> = &radic;(&mu;<em>gR</em><sub>curva</sub>).</li>
  <li><strong>Comparação sistemática com o estado da arte:</strong> implementação e avaliação reproduzível, sob o mesmo protocolo experimental, contra Filtro de Kalman Linear (CV), EKF Canônico, CTRV, IMM, <strong>Filtro de Partículas (SIR, 500 partículas)</strong>, redes recorrentes <strong>LSTM</strong> e <strong>GRU</strong> e <strong>Rede Neural em Grafos Temporal (T-GNN)</strong>, analisando simultaneamente erro em manobras, erro em apagão visual, acurácia topológica de rota, variância entre sementes e latência em CPU.</li>
  <li><strong>Validação em larga escala urbana e rastreabilidade de fontes de dados:</strong> avaliação de escalabilidade em malha urbana 8&times;8 (64 intersecções e 196 arestas direcionadas) escalando de 4 para <strong>24 câmeras não sobrepostas</strong> e de 20 para <strong>150 veículos simultâneos</strong> ({_fmt_int_pt(s4['total_detections'])} detecções e {s4['num_blind_transitions']} transições cegas), combinada à documentação integral das fontes bibliográficas e de acesso de todos os bancos de dados reais e sintéticos empregados.</li>
</ul>

<h2>2. Modelagem Matemática: Cinemática Não Linear e Filtragem Adaptativa</h2>
<p>A representação fiel da locomoção de veículos terrestres em vias urbanas requer modelos que capturem a restrição não holonômica de esterçamento e o acoplamento entre velocidade escalar e rotação do chassi. Esta seção apresenta as limitações dos filtros lineares, a formulação contínua e discreta do modelo <em>Two-Point-Bicycle</em>, sua linearização por Jacobianas analíticas e o mecanismo estabilizado de adaptação estocástica.</p>

<h3>2.1 Limitações da Filtragem Linear em Manobras Urbanas</h3>
<p>No Filtro de Kalman linear clássico (Kalman, 1960), assume-se o modelo de Velocidade Constante (CV) com vetor de estado <span class="vec">x</span><sub><em>k</em></sub> = [<em>X</em><sub><em>k</em></sub>, <em>Y</em><sub><em>k</em></sub>, <em>v</em><sub><em>x,k</em></sub>, <em>v</em><sub><em>y,k</em></sub>]<sup>&top;</sup> &isin; &Ropf;<sup>4</sup> evoluindo por uma matriz de transição invariante:</p>
<div class="eq-block">
  <div class="eq-content"><span class="vec">x</span><sub><em>k</em>+1</sub> = &Phi;(&Delta;<em>t</em>)&thinsp;<span class="vec">x</span><sub><em>k</em></sub> + <span class="vec">w</span><sub><em>k</em></sub>, &nbsp;&nbsp; com &nbsp; <span class="vec">w</span><sub><em>k</em></sub> &sim; &Nscr;(<strong>0</strong>, <span class="vec">Q</span><sub><em>k</em></sub>)</div>
  <div class="eq-num">(1)</div>
</div>
<p>Em interseções urbanas, rotatórias e mudanças bruscas de faixa, acelerações laterais centrípetas rompem a hipótese de linearidade de &Phi;. Durante curvas acentuadas ou intervalos sem observação visual, a extrapolação retilínea acumula defasagem crescente em relação ao traçado viário, fazendo a inovação divergir e comprometendo a reaquisição do veículo em câmeras subsequentes (Lou <em>et al.</em>, 2002; Bradler <em>et al.</em>, 2021).</p>

<h3>2.2 Modelo Cinemático <em>Two-Point-Bicycle</em> e Alternativas Não Lineares</h3>
<p>Adota-se o modelo cinemático <em>Two-Point-Bicycle</em> sob hipótese de chassi rígido com distância entre-eixos <em>L</em> = 2,5 m e ausência de escorregamento lateral (<em>no-slip</em>). O vetor de estado de quinta ordem no plano métrico UTM é definido como:</p>
<div class="eq-block">
  <div class="eq-content"><span class="vec">x</span><sub><em>k</em></sub> = [ <em>X</em><sub><em>k</em></sub>, <em>Y</em><sub><em>k</em></sub>, <em>v</em><sub><em>k</em></sub>, &theta;<sub><em>k</em></sub>, &delta;<sub><em>k</em></sub> ]<sup>&top;</sup> &isin; &Ropf;<sup>5</sup></div>
  <div class="eq-num">(2)</div>
</div>
<p>em que (<em>X</em><sub><em>k</em></sub>, <em>Y</em><sub><em>k</em></sub>) denotam as coordenadas métricas no plano de solo (m), <em>v</em><sub><em>k</em></sub> &ge; 0 representa a velocidade longitudinal (m/s), &theta;<sub><em>k</em></sub> &isin; (&minus;&pi;, &pi;] é a orientação azimutal do chassi (rad) e &delta;<sub><em>k</em></sub> &isin; [&minus;&delta;<sub>máx</sub>, +&delta;<sub>máx</sub>] é o ângulo de esterçamento das rodas dianteiras (rad). Definindo a taxa de guinada instantânea &omega;<sub><em>k</em></sub> = (<em>v</em><sub><em>k</em></sub> / <em>L</em>)&thinsp;tan(&delta;<sub><em>k</em></sub>), a integração exata por arco circular no passo &Delta;<em>t</em> (com tratamento analítico de singularidade para |&omega;<sub><em>k</em></sub>| &lt; 10<sup>&minus;6</sup> rad/s) é dada por:</p>
<div class="eq-block">
  <div class="eq-content">
    <em>X</em><sub><em>k</em>+1</sub> = <em>X</em><sub><em>k</em></sub> + (<em>v</em><sub><em>k</em></sub> / &omega;<sub><em>k</em></sub>)&thinsp;[sen(&theta;<sub><em>k</em></sub> + &omega;<sub><em>k</em></sub>&Delta;<em>t</em>) &minus; sen(&theta;<sub><em>k</em></sub>)]<br/>
    <em>Y</em><sub><em>k</em>+1</sub> = <em>Y</em><sub><em>k</em></sub> + (<em>v</em><sub><em>k</em></sub> / &omega;<sub><em>k</em></sub>)&thinsp;[&minus;cos(&theta;<sub><em>k</em></sub> + &omega;<sub><em>k</em></sub>&Delta;<em>t</em>) + cos(&theta;<sub><em>k</em></sub>)]<br/>
    <em>v</em><sub><em>k</em>+1</sub> = <em>v</em><sub><em>k</em></sub>, &nbsp;&nbsp; &theta;<sub><em>k</em>+1</sub> = wrap(&theta;<sub><em>k</em></sub> + &omega;<sub><em>k</em></sub>&Delta;<em>t</em>), &nbsp;&nbsp; &delta;<sub><em>k</em>+1</sub> = clip(&delta;<sub><em>k</em></sub>, &minus;&delta;<sub>máx</sub>, +&delta;<sub>máx</sub>)
  </div>
  <div class="eq-num">(3)</div>
</div>
<p>Para avaliação comparativa rigorosa, o <em>pipeline</em> implementa ainda o modelo CTRV (<em>Constant Turn Rate and Velocity</em>, Bradler <em>et al.</em>, 2021) — substituindo o estado &delta; diretamente pela taxa angular &omega; — e o filtro IMM (<em>Interacting Multiple Model</em>, Blom e Bar-Shalom, 1988), que combina dinamicamente o EKF Adaptativo <em>Two-Point-Bicycle</em> e o EKF CTRV via matriz de transição de Markov e fusão bayesiana de covariâncias ponderadas pela verossimilhança de inovação.</p>

<h3>2.3 Linearização Analítica e Estabilidade Numérica na Forma de Joseph</h3>
<p>A propagação da covariância de erro <span class="vec">P</span><sub><em>k</em>|<em>k</em>&minus;1</sub> utiliza a matriz Jacobiana analítica exata <span class="vec">F</span><sub><em>k</em></sub> = &part;<em>f</em><sub><em>d</em></sub>/&part;<span class="vec">x</span> &isin; &Ropf;<sup>5&times;5</sup> (verificada por diferenças finitas centradas nos testes automatizados com erro inferior a 10<sup>&minus;5</sup>). Para o regime quase retilíneo (|&omega;<sub><em>k</em></sub>| &lt; 10<sup>&minus;5</sup>), definindo <em>c</em> = cos(&theta;), <em>s</em> = sen(&theta;), &part;&omega;/&part;<em>v</em> = tan(&delta;)/<em>L</em> e &part;&omega;/&part;&delta; = (<em>v</em>/<em>L</em>)&thinsp;sec<sup>2</sup>(&delta;), tem-se:</p>
<div class="eq-block">
  <div class="eq-content">
    <span class="vec">F</span><sub><em>k</em></sub> = <span class="vec">I</span><sub>5</sub> + &Delta;<em>t</em> &middot;
    <table class="mat-inline">
      <tr><td>0</td><td>0</td><td><em>c</em></td><td>&minus;<em>v</em>&thinsp;<em>s</em></td><td>&minus;0,5&thinsp;<em>v</em>&thinsp;<em>s</em>&thinsp;&Delta;<em>t</em>&thinsp;&omega;<sub>&delta;</sub></td></tr>
      <tr><td>0</td><td>0</td><td><em>s</em></td><td><em>v</em>&thinsp;<em>c</em></td><td>0,5&thinsp;<em>v</em>&thinsp;<em>c</em>&thinsp;&Delta;<em>t</em>&thinsp;&omega;<sub>&delta;</sub></td></tr>
      <tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
      <tr><td>0</td><td>0</td><td>&omega;<sub><em>v</em></sub></td><td>0</td><td>&omega;<sub>&delta;</sub></td></tr>
      <tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
    </table>
  </div>
  <div class="eq-num">(4)</div>
</div>
<p>No regime curvilíneo (|&omega;<sub><em>k</em></sub>| &ge; 10<sup>&minus;5</sup>), empregam-se as derivadas exatas do raio de curva <em>R</em> = <em>L</em>/tan(&delta;) em função de <em>v</em>, &theta; e &delta;. A inicialização de cada trajetória é <strong>estritamente causal</strong> (sem qualquer acesso a quadros futuros): no primeiro quadro observado, a orientação inicial &theta;<sub>0</sub> é alinhada à aresta direcionada mais próxima do grafo viário; no segundo quadro, a velocidade inicial <em>v</em><sub>1</sub> e a orientação são refinadas causalmente pelo deslocamento observado. Na etapa de atualização com observação métrica <span class="vec">z</span><sub><em>k</em></sub> = [<em>X</em><sub>obs</sub>, <em>Y</em><sub>obs</sub>]<sup>&top;</sup> e covariância <span class="vec">R</span><sub><em>k</em></sub> &isin; &Ropf;<sup>2&times;2</sup>, emprega-se a forma estabilizada de Joseph seguida de projeção espectral simétrica positiva definida (SPD):</p>
<div class="eq-block">
  <div class="eq-content">
    <span class="vec">&gamma;</span><sub><em>k</em></sub> = <span class="vec">z</span><sub><em>k</em></sub> &minus; <span class="vec">H</span><span class="vec math-hat">x</span><sub><em>k</em>|<em>k</em>&minus;1</sub>, &nbsp;&nbsp;
    <span class="vec">S</span><sub><em>k</em></sub> = <span class="vec">H</span><span class="vec math-tilde">P</span><sub><em>k</em>|<em>k</em>&minus;1</sub><span class="vec">H</span><sup>&top;</sup> + <span class="vec">R</span><sub><em>k</em></sub>, &nbsp;&nbsp;
    <span class="vec">K</span><sub><em>k</em></sub> = <span class="vec math-tilde">P</span><sub><em>k</em>|<em>k</em>&minus;1</sub><span class="vec">H</span><sup>&top;</sup><span class="vec">S</span><sub><em>k</em></sub><sup>&minus;1</sup><br/>
    <span class="vec math-hat">x</span><sub><em>k</em>|<em>k</em></sub> = <span class="vec math-hat">x</span><sub><em>k</em>|<em>k</em>&minus;1</sub> + <span class="vec">K</span><sub><em>k</em></sub><span class="vec">&gamma;</span><sub><em>k</em></sub>, &nbsp;&nbsp;
    <span class="vec">P</span><sub><em>k</em>|<em>k</em></sub> = (<span class="vec">I</span> &minus; <span class="vec">K</span><sub><em>k</em></sub><span class="vec">H</span>)<span class="vec math-tilde">P</span><sub><em>k</em>|<em>k</em>&minus;1</sub>(<span class="vec">I</span> &minus; <span class="vec">K</span><sub><em>k</em></sub><span class="vec">H</span>)<sup>&top;</sup> + <span class="vec">K</span><sub><em>k</em></sub><span class="vec">R</span><sub><em>k</em></sub><span class="vec">K</span><sub><em>k</em></sub><sup>&top;</sup>
  </div>
  <div class="eq-num">(5)</div>
</div>

<h3>2.4 Estabilização do Fator de Desvanecimento &lambda;<sub><em>k</em></sub> sob Ruído Real de Detecção</h3>
<p>Na formulação original de Lou <em>et al.</em> (2002), um fator escalar de desvanecimento &lambda;<sub><em>k</em></sub> &ge; 1 infla a covariância predita (<span class="vec math-tilde">P</span><sub><em>k</em>|<em>k</em>&minus;1</sub> = &lambda;<sub><em>k</em></sub><span class="vec">P</span><sub><em>k</em>|<em>k</em>&minus;1</sub>) quando os resíduos de inovação perdem a propriedade de ruído branco durante manobras abruptas. Todavia, em câmeras urbanas reais com ruído substancial na caixa delimitadora (&sigma;<sub>pos</sub> &ge; 0,5 m), estimar a orientação instantânea por diferenças finitas simples e aplicar peso fixo elevado &alpha;<sub>&theta;</sub> = 2,8 faz com que oscilações de um único quadro disparem falsos picos de &lambda;<sub><em>k</em></sub>, degradando a filtragem ao seguir o ruído do detector.</p>
<p>Para superar essa fragilidade, o filtro proposto introduz três salvaguardas acopladas: (i) <strong>suavização circular da orientação</strong> em janela deslizante de <em>W</em> = 4 quadros via média circular ponderada atan2(&Sigma;<em>w</em><sub><em>i</em></sub>sen&thinsp;&theta;<sub><em>i</em></sub>, &Sigma;<em>w</em><sub><em>i</em></sub>cos&thinsp;&theta;<sub><em>i</em></sub>), protegida contra picos isolados; (ii) <strong>gatilho persistente de Inovação Normalizada ao Quadrado (NIS)</strong>, computando &epsilon;<sub><em>k</em></sub> = <span class="vec">&gamma;</span><sub><em>k</em></sub><sup>&top;</sup><span class="vec">S</span><sub><em>k</em></sub><sup>&minus;1</sup><span class="vec">&gamma;</span><sub><em>k</em></sub> e exigindo que &epsilon;<sub><em>k</em></sub> exceda o quantil &chi;<sup>2</sup><sub>0,95;2</sub> = 5,991 por pelo menos <em>N</em><sub>pers</sub> = 2 quadros consecutivos antes de acionar a adaptação; e (iii) <strong>ponderação direcional adaptativa à velocidade</strong> &alpha;<sub>&theta;</sub>(<em>v</em>) &isin; [0,2; 2,8] com soma normalizada &alpha;<sub><em>x</em></sub> + &alpha;<sub><em>y</em></sub> + &alpha;<sub>&theta;</sub> = 3,0 e limiar angular (<em>heading gate</em> = 0,35 rad):</p>
<div class="eq-block">
  <div class="eq-content">
    &lambda;<sub><em>k</em></sub> = clip(&thinsp;1 + &alpha;<sub><em>xy</em></sub>(<em>v</em>)&thinsp;(&epsilon;<sub><em>k</em></sub> / &chi;<sup>2</sup><sub>0,95;2</sub>) + &alpha;<sub>&theta;</sub>(<em>v</em>)&thinsp;|&Delta;&theta;<sub>circ</sub>|&thinsp;&Iopf;<sub>|&Delta;&theta;<sub>circ</sub>| &gt; 0,35</sub>,&nbsp; 1,15,&nbsp; &lambda;<sub>máx</sub>&thinsp;) &nbsp; se &nbsp; streak(&epsilon;<sub><em>k</em></sub> &gt; &chi;<sup>2</sup><sub>0,95;2</sub>) &ge; 2, &nbsp; senão &nbsp; 1,0
  </div>
  <div class="eq-num">(6)</div>
</div>
<p>Em trechos retilíneos e sob ruído pontual isolado, &lambda;<sub><em>k</em></sub> permanece igual a 1,0, preservando a suavidade do EKF canônico; ao detectar manobra genuína persistente, &lambda;<sub><em>k</em></sub> eleva-se de forma controlada, permitindo rápida convergência de posição, orientação e ângulo de esterçamento.</p>

<h2>3. Geometria Projetiva 2D/3D, Regiões Cegas e Associação Híbrida Multicâmera</h2>
<p>A integração de múltiplas câmeras estáticas sem sobreposição visual exige projetar todas as detecções para um sistema de coordenadas métrico unificado (projeção cartográfica UTM) e modelar rigorosamente a incerteza durante o intervalo sem observação visual.</p>

<h3>3.1 Projeção 2D/3D, Correção do Ponto de Contato e Covariância Métrica</h3>
<p>O módulo geométrico suporta tanto calibração planar por matriz de homografia <span class="vec">H</span><sub><em>i</em></sub> &isin; &Ropf;<sup>3&times;3</sup> (estimada por DLT normalizado de Hartley com rejeição robusta RANSAC) quanto projeção volumétrica direta por matrizes intrínseca <span class="vec">K</span> e extrínseca [<span class="vec">R</span> | <span class="vec">t</span>] (Bradler <em>et al.</em>, 2021) interceptando o plano de solo <em>Z</em> = 0. Um erro sistemático frequente na literatura consiste em projetar a borda inferior central bruta da caixa delimitadora 2D (<em>v</em> = <em>y</em><sub>2</sub>): em câmeras montadas em ângulo oblíquo (20&deg;–30&deg;), o volume tridimensional da carroceria desloca <em>y</em><sub>2</sub> para além do eixo real de contato dos pneus com o asfalto. Adota-se a correção de ponto de contato em perspectiva com fração calibrada <em>f</em> = 0,90:</p>
<div class="eq-block">
  <div class="eq-content">
    <em>u</em><sub><em>c</em></sub> = 0,5&thinsp;(<em>x</em><sub>1</sub> + <em>x</em><sub>2</sub>), &nbsp;&nbsp; <em>v</em><sub><em>c</em></sub> = <em>y</em><sub>1</sub> + <em>f</em>&thinsp;(<em>y</em><sub>2</sub> &minus; <em>y</em><sub>1</sub>), &nbsp;&nbsp;
    [<em>X</em><sub><em>h</em></sub>, <em>Y</em><sub><em>h</em></sub>, <em>w</em><sub><em>h</em></sub>]<sup>&top;</sup> = <span class="vec">H</span><sub><em>i</em></sub>&thinsp;[<em>u</em><sub><em>c</em></sub>, <em>v</em><sub><em>c</em></sub>, 1]<sup>&top;</sup>, &nbsp;&nbsp; (<em>X</em>, <em>Y</em>) = (<em>X</em><sub><em>h</em></sub>/<em>w</em><sub><em>h</em></sub>, <em>Y</em><sub><em>h</em></sub>/<em>w</em><sub><em>h</em></sub>)
  </div>
  <div class="eq-num">(7)</div>
</div>
<p>A incerteza de localização em pixels (&sigma;<sub><em>u</em></sub>, &sigma;<sub><em>v</em></sub>) é propagada analiticamente para o plano de solo por meio da Jacobiana da homografia <span class="vec">J</span><sub><em>H</em></sub>(<em>u</em><sub><em>c</em></sub>, <em>v</em><sub><em>c</em></sub>) = &part;(<em>X</em>, <em>Y</em>)/&part;(<em>u</em>, <em>v</em>), gerando uma matriz de covariância de medição anisotrópica dependente da distância à câmera:</p>
<div class="eq-block">
  <div class="eq-content">
    <span class="vec">R</span><sub><em>k</em></sub>(<em>u</em><sub><em>c</em></sub>, <em>v</em><sub><em>c</em></sub>) = <span class="vec">J</span><sub><em>H</em></sub>(<em>u</em><sub><em>c</em></sub>, <em>v</em><sub><em>c</em></sub>)&thinsp;diag(&sigma;<sub><em>u</em></sub><sup>2</sup>, &sigma;<sub><em>v</em></sub><sup>2</sup>)&thinsp;<span class="vec">J</span><sub><em>H</em></sub>(<em>u</em><sub><em>c</em></sub>, <em>v</em><sub><em>c</em></sub>)<sup>&top;</sup>
  </div>
  <div class="eq-num">(8)</div>
</div>

<h3>3.2 Propagação Multi-Hipótese em Regiões Cegas e Associação Híbrida</h3>
<p>Quando um veículo cruza uma zona de saída (<em>Exit Zone</em>) ou permanece sem detecção além do limiar local de oclusão, sua trajetória entra em estado de região cega. Diferentemente da extrapolação euclidiana simples, o sistema propaga simultaneamente um conjunto ponderado de hipóteses curvilíneas {{<em>h</em><sub>1</sub>, &hellip;, <em>h</em><sub><em>M</em></sub>}} ao longo do grafo viário direcionado, expandindo anisotropicamente a covariância 2&times;2 na direção tangente e normal a cada via. Quando uma nova aparição <em>b</em> surge em uma câmera a jusante no instante <em>t</em><sub><em>b</em></sub>, avalia-se sua compatibilidade com cada registro cego <em>a</em> por três fontes complementares:</p>
<ol>
  <li><strong>Distância espaço-temporal multi-hipótese (&chi;<sup>2</sup>):</strong> calcula-se a menor distância quadrática de Mahalanobis entre a posição observada <span class="vec">z</span><sub><em>b</em></sub> e as hipóteses fisicamente viáveis de <em>a</em> no instante <em>t</em><sub><em>b</em></sub>, incorporando também a projeção curvilínea longitudinal/lateral sobre a polilinha da aresta.</li>
  <li><strong>Similaridade de aparência visual (Deep Re-ID):</strong> extrai-se um vetor normalizado em <em>L</em><sub>2</sub> de 512 dimensões via rede leve OSNet x0.25 em ONNX Runtime (Zhou <em>et al.</em>, 2019), avaliando a distância cosseno <em>d</em><sub>ReID</sub>(<em>a</em>, <em>b</em>) = 1 &minus; &langle;<span class="vec">e</span><sub><em>a</em></sub>, <span class="vec">e</span><sub><em>b</em></sub>&rangle;.</li>
  <li><strong>Autocalibração de escala geométrica e leitura seletiva ALPR:</strong> estima-se continuamente, pela mediana dos pares confirmados entre câmeras (<em>i</em>, <em>j</em>), os fatores de escala aparente (<em>C</em><sub><em>w</em></sub><sup><em>ij</em></sup>, <em>C</em><sub><em>l</em></sub><sup><em>ij</em></sup>) que corrigem distorções de perspectiva nas dimensões projetadas (<em>W</em>, <em>L</em>) (Chilgunde <em>et al.</em>, 2004). Quando ambas as câmeras capturam leitura de placa válida com alta confiança (&ge; 0,85), aplica-se confirmação determinística ou veto de inconsistência.</li>
</ol>
<p>A função de custo híbrida para o par (<em>a</em>, <em>b</em>) combina as métricas normalizadas sujeitas às portas estatísticas <em>d</em><sub>&chi;<sup>2</sup></sub> &le; 2&chi;<sup>2</sup><sub>lim</sub> e <em>d</em><sub>ReID</sub> &le; <em>d</em><sub>ReID,máx</sub>, resolvendo-se a atribuição global ótima 1:1 pelo Algoritmo Húngaro em tempo <em>O</em>(<em>n</em><sup>3</sup>) (Kuhn, 1955):</p>
<div class="eq-block">
  <div class="eq-content">
    <em>C</em><sub>híbrido</sub>(<em>a</em>, <em>b</em>) = <em>w</em><sub>st</sub>&thinsp;(<em>d</em><sub>&chi;<sup>2</sup></sub> / &chi;<sup>2</sup><sub>lim</sub>) + <em>w</em><sub>ReID</sub>&thinsp;(<em>d</em><sub>ReID</sub> / <em>d</em><sub>ReID,máx</sub>) + 0,15&thinsp;<em>d</em><sub>dim</sub>(<em>a</em>, <em>b</em>)
  </div>
  <div class="eq-num">(9)</div>
</div>
<p>Confirmada a associação intercâmeras, executa-se a <strong>reconstrução retrospectiva de trajetória (<em>backtracking</em> topológico)</strong>, conectando a aresta de saída na câmera de origem à aresta de entrada na câmera de destino ao longo da hipótese vencedora e podando as ramificações alternativas não percorridas para registro analítico contínuo do fluxo viário.</p>

<h2>4. Acoplamento à Topologia Viária e Predição de Rotas Prováveis</h2>
<p>Em malhas urbanas reais, veículos trafegam restritos a vias pavimentadas, sentidos de mão única e limites físicos de curvatura. A Tabela 1 sintetiza as principais famílias de algoritmos de <em>map-matching</em> revisadas por Chao <em>et al.</em> (2020) e fundamenta a arquitetura híbrida adotada neste trabalho.</p>

<table class="data-table">
  <caption class="table-caption"><strong>Tabela 1</strong> – Comparação entre famílias de modelos de acoplamento viário (<em>map-matching</em>) adaptada de Chao <em>et al.</em> (2020) para aplicações em Sistemas Inteligentes de Transporte (ITS).</caption>
  <thead>
    <tr>
      <th style="width: 22%;">Família de Modelo</th>
      <th style="width: 26%;">Paradigma Operacional</th>
      <th style="width: 20%;">Robustez a Ruídos e Oclusões</th>
      <th style="width: 15%;">Complexidade / Latência</th>
      <th style="width: 17%;">Aplicabilidade em Gestão de Tráfego (ITS)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Similaridade Geométrica</strong> (Euclidiano / Fréchet)</td>
      <td>Projeção ponto a ponto na aresta mais próxima ou alinhamento de curvas</td>
      <td>Baixa; sensível a <em>outliers</em> e vias paralelas próximas</td>
      <td>Baixa — <em>O</em>(<em>N</em>) / Tempo real</td>
      <td>Limitada: produz saltos espúrios em cruzamentos densos</td>
    </tr>
    <tr>
      <td><strong>Transição de Estados</strong> (HMM / Viterbi)</td>
      <td>Estados ocultos sobre arestas com probabilidades de emissão e transição</td>
      <td>Alta; elimina desvios topologicamente desconexos</td>
      <td>Média — <em>O</em>(<em>T</em>&middot;|<em>S</em>|<sup>2</sup>)</td>
      <td>Excelente para consolidação de trechos visíveis e históricos</td>
    </tr>
    <tr>
      <td><strong>Evolução de Candidatos</strong> (Multi-Hipótese / Feixe)</td>
      <td>Manutenção e poda dinâmica de múltiplas rotas ponderadas simultâneas</td>
      <td>Muito Alta; acompanha bifurcações em regiões cegas</td>
      <td>Média — <em>O</em>(<em>M</em>) com poda física</td>
      <td>Excelente: prevê distribuição de fluxos em múltiplas saídas</td>
    </tr>
    <tr class="highlight">
      <td><strong>Híbrido Proposto</strong> (HMM + Árvore Física <em>v</em><sub>crit</sub>)</td>
      <td>HMM/Viterbi em observação + árvore multi-hipótese podada por física de curva</td>
      <td>Muito Alta; imune a curvas fisicamente inviáveis</td>
      <td>Baixa a Média (~1,5 ms/consulta)</td>
      <td>Ótima: alta acurácia topológica em tempo real sem retreinamento</td>
    </tr>
  </tbody>
</table>

<h3>4.1 Malha Viária Direcionada e <em>Map-Matching</em> HMM/Viterbi</h3>
<p>A malha viária urbana é modelada como um multigrafo direcionado <em>G</em> = (<em>V</em>, <em>E</em>) em coordenadas métricas UTM (importado diretamente de arquivos OpenStreetMap via OSMnx; Boeing, 2017). Cada aresta <em>e</em> = (<em>u</em>, <em>v</em>, <em>k</em>) &isin; <em>E</em> armazena sua polilinha métrica, extensão <em>L</em><sub><em>e</em></sub>, sentido regulamentar (mão única ou dupla), número de faixas, velocidade limite <em>v</em><sub>lim</sub> e raio mínimo de curvatura <em>R</em><sub>curva</sub>. Para eliminar projeções incoerentes sobre vias paralelas sob ruído de câmera (<em>unnecessary detours</em>), o <em>map-matching</em> sequencial emprega um Modelo Oculto de Markov (HMM) decodificado por programação dinâmica de Viterbi (Newson e Krumm, 2009), combinando verossimilhança gaussiana de emissão e penalização exponencial da discrepância entre distância euclidiana e distância em rede, com pré-filtragem espacial das 16 arestas candidatas mais próximas para garantir escalabilidade em malhas municipais extensas.</p>

<h3>4.2 Poda Física por Velocidade Crítica de Curva e Frenagem Pré-Intersecção</h3>
<p>Ao alcançar uma intersecção <em>v</em> durante a propagação em região cega, o conjunto de arestas eferentes Out(<em>v</em>) sofre poda topológica (respeito à mão única e proibição de retorno instantâneo de 180&deg;) seguida de <strong>poda dinâmica por velocidade crítica de derrapagem lateral</strong>. Para uma curva com ângulo de deflexão &Delta;&theta; &ge; 30&deg; e raio efetivo <em>R</em><sub>curva</sub>, sob coeficiente de atrito pneu-asfalto &mu; = 0,70 e aceleração gravitacional <em>g</em> = 9,81 m/s<sup>2</sup>, a velocidade limite fisicamente sustentável e a menor velocidade alcançável sob frenagem máxima a partir da distância restante <em>d</em><sub>aprox</sub> são:</p>
<div class="eq-block">
  <div class="eq-content">
    <em>v</em><sub>crit</sub> = &radic;(&mu;&thinsp;<em>g</em>&thinsp;<em>R</em><sub>curva</sub>), &nbsp;&nbsp;&nbsp;
    <em>v</em><sub>mín</sub> = &radic;(max(0,&nbsp; <em>v</em><sub>0</sub><sup>2</sup> &minus; 2&mu;<em>g</em>&thinsp;<em>d</em><sub>aprox</sub>))
  </div>
  <div class="eq-num">(10)</div>
</div>
<p>Se <em>v</em><sub>mín</sub> &gt; 1,05&thinsp;<em>v</em><sub>crit</sub>, a conversão fechada é marcada como fisicamente inviável e podada da árvore de hipóteses, concentrando a massa probabilística nas continuações viáveis. Para conversões factíveis, o perfil longitudinal simula frenagem pré-esquina até 0,90&thinsp;<em>v</em><sub>crit</sub> dentro do arco curvo e reaceleração progressiva até <em>v</em><sub>lim</sub> após a saída da curva. Cada ramificação viável recebe peso proporcional à hierarquia da via (faixas e velocidade limite) e à suavidade da deflexão angular, sendo normalizada ao final para que &Sigma;<sub><em>m</em></sub> <em>P</em>(<em>h</em><sub><em>m</em></sub>) = 1,0:</p>
<div class="eq-block">
  <div class="eq-content">
    <em>w</em>(<em>e</em><sub>próx</sub> | <em>e</em><sub>atual</sub>) = faixas(<em>e</em><sub>próx</sub>) &middot; (<em>v</em><sub>lim</sub>(<em>e</em><sub>próx</sub>) / 13,89) &middot; exp(&minus;0,6&thinsp;|&Delta;&theta;|), &nbsp;&nbsp;&nbsp;
    <em>P</em>(<em>h</em><sub><em>m</em></sub>) = <em>W</em>(<em>h</em><sub><em>m</em></sub>) / &Sigma;<sub><em>j</em></sub> <em>W</em>(<em>h</em><sub><em>j</em></sub>)
  </div>
  <div class="eq-num">(11)</div>
</div>
<p>A partir da distribuição resultante, o sistema identifica automaticamente os <strong>nós críticos de monitoramento viário</strong> (vértices a jusante com maior probabilidade acumulada de passagem e tempo estimado de chegada — ETA), subsidiando a gestão proativa de corredores, o controle semafórico adaptativo e o balanceamento de sensores em CGTs.</p>

<h2>5. Arquitetura Computacional, Baselines do Estado da Arte e Fontes de Dados</h2>
<p>A arquitetura computacional foi desenvolvida em Python 3.12 com contratos de dados estritamente tipados (Pydantic v2) e operação 100% causal quadro a quadro. A Figura 1 apresenta o fluxo de dados entre os seis estágios do sistema. Em conformidade com as boas práticas de redação científica, os trechos ilustrativos de implementação e os comandos de reprodutibilidade foram consolidados nos Apêndices A e B, após as Referências.</p>

<figure>
  <img src="{f1}" alt="Arquitetura modular causal do pipeline multicâmera" />
  <figcaption><strong>Figura 1</strong> – Arquitetura modular causal do <em>pipeline</em> implementado em seis estágios integrados: percepção local (YOLO11n + ByteTrack), geometria projetiva 2D/3D, filtragem cinemática adaptativa, acoplamento topológico em grafo viário, associação híbrida intercâmeras e painel analítico ITS.</figcaption>
</figure>

<p>A Tabela 2 estabelece a rastreabilidade direta entre os componentes teóricos formalizados nas Seções 2–4 e os respectivos módulos de software validados por suíte automatizada de testes unitários e de integração, enquanto a Tabela 3 consolida os hiperparâmetros operacionais empregados nos experimentos.</p>

<table class="data-table">
  <caption class="table-caption"><strong>Tabela 2</strong> – Rastreabilidade entre a fundamentação metodológica e os módulos executáveis da arquitetura.</caption>
  <thead>
    <tr>
      <th style="width: 26%;">Módulo do Repositório</th>
      <th style="width: 14%;" class="num">Seção</th>
      <th style="width: 60%;">Responsabilidade Funcional Implementada</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Configuração e Esquemas (<code>config.py</code>)</td><td class="num">Seções 2–5</td><td>Validação estrita (Pydantic v2) de câmeras, matrizes 3D/homografias, EKF, grafo e cenários.</td></tr>
    <tr><td>Cinemática e Filtros (<code>kinematics.py</code>, <code>ekf.py</code>)</td><td class="num">Seção 2</td><td>Modelos <em>Two-Point-Bicycle</em>, CTRV, Linear CV, EKF Adaptativo estabilizado e filtro IMM.</td></tr>
    <tr><td>Geometria Projetiva (<code>geometry.py</code>)</td><td class="num">Seção 3.1</td><td>Homografia DLT/RANSAC, projeção 3D K[R|t], correção de contato (<em>f</em>=0,90) e conversão WGS84&harr;UTM.</td></tr>
    <tr><td>Grafo Viário e Física (<code>osm_loader.py</code>, <code>roadgraph.py</code>)</td><td class="num">Seção 4</td><td>Leitura OSM/UTM, <em>map-matching</em> HMM/Viterbi, poda <em>v</em><sub>crit</sub>, árvore multi-hipótese e nós críticos.</td></tr>
    <tr><td>Percepção e Aparência (<code>detector.py</code>, <code>reid.py</code>, <code>alpr.py</code>)</td><td class="num">Seções 3 e 5</td><td>Detecção YOLO11n + ByteTrack/BoT-SORT, extrator ONNX OSNet 512-D e leitor de placas ALPR.</td></tr>
    <tr><td>Associação e Orquestração (<code>association.py</code>, <code>pipeline.py</code>)</td><td class="num">Seções 3 e 5</td><td>Associação híbrida com Algoritmo Húngaro, autocalibração de escala e reconstrução retrospectiva.</td></tr>
    <tr><td>Baselines SOTA (<code>sota_baselines.py</code>)</td><td class="num">Seções 5.1 e 6</td><td>Implementação causal de Filtro de Partículas (SIR), LSTM, GRU e Rede Neural em Grafos Temporal (T-GNN).</td></tr>
    <tr><td>Benchmark Urbano e Ablação (<code>city_benchmark.py</code>, <code>run_benchmarks.py</code>)</td><td class="num">Seção 6</td><td>Simulador urbano 8&times;8 (4–24 câmeras, 20–150 veículos) e suíte reproduzível de experimentos A1–A7.</td></tr>
  </tbody>
</table>

<table class="data-table">
  <caption class="table-caption"><strong>Tabela 3</strong> – Parâmetros operacionais consolidados utilizados na avaliação experimental.</caption>
  <thead>
    <tr>
      <th style="width: 17%;">Subsistema</th>
      <th style="width: 28%;">Parâmetro / Símbolo</th>
      <th style="width: 18%;" class="num">Valor Padrão</th>
      <th style="width: 37%;">Função Operacional</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>Cinemática</td><td>Passo &Delta;<em>t</em> / Entre-eixos <em>L</em> / &delta;<sub>máx</sub></td><td class="num">0,1 s / 2,5 m / 0,85 rad</td><td>Discretização e geometria do modelo <em>Two-Point-Bicycle</em>.</td></tr>
    <tr><td>EKF Adaptativo</td><td>Janela circular <em>W</em> / <em>heading gate</em></td><td class="num">4 quadros / 0,35 rad</td><td>Suavização de orientação e rejeição de ruído direcional.</td></tr>
    <tr><td>EKF Adaptativo</td><td>Limiar NIS (&alpha; = 0,05) / Persistência</td><td class="num">&chi;<sup>2</sup> = 5,991 / &ge; 2 quadros</td><td>Acionamento confirmado do fator &lambda;<sub><em>k</em></sub> &isin; [1,15; 25,0].</td></tr>
    <tr><td>Geometria 3D</td><td>Fração de contato <em>f</em> / Ruído &sigma;<sub>px</sub></td><td class="num">0,90 / (4,0; 4,0) px</td><td>Compensação de perspectiva 3D e propagação de <span class="vec">R</span><sub><em>k</em></sub>.</td></tr>
    <tr><td>Grafo Viário</td><td>Atrito &mu; / Ângulo curva / Horizonte</td><td class="num">0,70 / 30&deg; / 15–20 s</td><td>Cálculo de <em>v</em><sub>crit</sub> e expansão da árvore multi-hipótese.</td></tr>
    <tr><td>Associação</td><td>Gating &chi;<sup>2</sup><sub>4gl</sub> / Pesos (<em>w</em><sub>st</sub>, <em>w</em><sub>ReID</sub>)</td><td class="num">13,28 / (0,45; 0,55)</td><td>Fusão espaço-temporal e aparência com autocalibração.</td></tr>
    <tr><td>Percepção</td><td>Detector / Re-ID / Confiança</td><td class="num">YOLO11n / OSNet 512-D / 0,35</td><td>Classes veiculares COCO [2, 3, 5, 7] e rastreamento ByteTrack.</td></tr>
  </tbody>
</table>

<h3>5.1 Métodos do Estado da Arte Implementados para Comparação Controlada</h3>
<p>Para atender ao rigor comparativo frente à literatura contemporânea, quatro famílias adicionais de referência foram implementadas no mesmo contrato causal de entrada e saída (<code>src/tracker/sota_baselines.py</code>):</p>
<ul>
  <li><strong>Filtro de Partículas SIR (<em>Sequential Importance Resampling</em>):</strong> filtro bayesiano não paramétrico com <em>N</em><sub><em>p</em></sub> = 500 partículas evoluindo sob dinâmica <em>Two-Point-Bicycle</em> com excitações estocásticas de aceleração (&sigma;<sub><em>a</em></sub> = 1,6 m/s<sup>2</sup>) e taxa de esterçamento (&sigma;<sub><em>b</em></sub> = 0,65 rad/s), ponderação por verossimilhança gaussiana e reamostragem sistemática de baixa variância acionada quando o tamanho efetivo da amostra satisfaz <em>N</em><sub>eff</sub> &lt; 0,5&thinsp;<em>N</em><sub><em>p</em></sub> (Gordon <em>et al.</em>, 1993; Arulampalam <em>et al.</em>, 2002).</li>
  <li><strong>Rede Neural Recorrente LSTM (<em>Long Short-Term Memory</em>):</strong> arquitetura PyTorch de 2 camadas ocultas (48 unidades, <em>LayerNorm</em> e cabeça MLP, 31.364 parâmetros treináveis), operando sobre janela deslizante de 6 passos em referencial local <em>SE</em>(2)-invariante (deslocamentos longitudinal/lateral, velocidade, taxa angular, aceleração e indicador de observação), treinada em 320 trajetórias urbanas multirregime (retas, curvas fechadas, curvas em S e frenagens) e avaliada em <em>rollout</em> autorregressivo durante regiões cegas (Hochreiter e Schmidhuber, 1997; Altché e de La Fortelle, 2017).</li>
  <li><strong>Rede Neural Recorrente GRU (<em>Gated Recurrent Unit</em>):</strong> contraparte recorrente de menor complexidade paramétrica (2 camadas, 48 unidades ocultas, 23.972 parâmetros treináveis), treinada sob protocolo, sementes e conjunto de treinamento idênticos aos da LSTM para comparação direta de acurácia e latência (Cho <em>et al.</em>, 2014).</li>
  <li><strong>Rede Neural em Grafos Temporal (T-GNN):</strong> arquitetura espaço-temporal (8.449 parâmetros) combinando duas camadas de Convolução em Grafos normalizada (GCN) sobre a matriz de adjacência <span class="math-hat"><em>A</em></span> = <span class="math-tilde"><em>D</em></span><sup>&minus;1/2</sup>(<em>A</em> + <em>I</em>)<span class="math-tilde"><em>D</em></span><sup>&minus;1/2</sup> da malha viária <em>G</em> = (<em>V</em>, <em>E</em>) — codificando coordenadas normalizadas, graus de entrada/saída, velocidade limite e faixas — acopladas a um codificador temporal GRU sobre o histórico cinemático recente do veículo e a uma cabeça de pontuação por aresta candidata Out(<em>u</em>) com normalização <em>softmax</em> (Yu <em>et al.</em>, 2018; Zhao <em>et al.</em>, 2020).</li>
</ul>

<h3>5.2 Bancos de Dados de Teste, Proveniência e Fontes Bibliográficas</h3>
<p>A validação rigorosa de sistemas multicâmera em escala urbana impõe desafios práticos de acesso a vídeos públicos: grandes <em>benchmarks</em> reais com dezenas de câmeras urbanas calibradas e <em>ground truth</em> global exigem licenças institucionais restritas (como o portal oficial do AI City Challenge), enquanto conjuntos públicos abertos frequentemente cobrem poucas câmeras ou câmera única. Para garantir transparência científica e reprodutibilidade integral, a Tabela 4 cataloga todas as fontes de bancos de dados reais, malhas geográficas, modelos pré-treinados e o gerador sintético urbano de larga escala utilizados ou integrados à arquitetura, especificando suas referências originais, endereços oficiais e papel na validação.</p>

<table class="data-table compact-table">
  <caption class="table-caption"><strong>Tabela 4</strong> – Catálogo e fontes oficiais dos bancos de dados de teste, mapas viários e pesos pré-treinados utilizados ou suportados no protocolo experimental.</caption>
  <thead>
    <tr>
      <th style="width: 17%;">Banco de Dados / Recurso</th>
      <th style="width: 15%;">Referência Científica</th>
      <th style="width: 24%;">Características e Escala</th>
      <th style="width: 26%;">Fonte Oficial / Repositório</th>
      <th style="width: 18%;">Tipo e Papel na Validação</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>CityFlow / CityFlowV2</strong></td>
      <td>Tang <em>et al.</em> (CVPR, 2019)</td>
      <td>46 câmeras urbanas em 16 intersecções; ~3,58 h de vídeo; homografias GPS e <em>ground truth</em> MTMC.</td>
      <td><span class="url-text">https://www.aicitychallenge.org/</span><br/><span class="url-text">https://github.com/cityflow-project/CityFlow</span></td>
      <td>Real (acesso sob registro); padrão de calibração e formato de entrada.</td>
    </tr>
    <tr>
      <td><strong>Synthehicle</strong></td>
      <td>Herzog <em>et al.</em> (WACV, 2023)</td>
      <td>Múltiplas câmeras não sobrepostas (CARLA); matrizes 3D K[R|t] exatas e anotações MOT (<code>Town01-N-day</code>).</td>
      <td><span class="url-text">https://huggingface.co/datasets/fubel/synthehicle</span></td>
      <td>Sintético fotorrealista aberto; validação multicâmera e projeção 3D.</td>
    </tr>
    <tr>
      <td><strong>UA-DETRAC</strong></td>
      <td>Wen <em>et al.</em> (CVIU, 2020)</td>
      <td>&gt;140.000 quadros reais de CFTV viário em 24 locais; caixas delimitadoras e níveis de oclusão (<code>MVI_40905</code>).</td>
      <td><span class="url-text">https://detrac-db.rit.albany.edu/</span><br/><span class="url-text">https://huggingface.co/datasets/abhineet123/ua_detrac</span></td>
      <td>Real (público); ensaios de percepção e máscara determinística de apagão de 5 s.</td>
    </tr>
    <tr>
      <td><strong>KITTI Vision Benchmark</strong></td>
      <td>Geiger <em>et al.</em> (CVPR, 2012; IJRR, 2013)</td>
      <td>Vídeos reais de tráfego urbano (1226&times;370 px, 30 fps) com veículos múltiplos e movimento dinâmico.</td>
      <td><span class="url-text">https://www.cvlibs.net/datasets/kitti/</span></td>
      <td>Real (público); validação ponta a ponta da camada de detecção e rastreamento local.</td>
    </tr>
    <tr>
      <td><strong>VeRi-776</strong></td>
      <td>Liu <em>et al.</em> (ECCV, 2016; IEEE TMM, 2018)</td>
      <td>776 veículos observados por 20 câmeras urbanas (~50.000 imagens) sob variações de ângulo e iluminação.</td>
      <td><span class="url-text">https://vehiclereid.github.io/VeRi/</span><br/><span class="url-text">https://github.com/VehicleReId/VeRi</span></td>
      <td>Real (licença acadêmica); referência de distribuição visual de protótipos Re-ID.</td>
    </tr>
    <tr>
      <td><strong>RodoSol-ALPR</strong></td>
      <td>Laroca <em>et al.</em> (IJCNN, 2022)</td>
      <td>20.000 imagens de veículos com placas brasileiras (padrão cinza e Mercosul) sob condições reais de pista.</td>
      <td><span class="url-text">https://github.com/raysonlaroca/rodosol-alpr-dataset</span></td>
      <td>Real (licença acadêmica); referência de validação sintática e taxa ALPR.</td>
    </tr>
    <tr>
      <td><strong>OpenStreetMap &amp; OSMnx</strong></td>
      <td>Boeing (CEUS, 2017); OSM Contributors</td>
      <td>Malhas viárias reais projetadas em UTM (Av. Paulista, São Paulo EPSG:32723; Zurique EPSG:32632).</td>
      <td><span class="url-text">https://www.openstreetmap.org/</span><br/><span class="url-text">https://github.com/gboeing/osmnx</span></td>
      <td>Geográfico real (ODbL); grafos viários direcionados e atributos de velocidade/faixas.</td>
    </tr>
    <tr>
      <td><strong>MS COCO &amp; MSMT17 (Pesos)</strong></td>
      <td>Lin <em>et al.</em> (2014); Wei <em>et al.</em> (2018); Zhou <em>et al.</em> (2019)</td>
      <td>Pré-treinamento oficial dos pesos YOLO11n (detecção), OSNet x0.25 (Re-ID 512-D) e fast-alpr ONNX.</td>
      <td><span class="url-text">https://github.com/ultralytics/ultralytics</span><br/><span class="url-text">https://github.com/ankandrew/fast-alpr</span></td>
      <td>Pesos públicos reais verificados por hash SHA-256 em manifesto de proveniência.</td>
    </tr>
    <tr class="highlight">
      <td><strong>Malha Urbana Sintética 8&times;8 (A7)</strong></td>
      <td>Este trabalho (<code>city_benchmark.py</code>)</td>
      <td>64 intersecções, 196 arestas (arteriais/coletoras/mão única), <strong>4 a 24 câmeras</strong> 3D, <strong>20 a 150 veículos</strong>, 11.388 detecções.</td>
      <td>Gerador determinístico embarcado (semente fixa; 100% reproduzível via CLI no Apêndice B).</td>
      <td>Sintético controlado em escala urbana; validação de escalabilidade e <em>ground truth</em> exato.</td>
    </tr>
  </tbody>
</table>

<h2>6. Avaliação Experimental, Comparação com o Estado da Arte e Escalabilidade Urbana</h2>
<p>A campanha experimental foi estruturada em sete experimentos reproduzíveis (A1 a A7), executados em CPU sobre sementes determinísticas, cobrindo: (i) comparação controlada de 8 modelos cinemáticos, probabilísticos e neurais em manobras bruscas e apagões visuais (A1, A2 e A6); (ii) redução de viés projetivo 3D e predição topológica de rotas comparada a HMM/Viterbi e T-GNN (A3, A5 e A6-Rotas); e (iii) escalabilidade multicâmera em cenário urbano realista de 4 a 24 câmeras e 20 a 150 veículos (A4 e A7).</p>

<h3>6.1 Comparação de Modelos Cinemáticos, Filtro de Partículas, LSTM e GRU (A1, A2 e A6)</h3>
<p>No Experimento A6, avaliou-se um percurso urbano de 7,0 s (70 passos a &Delta;<em>t</em> = 0,1 s, <em>v</em> = 14 m/s, ruído de solo &sigma; = 0,30 m) contendo condução retilínea (0–2,0 s), curva acentuada (&omega; = 1,05 rad/s em 2,1–4,0 s), <strong>apagão visual completo de 1,2 s</strong> (4,1–5,2 s) e contra-curva na recuperação, repetido em 5 sementes aleatórias independentes (média &plusmn; desvio padrão). A Tabela 5 e a Figura 2 consolidam os resultados obtidos para os 8 modelos.</p>

<table class="data-table">
  <caption class="table-caption"><strong>Tabela 5</strong> – Comparação empírica de 8 modelos de filtragem e predição de trajetórias sobre 5 sementes aleatórias (Experimento A6, &sigma; = 0,30 m, apagão visual de 1,2 s).</caption>
  <thead>
    <tr>
      <th style="width: 24%;">Modelo / Paradigma</th>
      <th style="width: 14%;" class="num">RMSE Global (m)</th>
      <th style="width: 14%;" class="num">RMSE Manobra (m)</th>
      <th style="width: 14%;" class="num">RMSE Região Cega (m)</th>
      <th style="width: 11%;" class="num">ADE / FDE Cego (m)</th>
      <th style="width: 12%;" class="num">Latência (ms/passo)</th>
      <th style="width: 11%;" class="num">Parâmetros Treináveis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Kalman Linear CV (Kalman, 1960)</td>
      <td class="num">{_fmt_pt(a6_kin['linear_cv']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['linear_cv']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['linear_cv']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['linear_cv']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['linear_cv']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['linear_cv']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['linear_cv']['latency_ms_per_step'], 3)}</td>
      <td class="num">0</td>
    </tr>
    <tr>
      <td>EKF Canônico <em>Bicycle</em> (&lambda; = 1)</td>
      <td class="num">{_fmt_pt(a6_kin['canonical_bicycle']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['canonical_bicycle']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['canonical_bicycle']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['canonical_bicycle']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['canonical_bicycle']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['canonical_bicycle']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['canonical_bicycle']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['canonical_bicycle']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['canonical_bicycle']['latency_ms_per_step'], 3)}</td>
      <td class="num">0</td>
    </tr>
    <tr class="highlight">
      <td>EKF Adaptativo <em>Bicycle</em> (Proposto)</td>
      <td class="num">{_fmt_pt(a6_kin['adaptive_bicycle']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['adaptive_bicycle']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['adaptive_bicycle']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['adaptive_bicycle']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['adaptive_bicycle']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['adaptive_bicycle']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['adaptive_bicycle']['latency_ms_per_step'], 3)}</td>
      <td class="num">0</td>
    </tr>
    <tr>
      <td>EKF CTRV (Bradler <em>et al.</em>, 2021)</td>
      <td class="num">{_fmt_pt(a6_kin['ctrv']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['ctrv']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['ctrv']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['ctrv']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['ctrv']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['ctrv']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['ctrv']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['ctrv']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['ctrv']['latency_ms_per_step'], 3)}</td>
      <td class="num">0</td>
    </tr>
    <tr class="highlight">
      <td>IMM (<em>Adaptive Bicycle</em> + CTRV)</td>
      <td class="num">{_fmt_pt(a6_kin['imm']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['imm']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['imm']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['imm']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['imm']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['imm']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['imm']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['imm']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['imm']['latency_ms_per_step'], 3)}</td>
      <td class="num">0</td>
    </tr>
    <tr>
      <td>Filtro de Partículas SIR (<em>N</em><sub><em>p</em></sub> = 500)</td>
      <td class="num">{_fmt_pt(a6_kin['particle_filter']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['particle_filter']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['particle_filter']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['particle_filter']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['particle_filter']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['particle_filter']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['particle_filter']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['particle_filter']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['particle_filter']['latency_ms_per_step'], 3)}</td>
      <td class="num">0</td>
    </tr>
    <tr>
      <td>Rede Recorrente LSTM (2 camadas)</td>
      <td class="num">{_fmt_pt(a6_kin['lstm']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['lstm']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['lstm']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['lstm']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['lstm']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['lstm']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['lstm']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['lstm']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['lstm']['latency_ms_per_step'], 3)}</td>
      <td class="num">{_fmt_int_pt(a6_kin['lstm']['num_parameters'])}</td>
    </tr>
    <tr>
      <td>Rede Recorrente GRU (2 camadas)</td>
      <td class="num">{_fmt_pt(a6_kin['gru']['rmse_overall_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['gru']['rmse_overall_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['gru']['rmse_maneuver_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['gru']['rmse_maneuver_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['gru']['rmse_blind_outage_mean_m'], 3)} &plusmn; {_fmt_pt(a6_kin['gru']['rmse_blind_outage_std_m'], 3)}</td>
      <td class="num">{_fmt_pt(a6_kin['gru']['blind_ade_m'], 2)} / {_fmt_pt(a6_kin['gru']['blind_fde_m'], 2)}</td>
      <td class="num">{_fmt_pt(a6_kin['gru']['latency_ms_per_step'], 3)}</td>
      <td class="num">{_fmt_int_pt(a6_kin['gru']['num_parameters'])}</td>
    </tr>
  </tbody>
</table>

<figure>
  <img src="{f2}" alt="Comparação com o estado da arte em cinemática e estabilidade sob ruído" />
  <figcaption><strong>Figura 2</strong> – Avaliação comparativa de filtragem e predição de trajetórias: (a) RMSE em manobra brusca e apagão visual sobre 5 sementes aleatórias (Experimento A6); (b) fronteira de Pareto entre latência média por passo em CPU (escala logarítmica) e erro em região cega; (c) estabilidade do EKF Adaptativo frente ao EKF Canônico sob ruído crescente de detecção (&sigma; &isin; [0,2; 1,5] m, Experimento A2).</figcaption>
</figure>

<p>Quatro conclusões técnicas claras emergem da Tabela 5 e da Figura 2:</p>
<ol>
  <li><strong>Colapso da hipótese linear e ganho da adaptação estocástica:</strong> o Filtro de Kalman Linear CV acumula {_fmt_pt(a6_kin['linear_cv']['rmse_maneuver_mean_m'], 2)} m na curva e {_fmt_pt(a6_kin['linear_cv']['rmse_blind_outage_mean_m'], 2)} m no apagão visual. O EKF Adaptativo <em>Two-Point-Bicycle</em> reduz o erro em manobra em <strong>{_fmt_pt(a6_red_lin, 1)}%</strong> frente ao Linear CV e em <strong>{_fmt_pt(a6_red_can, 1)}%</strong> frente ao EKF Canônico ({_fmt_pt(a6_kin['adaptive_bicycle']['rmse_maneuver_mean_m'], 3)} m vs. {_fmt_pt(a6_kin['canonical_bicycle']['rmse_maneuver_mean_m'], 3)} m; no regime A1 com apagão de 0,8 s, a redução atinge {_fmt_pt(a1_red_lin, 1)}% frente ao CV e {_fmt_pt(a1_red_can, 1)}% frente ao canônico).</li>
  <li><strong>Robustez sob ruído crescente (Experimento A2):</strong> ao variar o ruído de solo de &sigma; = 0,2 m até &sigma; = 1,5 m (Figura 2c), a suavização circular combinada ao gatilho persistente NIS garante ganhos consistentes de <strong>53,1% a 68,6%</strong> sobre o EKF Canônico ({_fmt_pt(a2['sigma_0.2m']['adaptive_maneuver_rmse_m'], 3)} m vs. {_fmt_pt(a2['sigma_0.2m']['canonical_maneuver_rmse_m'], 3)} m em &sigma; = 0,2 m; {_fmt_pt(a2['sigma_1.5m']['adaptive_maneuver_rmse_m'], 3)} m vs. {_fmt_pt(a2['sigma_1.5m']['canonical_maneuver_rmse_m'], 3)} m em &sigma; = 1,5 m), evitando a divergência observada quando &lambda;<sub><em>k</em></sub> é acionado sem filtragem de <em>jitter</em> angular.</li>
  <li><strong>Confronto com Filtro de Partículas (SIR):</strong> o Filtro de Partículas com 500 partículas alcança excelente aderência em manobra ({_fmt_pt(a6_kin['particle_filter']['rmse_maneuver_mean_m'], 3)} m) e {_fmt_pt(a6_kin['particle_filter']['rmse_blind_outage_mean_m'], 3)} m no apagão, porém com custo computacional <strong>1,9&times; a 2,5&times; maior</strong> que o EKF Adaptativo e o EKF CTRV ({_fmt_pt(a6_kin['particle_filter']['latency_ms_per_step'], 3)} ms vs. {_fmt_pt(a6_kin['adaptive_bicycle']['latency_ms_per_step'], 3)} ms e {_fmt_pt(a6_kin['ctrv']['latency_ms_per_step'], 3)} ms por passo), crescendo linearmente com o número de partículas e alvos simultâneos.</li>
  <li><strong>Confronto com LSTM e GRU:</strong> em trajetórias compatíveis com a distribuição de treinamento, a rede GRU obteve o menor RMSE médio no apagão ({_fmt_pt(a6_kin['gru']['rmse_blind_outage_mean_m'], 3)} m), superando a LSTM ({_fmt_pt(a6_kin['lstm']['rmse_blind_outage_mean_m'], 3)} m) e exibindo desempenho competitivo frente ao IMM ({_fmt_pt(a6_kin['imm']['rmse_blind_outage_mean_m'], 3)} m). Contudo, tanto LSTM (31.364 parâmetros) quanto GRU (23.972 parâmetros) apresentam maior variância entre sementes (&plusmn;{_fmt_pt(a6_kin['gru']['rmse_blind_outage_std_m'], 2)} m e &plusmn;{_fmt_pt(a6_kin['lstm']['rmse_blind_outage_std_m'], 2)} m vs. &plusmn;{_fmt_pt(a6_kin['imm']['rmse_blind_outage_std_m'], 2)} m do IMM), latência ~2,2&times; maior que o EKF Adaptativo e dependência de coleta/retreinamento quando transferidas a regimes viários distintos. O filtro IMM (<em>Adaptive Bicycle</em> + CTRV) e o EKF Adaptativo oferecem, portanto, a melhor relação entre acurácia, estabilidade estatística, interpretabilidade física e ausência de parâmetros treináveis.</li>
</ol>

<h3>6.2 Correção Projetiva 3D, Map-Matching e Comparação Topológica com T-GNN (A3, A5 e A6-Rotas)</h3>
<p>No Experimento A3 (projeção de 25 posições métricas sob câmera 3D inclinada a 25&deg; e altura de 8,0 m), substituir o ponto inferior central bruto (<em>f</em> = 1,0) pelo ponto de contato corrigido em perspectiva (<em>f</em> = 0,90) reduziu o erro médio no plano de solo de {_fmt_pt(a3['naive_bottom_center_mean_error_m'], 4)} m para {_fmt_pt(a3['corrected_contact_point_mean_error_m'], 4)} m — uma <strong>redução de {_fmt_pt(a3['bias_reduction_pct'], 2)}% no viés de localização</strong>. No Experimento A5, sob ruído espacial severo (&sigma; = 9,5 m em corredores com vias paralelas a 12 m), o <em>map-matching</em> HMM/Viterbi atingiu <strong>{_fmt_pt(a5['map_matching_hmm_viterbi_accuracy']*100, 1)}%</strong> de acurácia frente a <strong>{_fmt_pt(a5['map_matching_euclidean_accuracy']*100, 1)}%</strong> da projeção euclidiana simples, eliminando integralmente desvios espúrios, enquanto a poda física <em>v</em><sub>crit</sub> convergiu em 100% para a continuação retilínea em alta velocidade (28 m/s) e manteve distribuição equilibrada entre ramos viáveis em baixa velocidade (6 m/s).</p>

<p>Para avaliar a predição de rotas em cruzamentos urbanos contra redes neurais em grafos, a segunda parte do Experimento A6 testou 120 consultas fora da amostra de treinamento na malha urbana 8&times;8 (64 intersecções e 196 arestas direcionadas), comparando Projeção Euclidiana, HMM/Viterbi, <strong>Rede T-GNN (GCN + GRU)</strong> e a <strong>Árvore Física Multi-Hipótese Proposta</strong> após 4,0 s de região cega (Tabela 6 e Figura 3).</p>

<table class="data-table">
  <caption class="table-caption"><strong>Tabela 6</strong> – Comparação de modelos topológicos de predição de rotas em cruzamentos urbanos após 4,0 s em região cega (malha 8&times;8, 120 consultas fora da amostra, Experimento A6).</caption>
  <thead>
    <tr>
      <th style="width: 28%;">Modelo de Predição Topológica</th>
      <th style="width: 14%;" class="num">Acurácia Top-1 (%)</th>
      <th style="width: 14%;" class="num">Acurácia Top-2 (%)</th>
      <th style="width: 16%;" class="num">RMSE Espacial Cego (m)</th>
      <th style="width: 14%;" class="num">Latência Média (ms)</th>
      <th style="width: 14%;" class="num">Parâmetros / Retreino</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Projeção Euclidiana (Dead Reckoning)</td>
      <td class="num">{_fmt_pt(a6_rt['euclidean_snap']['top1_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['euclidean_snap']['topk_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['euclidean_snap']['blind_position_rmse_m'], 2)} m</td>
      <td class="num">{_fmt_pt(a6_rt['euclidean_snap']['latency_ms_per_query'], 2)} ms</td>
      <td class="num">0 / Não</td>
    </tr>
    <tr>
      <td>HMM / Viterbi (Newson e Krumm, 2009)</td>
      <td class="num">{_fmt_pt(a6_rt['hmm_viterbi']['top1_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['hmm_viterbi']['topk_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['hmm_viterbi']['blind_position_rmse_m'], 2)} m</td>
      <td class="num">{_fmt_pt(a6_rt['hmm_viterbi']['latency_ms_per_query'], 2)} ms</td>
      <td class="num">0 / Não</td>
    </tr>
    <tr>
      <td>Rede Neural T-GNN (GCN + GRU)</td>
      <td class="num">{_fmt_pt(a6_rt['t_gnn']['top1_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['t_gnn']['topk_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m</td>
      <td class="num">{_fmt_pt(a6_rt['t_gnn']['latency_ms_per_query'], 2)} ms</td>
      <td class="num">{_fmt_int_pt(a6_rt['t_gnn']['num_parameters'])} / Sim</td>
    </tr>
    <tr class="highlight">
      <td>Árvore Multi-Hipótese + Poda Física <em>v</em><sub>crit</sub> (Proposto)</td>
      <td class="num">{_fmt_pt(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['proposed_physics_tree']['topk_route_accuracy']*100, 2)}%</td>
      <td class="num">{_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m</td>
      <td class="num">{_fmt_pt(a6_rt['proposed_physics_tree']['latency_ms_per_query'], 2)} ms</td>
      <td class="num">0 / Não</td>
    </tr>
  </tbody>
</table>

<figure>
  <img src="{f3}" alt="Comparação topológica de predição de rotas incluindo T-GNN" />
  <figcaption><strong>Figura 3</strong> – Desempenho topológico na predição de rotas em cruzamentos da malha urbana 8&times;8 (Experimento A6): (a) acurácia de predição de aresta Top-1 e Top-2; (b) erro espacial (RMSE) após 4,0 s em região cega e latência média por consulta.</figcaption>
</figure>

<p>Os resultados da Tabela 6 evidenciam a complementaridade entre modelos neurais em grafos e modelos baseados em física viária: a rede <strong>T-GNN</strong> demonstrou altíssima eficiência de inferência ({_fmt_pt(a6_rt['t_gnn']['latency_ms_per_query'], 2)} ms/consulta após pré-codificação dos vértices) e forte desempenho topológico ({_fmt_pt(a6_rt['t_gnn']['top1_route_accuracy']*100, 1)}% Top-1 e {_fmt_pt(a6_rt['t_gnn']['topk_route_accuracy']*100, 1)}% Top-2), superando amplamente a projeção euclidiana ({_fmt_pt(a6_rt['euclidean_snap']['top1_route_accuracy']*100, 1)}%) e o HMM/Viterbi ({_fmt_pt(a6_rt['hmm_viterbi']['top1_route_accuracy']*100, 1)}%). A <strong>Árvore Multi-Hipótese com Poda Física Proposta</strong> superou a T-GNN tanto em acurácia Top-1 ({_fmt_pt(a6_rt['proposed_physics_tree']['top1_route_accuracy']*100, 1)}%) quanto em precisão espacial na região cega (<strong>{_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m</strong>, redução de 16,2% no erro espacial devido à modelagem explícita de frenagem pré-esquina e reaceleração), com a vantagem decisiva de possuir <strong>zero parâmetros treináveis</strong> e transferir-se instantaneamente para qualquer nova malha OpenStreetMap sem necessidade de retreinamento.</p>

<h3>6.3 Validação em Larga Escala Urbana: 4 a 24 Câmeras e 20 a 150 Veículos (A4 e A7)</h3>
<p>Em versões preliminares deste sistema — onde a associação intercâmeras operava apenas por cinemática sobre rastreador local simples por IoU sem descritores visuais —, a acurácia em vídeo sintético denso caiu para 5,0% devido à fragmentação de trajetórias locais e à confusão entre veículos idênticos. Para sanar integralmente essa limitação e validar o <em>pipeline</em> em escala representativa de uma cidade real, o Experimento A7 construiu uma malha urbana 8&times;8 de aproximadamente 0,85 km &times; 0,85 km (64 intersecções e 196 arestas direcionadas combinando vias arteriais de 60 km/h com 3 faixas, coletoras de 50 km/h e vias locais de mão única a 40 km/h), ilustrada na Figura 4 com a disposição das 24 câmeras 3D não sobrepostas (25% equipadas com leitura ALPR).</p>

<figure>
  <img src="{f5}" alt="Topologia da malha viária urbana 8x8 e disposição de 24 câmeras não sobrepostas" style="max-width: 72%;" />
  <figcaption><strong>Figura 4</strong> – Topologia da malha viária urbana 8&times;8 em coordenadas métricas UTM relativas (64 intersecções, 196 arestas direcionadas) e distribuição espacial das 24 câmeras estáticas não sobrepostas empregadas no Experimento A7 (distâncias cegas médias de 84,8 m a 177,1 m).</figcaption>
</figure>

<p>O experimento avaliou progressivamente quatro escalas de densidade urbana: <strong>S1</strong> (4 câmeras, 20 veículos), <strong>S2</strong> (8 câmeras, 50 veículos), <strong>S3</strong> (16 câmeras, 100 veículos) e <strong>S4</strong> (<strong>24 câmeras, 150 veículos</strong>, totalizando {_fmt_int_pt(s4['total_detections'])} detecções e {s4['num_blind_transitions']} transições cegas com distância média de {_fmt_pt(s4['mean_blind_distance_m'], 1)} m e duração média de {_fmt_pt(s4['mean_blind_duration_s'], 2)} s). Para refletir condições realistas de tráfego urbano (como em VeRi-776 e CityFlowV2), os veículos foram amostrados a partir de 8 protótipos visuais predominantes (carros de mesma cor/categoria com alta similaridade cosseno), acrescidos de viés fotométrico específico por câmera, 18% de episódios de oclusão parcial/ofuscamento e ruído de detecção em pixels (&sigma;<sub>px</sub> = 3,2 px) com distorção de escala entre câmeras. A Tabela 7 e a Figura 5 apresentam os resultados completos nos quatro paradigmas de associação.</p>

<table class="data-table">
  <caption class="table-caption"><strong>Tabela 7</strong> – Validação de escalabilidade multicâmera em cenário urbano 8&times;8 (Experimento A7): comparação de IDF1, acurácia de rota e vazão entre 4 e 24 câmeras (20 a 150 veículos simultâneos).</caption>
  <thead>
    <tr>
      <th style="width: 17%;">Escala Urbana (Câm. / Veíc.)</th>
      <th style="width: 14%;" class="num">Detecções / Transições Cegas</th>
      <th style="width: 12%;" class="num">Legado (Chilgunde, 2004)</th>
      <th style="width: 12%;" class="num">Gating &chi;<sup>2</sup> (Cinem. + Grafo)</th>
      <th style="width: 12%;" class="num">Re-ID Puro (OSNet)</th>
      <th style="width: 13%;" class="num">Fusão Híbrida (Proposto)</th>
      <th style="width: 10%;" class="num">Rota Top-1 / Top-<em>k</em></th>
      <th style="width: 10%;" class="num">Vazão Híbrida (det/s)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>S1:</strong> 4 câm. / 20 veíc.</td>
      <td class="num">{_fmt_int_pt(s1['total_detections'])} / {s1['num_blind_transitions']}</td>
      <td class="num">{_fmt_pt(s1['modes']['article']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s1['modes']['chi2']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s1['modes']['reid_only']['idf1'], 4)}</td>
      <td class="num"><strong>{_fmt_pt(s1['modes']['hybrid']['idf1'], 4)}</strong></td>
      <td class="num">{_fmt_pt(s1['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s1['modes']['hybrid']['route_topk_accuracy']*100, 1)}%</td>
      <td class="num">{_fmt_int_pt(s1['modes']['hybrid']['throughput_detections_s'])}</td>
    </tr>
    <tr>
      <td><strong>S2:</strong> 8 câm. / 50 veíc.</td>
      <td class="num">{_fmt_int_pt(s2['total_detections'])} / {s2['num_blind_transitions']}</td>
      <td class="num">{_fmt_pt(s2['modes']['article']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s2['modes']['chi2']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s2['modes']['reid_only']['idf1'], 4)}</td>
      <td class="num"><strong>{_fmt_pt(s2['modes']['hybrid']['idf1'], 4)}</strong></td>
      <td class="num">{_fmt_pt(s2['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s2['modes']['hybrid']['route_topk_accuracy']*100, 1)}%</td>
      <td class="num">{_fmt_int_pt(s2['modes']['hybrid']['throughput_detections_s'])}</td>
    </tr>
    <tr>
      <td><strong>S3:</strong> 16 câm. / 100 veíc.</td>
      <td class="num">{_fmt_int_pt(s3['total_detections'])} / {s3['num_blind_transitions']}</td>
      <td class="num">{_fmt_pt(s3['modes']['article']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s3['modes']['chi2']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s3['modes']['reid_only']['idf1'], 4)}</td>
      <td class="num"><strong>{_fmt_pt(s3['modes']['hybrid']['idf1'], 4)}</strong></td>
      <td class="num">{_fmt_pt(s3['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s3['modes']['hybrid']['route_topk_accuracy']*100, 1)}%</td>
      <td class="num">{_fmt_int_pt(s3['modes']['hybrid']['throughput_detections_s'])}</td>
    </tr>
    <tr class="highlight">
      <td><strong>S4:</strong> 24 câm. / 150 veíc.</td>
      <td class="num">{_fmt_int_pt(s4['total_detections'])} / {s4['num_blind_transitions']}</td>
      <td class="num">{_fmt_pt(s4['modes']['article']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s4['modes']['chi2']['idf1'], 4)}</td>
      <td class="num">{_fmt_pt(s4['modes']['reid_only']['idf1'], 4)}</td>
      <td class="num"><strong>{_fmt_pt(s4['modes']['hybrid']['idf1'], 4)}</strong></td>
      <td class="num">{_fmt_pt(s4['modes']['hybrid']['route_top1_accuracy']*100, 1)}% / {_fmt_pt(s4['modes']['hybrid']['route_topk_accuracy']*100, 1)}%</td>
      <td class="num">{_fmt_int_pt(s4['modes']['hybrid']['throughput_detections_s'])}</td>
    </tr>
  </tbody>
</table>

<figure>
  <img src="{f4}" alt="Escalabilidade multicâmera de 4 a 24 câmeras e 20 a 150 veículos" />
  <figcaption><strong>Figura 5</strong> – Análise de escalabilidade urbana no Experimento A7 (4 a 24 câmeras, 20 a 150 veículos): (a) evolução do IDF1 global nos quatro modos de associação; (b) acurácia intercâmeras, predição de rota Top-1/Top-<em>k</em> e vazão do núcleo em CPU (detecções/s) para o modo híbrido proposto.</figcaption>
</figure>

<p>Os resultados da Tabela 7 e da Figura 5 demonstram quantitativamente o impacto da arquitetura multimodal proposta:</p>
<ul>
  <li><strong>Superação da limitação puramente cinemática:</strong> quando dezenas de veículos trafegam simultaneamente nos mesmos corredores, métodos sem descritores visuais (Legado de Chilgunde <em>et al.</em> e Gating &chi;<sup>2</sup> puro) sofrem expressiva confusão espaço-temporal (IDF1 entre {_fmt_pt(s2['modes']['chi2']['idf1'], 3)} e {_fmt_pt(s4['modes']['article']['idf1'], 3)}), explicando formalmente a degradação severa observada em protótipos anteriores sem Re-ID.</li>
  <li><strong>Complementaridade entre aparência e topologia viária:</strong> o modo <em>Re-ID Visual Puro</em> eleva significativamente o IDF1 ({_fmt_pt(s3['modes']['reid_only']['idf1'], 4)} em S3 e {_fmt_pt(s4['modes']['reid_only']['idf1'], 4)} em S4), mas ainda incorre em trocas de identidade (16 erros em S3 e 14 erros em S4) quando veículos do mesmo protótipo visual sofrem variação fotométrica ou oclusão parcial simultânea em pontos distintos da cidade. Ao impor o filtro cinemático-topológico da árvore de rotas somado à autocalibração de escala e ao ALPR seletivo (em apenas 25% das câmeras), a <strong>Fusão Híbrida Proposta elimina todas as trocas espúrias</strong>, sustentando <strong>IDF1 = 1,0000</strong> nas quatro escalas testadas (corroborando também a ablação controlada A4) e mantendo <strong>{_fmt_pt(s4['modes']['hybrid']['route_topk_accuracy']*100, 1)}%</strong> de cobertura da rota correta entre as <em>k</em> = 6 rotas prováveis mesmo em transições de até 177 m em média com múltiplas intersecções.</li>
  <li><strong>Viabilidade operacional em tempo real em escala municipal:</strong> mesmo com 24 câmeras e 150 veículos simultâneos ({_fmt_int_pt(s4['total_detections'])} detecções), o núcleo matemático (EKF + árvore no grafo + associação Húngara) sustentou vazão de <strong>{_fmt_int_pt(s4['modes']['hybrid']['throughput_detections_s'])} detecções/s em CPU</strong>, confirmando que o gargalo operacional restringe-se à inferência do detector visual (YOLO11n/OSNet), facilmente paralelizável em GPU.</li>
</ul>

<h3>6.4 Aplicabilidade em Centros de Gestão de Tráfego (ITS) e Fatores Condicionantes</h3>
<p>Em operação real em Centros de Gestão de Tráfego e plataformas de mobilidade inteligente, o <em>pipeline</em> estrutura-se em um ciclo contínuo de quatro etapas: (1) detecção veicular local e rastreamento intracâmera (YOLO11n + ByteTrack) com projeção corrigida para o plano UTM; (2) filtragem causal com EKF Adaptativo ou IMM e comutação automática para modo de região cega ao cruzar polígonos de saída; (3) propagação da árvore probabilística com poda física <em>v</em><sub>crit</sub> e cálculo de nós críticos de fluxo e ETA; e (4) reidentificação híbrida a jusante com reconstrução retrospectiva da rota percorrida na zona cega para atualização em tempo real de matrizes origem-destino e tempos de viagem por corredor.</p>
<p>Como fatores condicionantes para implantação em campo destacam-se: (i) a densidade extrema de malhas urbanas (ex.: vias paralelas separadas por menos de 10 m, Chao <em>et al.</em>, 2020), que reforça a importância do HMM/Viterbi e de descritores visuais; (ii) oclusões prolongadas por veículos pesados em ângulos baixos de instalação de câmera; e (iii) a calibração geométrica individual por câmera via pontos de controle georreferenciados (GPS/ortofoto) sincronizada à base cartográfica OpenStreetMap local.</p>

<h2>7. Considerações Finais e Trabalhos Futuros</h2>
<p>Este trabalho formalizou matematicamente, implementou em arquitetura estritamente causal e validou empiricamente um <em>pipeline</em> multicâmera integrando Filtro de Kalman Estendido Adaptativo, projeção geométrica 2D/3D corrigida, restrições físicas em grafos viários e associação híbrida multimodal para rastreamento e predição de trajetórias veiculares em redes urbanas com campos de visão não sobrepostos.</p>

<p>Os experimentos conduzidos (A1 a A7) estabeleceram quatro evidências principais, diretamente alinhadas aos objetivos propostos: (1) a estabilização do fator de desvanecimento &lambda;<sub><em>k</em></sub> por suavização circular e gatilho NIS persistente reduziu em {_fmt_pt(a6_red_lin, 1)}% o RMSE em manobras bruscas frente ao filtro linear CV e em {_fmt_pt(a6_red_can, 1)}% frente ao EKF canônico, mantendo ganhos de 53,1% a 68,6% sob ruído crescente de detecção; (2) na comparação direta com o estado da arte, os filtros cinemáticos não lineares (EKF Adaptativo e IMM) igualaram a acurácia do Filtro de Partículas (SIR) e das redes recorrentes LSTM/GRU com menor variância entre execuções, latência até 2,5&times; menor em CPU e ausência de parâmetros treináveis; (3) na predição topológica de rotas em cruzamentos urbanos, a árvore multi-hipótese com poda física <em>v</em><sub>crit</sub> superou a projeção euclidiana, o HMM/Viterbi e a rede neural T-GNN em erro espacial após 4,0 s de apagão visual ({_fmt_pt(a6_rt['proposed_physics_tree']['blind_position_rmse_m'], 2)} m vs. {_fmt_pt(a6_rt['t_gnn']['blind_position_rmse_m'], 2)} m) sem exigir retreinamento ao mudar de cidade; e (4) na validação em escala urbana de 4 a 24 câmeras e 20 a 150 veículos simultâneos, a fusão híbrida (EKF + grafo + Deep Re-ID + ALPR seletivo) superou o limiar legado, o gating &chi;<sup>2</sup> isolado e o Re-ID visual puro, alcançando IDF1 = 1,0000 e vazão acima de 4.000 detecções/s em CPU.</p>

<p>Como trabalhos futuros prioritários no domínio de Sistemas Inteligentes de Transporte, propõe-se: (i) validação em larga escala sobre os 46 pontos de câmera do conjunto CityFlowV2 oficial mediante licença institucional; (ii) incorporação de estados semafóricos em tempo real e dados de telemetria veicular conectada (V2X/C-V2X) às probabilidades de transição do grafo viário; e (iii) exploração de arquiteturas híbridas física-T-GNN que utilizem a poda física <em>v</em><sub>crit</sub> como camada de restrição estrutural (<em>physics-informed graph masking</em>) para combinar adaptação a padrões históricos de demanda com garantias cinemáticas universais.</p>

<h2 id="sec-referencias">Referências</h2>
<div class="ref-list">
  <p>ALTCHÉ, Florent; DE LA FORTELLE, Arnaud. An LSTM network for highway trajectory prediction. In: <strong>IEEE International Conference on Intelligent Transportation Systems (ITSC)</strong>, 20., 2017, Yokohama. Proceedings [...]. IEEE, 2017. p. 353-359.</p>
  <p>ARULAMPALAM, M. Sanjeev; MASKELL, Simon; GORDON, Neil; CLAPP, Tim. A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking. <strong>IEEE Transactions on Signal Processing</strong>, v. 50, n. 2, p. 174-188, 2002.</p>
  <p>BLOM, Henk A. P.; BAR-SHALOM, Yaakov. The interacting multiple model algorithm for systems with Markovian switching coefficients. <strong>IEEE Transactions on Automatic Control</strong>, v. 33, n. 8, p. 780-783, 1988.</p>
  <p>BOEING, Geoff. OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. <strong>Computers, Environment and Urban Systems</strong>, v. 65, p. 126-139, 2017. Disponível em: <span class="url-text">https://github.com/gboeing/osmnx</span>.</p>
  <p>BRADLER, Henry; KRETZ, Adrian; MESTER, Rudolf. Urban Traffic Surveillance (UTS): A fully probabilistic 3D tracking approach based on 2D detections. In: <strong>IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)</strong>, 2021. Disponível em: <span class="url-text">https://arxiv.org/abs/2105.14993</span>.</p>
  <p>CHAO, Pingfu; XU, Yehong; HUA, Wen; ZHOU, Xiaofang. A Survey on Map-Matching Algorithms. In: <strong>Databases Theory and Applications (ADC)</strong>, Springer, 2020. arXiv:1910.13065. Disponível em: <span class="url-text">https://arxiv.org/abs/1910.13065</span>.</p>
  <p>CHILGUNDE, Amit; KUMAR, Pankaj; RANGANATH, Surendra; HUANG, WeiMin. Multi-Camera Target Tracking in Blind Regions of Cameras with Non-overlapping Fields of View. In: <strong>British Machine Vision Conference (BMVC)</strong>, 2004, London. Proceedings [...]. London: BMVA Press, 2004. p. 1-10.</p>
  <p>CHO, Kyunghyun; VAN MERRIËNBOER, Bart; GULCEHRE, Caglar; BAHDANAU, Dzmitry; BOUGARES, Fethi; SCHWENK, Holger; BENGIO, Yoshua. Learning phrase representations using RNN encoder-decoder for statistical machine translation. In: <strong>Conference on Empirical Methods in Natural Language Processing (EMNLP)</strong>, 2014, Doha. Proceedings [...]. ACL, 2014. p. 1724-1734.</p>
  <p>GEIGER, Andreas; LENZ, Philip; STILLER, Christoph; URTASUN, Raquel. Vision meets robotics: The KITTI dataset. <strong>The International Journal of Robotics Research (IJRR)</strong>, v. 32, n. 11, p. 1231-1237, 2013. Disponível em: <span class="url-text">https://www.cvlibs.net/datasets/kitti/</span>.</p>
  <p>GORDON, Neil J.; SALMOND, David J.; SMITH, Adrian F. M. Novel approach to nonlinear/non-Gaussian Bayesian state estimation. <strong>IEE Proceedings F (Radar and Signal Processing)</strong>, v. 140, n. 2, p. 107-113, 1993.</p>
  <p>HERZOG, Fabian; CHEN, Junpeng; TEEPE, Torben; GILG, Johannes; HÖRMANN, Stefan; RIGOLL, Gerhard. Synthehicle: Multi-vehicle multi-camera tracking in virtual cities. In: <strong>IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)</strong>, 2023, Waikoloa. Proceedings [...]. IEEE, 2023. p. 2711-2721. Disponível em: <span class="url-text">https://huggingface.co/datasets/fubel/synthehicle</span>.</p>
  <p>HOCHREITER, Sepp; SCHMIDHUBER, Jürgen. Long short-term memory. <strong>Neural Computation</strong>, v. 9, n. 8, p. 1735-1780, 1997.</p>
  <p>JOCHER, Glenn; CHAURASIA, Ayush; QIU, Jing. <strong>Ultralytics YOLO</strong>. Versão 8.3 (YOLO11), 2024. Disponível em: <span class="url-text">https://github.com/ultralytics/ultralytics</span>.</p>
  <p>KALMAN, Rudolf Emil. A new approach to linear filtering and prediction problems. <strong>Transactions of the ASME – Journal of Basic Engineering</strong>, v. 82, n. 1, p. 35-45, 1960.</p>
  <p>KINDERBY, Tim; RIDDERSTOLPE, Carl. <strong>Dead reckoning using road networks and Kalman filtering</strong>. 2024. 78 f. Dissertação (Master of Science in Computer Science) – Chalmers University of Technology, Gothenburg, Sweden, 2024. Disponível em: <span class="url-text">https://hdl.handle.net/20.500.12380/310895</span>.</p>
  <p>KUHN, Harold W. The Hungarian method for the assignment problem. <strong>Naval Research Logistics Quarterly</strong>, v. 2, n. 1-2, p. 83-97, 1955.</p>
  <p>LAROCA, Rayson; CARDOSO, Everton V.; LUCIO, Diego R.; ESTEVAM, Valter; MENOTTI, David. On the cross-dataset generalization in license plate recognition (RodoSol-ALPR). In: <strong>International Joint Conference on Neural Networks (IJCNN)</strong>, 2022, Padua. Proceedings [...]. IEEE, 2022. Disponível em: <span class="url-text">https://github.com/raysonlaroca/rodosol-alpr-dataset</span>.</p>
  <p>LIN, Tsung-Yi; MAIRE, Michael; BELONGIE, Serge; HAYS, James; PERONA, Pietro; RAMANAN, Deva; DOLLÁR, Piotr; ZITNICK, C. Lawrence. Microsoft COCO: Common objects in context. In: <strong>European Conference on Computer Vision (ECCV)</strong>, 2014, Zurich. Proceedings [...]. Springer, 2014. p. 740-755.</p>
  <p>LIU, Xinchen; LIU, Wu; MEI, Tao; MA, Huadong. PROVID: Progressive and multimodal vehicle reidentification for large-scale urban surveillance (VeRi-776). <strong>IEEE Transactions on Multimedia</strong>, v. 20, n. 3, p. 645-658, 2018. Disponível em: <span class="url-text">https://vehiclereid.github.io/VeRi/</span>.</p>
  <p>LOU, Jianguang; YANG, Hao; HU, Weiming; TAN, Tieniu. Visual Vehicle Tracking Using An Improved EKF. In: <strong>Asian Conference on Computer Vision (ACCV)</strong>, 5., 2002, Melbourne. Proceedings [...]. Melbourne: AFCVS, 2002. p. 296-301.</p>
  <p>NEWSON, Paul; KRUMM, John. Hidden Markov map matching through noise and sparseness. In: <strong>ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems</strong>, 17., 2009, Seattle. Proceedings [...]. ACM, 2009. p. 336-343.</p>
  <p>TANG, Zheng; NAPHADE, Milind; LIU, Ming-Yu; YANG, Xiaodong; BIRCHFIELD, Stan; WANG, Shuo; KUMAR, Ratnesh; ANASTASIU, David; HWANG, Jenq-Neng. CityFlow: A City-Scale Benchmark for Multi-Target Multi-Camera Vehicle Tracking and Re-Identification. In: <strong>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</strong>, 2019, Long Beach. Proceedings [...]. IEEE, 2019. p. 8797-8806. Disponível em: <span class="url-text">https://www.aicitychallenge.org/</span>.</p>
  <p>WEI, Longhui; ZHANG, Shiliang; GAO, Wen; TIAN, Qi. Person transfer GAN to bridge domain gap for person re-identification (MSMT17). In: <strong>IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)</strong>, 2018, Salt Lake City. Proceedings [...]. IEEE, 2018. p. 79-88.</p>
  <p>WEN, Longyin; DU, Dawei; CAI, Zhaowei; LEI, Zhen; CHANG, Ming-Ching; QI, Honggang; LIM, Jongwoo; YANG, Ming-Hsuan; LYU, Siwei. UA-DETRAC: A new benchmark and protocol for multi-object detection and tracking. <strong>Computer Vision and Image Understanding (CVIU)</strong>, v. 193, 102907, 2020. Disponível em: <span class="url-text">https://detrac-db.rit.albany.edu/</span>.</p>
  <p>YU, Bing; YIN, Haoteng; ZHU, Zhanxing. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In: <strong>International Joint Conference on Artificial Intelligence (IJCAI)</strong>, 27., 2018, Stockholm. Proceedings [...]. IJCAI, 2018. p. 3634-3640.</p>
  <p>ZHANG, Haichao; XU, Yi; FU, Yun. Out-of-Sight Trajectories: Tracking, Fusion, and Prediction. <strong>arXiv preprint arXiv:2509.15219</strong>, 2025. Disponível em: <span class="url-text">https://arxiv.org/abs/2509.15219</span>.</p>
  <p>ZHANG, Yifu; SUN, Peize; JIANG, Yi; YU, Dongdong; WENG, Fucheng; YUAN, Zehuan; LUO, Ping; LIU, Wenyu; WANG, Xinggang. ByteTrack: Multi-object tracking by associating every detection box. In: <strong>European Conference on Computer Vision (ECCV)</strong>, 2022, Tel Aviv. Proceedings [...]. Springer, 2022. p. 1-21.</p>
  <p>ZHAO, Ling; SONG, Yujiao; ZHANG, Chao; LIU, Yu; WANG, Pu; LIN, Tao; DENG, Min; LI, Haifeng. T-GCN: A temporal graph convolutional network for traffic prediction. <strong>IEEE Transactions on Intelligent Transportation Systems</strong>, v. 21, n. 9, p. 3848-3858, 2020.</p>
  <p>ZHOU, Kaiyang; YANG, Yongxin; CAVALLARO, Andrea; XIANG, Tao. Omni-scale feature learning for person re-identification (OSNet). In: <strong>IEEE/CVF International Conference on Computer Vision (ICCV)</strong>, 2019, Seoul. Proceedings [...]. IEEE, 2019. p. 3702-3712.</p>
</div>

<div class="appendix-break"></div>
<h2>Apêndice A – Núcleos Algorítmicos de Filtragem Adaptativa e Baselines</h2>
<p>Em observância às convenções editoriais de artigos científicos, os trechos de código-fonte que concretizam a estabilização do fator de desvanecimento &lambda;<sub><em>k</em></sub> (Seção 2.4) e o núcleo do Filtro de Partículas SIR comparativo (Seção 5.1) são apresentados neste apêndice para auditoria técnica e reprodutibilidade.</p>

<p class="listing-title">Listagem A.1 – Estabilização do fator &lambda;<sub><em>k</em></sub> via gatilho persistente NIS e atualização de Joseph (<code>src/tracker/ekf.py</code>).</p>
<pre class="code-box"># Trecho resumido de AdaptiveEKF.update() — src/tracker/ekf.py
y_pos = z - self._x[:2]
s_pos = _enforce_spd(self._P[:2, :2] + r_pos, min_eig=1e-6)
nis = float(y_pos.T @ np.linalg.inv(s_pos) @ y_pos)

if self.adaptive and is_moving:
    self._nis_streak = (self._nis_streak + 1) if (nis &gt; self._chi2_thresh) else 0
    if self._nis_streak &gt;= int(self.cfg.nis_persistence_frames):
        adaptive_triggered = True
        speed_ratio = float(np.clip((v_state - v_min) / 8.0, 0.0, 1.0))
        alpha_theta = 0.2 + 2.6 * speed_ratio
        alpha_xy = 0.5 * (3.0 - alpha_theta)
        heading_contrib = alpha_theta * abs(d_theta) if abs(d_theta) &gt; self.cfg.heading_gate_rad else 0.0
        raw_lambda = 1.0 + alpha_xy * (nis / self._chi2_thresh) + heading_contrib
        lambda_k = float(np.clip(raw_lambda, 1.15, float(self.cfg.lambda_max)))

p_working = _enforce_spd(self._P * lambda_k) if adaptive_triggered else self._P.copy()
s_mat = _enforce_spd(h_pos @ p_working @ h_pos.T + r_pos, min_eig=1e-6)
k_gain = p_working @ h_pos.T @ np.linalg.inv(s_mat)
self._x = self._x + k_gain @ y_pos
i_kh = np.eye(5) - k_gain @ h_pos
self._P = _enforce_spd(i_kh @ p_working @ i_kh.T + k_gain @ r_pos @ k_gain.T)</pre>

<p class="listing-title">Listagem A.2 – Ponderação de verossimilhança e reamostragem sistemática no Filtro de Partículas SIR (<code>src/tracker/sota_baselines.py</code>).</p>
<pre class="code-box"># Trecho resumido de ParticleFilter.update() — src/tracker/sota_baselines.py
r_eff = _enforce_spd(r_pos + np.diag([0.04, 0.04]), min_eig=1e-3)
r_eff_inv = np.linalg.inv(r_eff)
innov = z[None, :] - self._particles[:, :2]
quad = np.einsum("ni,ij,nj-&gt;n", innov, r_eff_inv, innov)
log_w = np.log(np.maximum(self._weights, 1e-300)) - 0.5 * quad
log_w -= float(np.max(log_w))
w_unnorm = np.exp(log_w)
self._weights = w_unnorm / float(np.sum(w_unnorm))

self._neff = 1.0 / max(float(np.sum(self._weights ** 2)), 1e-12)
if self._neff &lt; 0.5 * self.num_particles:
    self._systematic_resample()</pre>

<h2>Apêndice B – Comandos de Reprodutibilidade Experimental</h2>
<p>Todos os resultados numéricos (Tabelas 5, 6 e 7), figuras (Figuras 1 a 5) e verificações automatizadas reportados neste artigo podem ser reproduzidos integralmente a partir do ambiente virtual Python 3.12 do repositório pelos comandos abaixo.</p>

<p class="listing-title">Listagem B.1 – Comandos de linha de comando para reprodução integral dos experimentos A1–A7, figuras e testes.</p>
<pre class="code-box"># 1) Preparação dos bancos de dados públicos e manifesto SHA-256 (data/MANIFEST.json)
.venv/bin/python scripts/prepare_datasets.py --all

# 2) Execução completa da suíte científica A1-A7 (incluindo PF, LSTM, GRU, T-GNN e escala 24 câm./150 veíc.)
.venv/bin/python scripts/run_benchmarks.py --all --out data/benchmarks

# 3) Geração das 5 figuras de alta resolução (300 DPI) diretamente a partir de data/benchmarks/ablation_results.json
.venv/bin/python scripts/generate_article_figures.py

# 4) Compilação automatizada do artigo científico em PDF (com auditoria de conformidade)
.venv/bin/python scripts/build_article_pdf.py

# 5) Verificação automatizada (testes unitários de cinemática, SOTA baselines e benchmark urbano)
.venv/bin/pytest tests/test_sota_and_city_benchmark.py tests/test_ekf.py tests/test_roadgraph.py -v</pre>

</body>
</html>
"""
    # Keep every table caption glued to its table header (prevent orphan captions across page breaks)
    html = re.sub(
        r'<table class="(data-table[^"]*)">\s*<caption class="table-caption">(.*?)</caption>(.*?)</table>',
        r'<div class="table-wrap">\n  <p class="table-caption">\2</p>\n  <table class="\1">\3</table>\n</div>',
        html,
        flags=re.DOTALL,
    )
    return html


def main() -> int:
    bench_path = Path("data/benchmarks/ablation_results.json")
    fig_dir = Path("artigo/figuras")
    out_html = Path("artigo/artigo_revisado.html")
    out_pdf = Path("artigo/Artigo_Rastreamento_Multicamera_EKF_Adaptativo_Redes_Urbanas.pdf")

    data = json.loads(bench_path.read_text(encoding="utf-8"))
    html_str = build_html(data, fig_dir)

    # Run automated editorial & terminology compliance audit
    _audit_article_compliance(html_str)

    out_html.write_text(html_str, encoding="utf-8")
    HTML(string=html_str, base_url=str(Path("artigo").resolve())).write_pdf(str(out_pdf))
    print(f"Compliance audit PASSED. Generated {out_html} and {out_pdf} ({out_pdf.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
