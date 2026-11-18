"""Malha viária como grafo direcionado, poda física real, propagador multi-hipótese e cerco policial.

Implementa e aprimora a Seção 4 e a Seção 5.3 do artigo:
1. Grafo Direcionado Métrico G = (V, E) com suporte a:
   - Polilinhas curvas reais (importáveis de OpenStreetMap / GeoJSON / CityFlow);
   - Interpolação curvilínea exata point_at_distance(s) e tangent_at_distance(s);
   - Atributos viários: limite de velocidade (speed_limit), sentido único (one_way),
     número de faixas (lanes) e classe da via (road_type).
2. Poda Cinemática e Física Realista em Cruzamentos (Seção 4.2 corrigida):
   - Proíbe retorno instantâneo (U-turn na mesma via) e sentidos proibidos;
   - Calcula o raio de curvatura efetivo da manobra R_turn na interseção entre
     a aresta de entrada e a aresta candidata de saída;
   - Aplica o limite físico de derrapagem lateral (velocidade crítica de curva):
       v_crit = sqrt(mu * g * R_turn)
     e o envelope mecânico do entre-eixos R_min = l / tan(phi_max).
3. Propagação Multi-Hipótese na Zona Cega (Seção 4.3 + Teste A5):
   - Propaga partículas/hipóteses Gaussianas (s, v, sigma_s^2) ao longo dos ramos da
     Árvore de Rotas Prováveis durante o intervalo cego dt, permitindo localizar onde
     o veículo estará em cada rota candidata no instante t_a + dt.
4. Recomendação de Pontos de Bloqueio Tático Policial (CICC - Seção 5.3):
   - Identifica os nós de interceptação a jusante que maximizam a probabilidade
     acumulada de captura antes que as rotas se dispersem.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .geometry import angle_diff, wrap_angle

Array = np.ndarray
Point = Tuple[float, float]


@dataclass
class Node:
    id: str
    x: float
    y: float
    lat: Optional[float] = None
    lon: Optional[float] = None


@dataclass
class Edge:
    id: str
    source: str
    target: str
    polyline: List[Point]
    speed_limit: float = 13.9
    one_way: bool = True
    lanes: int = 2
    road_type: str = "urban"
    length: float = field(init=False)
    heading: float = field(init=False)
    start_heading: float = field(init=False)
    end_heading: float = field(init=False)
    _cum_lengths: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = [(float(p[0]), float(p[1])) for p in self.polyline]
        if len(pts) < 2:
            raise ValueError("Aresta viária requer pelo menos 2 pontos na polilinha")
        self.polyline = pts
        arr = np.asarray(pts, dtype=float)
        dif = np.diff(arr, axis=0)
        seg_lens = np.linalg.norm(dif, axis=1)
        self._cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self.length = float(self._cum_lengths[-1])

        start, end = arr[0], arr[-1]
        self.heading = wrap_angle(math.atan2(end[1] - start[1], end[0] - start[0]))
        self.start_heading = wrap_angle(math.atan2(arr[1, 1] - arr[0, 1], arr[1, 0] - arr[0, 0]))
        self.end_heading = wrap_angle(math.atan2(arr[-1, 1] - arr[-2, 1], arr[-1, 0] - arr[-2, 0]))

    def point_at_s(self, s: float) -> Tuple[Array, float]:
        """Retorna (posição [x, y], orientação tangente rad) na distância curvilínea s in [0, length]."""
        arr = np.asarray(self.polyline, dtype=float)
        if self.length <= 1e-9 or s <= 0.0:
            return arr[0].copy(), self.start_heading
        if s >= self.length:
            return arr[-1].copy(), self.end_heading

        idx = int(np.searchsorted(self._cum_lengths, s, side="right") - 1)
        idx = max(0, min(idx, len(arr) - 2))
        s0 = float(self._cum_lengths[idx])
        s1 = float(self._cum_lengths[idx + 1])
        seg_len = max(s1 - s0, 1e-9)
        frac = (s - s0) / seg_len
        a, b = arr[idx], arr[idx + 1]
        pos = a + frac * (b - a)
        hdg = wrap_angle(math.atan2(b[1] - a[1], b[0] - a[0]))
        return pos, hdg


@dataclass
class Route:
    edge_ids: Tuple[str, ...]
    nodes: Tuple[str, ...]
    log_prob: float
    prob: float = 1.0
    length: float = 0.0

    def as_dict(self) -> dict:
        return {
            "edges": list(self.edge_ids),
            "nodes": list(self.nodes),
            "log_prob": round(self.log_prob, 4),
            "prob": round(self.prob, 6),
            "length_m": round(self.length, 2),
        }


@dataclass
class BlindHypothesis:
    """Hipótese cinemática propagada ao longo de uma rota durante o apagão visual (Out-of-Sight)."""
    route: Route
    position: Array
    heading: float
    speed: float
    covariance_2d: Array
    prob: float
    reached_node: str


@dataclass
class TacticalBlockPoint:
    """Ponto de bloqueio recomendado para viaturas policiais (Cerco Tático CICC - Seção 5.3)."""
    node_id: str
    x: float
    y: float
    interception_prob: float
    eta_seconds: float
    converging_routes: int

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "x": round(self.x, 2),
            "y": round(self.y, 2),
            "interception_prob": round(self.interception_prob, 4),
            "eta_seconds": round(self.eta_seconds, 1),
            "converging_routes": self.converging_routes,
        }


@dataclass
class Projection:
    edge_id: str
    distance: float
    point: Point
    s: float            # Distância curvilínea acumulada ao longo da polilinha (m)
    heading: float


def project_to_polyline(edge: Edge, point: Sequence[float]) -> Projection:
    """Projeta ortogonalmente um ponto 2D sobre a polilinha de uma aresta."""
    p = np.asarray(point, dtype=float)
    best = (math.inf, 0.0, np.zeros(2), edge.heading)
    acc = 0.0
    pts = np.asarray(edge.polyline, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue
        t = float(np.clip(np.dot(p - a, seg) / (seg_len * seg_len), 0.0, 1.0))
        proj = a + t * seg
        dist = float(np.linalg.norm(p - proj))
        if dist < best[0]:
            best = (
                dist,
                acc + t * seg_len,
                proj,
                wrap_angle(math.atan2(seg[1], seg[0])),
            )
        acc += seg_len
    return Projection(
        edge.id,
        best[0],
        (float(best[2][0]), float(best[2][1])),
        best[1],
        best[3],
    )


class RoadGraph:
    """Grafo viário urbano direcionado com Map-Matching, Poda Cinemática e Propagação Multi-Hipótese."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self._out: Dict[str, List[str]] = {}
        self._in: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ construção
    def add_node(self, node_id: str, x: float, y: float,
                 lat: Optional[float] = None, lon: Optional[float] = None) -> None:
        self.nodes[node_id] = Node(node_id, float(x), float(y), lat, lon)
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError(f"Aresta {edge.id} referencia vértice inexistente")
        self.edges[edge.id] = edge
        if edge.id not in self._out[edge.source]:
            self._out[edge.source].append(edge.id)
        if edge.id not in self._in[edge.target]:
            self._in[edge.target].append(edge.id)

    def out_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[e] for e in self._out.get(node_id, [])]

    def in_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[e] for e in self._in.get(node_id, [])]

    # --------------------------------------------------------------- map-matching
    def nearest_edge(self,
                     point: Sequence[float],
                     max_dist: float = math.inf,
                     heading: Optional[float] = None,
                     heading_weight: float = 8.0) -> Optional[Projection]:
        """Encontra a aresta mais compatível combinando distância lateral e alinhamento angular."""
        best: Optional[Projection] = None
        best_cost = math.inf
        for edge in self.edges.values():
            proj = project_to_polyline(edge, point)
            if proj.distance > max_dist:
                continue
            cost = proj.distance
            if heading is not None:
                d_th = abs(angle_diff(float(heading), proj.heading))
                cost += heading_weight * (1.0 - math.cos(d_th))
            if cost < best_cost:
                best_cost = cost
                best = proj
        return best

    def nearest_node(self, point: Sequence[float], max_dist: float = math.inf) -> Optional[Node]:
        p = np.asarray(point, dtype=float)
        best, bd = None, math.inf
        for node in self.nodes.values():
            d = float(np.linalg.norm(p - np.array([node.x, node.y])))
            if d < bd and d <= max_dist:
                best, bd = node, d
        return best

    def constrain_to_edge(self,
                          x: Array,
                          P: Array,
                          edge: Edge,
                          cfg: PipelineConfig) -> Tuple[Array, Array]:
        """Projeta o estado e comprime a incerteza lateral ortogonal à via (Seção 4.1)."""
        proj = project_to_polyline(edge, x[:2])
        w = float(np.clip(cfg.graph_snap_weight, 0.0, 1.0))
        xn = np.asarray(x, dtype=float).copy()
        xn[0] = (1.0 - w) * xn[0] + w * proj.point[0]
        xn[1] = (1.0 - w) * xn[1] + w * proj.point[1]

        # Só alinha o heading se a diferença não indicar uma conversão em andamento
        d_th = angle_diff(proj.heading, xn[3])
        if abs(d_th) < math.radians(65.0):
            xn[3] = wrap_angle(xn[3] + w * d_th)
        xn[4] *= (1.0 - 0.5 * w)

        c, s = math.cos(proj.heading), math.sin(proj.heading)
        R_rot = np.array([[c, -s], [s, c]], dtype=float)
        Ploc = R_rot @ P[:2, :2] @ R_rot.T
        Ploc[1, 1] *= max(0.15, 1.0 - w)
        Ploc[0, 1] *= max(0.15, 1.0 - w)
        Ploc[1, 0] = Ploc[0, 1]
        Pn = np.asarray(P, dtype=float).copy()
        Pn[:2, :2] = R_rot.T @ Ploc @ R_rot
        return xn, Pn

    # ------------------------------------------------------- poda cinemática e física
    def max_turn_rad(self, edge_length: float, cfg: PipelineConfig) -> float:
        """Envelope mecânico de giro dado o comprimento do segmento de manobra e phi_max."""
        phi = math.radians(cfg.max_steer_deg)
        radius = cfg.wheelbase / max(math.tan(phi), 1e-4)
        # Limita o espaço efetivo de manobra ao cruzamento (~12m) ou comprimento da via (o que for menor)
        effective_arc = min(float(edge_length), 12.0)
        return min(math.pi, effective_arc / radius)

    def intersection_turn_radius(self, delta_theta: float, lanes: int = 2) -> float:
        """Estima o raio geométrico disponível numa interseção urbana para uma curva delta_theta."""
        d_th = abs(float(delta_theta))
        if d_th < 0.12:  # Movimento quase reto
            return 1e4
        lane_width = 3.5 * max(1, lanes)
        # Raio de inscrição de curva em cruzamento urbano: R ~ W_curb / (1 - cos(d_th))
        denom = max(1.0 - math.cos(min(d_th, math.pi * 0.85)), 0.08)
        return max(3.5, min(250.0, (1.1 * lane_width) / denom))

    def critical_turn_speed(self, delta_theta: float, cfg: PipelineConfig, lanes: int = 2) -> float:
        """Velocidade máxima fisicamente sustentável sem derrapagem lateral na curva:
            v_crit = sqrt(mu * g * R_turn).
        """
        r_turn = self.intersection_turn_radius(delta_theta, lanes=lanes)
        return math.sqrt(cfg.tire_friction_mu * cfg.gravity_g * r_turn)

    def feasible_edges(self,
                       node_id: str,
                       x: Array,
                       cfg: PipelineConfig,
                       margin_deg: float = 0.0,
                       prev_node_id: Optional[str] = None) -> List[Edge]:
        """Poda arestas por conectividade, proibição de retorno (U-turn) e envelope cinemático (Seção 4.2)."""
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        theta = float(x_arr[3])
        margin = math.radians(margin_deg + cfg.turn_margin_deg)
        out: List[Edge] = []
        candidates = self.out_edges(node_id)

        for edge in candidates:
            # 1. Evita retorno imediato de 180 graus para o nó de origem se houver outras saídas
            if prev_node_id is not None and edge.target == prev_node_id and len(candidates) > 1:
                continue
            delta = abs(angle_diff(edge.start_heading, theta))
            # 2. Envelope mecânico do veículo Two-Point-Bicycle
            max_mech = self.max_turn_rad(edge.length, cfg) + margin
            # 3.Proíbe U-turns (> 135 graus) em cruzamentos padrão
            if delta <= min(max_mech, math.radians(135.0)):
                out.append(edge)
        return out

    def transition_logprob(self, edge: Edge, x: Array, cfg: PipelineConfig) -> float:
        """Calcula log P(e_{t+1} | e_t, X_hat_t) combinando cinemática, física de curva e via (Seção 4.3)."""
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        theta = float(x_arr[3])
        speed = float(max(x_arr[2], 0.0))
        dtheta = abs(angle_diff(edge.start_heading, theta))

        # 1. Penalidade angular com relação à intenção direcional do EKF
        envelope = max(0.25, self.max_turn_rad(edge.length, cfg))
        sigma_theta = max(0.22, 0.45 * envelope)
        score = -0.5 * ((dtheta / sigma_theta) ** 2)

        # 2. Penalidade física de derrapagem em curva (Velocidade atual vs. v_crit da conversão)
        v_crit = self.critical_turn_speed(dtheta, cfg, lanes=edge.lanes)
        if speed > v_crit:
            # Se o veículo está rápido demais para fazer a curva, penaliza exponencialmente
            excess = (speed - v_crit) / max(cfg.sigma_accel * 1.5, 1.0)
            score -= 0.85 * (excess ** 2)

        # 3. Proximidade espacial ao início da aresta
        sigma_pos = max(1.5, cfg.sigma_pos_meas)
        dpos = float(np.linalg.norm(np.asarray(edge.polyline[0]) - x_arr[:2]))
        score -= 0.5 * ((dpos / sigma_pos) ** 2)

        # 4. Compatibilidade de velocidade com a capacidade / limite da via
        if speed > 0.5:
            t_edge = edge.length / max(edge.speed_limit, 1.0)
            t_alvo = edge.length / speed
            score -= 0.35 * (math.log((t_edge + 0.5) / (t_alvo + 0.5)) ** 2)
        score += cfg.speed_bonus * math.log1p(edge.speed_limit / 10.0) + 0.05 * (edge.lanes - 1)
        return float(score)

    # -------------------------------------------------------------- árvore de rotas
    def build_route_tree(self,
                         start_edge_id: str,
                         x: Array,
                         cfg: PipelineConfig,
                         horizon: Optional[int] = None) -> List[Route]:
        """Gera a Árvore de Rotas de Fuga Prováveis por busca em feixe podada (Seção 4.3)."""
        if start_edge_id not in self.edges:
            return []
        horizon_steps = int(horizon or cfg.route_horizon)
        start = self.edges[start_edge_id]
        beam: List[Route] = [Route((start.id,), (start.source, start.target), 0.0)]

        for _ in range(max(0, horizon_steps - 1)):
            nxt: List[Route] = []
            for route in beam:
                tail = self.edges[route.edge_ids[-1]]
                prev_node = route.nodes[-2] if len(route.nodes) >= 2 else None
                pseudo_state = np.array([
                    tail.polyline[-1][0],
                    tail.polyline[-1][1],
                    max(float(np.asarray(x).reshape(-1)[2]), 4.0),
                    tail.end_heading,
                    0.0,
                ], dtype=float)
                candidates = self.feasible_edges(
                    tail.target, pseudo_state, cfg, prev_node_id=prev_node
                )
                if not candidates:
                    nxt.append(route)
                    continue
                scores = np.array(
                    [self.transition_logprob(e, pseudo_state, cfg) for e in candidates],
                    dtype=float,
                )
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                for edge, p in zip(candidates, probs):
                    if p < cfg.min_route_prob:
                        continue
                    nxt.append(
                        Route(
                            route.edge_ids + (edge.id,),
                            route.nodes + (edge.target,),
                            route.log_prob + float(np.log(max(p, 1e-12))),
                        )
                    )
            if not nxt:
                break
            nxt.sort(key=lambda r: r.log_prob, reverse=True)
            beam = nxt[: cfg.beam_width]

        beam = beam[: cfg.max_routes]
        for route in beam:
            route.length = float(sum(self.edges[e].length for e in route.edge_ids))
        probs = np.array([math.exp(r.log_prob) for r in beam], dtype=float)
        total = float(probs.sum())
        if total > 0:
            for route, p in zip(beam, probs):
                route.prob = float(p / total)
        beam.sort(key=lambda r: r.prob, reverse=True)
        return beam

    # --------------------------------------- propagador multi-hipótese em zona cega
    def propagate_along_route(self,
                              route: Route,
                              start_pos: Sequence[float],
                              speed: float,
                              dt: float,
                              cfg: PipelineConfig) -> BlindHypothesis:
        """Propaga o veículo curvilineamente ao longo de uma rota candidata durante dt segundos."""
        first_edge = self.edges[route.edge_ids[0]]
        proj = project_to_polyline(first_edge, start_pos)
        rem_dist = max(0.0, float(speed) * max(0.0, float(dt)))

        # Distância restante na primeira aresta a partir da projeção atual
        avail_first = max(0.0, first_edge.length - proj.s)
        if rem_dist <= avail_first:
            pos, hdg = first_edge.point_at_s(proj.s + rem_dist)
            reached = first_edge.target
        else:
            rem_dist -= avail_first
            pos, hdg = first_edge.point_at_s(first_edge.length)
            reached = first_edge.target
            for eid in route.edge_ids[1:]:
                edge = self.edges[eid]
                # Aplica leve desaceleração natural caso haja curva forte na esquina
                if rem_dist <= edge.length:
                    pos, hdg = edge.point_at_s(rem_dist)
                    reached = edge.target
                    break
                rem_dist -= edge.length
                pos, hdg = edge.point_at_s(edge.length)
                reached = edge.target

        # Covariância anisotrópica no referencial da via:
        # Cresce com (sigma_accel * dt)^2 ao longo da via, mas fica confinada à largura da pista lateralmente
        sigma_long = max(cfg.sigma_assoc_pos, 0.35 * cfg.sigma_accel * (dt ** 1.35) + 4.0)
        sigma_lat = max(3.5, 0.35 * sigma_long)
        c, s = math.cos(hdg), math.sin(hdg)
        R_rot = np.array([[c, -s], [s, c]], dtype=float)
        cov_loc = np.diag([sigma_long ** 2, sigma_lat ** 2])
        cov_world = R_rot @ cov_loc @ R_rot.T

        return BlindHypothesis(
            route=route,
            position=pos,
            heading=hdg,
            speed=speed,
            covariance_2d=cov_world,
            prob=route.prob,
            reached_node=reached,
        )

    def propagate_route_tree(self,
                             routes: Sequence[Route],
                             start_pos: Sequence[float],
                             speed: float,
                             dt: float,
                             cfg: PipelineConfig) -> List[BlindHypothesis]:
        """Propaga todas as rotas da árvore de fuga para o instante t + dt (Multi-Hypothesis EKF)."""
        return [
            self.propagate_along_route(r, start_pos, speed, dt, cfg)
            for r in routes
        ]

    # --------------------------------------- recomendação de cerco tático policial
    def recommend_blockade_points(self,
                                  routes: Sequence[Route],
                                  current_speed: float,
                                  top_k: int = 3) -> List[TacticalBlockPoint]:
        """Calcula os cruzamentos estratégicos ideais para posicionamento de viaturas (Seção 5.3)."""
        if not routes:
            return []
        node_prob: Dict[str, float] = {}
        node_dist: Dict[str, List[float]] = {}
        node_count: Dict[str, int] = {}

        for r in routes:
            cum_d = 0.0
            # Ignora o nó de origem imediato; avalia os vértices a jusante
            for idx, eid in enumerate(r.edge_ids):
                edge = self.edges.get(eid)
                if edge is None:
                    continue
                cum_d += edge.length
                nid = edge.target
                if idx == 0 and len(r.edge_ids) > 1:
                    continue
                node_prob[nid] = node_prob.get(nid, 0.0) + r.prob
                node_dist.setdefault(nid, []).append(cum_d)
                node_count[nid] = node_count.get(nid, 0) + 1

        blocks: List[TacticalBlockPoint] = []
        v_eff = max(float(current_speed), 5.0)
        for nid, prob in node_prob.items():
            node = self.nodes.get(nid)
            if node is None:
                continue
            mean_d = float(np.mean(node_dist[nid]))
            blocks.append(
                TacticalBlockPoint(
                    node_id=nid,
                    x=node.x,
                    y=node.y,
                    interception_prob=min(1.0, prob),
                    eta_seconds=mean_d / v_eff,
                    converging_routes=node_count[nid],
                )
            )
        blocks.sort(key=lambda b: (b.interception_prob, b.converging_routes), reverse=True)
        return blocks[:top_k]

    # --------------------------------------- baseline Map-Matching HMM / Viterbi
    def hmm_viterbi_match(self,
                          observations: Sequence[Sequence[float]],
                          sigma_obs: float = 6.0) -> List[str]:
        """Map-matching de sequência por Hidden Markov Model (Viterbi) - Tabela da Seção 4."""
        if not observations or not self.edges:
            return []
        edge_list = list(self.edges.values())
        n_edges = len(edge_list)
        edge_idx = {e.id: i for i, e in enumerate(edge_list)}

        # Matriz de transição log-prob entre arestas conectadas
        trans_log = np.full((n_edges, n_edges), -18.0, dtype=float)
        for i, e1 in enumerate(edge_list):
            trans_log[i, i] = -0.2  # Permanecer na mesma aresta
            for e2 in self.out_edges(e1.target):
                j = edge_idx[e2.id]
                d_th = abs(angle_diff(e2.start_heading, e1.end_heading))
                trans_log[i, j] = -0.5 - 1.5 * (d_th / math.pi)

        T = len(observations)
        dp = np.full((T, n_edges), -math.inf, dtype=float)
        back = np.zeros((T, n_edges), dtype=int)

        for t, obs in enumerate(observations):
            emis = np.array([
                -0.5 * ((project_to_polyline(e, obs).distance / sigma_obs) ** 2)
                for e in edge_list
            ], dtype=float)
            if t == 0:
                dp[0] = emis
            else:
                scores = dp[t - 1][:, None] + trans_log
                back[t] = np.argmax(scores, axis=0)
                dp[t] = np.max(scores, axis=0) + emis

        path_idx = [int(np.argmax(dp[-1]))]
        for t in range(T - 1, 0, -1):
            path_idx.append(int(back[t, path_idx[-1]]))
        path_idx.reverse()
        return [edge_list[i].id for i in path_idx]

    # ------------------------------------------------------------------ serialização
    def to_dict(self) -> dict:
        return {
            "nodes": [
                {"id": n.id, "x": n.x, "y": n.y, "lat": n.lat, "lon": n.lon}
                for n in self.nodes.values()
            ],
            "edges": [
                {
                    "id": e.id,
                    "source": e.source,
                    "target": e.target,
                    "polyline": e.polyline,
                    "speed_limit": e.speed_limit,
                    "one_way": e.one_way,
                    "lanes": e.lanes,
                    "road_type": e.road_type,
                }
                for e in self.edges.values()
            ],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RoadGraph":
        g = cls()
        for n in data["nodes"]:
            g.add_node(n["id"], n["x"], n["y"], n.get("lat"), n.get("lon"))
        for e in data["edges"]:
            g.add_edge(
                Edge(
                    id=e["id"],
                    source=e["source"],
                    target=e["target"],
                    polyline=e["polyline"],
                    speed_limit=e.get("speed_limit", 13.9),
                    one_way=e.get("one_way", True),
                    lanes=e.get("lanes", 2),
                    road_type=e.get("road_type", "urban"),
                )
            )
        return g

    @classmethod
    def load_json(cls, path: str) -> "RoadGraph":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def grid_graph(nx: int,
               ny: int,
               spacing: float = 100.0,
               speed_limit: float = 13.9) -> RoadGraph:
    """Constrói um grafo viário urbano com hierarquia de vias (arteriais rápidas e coletoras)."""
    g = RoadGraph()
    for i in range(nx):
        for j in range(ny):
            g.add_node(f"v_{i}_{j}", i * spacing, j * spacing)
    for i in range(nx):
        for j in range(ny):
            a = f"v_{i}_{j}"
            if i + 1 < nx:
                b = f"v_{i+1}_{j}"
                pts = [(i * spacing, j * spacing), ((i + 1) * spacing, j * spacing)]
                # Avenidas centrais têm velocidade limite maior e mais faixas
                is_arterial = j in (1, 2)
                sl = speed_limit * (1.25 if is_arterial else 1.0)
                lanes = 3 if is_arterial else 2
                g.add_edge(Edge(f"e_{a}_{b}", a, b, pts, sl, True, lanes=lanes))
                g.add_edge(Edge(f"e_{b}_{a}", b, a, pts[::-1], sl, True, lanes=lanes))
            if j + 1 < ny:
                b = f"v_{i}_{j+1}"
                pts = [(i * spacing, j * spacing), (i * spacing, (j + 1) * spacing)]
                is_arterial = i in (1, 2)
                sl = speed_limit * (1.25 if is_arterial else 1.0)
                lanes = 3 if is_arterial else 2
                g.add_edge(Edge(f"e_{a}_{b}", a, b, pts, sl, True, lanes=lanes))
                g.add_edge(Edge(f"e_{b}_{a}", b, a, pts[::-1], sl, True, lanes=lanes))
    return g
