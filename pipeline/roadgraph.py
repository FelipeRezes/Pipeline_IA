"""Malha viária como grafo direcionado e poda topológica (Sec. 4 do artigo).

Restrições aplicadas:
1. conectividade/sentido: só arestas em Out(v) são elegíveis;
2. envelope cinemático: uma conversão de delta_theta exige raio R = l/tan(phi_max)
   e comprimento de arco >= R * |delta_theta|; arestas inviáveis são podadas;
3. transições ponderadas por coerência de heading, distância até a posição
   predita e velocidade limite da via, gerando a Árvore de Rotas Prováveis.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import PipelineConfig
from .geometry import wrap_angle

Array = np.ndarray
Point = Tuple[float, float]


@dataclass
class Node:
    id: str
    x: float
    y: float


@dataclass
class Edge:
    id: str
    source: str
    target: str
    polyline: List[Point]
    speed_limit: float = 13.9
    one_way: bool = True
    length: float = field(init=False)
    heading: float = field(init=False)

    def __post_init__(self) -> None:
        pts = [tuple(map(float, p)) for p in self.polyline]
        if len(pts) < 2:
            raise ValueError("aresta requer pelo menos 2 pontos")
        self.polyline = pts
        dif = np.diff(np.asarray(pts, dtype=float), axis=0)
        self.length = float(np.linalg.norm(dif, axis=1).sum())
        start, end = np.asarray(pts[0]), np.asarray(pts[-1])
        self.heading = wrap_angle(math.atan2(end[1] - start[1], end[0] - start[0]))


@dataclass
class Route:
    edge_ids: Tuple[str, ...]
    nodes: Tuple[str, ...]
    log_prob: float
    prob: float = 1.0
    length: float = 0.0

    def as_dict(self) -> dict:
        return {"edges": list(self.edge_ids), "nodes": list(self.nodes),
                "log_prob": round(self.log_prob, 4),
                "prob": round(self.prob, 6),
                "length_m": round(self.length, 2)}


@dataclass
class Projection:
    edge_id: str
    distance: float
    point: Point
    s: float            # distância acumulada ao longo da polilinha (m)
    heading: float


def project_to_polyline(edge: Edge, point: Sequence[float]) -> Projection:
    """Projeta um ponto no plano de solo sobre a polilinha da aresta."""
    p = np.asarray(point, dtype=float)
    best = (math.inf, 0.0, np.zeros(2), 0.0)
    acc = 0.0
    pts = np.asarray(edge.polyline, dtype=float)
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        seg_len = float(np.linalg.norm(seg))
        if seg_len < 1e-9:
            continue
        t = float(np.clip(np.dot(p - a, seg) / seg_len ** 2, 0.0, 1.0))
        proj = a + t * seg
        dist = float(np.linalg.norm(p - proj))
        if dist < best[0]:
            best = (dist, acc + t * seg_len, proj,
                    wrap_angle(math.atan2(seg[1], seg[0])))
        acc += seg_len
    return Projection(edge.id, best[0], (float(best[2][0]), float(best[2][1])),
                      best[1], best[3])


class RoadGraph:
    """Grafo viário direcionado com consultas de map-matching e árvore de rotas."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self._out: Dict[str, List[str]] = {}
        self._in: Dict[str, List[str]] = {}

    # ------------------------------------------------------------------ construção
    def add_node(self, node_id: str, x: float, y: float) -> None:
        self.nodes[node_id] = Node(node_id, float(x), float(y))
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])

    def add_edge(self, edge: Edge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("aresta referencia vértice inexistente")
        self.edges[edge.id] = edge
        self._out[edge.source].append(edge.id)
        self._in[edge.target].append(edge.id)

    def out_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[e] for e in self._out.get(node_id, [])]

    def in_edges(self, node_id: str) -> List[Edge]:
        return [self.edges[e] for e in self._in.get(node_id, [])]

    # --------------------------------------------------------------- map-matching
    def nearest_edge(self, point: Sequence[float], max_dist: float = math.inf
                     ) -> Optional[Projection]:
        best: Optional[Projection] = None
        for edge in self.edges.values():
            proj = project_to_polyline(edge, point)
            if proj.distance <= max_dist and (best is None or proj.distance < best.distance):
                best = proj
        return best

    def nearest_node(self, point: Sequence[float], max_dist: float = math.inf
                     ) -> Optional[Node]:
        p = np.asarray(point, dtype=float)
        best, bd = None, math.inf
        for node in self.nodes.values():
            d = float(np.linalg.norm(p - np.array([node.x, node.y])))
            if d < bd and d <= max_dist:
                best, bd = node, d
        return best

    def constrain_to_edge(self, x: Array, P: Array, edge: Edge,
                          cfg: PipelineConfig) -> tuple[Array, Array]:
        """Projeta estado/covariância na aresta viária (reduz o ruído lateral)."""
        proj = project_to_polyline(edge, x[:2])
        w = float(np.clip(cfg.graph_snap_weight, 0.0, 1.0))
        xn = np.asarray(x, dtype=float).copy()
        xn[0] = (1.0 - w) * xn[0] + w * proj.point[0]
        xn[1] = (1.0 - w) * xn[1] + w * proj.point[1]
        xn[3] = wrap_angle(xn[3] + w * wrap_angle(edge.heading - xn[3]))
        xn[4] *= (1.0 - 0.5 * w)

        # covariância no referencial da via: eixo longitudinal / lateral
        c, s = np.cos(edge.heading), np.sin(edge.heading)
        R = np.array([[c, -s], [s, c]])
        Ploc = R @ P[:2, :2] @ R.T
        Ploc[1, 1] *= max(0.15, 1.0 - w)
        Ploc[0, 1] *= max(0.15, 1.0 - w)
        Ploc[1, 0] = Ploc[0, 1]
        Pn = np.asarray(P, dtype=float).copy()
        Pn[:2, :2] = R.T @ Ploc @ R
        return xn, Pn

    # ------------------------------------------------------- restrições e transições
    def max_turn_rad(self, edge_length: float, cfg: PipelineConfig) -> float:
        """Maior mudança de heading fisicamente atingível na aresta."""
        phi = math.radians(cfg.max_steer_deg)
        radius = cfg.wheelbase / math.tan(phi)
        return min(math.pi, edge_length / radius)

    def feasible_edges(self, node_id: str, x: Array, cfg: PipelineConfig,
                       margin_deg: float = 0.0) -> List[Edge]:
        """Poda por conectividade/sentido e envelope de esterçamento (Sec. 4.2)."""
        theta = float(np.asarray(x).reshape(-1)[3])
        margin = math.radians(margin_deg + cfg.turn_margin_deg)
        out: List[Edge] = []
        for edge in self.out_edges(node_id):
            delta = abs(wrap_angle(edge.heading - theta))
            if delta <= self.max_turn_rad(edge.length, cfg) + margin:
                out.append(edge)
        return out

    def transition_logprob(self, edge: Edge, x: Array, cfg: PipelineConfig) -> float:
        """log P(e_{t+1} | e_t, X_hat_t): heading, posição e capacidade da via."""
        x = np.asarray(x, dtype=float).reshape(-1)
        theta, speed = float(x[3]), float(max(x[2], 0.0))
        dtheta = wrap_angle(edge.heading - theta)
        envelope = self.max_turn_rad(edge.length, cfg)
        sigma_theta = max(0.18, 0.5 * envelope)
        score = -0.5 * (dtheta / sigma_theta) ** 2

        sigma_pos = max(1.0, cfg.sigma_pos_meas)
        dpos = float(np.linalg.norm(np.asarray(edge.polyline[0]) - x[:2]))
        score -= 0.5 * (dpos / sigma_pos) ** 2

        if speed > 0.5:
            # tempo para percorrer a aresta vs. velocidade estimada do alvo
            t_edge = edge.length / max(edge.speed_limit, 1.0)
            t_alvo = edge.length / speed
            score -= 0.4 * math.log((t_edge + 0.5) / (t_alvo + 0.5)) ** 2
        score += cfg.speed_bonus * math.log1p(edge.speed_limit / 10.0)
        return score

    # -------------------------------------------------------------- árvore de rotas
    def build_route_tree(self, start_edge_id: str, x: Array,
                         cfg: PipelineConfig, horizon: Optional[int] = None
                         ) -> List[Route]:
        """Expansão em feixe das rotas prováveis a partir da aresta atual."""
        if start_edge_id not in self.edges:
            return []
        horizon = int(horizon or cfg.route_horizon)
        start = self.edges[start_edge_id]
        beam: List[Route] = [Route((start.id,), (start.source, start.target), 0.0)]

        for _ in range(max(0, horizon - 1)):
            nxt: List[Route] = []
            for route in beam:
                tail = self.edges[route.edge_ids[-1]]
                pseudo = np.array([tail.polyline[-1][0], tail.polyline[-1][1],
                                   max(float(np.asarray(x).reshape(-1)[2]), 3.0),
                                   tail.heading, 0.0])
                candidates = self.feasible_edges(tail.target, pseudo, cfg)
                if not candidates:
                    nxt.append(route)
                    continue
                scores = np.array([self.transition_logprob(e, pseudo, cfg)
                                   for e in candidates])
                scores -= scores.max()
                probs = np.exp(scores)
                probs /= probs.sum()
                for edge, p in zip(candidates, probs):
                    nxt.append(Route(
                        route.edge_ids + (edge.id,),
                        route.nodes + (edge.target,),
                        route.log_prob + float(np.log(max(p, 1e-12))),
                    ))
            if not nxt:
                break
            nxt.sort(key=lambda r: r.log_prob, reverse=True)
            beam = nxt[:cfg.beam_width]

        # corta o feixe ANTES de normalizar: as probabilidades retornadas devem
        # somar 1 sobre exatamente as rotas entregues ao módulo de decisão.
        beam = beam[:cfg.max_routes]
        for route in beam:
            route.length = float(sum(self.edges[e].length for e in route.edge_ids))
        probs = np.array([math.exp(r.log_prob) for r in beam])
        total = probs.sum()
        if total > 0:
            for route, p in zip(beam, probs):
                route.prob = float(p / total)
        beam.sort(key=lambda r: r.prob, reverse=True)
        return beam

    # ------------------------------------------------------------------ serialização
    def to_dict(self) -> dict:
        return {
            "nodes": [vars(n) for n in self.nodes.values()],
            "edges": [{"id": e.id, "source": e.source, "target": e.target,
                       "polyline": e.polyline, "speed_limit": e.speed_limit,
                       "one_way": e.one_way} for e in self.edges.values()],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RoadGraph":
        g = cls()
        for n in data["nodes"]:
            g.add_node(n["id"], n["x"], n["y"])
        for e in data["edges"]:
            g.add_edge(Edge(id=e["id"], source=e["source"], target=e["target"],
                            polyline=e["polyline"],
                            speed_limit=e.get("speed_limit", 13.9),
                            one_way=e.get("one_way", True)))
        return g

    @classmethod
    def load_json(cls, path: str) -> "RoadGraph":
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def grid_graph(nx: int, ny: int, spacing: float = 100.0,
               speed_limit: float = 13.9) -> RoadGraph:
    """Grafo de malha ortogonal (usado na demo sintética)."""
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
                g.add_edge(Edge(f"e_{a}_{b}", a, b, pts, speed_limit))
                g.add_edge(Edge(f"e_{b}_{a}", b, a, pts[::-1], speed_limit))
            if j + 1 < ny:
                b = f"v_{i}_{j+1}"
                pts = [(i * spacing, j * spacing), (i * spacing, (j + 1) * spacing)]
                g.add_edge(Edge(f"e_{a}_{b}", a, b, pts, speed_limit))
                g.add_edge(Edge(f"e_{b}_{a}", b, a, pts[::-1], speed_limit))
    return g
