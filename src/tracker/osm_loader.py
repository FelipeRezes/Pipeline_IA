"""OpenStreetMap UTM directed road network loader with Overpass mirror fallback and GraphML cache."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Sequence

import networkx as nx
import numpy as np

from tracker.geometry import wgs84_to_utm

DEFAULT_OVERPASS_URLS: tuple[str, ...] = (
    "https://overpass.osm.ch/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
)


@dataclass
class RoadEdge:
    """Directed metric UTM road edge with curvature and speed constraints."""

    edge_id: tuple[int, int, int]
    u: int
    v: int
    polyline_utm: np.ndarray
    length_m: float
    oneway: bool
    maxspeed_mps: float
    lanes: int
    heading_start_rad: float
    heading_end_rad: float
    min_curve_radius_m: float


@dataclass
class RoadNetwork:
    """Metric UTM directed road network container."""

    graph: nx.MultiDiGraph
    epsg_utm: int
    nodes_utm: dict[int, tuple[float, float]]
    edges: dict[tuple[int, int, int], RoadEdge]
    bounds_utm: tuple[float, float, float, float]


def _parse_int_id(val: Any) -> int:
    try:
        return int(val)
    except (ValueError, TypeError):
        return int(abs(hash(str(val))) % 1_000_000_000)


def _parse_bool_oneway(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    s = str(val).strip().lower()
    return s in {"true", "1", "yes", "y", "-1"}


def _parse_maxspeed_mps(val: Any, default_kmh: float = 50.0) -> float:
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    if val is None:
        return default_kmh / 3.6
    if isinstance(val, (int, float)) and math.isfinite(float(val)) and float(val) > 0:
        return float(val) / 3.6
    s = str(val).strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    if not m:
        return default_kmh / 3.6
    num = float(m.group(1))
    if "mph" in s:
        return num * 0.44704
    return max(num / 3.6, 1.0)


def _parse_lanes(val: Any, default_lanes: int = 1) -> int:
    if isinstance(val, (list, tuple)) and val:
        val = val[0]
    if val is None:
        return default_lanes
    m = re.search(r"(\d+)", str(val))
    if not m:
        return default_lanes
    return max(1, int(m.group(1)))


def _parse_wkt_linestring(geom_val: Any) -> np.ndarray | None:
    if geom_val is None:
        return None
    if hasattr(geom_val, "coords"):
        pts = np.asarray(list(geom_val.coords), dtype=np.float64)
        if pts.ndim == 2 and pts.shape[0] >= 2 and pts.shape[1] >= 2:
            return pts[:, :2].copy()
    s = str(geom_val).strip()
    if not s.upper().startswith("LINESTRING"):
        return None
    inner = s[s.find("(") + 1 : s.rfind(")")]
    if not inner:
        return None
    coords: list[tuple[float, float]] = []
    for part in inner.split(","):
        tokens = part.strip().split()
        if len(tokens) >= 2:
            try:
                coords.append((float(tokens[0]), float(tokens[1])))
            except ValueError:
                continue
    if len(coords) < 2:
        return None
    return np.asarray(coords, dtype=np.float64)


def _compute_polyline_length(poly: np.ndarray) -> float:
    diffs = np.diff(poly, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total = float(np.sum(seg_lens))
    return max(total, 1e-3)


def _compute_min_curve_radius(poly: np.ndarray) -> float:
    """Compute minimum local circumcircle radius (Menger curvature) along a polyline."""
    if len(poly) < 3:
        return 1e6
    min_r = 1e6
    for i in range(1, len(poly) - 1):
        p0 = poly[i - 1]
        p1 = poly[i]
        p2 = poly[i + 1]
        v1 = p1 - p0
        v2 = p2 - p1
        a = float(math.hypot(v1[0], v1[1]))
        b = float(math.hypot(v2[0], v2[1]))
        if a < 1e-3 or b < 1e-3:
            continue
        # Check deflection angle so nearly straight segments do not produce numerical noise
        dot = float(v1[0] * v2[0] + v1[1] * v2[1]) / (a * b)
        dot = max(-1.0, min(1.0, dot))
        defl = math.acos(dot)
        if defl < 1e-3:
            continue
        c_vec = p2 - p0
        c = float(math.hypot(c_vec[0], c_vec[1]))
        area2 = abs(float(v1[0] * c_vec[1] - v1[1] * c_vec[0]))
        if area2 > 1e-6:
            radius = (a * b * c) / (2.0 * area2)
            if radius < min_r:
                min_r = float(radius)
    return max(float(min_r), 1.0)


def _compute_headings(poly: np.ndarray) -> tuple[float, float]:
    # Start heading from first non-zero segment
    h_start = 0.0
    for i in range(len(poly) - 1):
        dx = float(poly[i + 1, 0] - poly[i, 0])
        dy = float(poly[i + 1, 1] - poly[i, 1])
        if math.hypot(dx, dy) > 1e-6:
            h_start = math.atan2(dy, dx)
            break
    # End heading from last non-zero segment
    h_end = h_start
    for i in range(len(poly) - 2, -1, -1):
        dx = float(poly[i + 1, 0] - poly[i, 0])
        dy = float(poly[i + 1, 1] - poly[i, 1])
        if math.hypot(dx, dy) > 1e-6:
            h_end = math.atan2(dy, dx)
            break
    return float(h_start), float(h_end)


def build_road_network_from_multidigraph(raw_graph: nx.MultiDiGraph, default_epsg: int = 32723) -> RoadNetwork:
    """Convert a NetworkX / OSMnx MultiDiGraph into a normalized metric UTM RoadNetwork."""
    crs_str = str(raw_graph.graph.get("crs", f"EPSG:{default_epsg}")).upper()
    epsg_match = re.search(r"(\d{4,5})", crs_str)
    epsg_code = int(epsg_match.group(1)) if epsg_match else default_epsg

    # Detect whether node coordinates are WGS84 degrees (EPSG:4326) and project if needed
    raw_xs = [float(d.get("x", 0.0)) for _, d in raw_graph.nodes(data=True)]
    raw_ys = [float(d.get("y", 0.0)) for _, d in raw_graph.nodes(data=True)]
    if not raw_xs:
        raise ValueError("Road graph contains no nodes")

    is_wgs84 = epsg_code == 4326 or (max(abs(x) for x in raw_xs) <= 180.0 and max(abs(y) for y in raw_ys) <= 90.0)
    detected_epsg = default_epsg

    norm_graph = nx.MultiDiGraph()
    nodes_utm: dict[int, tuple[float, float]] = {}

    for raw_nid, data in raw_graph.nodes(data=True):
        nid = _parse_int_id(raw_nid)
        rx = float(data.get("x", 0.0))
        ry = float(data.get("y", 0.0))
        if is_wgs84:
            ux, uy, detected_epsg = wgs84_to_utm(ry, rx)
        else:
            ux, uy = rx, ry
            detected_epsg = epsg_code
        nodes_utm[nid] = (ux, uy)
        norm_graph.add_node(nid, x=ux, y=uy)

    final_epsg = detected_epsg if is_wgs84 else epsg_code
    norm_graph.graph["crs"] = f"EPSG:{final_epsg}"

    edges: dict[tuple[int, int, int], RoadEdge] = {}
    key_counters: dict[tuple[int, int], int] = {}

    for raw_u, raw_v, raw_k, edata in raw_graph.edges(keys=True, data=True):
        u = _parse_int_id(raw_u)
        v = _parse_int_id(raw_v)
        try:
            k = int(raw_k)
        except (ValueError, TypeError):
            k = key_counters.get((u, v), 0)
        while (u, v, k) in edges:
            k += 1
        key_counters[(u, v)] = k + 1

        poly = _parse_wkt_linestring(edata.get("geometry"))
        if poly is None:
            u_xy = nodes_utm[u]
            v_xy = nodes_utm[v]
            poly = np.array([u_xy, v_xy], dtype=np.float64)
        elif is_wgs84:
            proj_pts = [wgs84_to_utm(float(lat), float(lon), epsg_utm=final_epsg)[:2] for lon, lat in poly]
            poly = np.asarray(proj_pts, dtype=np.float64)

        length_m = _compute_polyline_length(poly)
        if "length" in edata:
            try:
                l_attr = float(edata["length"])
                if math.isfinite(l_attr) and l_attr > 0.0 and not is_wgs84:
                    length_m = l_attr
            except (ValueError, TypeError):
                pass

        oneway = _parse_bool_oneway(edata.get("oneway", True))
        maxspeed_mps = _parse_maxspeed_mps(edata.get("maxspeed", 50.0))
        lanes = _parse_lanes(edata.get("lanes", 1))
        h_start, h_end = _compute_headings(poly)
        min_r = _compute_min_curve_radius(poly)

        edge_id = (u, v, k)
        road_edge = RoadEdge(
            edge_id=edge_id,
            u=u,
            v=v,
            polyline_utm=poly,
            length_m=float(length_m),
            oneway=oneway,
            maxspeed_mps=float(maxspeed_mps),
            lanes=int(lanes),
            heading_start_rad=h_start,
            heading_end_rad=h_end,
            min_curve_radius_m=min_r,
        )
        edges[edge_id] = road_edge
        norm_graph.add_edge(
            u,
            v,
            key=k,
            length=float(length_m),
            oneway=oneway,
            maxspeed=float(maxspeed_mps * 3.6),
            lanes=int(lanes),
            geometry=f"LINESTRING ({', '.join(f'{pt[0]:.6f} {pt[1]:.6f}' for pt in poly)})",
        )

    all_x = [xy[0] for xy in nodes_utm.values()]
    all_y = [xy[1] for xy in nodes_utm.values()]
    for e in edges.values():
        all_x.extend(e.polyline_utm[:, 0].tolist())
        all_y.extend(e.polyline_utm[:, 1].tolist())
    bounds_utm = (
        float(min(all_x)),
        float(min(all_y)),
        float(max(all_x)),
        float(max(all_y)),
    )

    return RoadNetwork(
        graph=norm_graph,
        epsg_utm=final_epsg,
        nodes_utm=nodes_utm,
        edges=edges,
        bounds_utm=bounds_utm,
    )


def load_utm_road_graph(
    place_name: str | None = None,
    bbox_wgs84: tuple[float, float, float, float] | None = None,
    graphml_cache: Path | str | None = None,
    overpass_urls: Sequence[str] = DEFAULT_OVERPASS_URLS,
) -> RoadNetwork:
    """Load an OpenStreetMap road network in metric UTM from local GraphML cache or Overpass mirrors."""
    cache_path = Path(graphml_cache) if graphml_cache is not None else None
    if cache_path is not None and cache_path.is_file():
        raw_g = nx.read_graphml(cache_path)
        if not isinstance(raw_g, nx.MultiDiGraph):
            raw_g = nx.MultiDiGraph(raw_g)
        return build_road_network_from_multidigraph(raw_g)

    urls = list(overpass_urls) if overpass_urls is not None else []
    if not urls:
        raise FileNotFoundError(
            f"GraphML cache not found ({cache_path}) and no Overpass URLs configured for download."
        )
    if place_name is None and bbox_wgs84 is None:
        raise FileNotFoundError(
            f"GraphML cache {cache_path} does not exist and neither place_name nor bbox_wgs84 was provided."
        )

    import osmnx as ox

    last_err: Exception | None = None
    for url in urls:
        try:
            ox.settings.overpass_url = url
            ox.settings.requests_timeout = 15
            if bbox_wgs84 is not None:
                try:
                    raw_g = ox.graph_from_bbox(bbox=bbox_wgs84, network_type="drive", simplify=True)
                except TypeError:
                    west, south, east, north = bbox_wgs84
                    raw_g = ox.graph_from_bbox(north, south, east, west, network_type="drive", simplify=True)
            else:
                assert place_name is not None
                raw_g = ox.graph_from_place(place_name, network_type="drive", simplify=True)
            proj_g = ox.project_graph(raw_g)
            net = build_road_network_from_multidigraph(proj_g)
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                nx.write_graphml(net.graph, cache_path)
            return net
        except Exception as exc:
            last_err = exc
            continue

    raise ConnectionError(f"Failed to download road network from all Overpass mirrors: {last_err}")
