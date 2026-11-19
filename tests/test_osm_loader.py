"""Unit tests for src/tracker/osm_loader.py."""

from __future__ import annotations

from pathlib import Path
import networkx as nx
from tracker.osm_loader import RoadEdge, RoadNetwork, load_utm_road_graph


def test_load_utm_road_graph_from_graphml(tmp_path: Path) -> None:
    g = nx.MultiDiGraph(crs="EPSG:32723")
    g.add_node(1, x=330000.0, y=7390000.0)
    g.add_node(2, x=330100.0, y=7390000.0)
    g.add_edge(1, 2, key=0, length=100.0, oneway=True, maxspeed="50", lanes="2")
    p = tmp_path / "net.graphml"
    nx.write_graphml(g, p)

    net = load_utm_road_graph(graphml_cache=p, overpass_urls=[])
    assert isinstance(net, RoadNetwork)
    assert net.epsg_utm == 32723
    assert (1, 2, 0) in net.edges
    assert isinstance(net.edges[(1, 2, 0)], RoadEdge)
