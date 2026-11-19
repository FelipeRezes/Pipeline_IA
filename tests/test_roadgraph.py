"""Unit tests for src/tracker/roadgraph.py."""

from __future__ import annotations

import numpy as np
from scripts.run_benchmarks import _build_benchmark_road_network
from tracker.ekf import FilterState


def test_roadgraph_engine_core() -> None:
    _, eng = _build_benchmark_road_network()
    eid, hdg = eng.snap_initial_heading(330050.0, 7390000.0)
    assert eid[:2] == (100, 101)
    assert abs(hdg) < 0.1

    path = eng.shortest_feasible_path(100, 103)
    assert len(path) == 3

    pts = np.array([[330020.0, 7390000.0], [330150.0, 7390000.0], [330250.0, 7390000.0]])
    matched = eng.map_match_hmm_viterbi(pts)
    assert len(matched) == 3

    p5 = np.eye(5)
    st = FilterState(np.array([330085.0, 7390000.0, 7.0, 0.0, 0.0]), p5, 330085.0, 7390000.0, 7.0, 0.0, p5[:2, :2])
    hyps = eng.propagate_route_tree(st, elapsed_s=4.0)
    assert len(hyps) >= 1
    blks = eng.recommend_blockade_points(hyps, top_k=2)
    assert len(blks) >= 1
