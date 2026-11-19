from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.area4]


def _dedup_edges(edges: list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for eid in edges:
        if not out or out[-1] != eid:
            out.append(eid)
    return out


# ---------------------------------------------------------------------------
# Tier 1 — Happy Path (6 tests)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_t1_a4_01_load_offline_graphml_cache(
    synthetic_graphml_path: Path,
    utm_origin: tuple[float, float, int],
) -> None:
    osm_loader = importlib.import_module("tracker.osm_loader")
    _, _, expected_epsg = utm_origin

    net = osm_loader.load_utm_road_graph(
        graphml_cache=synthetic_graphml_path,
        overpass_urls=[],
    )
    assert isinstance(net, osm_loader.RoadNetwork)
    assert net.epsg_utm == expected_epsg
    assert len(net.nodes_utm) >= 8
    assert len(net.edges) >= 8
    assert net.bounds_utm[0] < net.bounds_utm[2]
    assert net.bounds_utm[1] < net.bounds_utm[3]

    for edge in net.edges.values():
        assert isinstance(edge, osm_loader.RoadEdge)
        assert edge.length_m > 0.0
        assert edge.maxspeed_mps > 0.0
        assert edge.lanes >= 1
        assert edge.polyline_utm.ndim == 2 and edge.polyline_utm.shape[1] == 2
        assert np.all(np.isfinite(edge.polyline_utm))
        assert math.isfinite(edge.heading_start_rad)
        assert math.isfinite(edge.heading_end_rad)


@pytest.mark.tier1
def test_t1_a4_02_snap_heading_and_shortest_path(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ox, oy, _ = utm_origin

    eid, hdg = synthetic_road_engine.snap_initial_heading(ox + 50.0, oy + 2.0)
    assert eid[:2] == (100, 101)
    assert abs(hdg - 0.0) < 0.1

    path = synthetic_road_engine.shortest_feasible_path(100, 103)
    assert [e[:2] for e in path] == [(100, 101), (101, 102), (102, 103)]
    for i in range(len(path) - 1):
        assert path[i][1] == path[i + 1][0]


@pytest.mark.tier1
def test_t1_a4_03_hmm_viterbi_eliminates_euclidean_detour(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ox, oy, _ = utm_origin
    rng = np.random.default_rng(7)

    xs = np.linspace(ox + 10.0, ox + 290.0, 20)
    ys = np.full(20, oy, dtype=np.float64) + rng.normal(0.0, 4.0, size=20)
    # Deterministic noisy GPS observations near N_side (y=+12m) and N_cross (x=100, y=+15m)
    # within 9.5m sigma envelope, triggering Euclidean detour onto side/cross streets.
    ys[5] = oy + 9.2
    ys[6] = oy + 10.5
    ys[7] = oy + 9.8
    noisy_pts = np.column_stack([xs, ys])

    hmm_edges = synthetic_road_engine.map_match_hmm_viterbi(noisy_pts, obs_noise_m=9.5)
    euc_edges = synthetic_road_engine.map_match_euclidean(noisy_pts)

    corridor_pairs = {(100, 101), (101, 102), (102, 103)}
    detour_pairs = {(101, 104), (104, 101), (105, 106), (106, 105)}

    hmm_seq = _dedup_edges(hmm_edges)
    assert [e[:2] for e in hmm_seq] == [(100, 101), (101, 102), (102, 103)]
    assert all(e[:2] not in detour_pairs for e in hmm_edges)

    hmm_acc = sum(1 for e in hmm_edges if e[:2] in corridor_pairs) / max(len(hmm_edges), 1)
    euc_acc = sum(1 for e in euc_edges if e[:2] in corridor_pairs) / max(len(euc_edges), 1)
    assert hmm_acc >= euc_acc
    assert len(hmm_seq) <= len(_dedup_edges(euc_edges))


@pytest.mark.tier1
def test_t1_a4_04_propagate_tree_pre_corner_braking_and_reaccel(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    rg_mod = importlib.import_module("tracker.roadgraph")
    ox, oy, _ = utm_origin

    # Approaching N1 at 7.0 m/s ahead of R=8.0m 90-deg corner N1->N4 (v_crit ~= 7.41 m/s)
    state_exit = ekf_mod.FilterState(
        x=np.array([ox + 85.0, oy + 0.0, 7.0, 0.0, 0.0], dtype=np.float64),
        P=np.diag([1.0, 1.0, 0.5, 0.05, 0.05]),
        x_utm=ox + 85.0,
        y_utm=oy + 0.0,
        v_mps=7.0,
        theta_rad=0.0,
        P_pos_2x2=np.diag([1.0, 1.0]),
    )

    v_crit = math.sqrt(0.7 * 9.81 * 8.0)

    # Short horizon: right around/exiting the 90-deg curve
    hyps_turn = synthetic_road_engine.propagate_route_tree(state_exit, elapsed_s=3.5, dt_step=0.5)
    assert len(hyps_turn) > 0
    turn_corner_hyps = [h for h in hyps_turn if any(e[:2] == (101, 107) for e in h.edge_sequence)]
    assert turn_corner_hyps, "Expected N1->N4 corner branch to survive at 7.0 m/s"
    assert all(h.v_mps <= v_crit + 1.0 for h in turn_corner_hyps)

    # Longer horizon: vehicle clears corner and re-accelerates on straight toward maxspeed (13.89 m/s)
    hyps_long = synthetic_road_engine.propagate_route_tree(state_exit, elapsed_s=10.0, dt_step=0.5)
    assert len(hyps_long) > 0
    surviving = [h for h in hyps_long if not h.pruned_by_vcrit and h.probability > 0.0]
    assert surviving
    assert sum(h.probability for h in surviving) == pytest.approx(1.0, abs=1e-3)

    for h in surviving:
        assert isinstance(h, rg_mod.BlindHypothesis)
        assert h.P_2x2.shape == (2, 2)
        assert np.allclose(h.P_2x2, h.P_2x2.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(h.P_2x2) > 0.0)

    long_corner_hyps = [h for h in surviving if any(e[:2] == (101, 107) for e in h.edge_sequence)]
    assert long_corner_hyps
    assert any(h.v_mps > v_crit for h in long_corner_hyps)
    assert all(h.eta_s > 0.0 for h in long_corner_hyps)


@pytest.mark.tier1
def test_t1_a4_05_recommend_blockade_points_cicc(
    synthetic_road_engine: Any,
    synthetic_road_network: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    rg_mod = importlib.import_module("tracker.roadgraph")
    ox, oy, _ = utm_origin

    hyp_a = rg_mod.BlindHypothesis(
        hypothesis_id="h_n4_n6",
        edge_sequence=[(101, 107, 0), (107, 110, 0)],
        current_edge=(107, 110, 0),
        s_m=60.0,
        x_utm=ox + 150.0,
        y_utm=oy - 160.0,
        v_mps=12.0,
        theta_rad=-0.7,
        P_2x2=np.diag([4.0, 4.0]),
        probability=0.6,
        eta_s=8.0,
        pruned_by_vcrit=False,
    )
    hyp_b = rg_mod.BlindHypothesis(
        hypothesis_id="h_n5_n6",
        edge_sequence=[(101, 102, 0), (102, 109, 0), (109, 110, 0)],
        current_edge=(109, 110, 0),
        s_m=40.0,
        x_utm=ox + 200.0,
        y_utm=oy - 160.0,
        v_mps=12.0,
        theta_rad=-math.pi / 2.0,
        P_2x2=np.diag([4.0, 4.0]),
        probability=0.4,
        eta_s=6.0,
        pruned_by_vcrit=False,
    )

    pts = synthetic_road_engine.recommend_blockade_points([hyp_a, hyp_b], top_k=3)
    assert 1 <= len(pts) <= 3
    assert isinstance(pts[0], rg_mod.BlockadePoint)
    probs = [p.interception_prob for p in pts]
    assert probs == sorted(probs, reverse=True)

    top = pts[0]
    assert top.node_id == 110
    assert top.interception_prob == pytest.approx(1.0, abs=1e-3)
    assert top.eta_seconds > 0.0
    assert (top.x_utm, top.y_utm) == pytest.approx(synthetic_road_network.nodes_utm[110])
    assert len(top.converging_routes) == 2


@pytest.mark.tier1
def test_t1_a4_06_calibrate_camera_headless_cli_yaml(
    repo_root: Path,
    synthetic_calibration_bundle: dict[str, Any],
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    import yaml

    geom = importlib.import_module("tracker.geometry")
    script = repo_root / "scripts" / "calibrate_camera.py"
    assert script.is_file(), f"Missing {script}"

    out_yaml = tmp_path / "calibrated_cam.yaml"
    res = run_cli(
        [
            str(script),
            "--image",
            str(synthetic_calibration_bundle["ref_image_path"]),
            "--points",
            str(synthetic_calibration_bundle["points_6_json"]),
            "--out",
            str(out_yaml),
            "--headless",
        ]
    )
    assert res.returncode == 0, f"calibrate_camera failed:\n{res.stdout}\n{res.stderr}"
    assert out_yaml.is_file()

    payload = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert "homography" in payload
    h_mat = np.asarray(payload["homography"], dtype=np.float64)
    assert h_mat.shape == (3, 3) and np.all(np.isfinite(h_mat))
    if "rmse" in payload:
        assert float(payload["rmse"]) < 1.0

    pts_data = json.loads(synthetic_calibration_bundle["points_6_json"].read_text(encoding="utf-8"))
    pt0 = pts_data["points"][0]
    proj = geom.CameraProjector.from_yaml(out_yaml)
    gx, gy = proj.pixel_to_ground(float(pt0["u"]), float(pt0["v"]))
    assert math.hypot(gx - float(pt0["x_utm"]), gy - float(pt0["y_utm"])) < 1.0


# ---------------------------------------------------------------------------
# Tier 2 — Boundary & Corner Cases (7 tests)
# ---------------------------------------------------------------------------


@pytest.mark.tier2
def test_t2_a4_01_oneway_violation_pruning(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    ox, oy, _ = utm_origin

    rev = synthetic_road_engine.shortest_feasible_path(102, 108)
    assert rev == []

    state_at_n2 = ekf_mod.FilterState(
        x=np.array([ox + 195.0, oy + 0.0, 10.0, 0.0, 0.0], dtype=np.float64),
        P=np.diag([1.0, 1.0, 0.5, 0.05, 0.05]),
        x_utm=ox + 195.0,
        y_utm=oy + 0.0,
        v_mps=10.0,
        theta_rad=0.0,
        P_pos_2x2=np.diag([1.0, 1.0]),
    )
    hyps = synthetic_road_engine.propagate_route_tree(state_at_n2, elapsed_s=5.0, dt_step=0.5)
    for h in hyps:
        for edge in h.edge_sequence:
            assert edge[:2] != (102, 108)


@pytest.mark.tier2
def test_t2_a4_02_vcrit_sharp_corner_high_speed_pruning(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    ox, oy, _ = utm_origin

    state_fast = ekf_mod.FilterState(
        x=np.array([ox + 98.0, oy + 0.0, 28.0, 0.0, 0.0], dtype=np.float64),
        P=np.diag([1.0, 1.0, 0.5, 0.05, 0.05]),
        x_utm=ox + 98.0,
        y_utm=oy + 0.0,
        v_mps=28.0,
        theta_rad=0.0,
        P_pos_2x2=np.diag([1.0, 1.0]),
    )
    hyps_fast = synthetic_road_engine.propagate_route_tree(state_fast, elapsed_s=4.0, dt_step=0.5)
    corner_fast = [h for h in hyps_fast if any(e[:2] == (101, 107) for e in h.edge_sequence)]
    for h in corner_fast:
        assert h.pruned_by_vcrit is True or h.probability < 0.02

    straight_fast = [
        h
        for h in hyps_fast
        if not h.pruned_by_vcrit and any(e[:2] == (101, 102) for e in h.edge_sequence)
    ]
    assert sum(h.probability for h in straight_fast) >= 0.90

    state_slow = ekf_mod.FilterState(
        x=np.array([ox + 98.0, oy + 0.0, 6.0, 0.0, 0.0], dtype=np.float64),
        P=np.diag([1.0, 1.0, 0.5, 0.05, 0.05]),
        x_utm=ox + 98.0,
        y_utm=oy + 0.0,
        v_mps=6.0,
        theta_rad=0.0,
        P_pos_2x2=np.diag([1.0, 1.0]),
    )
    hyps_slow = synthetic_road_engine.propagate_route_tree(state_slow, elapsed_s=4.0, dt_step=0.5)
    corner_slow = [
        h
        for h in hyps_slow
        if not h.pruned_by_vcrit and any(e[:2] == (101, 107) for e in h.edge_sequence)
    ]
    assert corner_slow
    assert sum(h.probability for h in corner_slow) > 0.02


@pytest.mark.tier2
def test_t2_a4_03_unreachable_node_and_empty_hypotheses(synthetic_road_engine: Any) -> None:
    rg_mod = importlib.import_module("tracker.roadgraph")

    path = synthetic_road_engine.shortest_feasible_path(100, 201)
    assert path == []

    assert synthetic_road_engine.recommend_blockade_points([]) == []

    pruned_hyp = rg_mod.BlindHypothesis(
        hypothesis_id="pruned_only",
        edge_sequence=[(101, 107, 0)],
        current_edge=(101, 107, 0),
        s_m=0.0,
        x_utm=330100.0,
        y_utm=7390000.0,
        v_mps=28.0,
        theta_rad=0.0,
        P_2x2=np.eye(2),
        probability=0.0,
        eta_s=0.0,
        pruned_by_vcrit=True,
    )
    assert synthetic_road_engine.recommend_blockade_points([pruned_hyp]) == []


@pytest.mark.tier2
def test_t2_a4_04_missing_cache_and_offline_overpass_raises(tmp_path: Path) -> None:
    osm_loader = importlib.import_module("tracker.osm_loader")

    with pytest.raises((FileNotFoundError, ConnectionError, OSError, RuntimeError, ValueError)):
        osm_loader.load_utm_road_graph(
            graphml_cache=tmp_path / "missing_network.graphml",
            overpass_urls=["http://127.0.0.1:1/api/interpreter"],
        )


@pytest.mark.tier2
def test_t2_a4_05_map_matching_degenerate_and_far_offroad(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ox, oy, _ = utm_origin

    assert synthetic_road_engine.map_match_hmm_viterbi(np.empty((0, 2), dtype=np.float64)) == []

    single = synthetic_road_engine.map_match_hmm_viterbi(np.array([[ox + 50.0, oy]], dtype=np.float64))
    assert isinstance(single, list)
    assert len(single) <= 1

    far_pts = np.array([[ox + 50.0 + i * 10.0, oy - 600.0] for i in range(5)], dtype=np.float64)
    far_res = synthetic_road_engine.map_match_hmm_viterbi(far_pts)
    assert isinstance(far_res, list)


@pytest.mark.tier2
def test_t2_a4_06_zero_elapsed_time_and_zero_speed(
    synthetic_road_engine: Any,
    utm_origin: tuple[float, float, int],
) -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    ox, oy, _ = utm_origin

    state_v0 = ekf_mod.FilterState(
        x=np.array([ox + 50.0, oy + 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        P=np.diag([1.0, 1.0, 0.2, 0.05, 0.05]),
        x_utm=ox + 50.0,
        y_utm=oy + 0.0,
        v_mps=0.0,
        theta_rad=0.0,
        P_pos_2x2=np.diag([1.0, 1.0]),
    )
    hyps = synthetic_road_engine.propagate_route_tree(state_v0, elapsed_s=0.0, dt_step=0.5)
    assert len(hyps) >= 1
    for h in hyps:
        assert math.isfinite(h.x_utm) and math.isfinite(h.y_utm)
        assert h.eta_s >= 0.0
        assert h.s_m >= 0.0
        assert np.all(np.isfinite(h.P_2x2))
        assert np.all(np.linalg.eigvalsh(h.P_2x2) > 0.0)


@pytest.mark.tier2
def test_t2_a4_07_calibrate_camera_rejects_under4_and_collinear(
    repo_root: Path,
    synthetic_calibration_bundle: dict[str, Any],
    run_cli: Callable[..., Any],
    tmp_path: Path,
) -> None:
    script = repo_root / "scripts" / "calibrate_camera.py"

    for label, pts_key in (("under4", "points_3_json"), ("collinear", "points_4_collinear_json")):
        out_yaml = tmp_path / f"bad_{label}.yaml"
        res = run_cli(
            [
                str(script),
                "--image",
                str(synthetic_calibration_bundle["ref_image_path"]),
                "--points",
                str(synthetic_calibration_bundle[pts_key]),
                "--out",
                str(out_yaml),
                "--headless",
            ]
        )
        assert res.returncode != 0, f"Expected calibrate_camera to reject {label} points"
        assert not out_yaml.exists(), f"Output YAML should not be written for {label}"
