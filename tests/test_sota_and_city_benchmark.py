"""Unit and integration tests for SOTA baselines (PF, LSTM, GRU, T-GNN) and City-Scale Multi-Camera Benchmark."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from scripts.run_benchmarks import main as benchmarks_main
from tracker.city_benchmark import (
    build_city_road_network,
    evaluate_city_scale_scenario,
    place_non_overlapping_cameras,
)
from tracker.sota_baselines import (
    GRUTrajectoryPredictor,
    LSTMTrajectoryPredictor,
    ParticleFilter,
    TemporalGNNPredictor,
)


def test_particle_filter_tracks_curved_trajectory() -> None:
    pf = ParticleFilter(num_particles=300, seed=7)
    pf.initialize(0.0, 0.0, v0=12.0, theta0=0.0)
    r_k = np.diag([0.25, 0.25])
    x, y, th = 0.0, 0.0, 0.0
    dt = 0.1
    errs = []
    for k in range(1, 25):
        omega = 0.6 if k > 10 else 0.0
        if abs(omega) < 1e-9:
            x += 12.0 * math.cos(th) * dt
            y += 12.0 * math.sin(th) * dt
        else:
            x += (12.0 / omega) * (math.sin(th + omega * dt) - math.sin(th))
            y += (12.0 / omega) * (-math.cos(th + omega * dt) + math.cos(th))
            th += omega * dt
        pf.predict(dt)
        st = pf.update(np.array([x, y]), r_k, timestamp=k * dt)
        errs.append(math.hypot(st.x_utm - x, st.y_utm - y))
    assert np.isfinite(errs).all()
    assert float(np.sqrt(np.mean(np.square(errs)))) < 1.5
    assert pf.neff > 0.0


def test_lstm_and_gru_causal_predict_and_update() -> None:
    r_k = np.diag([0.2, 0.2])
    for cls in (LSTMTrajectoryPredictor, GRUTrajectoryPredictor):
        model = cls(hidden_size=48, seq_len=6, seed=42)
        assert model.num_parameters > 1000
        model.initialize(0.0, 0.0, v0=12.0, theta0=0.0)
        for k in range(1, 15):
            st_pred = model.predict(0.1)
            assert math.isfinite(st_pred.x_utm) and math.isfinite(st_pred.y_utm)
            if k <= 10:
                st_upd = model.update(np.array([1.2 * k, 0.0]), r_k, timestamp=0.1 * k)
                assert math.isfinite(st_upd.x_utm)
        # After 14 steps at 12 m/s (1.2 m/step), position should be in a realistic range near 16.8 m
        assert 10.0 < model.state.x_utm < 24.0


def test_tgnn_route_prediction_probabilities_normalized() -> None:
    net, _ = build_city_road_network(grid_rows=4, grid_cols=4, block_m=120.0)
    tgnn = TemporalGNNPredictor(net, hidden_dim=16, seed=42)
    assert tgnn.num_parameters > 500

    # Pick an interior edge with multiple outgoing choices
    outgoing: dict[int, list[tuple[int, int, int]]] = {}
    for eid, e in net.edges.items():
        outgoing.setdefault(e.u, []).append(eid)
    interior = next(e for e in net.edges.values() if len(outgoing.get(e.v, [])) >= 2)

    hist = [(12.0, interior.heading_end_rad, 0.6 + 0.08 * i) for i in range(5)]
    probs = tgnn.predict_next_edge(interior.edge_id, hist)
    assert len(probs) >= 1
    total_p = sum(p for _, p in probs)
    assert math.isclose(total_p, 1.0, abs_tol=1e-4)

    pred = tgnn.predict_blind_position(interior.edge_id, interior.length_m - 15.0, hist, elapsed_s=3.5)
    assert math.isfinite(pred.predicted_x_utm) and math.isfinite(pred.predicted_y_utm)


def test_city_scale_network_and_cameras() -> None:
    net, _ = build_city_road_network(grid_rows=8, grid_cols=8, block_m=120.0)
    assert len(net.nodes_utm) == 64
    assert len(net.edges) >= 150

    cams = place_non_overlapping_cameras(net, num_cameras=24, seed=42)
    assert len(cams) == 24
    assert len({c.cam_id for c in cams}) == 24

    res = evaluate_city_scale_scenario(num_cameras=8, num_vehicles=24, seed=77)
    assert res["num_cameras"] == 8
    assert res["num_vehicles"] == 24
    assert res["num_blind_transitions"] >= 20
    assert res["modes"]["hybrid"]["idf1"] >= res["modes"]["article"]["idf1"]
    assert res["modes"]["hybrid"]["idf1"] >= 0.70


def test_benchmarks_a6_and_a7_cli(tmp_path: Path) -> None:
    out_dir = tmp_path / "bench_a6_a7"
    rc = benchmarks_main(["--out", str(out_dir), "--experiments", "A6", "A7"])
    assert rc == 0
    data = json.loads((out_dir / "ablation_results.json").read_text(encoding="utf-8"))
    assert "A6" in data and "A7" in data
    kin = data["A6"]["results"]["kinematic_and_sequence_models"]
    assert {"linear_cv", "adaptive_bicycle", "particle_filter", "lstm", "gru"}.issubset(kin.keys())
    rt = data["A6"]["results"]["road_graph_route_models"]
    assert {"euclidean_snap", "hmm_viterbi", "t_gnn", "proposed_physics_tree"}.issubset(rt.keys())
    a7 = data["A7"]["results"]
    assert "S4_24cam_150veh" in a7
    assert a7["S4_24cam_150veh"]["num_cameras"] == 24
    assert a7["S4_24cam_150veh"]["num_vehicles"] == 150
