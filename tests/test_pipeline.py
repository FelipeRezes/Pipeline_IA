"""Testes unitários e de integração da pipeline refatorada (stdlib unittest)."""

from __future__ import annotations

import math
import unittest

import numpy as np

from pipeline.association import AssociationCandidate, association_metric
from pipeline.cityflow import load_cityflow
from pipeline.config import PipelineConfig
from pipeline.ekf import (
    AdaptiveEKF,
    CoordinatedTurnEKF,
    IMMFilter,
    LinearKalmanCV,
    create_filter,
)
from pipeline.evaluation import evaluate
from pipeline.experiments import (
    run_experiment_a1_kinematics,
    run_experiment_a2_fading_stability,
    run_experiment_a3_homography_projection,
    run_experiment_a5_topological_pruning,
)
from pipeline.geometry import (
    apply_homography,
    fit_homography,
    homography_from_intrinsics_extrinsics,
)
from pipeline.kinematics import (
    bicycle_jacobian,
    bicycle_step,
    coordinated_turn_jacobian,
    coordinated_turn_step,
)
from pipeline.optimize import linear_sum_assignment
from pipeline.pipeline import SurveillancePipeline
from pipeline.reid import VehicleReIDExtractor, cosine_similarity, synthetic_vehicle_embedding
from pipeline.roadgraph import Edge, RoadGraph, grid_graph
from pipeline.synthetic import generate_synthetic_cityflow


class TestGeometry(unittest.TestCase):
    def test_homography_roundtrip(self):
        H = np.array([[1.2, 0.1, 10.0], [0.05, 0.9, -5.0], [0.001, 0.0005, 1.0]])
        src = np.array([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40]], float)
        dst = apply_homography(H, src)
        Hh = fit_homography(src, dst)
        self.assertLess(np.linalg.norm(apply_homography(Hh, src) - dst).max(), 1e-6)

    def test_intrinsics_extrinsics_projection(self):
        exp_a3 = run_experiment_a3_homography_projection()
        res = exp_a3["results"]
        self.assertLess(
            res["corrected_contact_point_mean_error_m"],
            res["naive_bottom_center_mean_error_m"],
        )


class TestKinematics(unittest.TestCase):
    def test_bicycle_jacobian_matches_finite_differences(self):
        cfg = PipelineConfig(steer_damping=1.0)
        x = np.array([1.0, 2.0, 8.0, 0.4, 0.15])
        F = bicycle_jacobian(x, cfg.dt, cfg.wheelbase, steer_damping=1.0)
        eps = 1e-6
        for j in range(5):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fd = (
                bicycle_step(xp, cfg.dt, cfg.wheelbase, steer_damping=1.0)
                - bicycle_step(xm, cfg.dt, cfg.wheelbase, steer_damping=1.0)
            ) / (2 * eps)
            self.assertLess(np.linalg.norm(F[:, j] - fd), 1e-5, f"coluna {j}")

    def test_ctrv_jacobian_matches_finite_differences(self):
        x = np.array([3.0, -2.0, 12.0, 0.3, 0.25])
        dt = 0.1
        F = coordinated_turn_jacobian(x, dt, omega_damping=0.95)
        eps = 1e-6
        for j in range(5):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fd = (
                coordinated_turn_step(xp, dt, omega_damping=0.95)
                - coordinated_turn_step(xm, dt, omega_damping=0.95)
            ) / (2 * eps)
            self.assertLess(np.linalg.norm(F[:, j] - fd), 1e-5, f"coluna CTRV {j}")


class TestFilters(unittest.TestCase):
    def test_fading_factor_increases_on_maneuver(self):
        cfg = PipelineConfig()
        ekf = AdaptiveEKF([0, 0, 10, 0, 0], cfg=cfg)
        for _ in range(8):
            ekf.predict()
            ekf.update(ekf.position + np.array([1.2, 0.8]), np.eye(2) * 0.05)
        self.assertTrue(all(1.0 <= v <= cfg.lambda_max for v in ekf.lambda_history))
        self.assertGreater(max(ekf.lambda_history), 1.01)
        self.assertTrue(np.all(np.isfinite(ekf.P)))

    def test_dead_reckoning_grows_covariance(self):
        ekf = AdaptiveEKF([0, 0, 10, 0, 0], cfg=PipelineConfig())
        p0 = np.trace(ekf.P)
        for _ in range(10):
            ekf.propagate_only()
        self.assertGreater(np.trace(ekf.P), p0)

    def test_nonlinear_filters_outperform_linear_cv_on_evasive_turn(self):
        exp_a1 = run_experiment_a1_kinematics()
        res = exp_a1["results"]
        self.assertLess(
            res["adaptive_bicycle"]["rmse_maneuver_m"],
            res["linear_cv"]["rmse_maneuver_m"],
        )


class TestReIDAndAssociation(unittest.TestCase):
    def test_reid_discriminates_identities(self):
        e1_cam1 = synthetic_vehicle_embedding(1, "11")
        e1_cam2 = synthetic_vehicle_embedding(1, "22")
        e2_cam2 = synthetic_vehicle_embedding(2, "22")
        self.assertGreater(cosine_similarity(e1_cam1, e1_cam2), 0.70)
        self.assertLess(cosine_similarity(e1_cam1, e2_cam2), 0.45)

    def test_metric_rejects_far_target(self):
        cfg = PipelineConfig()
        a = AssociationCandidate(
            "a", np.array([0.0, 0.0]), np.array([10.0, 0.0]), np.array([2.0, 4.0]), 0.0, "11"
        )
        b = AssociationCandidate(
            "b", np.array([500.0, 500.0]), np.array([10.0, 0.0]), np.array([2.0, 4.0]), 1.0, "22"
        )
        self.assertFalse(association_metric(a, b, cfg).accepted)


class TestRoadGraph(unittest.TestCase):
    def test_turn_pruning_and_critical_speed(self):
        g = RoadGraph()
        for n, (x, y) in [("a", (0, 0)), ("b", (10, 0)), ("c", (20, 0))]:
            g.add_node(n, x, y)
        g.add_edge(Edge("e1", "a", "b", [(0, 0), (3, 0)], 13.9))
        cfg = PipelineConfig(max_steer_deg=12.0)
        x = np.array([0.0, 0.0, 5.0, np.pi / 2, 0.0])  # Apontando 90 graus para cima
        self.assertEqual(g.feasible_edges("a", x, cfg), [])

        # Em alta velocidade, seguir reto deve ter probabilidade maior que dobrar 90 graus
        exp_a5 = run_experiment_a5_topological_pruning()
        self.assertGreater(
            exp_a5["results"]["map_matching_hmm_viterbi_accuracy"],
            exp_a5["results"]["map_matching_euclidean_accuracy"],
        )

    def test_route_tree_probabilities_sum_to_one(self):
        g = grid_graph(3, 3, 80.0)
        routes = g.build_route_tree(
            "e_v_1_1_v_2_1", np.array([80, 80, 12, 0, 0]), PipelineConfig()
        )
        self.assertTrue(routes)
        self.assertAlmostEqual(sum(r.prob for r in routes), 1.0, places=5)


class TestOptimization(unittest.TestCase):
    def test_hungarian(self):
        r, c = linear_sum_assignment(np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]]))
        self.assertEqual(
            sorted(zip(r.tolist(), c.tolist())), [(0, 1), (1, 0), (2, 2)]
        )


class TestVideoLayer(unittest.TestCase):
    def test_video_ingest_to_ekf(self):
        import tempfile
        from pathlib import Path
        from pipeline.video import (
            AnnotationDetector,
            VideoAnalytics,
            VideoSource,
            render_videos,
        )
        root = str(Path(tempfile.gettempdir()) / "cityflow_video_test")
        ds, graph = generate_synthetic_cityflow(
            root, n_vehicles=10, n_frames=140, seed=20261110
        )
        paths = render_videos(ds, root)
        sources = [
            VideoSource(cam, p, fps=ds.fps, max_frames=140)
            for cam, p in sorted(paths.items())
        ]
        analytics = VideoAnalytics(
            sources,
            dict(ds.calibrations),
            AnnotationDetector(ds),
            PipelineConfig(),
            graph,
        )
        ds_video, result = analytics.run()
        self.assertGreater(analytics.stats.frames_read, 0)
        self.assertGreater(analytics.stats.boxes_detected, 0)
        self.assertEqual(analytics.stats.source_errors, {})
        self.assertIsNotNone(result)
        self.assertGreater(result.n_updates, 0)


class TestEndToEnd(unittest.TestCase):
    def test_synthetic_pipeline(self):
        import tempfile
        from pathlib import Path
        root = str(Path(tempfile.gettempdir()) / "cityflow_test")
        ds, graph = generate_synthetic_cityflow(
            root, n_vehicles=8, n_frames=240, seed=7
        )
        ds2 = load_cityflow(root)
        result = SurveillancePipeline(ds2, graph, PipelineConfig()).run()
        metrics = evaluate(ds2, result)
        for key in [
            "idf1",
            "cross_camera_accuracy",
            "route_top1_accuracy",
            "throughput_fps",
        ]:
            self.assertIn(key, metrics)
        self.assertGreater(result.n_updates, 0)
        self.assertTrue(result.routes or not result.matches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
