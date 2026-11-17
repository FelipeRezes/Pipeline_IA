"""Testes de regressão (stdlib unittest — sem dependência de pytest)."""

from __future__ import annotations

import unittest

import numpy as np

from pipeline.association import AssociationCandidate, association_metric
from pipeline.cityflow import load_cityflow
from pipeline.config import PipelineConfig
from pipeline.ekf import AdaptiveEKF
from pipeline.evaluation import evaluate
from pipeline.geometry import apply_homography, fit_homography
from pipeline.kinematics import bicycle_jacobian, bicycle_step
from pipeline.optimize import linear_sum_assignment
from pipeline.pipeline import SurveillancePipeline
from pipeline.roadgraph import RoadGraph, grid_graph
from pipeline.synthetic import generate_synthetic_cityflow


class TestGeometry(unittest.TestCase):
    def test_homography_roundtrip(self):
        H = np.array([[1.2, 0.1, 10.0], [0.05, 0.9, -5.0], [0.001, 0.0005, 1.0]])
        src = np.array([[0, 0], [100, 0], [100, 80], [0, 80], [50, 40]], float)
        dst = apply_homography(H, src)
        Hh = fit_homography(src, dst)
        self.assertLess(np.linalg.norm(apply_homography(Hh, src) - dst).max(), 1e-6)


class TestKinematics(unittest.TestCase):
    def test_jacobian_matches_finite_differences(self):
        cfg = PipelineConfig()
        x = np.array([1.0, 2.0, 8.0, 0.4, 0.15])
        F = bicycle_jacobian(x, cfg.dt, cfg.wheelbase)
        eps = 1e-6
        for j in range(5):
            xp, xm = x.copy(), x.copy()
            xp[j] += eps
            xm[j] -= eps
            fd = (bicycle_step(xp, cfg.dt, cfg.wheelbase) -
                  bicycle_step(xm, cfg.dt, cfg.wheelbase)) / (2 * eps)
            self.assertLess(np.linalg.norm(F[:, j] - fd), 1e-5, f"coluna {j}")


class TestEKF(unittest.TestCase):
    def test_fading_factor_increases_on_maneuver(self):
        cfg = PipelineConfig()
        ekf = AdaptiveEKF([0, 0, 10, 0, 0], cfg=cfg)
        for _ in range(8):
            ekf.predict()
            ekf.update(ekf.position + np.array([1.0, 0.0]), np.eye(2) * 0.05)
        self.assertTrue(all(1.0 <= v <= cfg.lambda_max for v in ekf.lambda_history))
        # resíduo sistemático deve inflar lambda em algum instante
        self.assertGreater(max(ekf.lambda_history), 1.01)
        self.assertTrue(np.all(np.isfinite(ekf.P)))

    def test_dead_reckoning_grows_covariance(self):
        ekf = AdaptiveEKF([0, 0, 10, 0, 0], cfg=PipelineConfig())
        p0 = np.trace(ekf.P)
        for _ in range(10):
            ekf.propagate_only()
        self.assertGreater(np.trace(ekf.P), p0)


class TestAssociation(unittest.TestCase):
    def test_metric_rejects_far_target(self):
        cfg = PipelineConfig()
        a = AssociationCandidate("a", np.array([0.0, 0.0]), np.array([10.0, 0.0]),
                                 np.array([2.0, 4.0]), 0.0, "11")
        b = AssociationCandidate("b", np.array([500.0, 500.0]), np.array([10.0, 0.0]),
                                 np.array([2.0, 4.0]), 1.0, "22")
        self.assertFalse(association_metric(a, b, cfg).accepted)


class TestRoadGraph(unittest.TestCase):
    def test_turn_pruning(self):
        g = RoadGraph()
        for n in ["a", "b", "c"]:
            g.add_node(n, 0, 0)
        g.add_edge(__import__("pipeline.roadgraph", fromlist=["Edge"]).Edge(
            "e1", "a", "b", [(0, 0), (10, 0)], 13.9))
        cfg = PipelineConfig(max_steer_deg=12.0)
        x = np.array([5.0, 0.0, 5.0, np.pi / 2, 0.0])  # apontando para cima
        self.assertEqual(g.feasible_edges("b", x, cfg), [])

    def test_route_tree_probabilities_sum_to_one(self):
        g = grid_graph(3, 3, 80.0)
        routes = g.build_route_tree("e_v_1_1_v_2_1",
                                    np.array([80, 80, 12, 0, 0]),
                                    PipelineConfig())
        self.assertTrue(routes)
        self.assertAlmostEqual(sum(r.prob for r in routes), 1.0, places=5)


class TestOptimization(unittest.TestCase):
    def test_hungarian(self):
        r, c = linear_sum_assignment(np.array([[4, 1, 3], [2, 0, 5], [3, 2, 2]]))
        self.assertEqual(sorted(zip(r.tolist(), c.tolist())),
                         [(0, 1), (1, 0), (2, 2)])


class TestVideoLayer(unittest.TestCase):
    def test_video_ingest_to_ekf(self):
        from pipeline.video import (AnnotationDetector, VideoAnalytics,
                                    VideoSource, render_videos)
        root = "/tmp/opencode/cityflow_video_test"
        ds, graph = generate_synthetic_cityflow(root, n_vehicles=12,
                                                n_frames=150, seed=20261110)
        paths = render_videos(ds, root)
        sources = [VideoSource(cam, p, fps=ds.fps, max_frames=150)
                   for cam, p in sorted(paths.items())]
        analytics = VideoAnalytics(
            sources, dict(ds.calibrations), AnnotationDetector(ds),
            PipelineConfig(), graph)
        ds_video, result = analytics.run()
        self.assertGreater(analytics.stats.frames_read, 0)
        self.assertGreater(analytics.stats.boxes_detected, 0)
        self.assertEqual(analytics.stats.source_errors, {})
        self.assertIsNotNone(result)
        self.assertGreater(result.n_updates, 0)


class TestEndToEnd(unittest.TestCase):
    def test_synthetic_pipeline(self):
        root = "/tmp/opencode/cityflow_test"
        ds, graph = generate_synthetic_cityflow(root, n_vehicles=6,
                                                n_frames=200, seed=7)
        ds2 = load_cityflow(root)
        result = SurveillancePipeline(ds2, graph, PipelineConfig()).run()
        metrics = evaluate(ds2, result)
        for key in ["idf1", "cross_camera_accuracy", "route_top1_accuracy",
                    "throughput_fps"]:
            self.assertIn(key, metrics)
        self.assertGreater(result.n_updates, 0)
        self.assertTrue(result.routes or not result.matches)


if __name__ == "__main__":
    unittest.main(verbosity=2)
