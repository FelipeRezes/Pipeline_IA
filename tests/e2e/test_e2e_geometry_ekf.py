from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.e2e]


def _make_tilted_3d_params() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create pinhole K, 25-deg downward pitch R, and t for camera 8m above Z=0."""
    k_mat = np.array(
        [
            [800.0, 0.0, 640.0],
            [0.0, 800.0, 360.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    pitch = math.radians(25.0)
    r_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -math.sin(pitch), -math.cos(pitch)],
            [0.0, math.cos(pitch), -math.sin(pitch)],
        ],
        dtype=np.float64,
    )
    c_world = np.array([330000.0, 7390000.0, 8.0], dtype=np.float64)
    t_vec = -r_mat @ c_world
    return k_mat, r_mat, t_vec


def _project_3d_ground(
    k_mat: np.ndarray,
    r_mat: np.ndarray,
    t_vec: np.ndarray,
    pts_xy: np.ndarray,
) -> np.ndarray:
    p_mat = k_mat @ np.column_stack([r_mat, t_vec])
    pts_h = np.column_stack(
        [
            pts_xy[:, 0],
            pts_xy[:, 1],
            np.zeros(len(pts_xy), dtype=np.float64),
            np.ones(len(pts_xy), dtype=np.float64),
        ]
    )
    uvw = (p_mat @ pts_h.T).T
    return uvw[:, :2] / uvw[:, 2:3]


# ===========================================================================
# Area 2 — Geometry (Features #8–#12): 6 Tier 1 + 6 Tier 2
# ===========================================================================


@pytest.mark.area2
@pytest.mark.tier1
def test_t1_a2_01_planar_homography_dlt_and_roundtrip(
    utm_origin: tuple[float, float, int],
) -> None:
    geom = importlib.import_module("tracker.geometry")
    ox, oy, epsg = utm_origin

    # True homography mapping 640x480 trapezoid to metric UTM patch
    h_true = np.array(
        [
            [0.10, 0.01, ox + 3000.0],
            [-0.005, -0.08, oy + 5040.0],
            [0.0, -0.0005, 1.0],
        ],
        dtype=np.float64,
    )
    src_pts = np.array(
        [
            [80.0, 140.0],
            [560.0, 140.0],
            [580.0, 420.0],
            [60.0, 420.0],
            [200.0, 200.0],
            [440.0, 220.0],
            [320.0, 310.0],
            [180.0, 380.0],
        ],
        dtype=np.float64,
    )
    src_h = np.column_stack([src_pts, np.ones(len(src_pts))])
    proj_h = (h_true @ src_h.T).T
    dst_pts = proj_h[:, :2] / proj_h[:, 2:3]

    h_est, rmse = geom.compute_homography_dlt(src_pts, dst_pts, use_ransac=False)
    assert h_est.shape == (3, 3)
    assert rmse < 1e-4

    projector = geom.CameraProjector.from_homography(h_est, epsg_utm=epsg)
    for (u, v), (gx_true, gy_true) in zip(src_pts, dst_pts):
        gx, gy = projector.pixel_to_ground(float(u), float(v))
        assert math.hypot(gx - gx_true, gy - gy_true) < 1e-3
        u2, v2 = projector.ground_to_pixel(gx, gy)
        assert math.hypot(u2 - u, v2 - v) < 1e-3


@pytest.mark.area2
@pytest.mark.tier1
def test_t1_a2_02_homography_ransac_outlier_rejection(
    utm_origin: tuple[float, float, int],
) -> None:
    geom = importlib.import_module("tracker.geometry")
    ox, oy, epsg = utm_origin

    h_true = np.array(
        [
            [0.12, 0.0, ox + 100.0],
            [0.0, -0.10, oy + 80.0],
            [0.0, -0.0004, 1.0],
        ],
        dtype=np.float64,
    )
    inlier_uv = np.array(
        [
            [100.0, 150.0],
            [250.0, 150.0],
            [450.0, 160.0],
            [540.0, 170.0],
            [120.0, 260.0],
            [320.0, 270.0],
            [500.0, 280.0],
            [140.0, 390.0],
            [330.0, 400.0],
            [520.0, 410.0],
        ],
        dtype=np.float64,
    )
    inlier_h = np.column_stack([inlier_uv, np.ones(len(inlier_uv))])
    tmp = (h_true @ inlier_h.T).T
    inlier_xy = tmp[:, :2] / tmp[:, 2:3]

    outlier_uv = np.array([[200.0, 220.0], [420.0, 350.0]], dtype=np.float64)
    outlier_xy = np.array([[ox + 500.0, oy - 300.0], [ox - 400.0, oy + 350.0]], dtype=np.float64)

    src_all = np.vstack([inlier_uv, outlier_uv + np.array([180.0, -180.0])])
    dst_all = np.vstack([inlier_xy, outlier_xy])

    h_est, _ = geom.compute_homography_dlt(src_all, dst_all, use_ransac=True)
    projector = geom.CameraProjector.from_homography(h_est, epsg_utm=epsg)

    max_err = max(
        math.hypot(
            projector.pixel_to_ground(float(u), float(v))[0] - float(gx),
            projector.pixel_to_ground(float(u), float(v))[1] - float(gy),
        )
        for (u, v), (gx, gy) in zip(inlier_uv, inlier_xy)
    )
    assert max_err < 0.5


@pytest.mark.area2
@pytest.mark.tier1
def test_t1_a2_03_3d_camera_projection_ray_ground(
    utm_origin: tuple[float, float, int],
) -> None:
    geom = importlib.import_module("tracker.geometry")
    ox, oy, epsg = utm_origin
    k_mat, r_mat, t_vec = _make_tilted_3d_params()

    ground_pts = np.array(
        [
            [ox - 4.0, oy + 14.0],
            [ox + 3.0, oy + 18.0],
            [ox - 2.0, oy + 26.0],
            [ox + 5.0, oy + 34.0],
        ],
        dtype=np.float64,
    )
    uv_pts = _project_3d_ground(k_mat, r_mat, t_vec, ground_pts)

    projector = geom.CameraProjector.from_3d_params(k_mat, r_mat, t_vec, epsg_utm=epsg)
    for (u, v), (gx_true, gy_true) in zip(uv_pts, ground_pts):
        gx_hat, gy_hat = projector.pixel_to_ground(float(u), float(v))
        assert math.hypot(gx_hat - gx_true, gy_hat - gy_true) < 1e-3
        u_hat, v_hat = projector.ground_to_pixel(gx_true, gy_true)
        assert math.hypot(u_hat - u, v_hat - v) < 1e-3


@pytest.mark.area2
@pytest.mark.tier1
def test_t1_a2_04_wgs84_utm_roundtrip() -> None:
    geom = importlib.import_module("tracker.geometry")

    # São Paulo
    x_sp, y_sp, epsg_sp = geom.wgs84_to_utm(-23.55052, -46.633308)
    assert epsg_sp == 32723
    assert 300_000.0 < x_sp < 400_000.0
    assert 7_300_000.0 < y_sp < 7_500_000.0
    lat_sp, lon_sp = geom.utm_to_wgs84(x_sp, y_sp, epsg_sp)
    assert abs(lat_sp - (-23.55052)) < 1e-6
    assert abs(lon_sp - (-46.633308)) < 1e-6

    # Boston
    x_bo, y_bo, epsg_bo = geom.wgs84_to_utm(42.3601, -71.0589)
    assert epsg_bo == 32619
    lat_bo, lon_bo = geom.utm_to_wgs84(x_bo, y_bo, epsg_bo)
    assert abs(lat_bo - 42.3601) < 1e-6
    assert abs(lon_bo - (-71.0589)) < 1e-6


@pytest.mark.area2
@pytest.mark.tier1
def test_t1_a2_05_bbox_contact_correction_and_ground_dims(
    utm_origin: tuple[float, float, int],
) -> None:
    geom = importlib.import_module("tracker.geometry")
    _, _, epsg = utm_origin
    k_mat, r_mat, t_vec = _make_tilted_3d_params()
    projector = geom.CameraProjector.from_3d_params(k_mat, r_mat, t_vec, epsg_utm=epsg)

    bbox = (520.0, 440.0, 680.0, 580.0)
    u_c, v_c = projector.bbox_contact_point(bbox, use_perspective_correction=True, fraction=0.90)
    u_raw, v_raw = projector.bbox_contact_point(bbox, use_perspective_correction=False)

    assert v_c < bbox[3]
    assert abs(v_c - (440.0 + 0.90 * 140.0)) < 1e-6
    assert (u_c, v_c) != (u_raw, v_raw)

    w_m, l_m = projector.bbox_dimensions_ground(bbox)
    assert math.isfinite(w_m) and w_m > 0.0
    assert math.isfinite(l_m) and l_m > 0.0


@pytest.mark.area2
@pytest.mark.tier1
def test_t1_a2_06_covariance_propagation_and_yaml_load(
    tmp_path: Path,
    utm_origin: tuple[float, float, int],
) -> None:
    import yaml

    geom = importlib.import_module("tracker.geometry")
    _, _, epsg = utm_origin
    k_mat, r_mat, t_vec = _make_tilted_3d_params()

    cam_yaml = tmp_path / "cam_3d.yaml"
    cam_yaml.write_text(
        yaml.safe_dump(
            {
                "cam_id": "cam_test_3d",
                "epsg_utm": epsg,
                "K": k_mat.tolist(),
                "R": r_mat.tolist(),
                "t": t_vec.tolist(),
            }
        ),
        encoding="utf-8",
    )

    projector = geom.CameraProjector.from_yaml(cam_yaml)

    bbox_near = (560.0, 540.0, 700.0, 660.0)
    bbox_far = (600.0, 385.0, 660.0, 430.0)

    x_n, y_n, r_near = projector.bbox_to_ground(bbox_near, sigma_px=(4.0, 4.0))
    x_f, y_f, r_far = projector.bbox_to_ground(bbox_far, sigma_px=(4.0, 4.0))

    for val in (x_n, y_n, x_f, y_f):
        assert math.isfinite(val)

    for r_k in (r_near, r_far):
        assert r_k.shape == (2, 2)
        assert r_k.dtype == np.float64
        assert np.allclose(r_k, r_k.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(r_k) > 0.0)

    assert np.trace(r_far) > np.trace(r_near)


@pytest.mark.area2
@pytest.mark.tier2
def test_t2_a2_01_homography_rejects_under_four_points() -> None:
    geom = importlib.import_module("tracker.geometry")
    src_base = np.array([[10.0, 20.0], [50.0, 20.0], [50.0, 60.0]], dtype=np.float64)
    dst_base = np.array([[1.0, 2.0], [5.0, 2.0], [5.0, 6.0]], dtype=np.float64)

    for n in (0, 1, 2, 3):
        for use_ransac in (False, True):
            with pytest.raises(ValueError):
                geom.compute_homography_dlt(src_base[:n], dst_base[:n], use_ransac=use_ransac)


@pytest.mark.area2
@pytest.mark.tier2
def test_t2_a2_02_homography_rejects_collinear_points() -> None:
    geom = importlib.import_module("tracker.geometry")
    src = np.array([[i * 20.0, i * 10.0] for i in range(5)], dtype=np.float64)
    dst = np.array([[i * 2.0, i * 1.0] for i in range(5)], dtype=np.float64)

    for use_ransac in (False, True):
        with pytest.raises(ValueError):
            geom.compute_homography_dlt(src, dst, use_ransac=use_ransac)


@pytest.mark.area2
@pytest.mark.tier2
def test_t2_a2_03_horizon_singularity_raises_value_error() -> None:
    geom = importlib.import_module("tracker.geometry")

    # Homography whose denominator w = 0*u + 1*v - 100 vanishes at v = 100
    h_horiz = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, -100.0],
        ],
        dtype=np.float64,
    )
    proj_h = geom.CameraProjector.from_homography(h_horiz, epsg_utm=32723)
    with pytest.raises(ValueError):
        proj_h.pixel_to_ground(320.0, 100.0)
    with pytest.raises(ValueError):
        # contact point at v = 80 + 0.9*(102.2222 - 80) -> v == 100
        proj_h.bbox_to_ground((300.0, 80.0, 340.0, 102.222222222))

    # Horizontal 3D camera whose center ray (640, 360) is parallel to Z=0
    k_mat = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    r_flat = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    t_flat = -r_flat @ np.array([330000.0, 7390000.0, 8.0], dtype=np.float64)
    proj_3d = geom.CameraProjector.from_3d_params(k_mat, r_flat, t_flat, epsg_utm=32723)
    with pytest.raises(ValueError):
        proj_3d.pixel_to_ground(640.0, 360.0)


@pytest.mark.area2
@pytest.mark.tier2
def test_t2_a2_04_extreme_and_degenerate_bboxes(
    utm_origin: tuple[float, float, int],
) -> None:
    geom = importlib.import_module("tracker.geometry")
    _, _, epsg = utm_origin
    k_mat, r_mat, t_vec = _make_tilted_3d_params()
    projector = geom.CameraProjector.from_3d_params(k_mat, r_mat, t_vec, epsg_utm=epsg)

    for valid_box in ((600.0, 500.0, 601.0, 501.0), (600.0, 420.0, 610.0, 620.0)):
        u_c, v_c = projector.bbox_contact_point(valid_box)
        assert math.isfinite(u_c) and math.isfinite(v_c)
        x, y, r_k = projector.bbox_to_ground(valid_box)
        assert math.isfinite(x) and math.isfinite(y)
        assert np.all(np.isfinite(r_k))
        assert np.all(np.linalg.eigvalsh(r_k) > 0.0)

    for degen_box in ((600.0, 500.0, 600.0, 500.0), (620.0, 520.0, 580.0, 480.0)):
        try:
            x, y, r_k = projector.bbox_to_ground(degen_box)
        except ValueError:
            continue
        assert math.isfinite(x) and math.isfinite(y)
        assert np.all(np.isfinite(r_k))
        assert np.all(np.linalg.eigvalsh(r_k) > 0.0)


@pytest.mark.area2
@pytest.mark.tier2
def test_t2_a2_05_invalid_lat_lon_and_antimeridian() -> None:
    geom = importlib.import_module("tracker.geometry")

    for bad_lat, bad_lon in ((-91.0, 0.0), (91.0, 0.0), (0.0, 181.0), (0.0, -181.0)):
        with pytest.raises(ValueError):
            geom.wgs84_to_utm(bad_lat, bad_lon)

    for lat, lon in ((0.0, -180.0), (0.0, 180.0)):
        x, y, epsg = geom.wgs84_to_utm(lat, lon)
        assert math.isfinite(x) and math.isfinite(y)
        lat2, lon2 = geom.utm_to_wgs84(x, y, epsg)
        assert abs(lat2 - lat) < 1e-5
        assert abs(abs(lon2) - 180.0) < 1e-5


@pytest.mark.area2
@pytest.mark.tier2
def test_t2_a2_06_zero_sigma_px_spd_floor(
    utm_origin: tuple[float, float, int],
) -> None:
    geom = importlib.import_module("tracker.geometry")
    _, _, epsg = utm_origin
    k_mat, r_mat, t_vec = _make_tilted_3d_params()
    projector = geom.CameraProjector.from_3d_params(k_mat, r_mat, t_vec, epsg_utm=epsg)

    bbox = (540.0, 460.0, 680.0, 580.0)
    for sigma in ((0.0, 0.0), (1e-9, 1e-9)):
        _, _, r_k = projector.bbox_to_ground(bbox, sigma_px=sigma)
        assert np.all(np.isfinite(r_k))
        assert np.allclose(r_k, r_k.T, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(r_k) > 0.0)
        inv_r = np.linalg.inv(r_k)
        assert np.all(np.isfinite(inv_r))


# ===========================================================================
# Area 3 — Kinematics & EKF (Features #13–#19): 6 Tier 1 + 7 Tier 2
# ===========================================================================

_ALL_FILTER_NAMES = [
    "AdaptiveEKF",
    "CanonicalEKF",
    "CTRVFilter",
    "LinearCVFilter",
    "IMMFilter",
]


def _make_filter(cls_name: str):
    ekf_mod = importlib.import_module("tracker.ekf")
    cls = getattr(ekf_mod, cls_name)
    return cls()


@pytest.mark.area3
@pytest.mark.tier1
@pytest.mark.parametrize("filter_cls_name", _ALL_FILTER_NAMES)
def test_t1_a3_01_unified_filter_contract_straight_track(filter_cls_name: str) -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    FilterState = ekf_mod.FilterState

    assert ekf_mod.CTRV is ekf_mod.CTRVFilter
    assert ekf_mod.LinearCV is ekf_mod.LinearCVFilter
    assert ekf_mod.IMM is ekf_mod.IMMFilter

    rng = np.random.default_rng(101)
    v_true = 12.0
    theta_true = 0.3
    dt = 0.1
    sigma = 0.3
    r_k = np.diag([sigma**2, sigma**2])

    flt = _make_filter(filter_cls_name)
    flt.initialize(0.0, 0.0, v0=v_true, theta0=theta_true)

    last_state = flt.state
    assert isinstance(last_state, FilterState)

    true_xy = np.zeros((40, 2), dtype=np.float64)
    for k in range(1, 41):
        t = k * dt
        gt_x = v_true * math.cos(theta_true) * t
        gt_y = v_true * math.sin(theta_true) * t
        true_xy[k - 1] = [gt_x, gt_y]
        z_xy = np.array([gt_x, gt_y]) + rng.normal(0.0, sigma, size=2)
        flt.predict(dt)
        st = flt.update(z_xy, r_k, timestamp=t)
        assert isinstance(st, FilterState)
        assert np.all(np.isfinite(st.x))
        assert st.P_pos_2x2.shape == (2, 2)
        assert np.allclose(st.P, st.P.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(st.P) > 0.0)
        last_state = st

    final_err = math.hypot(last_state.x_utm - true_xy[-1, 0], last_state.y_utm - true_xy[-1, 1])
    assert final_err < 1.0
    assert abs(last_state.v_mps - v_true) < 1.5


@pytest.mark.area3
@pytest.mark.tier1
def test_t1_a3_02_ctrv_circular_arc_beats_linear_cv() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    rng = np.random.default_rng(202)
    v_true = 10.0
    omega_true = 0.25
    dt = 0.1
    sigma = 0.2
    r_k = np.diag([sigma**2, sigma**2])

    ctrv = ekf_mod.CTRVFilter()
    lcv = ekf_mod.LinearCVFilter()
    ctrv.initialize(0.0, 0.0, v0=v_true, theta0=0.0)
    lcv.initialize(0.0, 0.0, v0=v_true, theta0=0.0)

    x_gt, y_gt, th = 0.0, 0.0, 0.0
    errs_ctrv: list[float] = []
    errs_lcv: list[float] = []

    for k in range(1, 51):
        th_mid = th + 0.5 * omega_true * dt
        x_gt += (v_true / omega_true) * (math.sin(th + omega_true * dt) - math.sin(th))
        y_gt += (v_true / omega_true) * (-math.cos(th + omega_true * dt) + math.cos(th))
        th += omega_true * dt
        _ = th_mid

        z_xy = np.array([x_gt, y_gt]) + rng.normal(0.0, sigma, size=2)
        ctrv.predict(dt)
        st_c = ctrv.update(z_xy, r_k, timestamp=k * dt)
        lcv.predict(dt)
        st_l = lcv.update(z_xy, r_k, timestamp=k * dt)

        errs_ctrv.append(math.hypot(st_c.x_utm - x_gt, st_c.y_utm - y_gt))
        errs_lcv.append(math.hypot(st_l.x_utm - x_gt, st_l.y_utm - y_gt))

    rmse_ctrv = float(np.sqrt(np.mean(np.square(errs_ctrv[25:]))))
    rmse_lcv = float(np.sqrt(np.mean(np.square(errs_lcv[25:]))))
    assert rmse_ctrv < 0.8
    assert rmse_ctrv < rmse_lcv


@pytest.mark.area3
@pytest.mark.tier1
def test_t1_a3_03_adaptive_ekf_evasive_turn_lambda_k() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    rng = np.random.default_rng(303)
    v_true = 15.0
    dt = 0.1
    sigma = 0.25
    r_k = np.diag([sigma**2, sigma**2])

    aekf = ekf_mod.AdaptiveEKF()
    cekf = ekf_mod.CanonicalEKF()
    aekf.initialize(0.0, 0.0, v0=v_true, theta0=0.0)
    cekf.initialize(0.0, 0.0, v0=v_true, theta0=0.0)

    x_gt, y_gt, th = 0.0, 0.0, 0.0
    turn_errs_a: list[float] = []
    turn_errs_c: list[float] = []
    turn_triggered = False
    max_lambda = 1.0

    for k in range(1, 36):
        omega = 0.0 if k <= 20 else 1.2
        if abs(omega) < 1e-9:
            x_gt += v_true * math.cos(th) * dt
            y_gt += v_true * math.sin(th) * dt
        else:
            x_gt += (v_true / omega) * (math.sin(th + omega * dt) - math.sin(th))
            y_gt += (v_true / omega) * (-math.cos(th + omega * dt) + math.cos(th))
            th += omega * dt

        z_xy = np.array([x_gt, y_gt]) + rng.normal(0.0, sigma, size=2)
        aekf.predict(dt)
        st_a = aekf.update(z_xy, r_k, timestamp=k * dt)
        cekf.predict(dt)
        st_c = cekf.update(z_xy, r_k, timestamp=k * dt)

        if k <= 20:
            assert st_a.adaptive_triggered is False
            assert st_a.lambda_k == pytest.approx(1.0, abs=1e-6)
        else:
            if st_a.adaptive_triggered:
                turn_triggered = True
            max_lambda = max(max_lambda, float(st_a.lambda_k))
            turn_errs_a.append(math.hypot(st_a.x_utm - x_gt, st_a.y_utm - y_gt))
            turn_errs_c.append(math.hypot(st_c.x_utm - x_gt, st_c.y_utm - y_gt))

    assert turn_triggered is True
    assert max_lambda > 1.0
    rmse_a = float(np.sqrt(np.mean(np.square(turn_errs_a))))
    rmse_c = float(np.sqrt(np.mean(np.square(turn_errs_c))))
    assert rmse_a < rmse_c


@pytest.mark.area3
@pytest.mark.tier1
def test_t1_a3_04_adaptive_ekf_circular_theta_smoothing_across_pi() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    aekf = ekf_mod.AdaptiveEKF()
    aekf.initialize(100.0, 0.0, v0=12.0, theta0=math.pi)

    dt = 0.1
    v_true = 12.0
    r_k = np.diag([0.04, 0.04])

    for k in range(1, 31):
        # Alternate slight cross-track offsets (+0.04m / -0.04m) around heading pi (+-3.11 rad)
        x_gt = 100.0 - v_true * k * dt
        y_gt = 0.04 if (k % 2 == 0) else -0.04
        aekf.predict(dt)
        st = aekf.update(np.array([x_gt, y_gt]), r_k, timestamp=k * dt)
        assert -math.pi - 1e-9 <= st.theta_rad <= math.pi + 1e-9
        assert st.lambda_k < 1.5

    assert abs(math.cos(aekf.state.theta_rad) - (-1.0)) < 0.1


@pytest.mark.area3
@pytest.mark.tier1
def test_t1_a3_05_imm_filter_regime_switching() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    rng = np.random.default_rng(505)
    v_true = 12.0
    dt = 0.1
    sigma = 0.2
    r_k = np.diag([sigma**2, sigma**2])

    imm = ekf_mod.IMMFilter()
    lcv = ekf_mod.LinearCVFilter()
    imm.initialize(0.0, 0.0, v0=v_true, theta0=0.0)
    lcv.initialize(0.0, 0.0, v0=v_true, theta0=0.0)

    x_gt, y_gt, th = 0.0, 0.0, 0.0
    errs_imm: list[float] = []
    errs_lcv: list[float] = []

    for k in range(1, 61):
        omega = 0.6 if 20 < k <= 40 else 0.0
        if abs(omega) < 1e-9:
            x_gt += v_true * math.cos(th) * dt
            y_gt += v_true * math.sin(th) * dt
        else:
            x_gt += (v_true / omega) * (math.sin(th + omega * dt) - math.sin(th))
            y_gt += (v_true / omega) * (-math.cos(th + omega * dt) + math.cos(th))
            th += omega * dt

        z_xy = np.array([x_gt, y_gt]) + rng.normal(0.0, sigma, size=2)
        imm.predict(dt)
        st_i = imm.update(z_xy, r_k, timestamp=k * dt)
        lcv.predict(dt)
        st_l = lcv.update(z_xy, r_k, timestamp=k * dt)

        assert np.allclose(st_i.P, st_i.P.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(st_i.P) > 0.0)
        errs_imm.append(math.hypot(st_i.x_utm - x_gt, st_i.y_utm - y_gt))
        errs_lcv.append(math.hypot(st_l.x_utm - x_gt, st_l.y_utm - y_gt))

    overall_rmse = float(np.sqrt(np.mean(np.square(errs_imm))))
    turn_rmse_imm = float(np.sqrt(np.mean(np.square(errs_imm[20:40]))))
    turn_rmse_lcv = float(np.sqrt(np.mean(np.square(errs_lcv[20:40]))))
    assert overall_rmse < 1.0
    assert turn_rmse_imm < turn_rmse_lcv


@pytest.mark.area3
@pytest.mark.tier1
@pytest.mark.parametrize("filter_cls_name", _ALL_FILTER_NAMES)
def test_t1_a3_06_alpr_reset_position_covariance(filter_cls_name: str) -> None:
    flt = _make_filter(filter_cls_name)
    flt.initialize(100.0, 200.0, v0=12.0, theta0=0.0)
    for _ in range(30):
        flt.predict(0.1)

    p_before = flt.state.P_pos_2x2.copy()
    z_alpr = np.array([136.5, 200.4], dtype=np.float64)
    r_tight = np.diag([0.05, 0.05])

    st = flt.reset_position_covariance(z_alpr, r_tight)
    assert math.hypot(st.x_utm - z_alpr[0], st.y_utm - z_alpr[1]) < 1.0
    assert float(np.trace(st.P_pos_2x2)) < float(np.trace(p_before))
    assert np.allclose(st.P, st.P.T, atol=1e-10)
    assert np.all(np.linalg.eigvalsh(st.P) > 0.0)


@pytest.mark.area3
@pytest.mark.tier2
def test_t2_a3_01_ctrv_omega_zero_limit() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    ctrv = ekf_mod.CTRVFilter()
    ctrv.initialize(0.0, 0.0, v0=10.0, theta0=0.5)
    st = ctrv.predict(0.1)

    assert st.x_utm == pytest.approx(10.0 * math.cos(0.5) * 0.1, abs=1e-6)
    assert st.y_utm == pytest.approx(10.0 * math.sin(0.5) * 0.1, abs=1e-6)
    assert np.all(np.isfinite(st.x))
    assert np.all(np.isfinite(st.P))
    assert np.all(np.linalg.eigvalsh(st.P) > 0.0)


@pytest.mark.area3
@pytest.mark.tier2
def test_t2_a3_02_adaptive_ekf_stationary_low_speed_jitter() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    rng = np.random.default_rng(707)
    aekf = ekf_mod.AdaptiveEKF()
    aekf.initialize(100.0, 200.0, v0=0.05, theta0=0.0)

    r_k = np.diag([0.64, 0.64])
    for k in range(1, 31):
        z_xy = np.array([100.0, 200.0]) + rng.normal(0.0, 0.8, size=2)
        aekf.predict(0.1)
        st = aekf.update(z_xy, r_k, timestamp=k * 0.1)
        assert st.adaptive_triggered is False

    assert aekf.state.lambda_k == pytest.approx(1.0, abs=0.05)
    assert math.hypot(aekf.state.x_utm - 100.0, aekf.state.y_utm - 200.0) < 1.5


@pytest.mark.area3
@pytest.mark.tier2
def test_t2_a3_03_adaptive_ekf_single_frame_outlier_nis_guard() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    aekf = ekf_mod.AdaptiveEKF()
    aekf.initialize(0.0, 0.0, v0=12.0, theta0=0.0)

    dt = 0.1
    r_k = np.diag([0.04, 0.04])
    for k in range(1, 16):
        aekf.predict(dt)
        aekf.update(np.array([12.0 * k * dt, 0.0]), r_k, timestamp=k * dt)

    # Single 8.0m lateral spike
    aekf.predict(dt)
    st_out = aekf.update(np.array([12.0 * 16 * dt, 8.0]), r_k, timestamp=16 * dt)
    assert st_out.nis > 5.99
    assert st_out.adaptive_triggered is False
    assert st_out.lambda_k == pytest.approx(1.0, abs=1e-6)


@pytest.mark.area3
@pytest.mark.tier2
def test_t2_a3_04_adaptive_ekf_sustained_nis_persistence_trigger() -> None:
    ekf_mod = importlib.import_module("tracker.ekf")
    aekf = ekf_mod.AdaptiveEKF()
    aekf.initialize(0.0, 0.0, v0=12.0, theta0=0.0)

    dt = 0.1
    r_k = np.diag([0.04, 0.04])
    for k in range(1, 16):
        aekf.predict(dt)
        aekf.update(np.array([12.0 * k * dt, 0.0]), r_k, timestamp=k * dt)

    states = []
    for j in range(3):
        k = 16 + j
        aekf.predict(dt)
        st = aekf.update(np.array([12.0 * k * dt, 6.0]), r_k, timestamp=k * dt)
        states.append(st)

    assert states[0].nis > 5.99
    assert states[0].adaptive_triggered is False
    assert states[0].lambda_k == pytest.approx(1.0, abs=1e-6)
    assert any(s.adaptive_triggered and s.lambda_k > 1.0 for s in states[1:])


@pytest.mark.area3
@pytest.mark.tier2
@pytest.mark.parametrize("filter_cls_name", _ALL_FILTER_NAMES)
def test_t2_a3_05_filters_dt_zero_tiny_and_negative(filter_cls_name: str) -> None:
    flt = _make_filter(filter_cls_name)
    flt.initialize(10.0, 20.0, v0=5.0, theta0=0.4)

    st_zero = flt.predict(0.0)
    assert st_zero.x_utm == pytest.approx(10.0, abs=1e-12)
    assert st_zero.y_utm == pytest.approx(20.0, abs=1e-12)
    assert st_zero.v_mps == pytest.approx(5.0, abs=1e-12)
    assert st_zero.theta_rad == pytest.approx(0.4, abs=1e-12)
    assert np.all(np.isfinite(st_zero.P))
    assert np.all(np.linalg.eigvalsh(st_zero.P) > 0.0)

    st_tiny = flt.predict(1e-6)
    assert np.all(np.isfinite(st_tiny.x))
    assert np.all(np.isfinite(st_tiny.P))
    assert np.all(np.linalg.eigvalsh(st_tiny.P) > 0.0)

    with pytest.raises(ValueError):
        flt.predict(-0.1)


@pytest.mark.area3
@pytest.mark.tier2
@pytest.mark.parametrize("filter_cls_name", _ALL_FILTER_NAMES)
def test_t2_a3_06_filters_100_step_blind_and_ill_conditioned_R(filter_cls_name: str) -> None:
    flt = _make_filter(filter_cls_name)
    flt.initialize(0.0, 0.0, v0=10.0, theta0=0.2)

    for _ in range(100):
        st_p = flt.predict(0.1)
        assert np.all(np.isfinite(st_p.x))

    r_ill = np.diag([1e-9, 1e6])
    st_u = flt.update(np.array([98.0, 20.0]), r_ill, timestamp=10.0)
    assert np.all(np.isfinite(st_u.x))
    assert np.all(np.isfinite(st_u.P))
    assert np.allclose(st_u.P, st_u.P.T, atol=1e-7)
    assert np.all(np.linalg.eigvalsh(st_u.P) > 0.0)


@pytest.mark.area3
@pytest.mark.tier2
@pytest.mark.parametrize("filter_cls_name", ["CanonicalEKF", "AdaptiveEKF"])
def test_t2_a3_07_two_point_bicycle_uturn_extreme_steering(filter_cls_name: str) -> None:
    flt = _make_filter(filter_cls_name)
    v_true = 10.0
    dt = 0.1
    r_turn = 6.0
    omega = v_true / r_turn  # ~1.67 rad/s -> 180 deg U-turn in ~1.9 s

    flt.initialize(0.0, 0.0, v0=v_true, theta0=0.0)
    x_gt, y_gt, th = 0.0, 0.0, 0.0
    r_k = np.diag([0.09, 0.09])

    for k in range(1, 25):
        x_gt += (v_true / omega) * (math.sin(th + omega * dt) - math.sin(th))
        y_gt += (v_true / omega) * (-math.cos(th + omega * dt) + math.cos(th))
        th += omega * dt
        flt.predict(dt)
        st = flt.update(np.array([x_gt, y_gt]), r_k, timestamp=k * dt)
        assert np.all(np.isfinite(st.x))
        assert -math.pi - 1e-9 <= st.theta_rad <= math.pi + 1e-9
        assert np.all(np.linalg.eigvalsh(st.P) > 0.0)
