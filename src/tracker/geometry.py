"""Metric ground-plane geometry, normalized DLT/RANSAC homography, 3D projection, and covariance propagation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pyproj import Transformer


def _check_non_collinear(pts: np.ndarray, name: str = "points") -> None:
    """Raise ValueError if 2D points are degenerate or nearly collinear."""
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2)")
    if len(pts) < 4:
        raise ValueError(f"At least 4 point correspondences are required (got {len(pts)})")
    if not np.all(np.isfinite(pts)):
        raise ValueError(f"{name} contains non-finite coordinates")

    centered = pts - np.mean(pts, axis=0, keepdims=True)
    _, s_vals, _ = np.linalg.svd(centered, full_matrices=False)
    if s_vals[0] <= 1e-9 or (len(s_vals) > 1 and s_vals[1] <= 1e-5 * s_vals[0]):
        raise ValueError(f"{name} are collinear or rank-deficient")

    # Also verify no 3 of the first 4 (or all) form zero-area triangles only
    n = len(pts)
    max_twice_area = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area2 = abs(
                    (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1])
                    - (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
                )
                if area2 > max_twice_area:
                    max_twice_area = area2
    if max_twice_area <= 1e-6:
        raise ValueError(f"{name} are collinear")


def _normalized_dlt(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Compute 3x3 planar homography H via Hartley-normalized DLT."""
    _check_non_collinear(src, "src_pts")
    _check_non_collinear(dst, "dst_pts")

    def _similarity_norm(pts: np.ndarray) -> np.ndarray:
        mean = np.mean(pts, axis=0)
        diffs = pts - mean
        rms = float(np.sqrt(np.mean(np.sum(diffs * diffs, axis=1))))
        if rms <= 1e-12:
            raise ValueError("Degenerate coincident points in homography estimation")
        scale = math.sqrt(2.0) / rms
        return np.array(
            [
                [scale, 0.0, -scale * float(mean[0])],
                [0.0, scale, -scale * float(mean[1])],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    t_src = _similarity_norm(src)
    t_dst = _similarity_norm(dst)

    n = len(src)
    src_h = np.column_stack([src, np.ones(n, dtype=np.float64)])
    dst_h = np.column_stack([dst, np.ones(n, dtype=np.float64)])
    src_n = (t_src @ src_h.T).T
    dst_n = (t_dst @ dst_h.T).T

    a_rows: list[list[float]] = []
    for (u, v, _), (x, y, _) in zip(src_n, dst_n):
        a_rows.append([0.0, 0.0, 0.0, -u, -v, -1.0, y * u, y * v, y])
        a_rows.append([u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v, -x])

    a_mat = np.asarray(a_rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(a_mat)
    h_tilde = vt[-1].reshape(3, 3)
    h_mat = np.linalg.inv(t_dst) @ h_tilde @ t_src

    if abs(np.linalg.det(h_mat)) <= 1e-15:
        raise ValueError("Computed homography is singular")
    if abs(h_mat[2, 2]) > 1e-12:
        h_mat = h_mat / h_mat[2, 2]
    else:
        h_mat = h_mat / np.linalg.norm(h_mat)
    return h_mat.astype(np.float64)


def _project_homography_array(h_mat: np.ndarray, pts_uv: np.ndarray) -> np.ndarray:
    pts_h = np.column_stack([pts_uv, np.ones(len(pts_uv), dtype=np.float64)])
    uvw = (h_mat @ pts_h.T).T
    w = uvw[:, 2:3]
    if np.any(np.abs(w) < 1e-9):
        raise ValueError("Point lies on or near the camera horizon (|w| < 1e-9)")
    return uvw[:, :2] / w


def compute_homography_dlt(
    src_pts: np.ndarray | list[tuple[float, float]] | list[list[float]],
    dst_pts: np.ndarray | list[tuple[float, float]] | list[list[float]],
    use_ransac: bool = True,
    ransac_thresh: float = 2.0,
) -> tuple[np.ndarray, float]:
    """Estimate planar homography H (3x3 float64) and destination RMSE in meters.

    Raises ValueError if fewer than 4 points are given or if points are collinear.
    """
    src = np.asarray(src_pts, dtype=np.float64)
    dst = np.asarray(dst_pts, dtype=np.float64)
    if src.ndim != 2 or dst.ndim != 2 or src.shape[1] != 2 or dst.shape[1] != 2:
        raise ValueError("src_pts and dst_pts must be (N, 2) arrays")
    if len(src) != len(dst):
        raise ValueError("src_pts and dst_pts must have the same length")
    _check_non_collinear(src, "src_pts")
    _check_non_collinear(dst, "dst_pts")

    n = len(src)
    inlier_mask = np.ones(n, dtype=bool)

    if use_ransac and n > 4:
        best_inliers: np.ndarray | None = None
        best_count = 0
        best_err_sum = float("inf")

        # Try OpenCV RANSAC first as an initial candidate
        try:
            h_cv, mask_cv = cv2.findHomography(
                src.astype(np.float64),
                dst.astype(np.float64),
                method=cv2.RANSAC,
                ransacReprojThreshold=float(ransac_thresh),
            )
            if h_cv is not None and mask_cv is not None:
                cand_mask = mask_cv.ravel().astype(bool)
                if int(np.sum(cand_mask)) >= 4:
                    proj = _project_homography_array(h_cv, src)
                    errs = np.linalg.norm(proj - dst, axis=1)
                    cand_mask = errs <= float(ransac_thresh)
                    cnt = int(np.sum(cand_mask))
                    if cnt >= 4:
                        best_inliers = cand_mask
                        best_count = cnt
                        best_err_sum = float(np.sum(errs[cand_mask] ** 2))
        except Exception:
            pass

        # Deterministic combinatorial / seeded RANSAC over 4-point subsets
        from itertools import combinations

        all_combos = list(combinations(range(n), 4))
        if len(all_combos) > 250:
            rng = np.random.default_rng(42)
            idxs = rng.choice(len(all_combos), size=250, replace=False)
            combos_to_test = [all_combos[int(i)] for i in idxs]
        else:
            combos_to_test = all_combos

        for subset in combos_to_test:
            idx = list(subset)
            try:
                h_sub = _normalized_dlt(src[idx], dst[idx])
                proj = _project_homography_array(h_sub, src)
            except ValueError:
                continue
            errs = np.linalg.norm(proj - dst, axis=1)
            cand_mask = errs <= float(ransac_thresh)
            cnt = int(np.sum(cand_mask))
            if cnt < 4:
                continue
            err_sum = float(np.sum(errs[cand_mask] ** 2))
            if cnt > best_count or (cnt == best_count and err_sum < best_err_sum):
                try:
                    _check_non_collinear(src[cand_mask], "inlier_src")
                    _check_non_collinear(dst[cand_mask], "inlier_dst")
                except ValueError:
                    continue
                best_count = cnt
                best_err_sum = err_sum
                best_inliers = cand_mask

        if best_inliers is not None and int(np.sum(best_inliers)) >= 4:
            inlier_mask = best_inliers

    h_final = _normalized_dlt(src[inlier_mask], dst[inlier_mask])
    proj_inliers = _project_homography_array(h_final, src[inlier_mask])
    sq_errs = np.sum((proj_inliers - dst[inlier_mask]) ** 2, axis=1)
    rmse = float(np.sqrt(np.mean(sq_errs)))
    return h_final, rmse


def wgs84_to_utm(
    lat: float,
    lon: float,
    epsg_utm: int | None = None,
) -> tuple[float, float, int]:
    """Convert WGS84 (lat, lon) in degrees to metric UTM (x_east, y_north, epsg_utm)."""
    lat_f = float(lat)
    lon_f = float(lon)
    if not (math.isfinite(lat_f) and math.isfinite(lon_f)):
        raise ValueError("lat and lon must be finite")
    if not (-90.0 <= lat_f <= 90.0):
        raise ValueError(f"Latitude out of bounds [-90, 90]: {lat_f}")
    if not (-180.0 <= lon_f <= 180.0):
        raise ValueError(f"Longitude out of bounds [-180, 180]: {lon_f}")

    if epsg_utm is None:
        # Handle antimeridian lon == 180.0 clamping to zone 60
        zone = int(math.floor((lon_f + 180.0) / 6.0)) + 1
        zone = min(60, max(1, zone))
        base = 32600 if lat_f >= 0.0 else 32700
        epsg = base + zone
    else:
        epsg = int(epsg_utm)

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x_utm, y_utm = transformer.transform(lon_f, lat_f)
    if not (math.isfinite(x_utm) and math.isfinite(y_utm)):
        raise ValueError("Projected UTM coordinates are non-finite")
    return float(x_utm), float(y_utm), int(epsg)


def utm_to_wgs84(x: float, y: float, epsg_utm: int) -> tuple[float, float]:
    """Convert metric UTM (x_east, y_north, epsg_utm) back to WGS84 (lat, lon)."""
    xf, yf = float(x), float(y)
    if not (math.isfinite(xf) and math.isfinite(yf)):
        raise ValueError("UTM coordinates must be finite")
    epsg = int(epsg_utm)
    transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(xf, yf)
    return float(lat), float(lon)


def _enforce_spd_2x2(mat: np.ndarray, min_eig: float = 1e-4) -> np.ndarray:
    """Symmetrize and clamp eigenvalues of a 2x2 covariance matrix to guarantee SPD."""
    sym = 0.5 * (np.asarray(mat, dtype=np.float64) + np.asarray(mat, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(sym)
    vals_clamped = np.maximum(vals, float(min_eig))
    spd = (vecs * vals_clamped) @ vecs.T
    return 0.5 * (spd + spd.T)


class CameraProjector:
    """Bidirectional pixel <-> ground plane (UTM Z=0) projector with covariance propagation."""

    def __init__(
        self,
        *,
        homography: np.ndarray | None = None,
        K: np.ndarray | None = None,
        R: np.ndarray | None = None,
        t: np.ndarray | None = None,
        epsg_utm: int | None = None,
    ) -> None:
        self.epsg_utm = int(epsg_utm) if epsg_utm is not None else None
        self._mode: str
        self.H: np.ndarray | None = None
        self.H_inv: np.ndarray | None = None
        self.K: np.ndarray | None = None
        self.K_inv: np.ndarray | None = None
        self.R: np.ndarray | None = None
        self.t: np.ndarray | None = None
        self.cam_center_world: np.ndarray | None = None

        if K is not None and R is not None and t is not None:
            k_arr = np.asarray(K, dtype=np.float64)
            r_arr = np.asarray(R, dtype=np.float64)
            t_arr = np.asarray(t, dtype=np.float64).reshape(3)
            if k_arr.shape != (3, 3) or r_arr.shape != (3, 3) or t_arr.shape != (3,):
                raise ValueError("K (3x3), R (3x3), and t (3,) shapes are invalid")
            if not (np.all(np.isfinite(k_arr)) and np.all(np.isfinite(r_arr)) and np.all(np.isfinite(t_arr))):
                raise ValueError("3D calibration parameters must be finite")
            self._mode = "3d"
            self.K = k_arr
            self.K_inv = np.linalg.inv(k_arr)
            self.R = r_arr
            self.t = t_arr
            self.cam_center_world = -np.linalg.solve(r_arr, t_arr)
            # Induced planar homography ground->pixel: P_z0 = K @ [r1, r2, t]
            h_g2p = k_arr @ np.column_stack([r_arr[:, 0], r_arr[:, 1], t_arr])
            self.H_inv = h_g2p
            if abs(np.linalg.det(h_g2p)) > 1e-12:
                h_p2g = np.linalg.inv(h_g2p)
                if abs(h_p2g[2, 2]) > 1e-12:
                    h_p2g = h_p2g / h_p2g[2, 2]
                self.H = h_p2g
        elif homography is not None:
            h_arr = np.asarray(homography, dtype=np.float64)
            if h_arr.shape != (3, 3) or not np.all(np.isfinite(h_arr)):
                raise ValueError("homography must be a finite 3x3 matrix")
            if abs(np.linalg.det(h_arr)) <= 1e-12:
                raise ValueError("homography matrix must be non-singular")
            self._mode = "homography"
            self.H = h_arr.copy()
            self.H_inv = np.linalg.inv(h_arr)
        else:
            raise ValueError("Either homography or (K, R, t) must be provided")

    @classmethod
    def from_homography(
        cls,
        H: np.ndarray | list[list[float]],
        epsg_utm: int | None = None,
    ) -> CameraProjector:
        return cls(homography=np.asarray(H, dtype=np.float64), epsg_utm=epsg_utm)

    @classmethod
    def from_3d_params(
        cls,
        K: np.ndarray | list[list[float]],
        R: np.ndarray | list[list[float]],
        t: np.ndarray | list[float],
        epsg_utm: int | None = None,
    ) -> CameraProjector:
        return cls(
            K=np.asarray(K, dtype=np.float64),
            R=np.asarray(R, dtype=np.float64),
            t=np.asarray(t, dtype=np.float64),
            epsg_utm=epsg_utm,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> CameraProjector:
        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Camera calibration YAML not found: {yaml_path}")
        raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        epsg = raw.get("epsg_utm")
        if raw.get("K") is not None and raw.get("R") is not None and raw.get("t") is not None:
            return cls.from_3d_params(raw["K"], raw["R"], raw["t"], epsg_utm=epsg)
        if raw.get("homography") is not None:
            return cls.from_homography(raw["homography"], epsg_utm=epsg)
        raise ValueError(f"Calibration YAML {yaml_path} missing 'homography' or ('K', 'R', 't')")

    def bbox_contact_point(
        self,
        bbox_xyxy: tuple[float, float, float, float] | list[float] | np.ndarray,
        use_perspective_correction: bool = True,
        fraction: float = 0.90,
    ) -> tuple[float, float]:
        """Compute ground contact pixel (u_c, v_c) from a 2D bounding box (x1, y1, x2, y2)."""
        arr = np.asarray(bbox_xyxy, dtype=np.float64).ravel()
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            raise ValueError("bbox_xyxy must contain 4 finite numbers")
        x1, y1, x2, y2 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        u_c = 0.5 * (x1 + x2)
        if use_perspective_correction:
            frac = float(fraction)
            v_c = y1 + frac * (y2 - y1)
        else:
            v_c = y2
        return float(u_c), float(v_c)

    def pixel_to_ground(self, u: float, v: float) -> tuple[float, float]:
        """Project image pixel (u, v) to ground plane Z=0 in UTM meters (x_utm, y_utm)."""
        uf, vf = float(u), float(v)
        if not (math.isfinite(uf) and math.isfinite(vf)):
            raise ValueError("Pixel coordinates (u, v) must be finite")

        if self._mode == "3d":
            assert self.K_inv is not None and self.R is not None and self.cam_center_world is not None
            d_cam = self.K_inv @ np.array([uf, vf, 1.0], dtype=np.float64)
            d_world = np.linalg.solve(self.R, d_cam)
            dz = float(d_world[2])
            if abs(dz) < 1e-9:
                raise ValueError("Ray is parallel to ground plane Z=0 (horizon singularity)")
            s = -float(self.cam_center_world[2]) / dz
            if not math.isfinite(s) or s <= 0.0:
                raise ValueError("Pixel lies above the horizon (ray does not hit ground in front of camera)")
            x_g = float(self.cam_center_world[0] + s * d_world[0])
            y_g = float(self.cam_center_world[1] + s * d_world[1])
            return x_g, y_g

        assert self.H is not None
        vec = self.H @ np.array([uf, vf, 1.0], dtype=np.float64)
        w = float(vec[2])
        if abs(w) < 1e-9:
            raise ValueError("Pixel lies on the homography horizon singularity (|w| < 1e-9)")
        return float(vec[0] / w), float(vec[1] / w)

    def ground_to_pixel(self, x_utm: float, y_utm: float) -> tuple[float, float]:
        """Project ground plane coordinate (x_utm, y_utm, Z=0) back to pixel (u, v)."""
        xf, yf = float(x_utm), float(y_utm)
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise ValueError("Ground coordinates must be finite")
        assert self.H_inv is not None
        uvw = self.H_inv @ np.array([xf, yf, 1.0], dtype=np.float64)
        w = float(uvw[2])
        if abs(w) < 1e-9:
            raise ValueError("Ground point projects to camera center singularity (|w| < 1e-9)")
        return float(uvw[0] / w), float(uvw[1] / w)

    def _jacobian_at_pixel(self, u: float, v: float) -> np.ndarray:
        """Compute 2x2 analytical Jacobian d(x_utm, y_utm)/d(u, v) at pixel (u, v)."""
        if self.H is not None:
            h = self.H
            num_x = h[0, 0] * u + h[0, 1] * v + h[0, 2]
            num_y = h[1, 0] * u + h[1, 1] * v + h[1, 2]
            w = h[2, 0] * u + h[2, 1] * v + h[2, 2]
            if abs(w) < 1e-9:
                raise ValueError("Jacobian evaluated at horizon singularity")
            w2 = w * w
            dx_du = (h[0, 0] * w - num_x * h[2, 0]) / w2
            dx_dv = (h[0, 1] * w - num_x * h[2, 1]) / w2
            dy_du = (h[1, 0] * w - num_y * h[2, 0]) / w2
            dy_dv = (h[1, 1] * w - num_y * h[2, 1]) / w2
            return np.array([[dx_du, dx_dv], [dy_du, dy_dv]], dtype=np.float64)

        eps = 1e-3
        x_pu, y_pu = self.pixel_to_ground(u + eps, v)
        x_mu, y_mu = self.pixel_to_ground(u - eps, v)
        x_pv, y_pv = self.pixel_to_ground(u, v + eps)
        x_mv, y_mv = self.pixel_to_ground(u, v - eps)
        return np.array(
            [
                [(x_pu - x_mu) / (2.0 * eps), (x_pv - x_mv) / (2.0 * eps)],
                [(y_pu - y_mu) / (2.0 * eps), (y_pv - y_mv) / (2.0 * eps)],
            ],
            dtype=np.float64,
        )

    def bbox_to_ground(
        self,
        bbox_xyxy: tuple[float, float, float, float] | list[float] | np.ndarray,
        sigma_px: tuple[float, float] = (4.0, 4.0),
        use_perspective_correction: bool = True,
        fraction: float = 0.90,
    ) -> tuple[float, float, np.ndarray]:
        """Project 2D bounding box to (x_utm, y_utm, R_k) with 2x2 SPD covariance R_k."""
        u_c, v_c = self.bbox_contact_point(
            bbox_xyxy,
            use_perspective_correction=use_perspective_correction,
            fraction=fraction,
        )
        x_utm, y_utm = self.pixel_to_ground(u_c, v_c)
        j_mat = self._jacobian_at_pixel(u_c, v_c)
        su = max(float(sigma_px[0]), 0.0)
        sv = max(float(sigma_px[1]), 0.0)
        sigma_mat = np.diag([su * su, sv * sv])
        r_raw = j_mat @ sigma_mat @ j_mat.T
        r_k = _enforce_spd_2x2(r_raw, min_eig=1e-4)
        return x_utm, y_utm, r_k

    def bbox_dimensions_ground(
        self,
        bbox_xyxy: tuple[float, float, float, float] | list[float] | np.ndarray,
    ) -> tuple[float, float]:
        """Estimate metric ground (W_m, L_m) footprint of a bounding box for scale calibration."""
        arr = np.asarray(bbox_xyxy, dtype=np.float64).ravel()
        if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
            raise ValueError("bbox_xyxy must contain 4 finite numbers")
        x1, y1, x2, y2 = float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])
        w_px = max(abs(x2 - x1), 1.0)
        h_px = max(abs(y2 - y1), 1.0)
        u_c, v_c = self.bbox_contact_point(bbox_xyxy, use_perspective_correction=True)
        j_mat = self._jacobian_at_pixel(u_c, v_c)
        col_u = float(np.linalg.norm(j_mat[:, 0]))
        col_v = float(np.linalg.norm(j_mat[:, 1]))
        w_m = max(col_u * w_px, 1e-3)
        l_m = max(col_v * h_px, 1e-3)
        return float(w_m), float(l_m)
