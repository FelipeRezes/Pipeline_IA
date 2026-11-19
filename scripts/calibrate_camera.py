"""Interactive and headless >=4-point image <-> UTM planar homography calibration CLI tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
import yaml

from tracker.geometry import compute_homography_dlt


def _load_points_json(points_path: Path) -> tuple[np.ndarray, np.ndarray, int | None]:
    if not points_path.is_file():
        raise FileNotFoundError(f"Points file not found: {points_path}")
    raw: Any = json.loads(points_path.read_text(encoding="utf-8"))

    epsg_utm: int | None = None
    if isinstance(raw, dict):
        if raw.get("epsg_utm") is not None:
            epsg_utm = int(raw["epsg_utm"])
        pts_list = raw.get("points", raw.get("correspondences", []))
    elif isinstance(raw, list):
        pts_list = raw
    else:
        raise ValueError(f"Unsupported points JSON format in {points_path}")

    if not isinstance(pts_list, list) or len(pts_list) < 4:
        raise ValueError(f"At least 4 point correspondences required (got {len(pts_list) if isinstance(pts_list, list) else 0})")

    src_uv: list[tuple[float, float]] = []
    dst_utm: list[tuple[float, float]] = []
    for item in pts_list:
        if isinstance(item, dict):
            u = float(item["u"])
            v = float(item["v"])
            x = float(item.get("x_utm", item.get("x")))
            y = float(item.get("y_utm", item.get("y")))
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            u, v, x, y = float(item[0]), float(item[1]), float(item[2]), float(item[3])
        else:
            raise ValueError(f"Invalid point entry: {item!r}")
        src_uv.append((u, v))
        dst_utm.append((x, y))

    return (
        np.asarray(src_uv, dtype=np.float64),
        np.asarray(dst_utm, dtype=np.float64),
        epsg_utm,
    )


def calibrate_from_files(
    image_path: Path | None,
    points_path: Path,
    out_yaml_path: Path,
    *,
    cam_id: str = "calibrated_cam",
    epsg_utm: int | None = None,
    use_ransac: bool = True,
) -> dict[str, Any]:
    """Compute homography H from image & points JSON and write calibration YAML."""
    resolution = [1280, 720]
    if image_path is not None:
        if not image_path.is_file():
            raise FileNotFoundError(f"Reference image not found: {image_path}")
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            resolution = [int(w), int(h)]

    src_uv, dst_utm, json_epsg = _load_points_json(points_path)
    final_epsg = epsg_utm if epsg_utm is not None else (json_epsg if json_epsg is not None else 32723)

    h_mat, rmse = compute_homography_dlt(src_uv, dst_utm, use_ransac=use_ransac)
    payload: dict[str, Any] = {
        "cam_id": str(cam_id),
        "epsg_utm": int(final_epsg),
        "resolution": resolution,
        "homography": h_mat.tolist(),
        "rmse": float(rmse),
        "num_points": int(len(src_uv)),
    }
    out_yaml_path.parent.mkdir(parents=True, exist_ok=True)
    out_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate planar camera homography (image pixels <-> UTM meters)."
    )
    parser.add_argument("--image", type=Path, default=None, help="Reference camera frame image path.")
    parser.add_argument("--points", type=Path, required=True, help="JSON file with >= 4 point correspondences.")
    parser.add_argument("--out", type=Path, required=True, help="Output camera calibration YAML path.")
    parser.add_argument("--cam-id", type=str, default="calibrated_cam", help="Camera identifier.")
    parser.add_argument("--epsg", type=int, default=None, help="Optional UTM EPSG override.")
    parser.add_argument("--no-ransac", action="store_true", help="Disable RANSAC outlier rejection.")
    parser.add_argument("--headless", action="store_true", help="Run non-interactively without GUI windows.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        payload = calibrate_from_files(
            image_path=args.image,
            points_path=args.points,
            out_yaml_path=args.out,
            cam_id=args.cam_id,
            epsg_utm=args.epsg,
            use_ransac=not args.no_ransac,
        )
        print(
            f"Calibrated {payload['cam_id']} with {payload['num_points']} points "
            f"(RMSE={payload['rmse']:.6f} m) -> {args.out}"
        )
        return 0
    except Exception as exc:
        print(f"Calibration error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
