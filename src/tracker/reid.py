"""Deep Vehicle Re-ID 512-D ONNX feature extractor (OSNet / ResNet50-IBN) with explicit offline HSV+HOG test fallback."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tracker.config import ReIDConfig


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute bounded cosine distance 1 - cos(a, b) in [0.0, 2.0]."""
    va = np.asarray(a, dtype=np.float64).ravel()
    vb = np.asarray(b, dtype=np.float64).ravel()
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        raise ValueError("Vectors a and b must be non-empty and have identical dimension")
    if not (np.all(np.isfinite(va)) and np.all(np.isfinite(vb))):
        raise ValueError("Vectors a and b must contain finite entries")

    na = float(np.linalg.norm(va))
    nb = float(np.linalg.norm(vb))
    if na <= 1e-12 or nb <= 1e-12:
        return 1.0

    sim = float(np.dot(va, vb) / (na * nb))
    sim = max(-1.0, min(1.0, sim))
    dist = 1.0 - sim
    return float(max(0.0, min(2.0, dist)))


def _l2_normalize_512(vec: np.ndarray, target_dim: int = 512) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32).ravel()
    if v.size < target_dim:
        padded = np.zeros(target_dim, dtype=np.float32)
        padded[: v.size] = v
        v = padded
    elif v.size > target_dim:
        v = v[:target_dim].copy()

    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(v))
    if norm > 1e-12:
        v = (v / norm).astype(np.float32)
    else:
        v = np.zeros(target_dim, dtype=np.float32)
        v[0] = 1.0
    return v


def _extract_hsv_hog_512(crop_bgr: np.ndarray) -> np.ndarray:
    """Deterministic 512-D 3-strip HSV + spatial gradient descriptor for explicit unit-test fallback."""
    resized = cv2.resize(crop_bgr, (64, 128), interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 3 horizontal strips (top, middle, bottom)
    strips = [hsv[:42, :], hsv[42:86, :], hsv[86:, :]]
    feats: list[np.ndarray] = []
    for st in strips:
        h_hist = cv2.calcHist([st], [0], None, [32], [0, 180]).ravel()
        s_hist = cv2.calcHist([st], [1], None, [16], [0, 256]).ravel()
        v_hist = cv2.calcHist([st], [2], None, [16], [0, 256]).ravel()
        b_hist = cv2.calcHist([resized], [0], None, [16], [0, 256]).ravel()
        g_hist = cv2.calcHist([resized], [1], None, [16], [0, 256]).ravel()
        r_hist = cv2.calcHist([resized], [2], None, [16], [0, 256]).ravel()
        feats.extend([h_hist, s_hist, v_hist, b_hist, g_hist, r_hist])

    # Spatial gradient magnitudes on 8x8 pooled grid (64 dims) + mean RGB per 4x4 cell (48 dims)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag_grid = cv2.resize(mag, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    rgb_grid = cv2.resize(resized.astype(np.float32) / 255.0, (4, 4), interpolation=cv2.INTER_AREA).ravel()
    feats.extend([mag_grid, rgb_grid])

    raw_vec = np.concatenate(feats).astype(np.float32)
    return _l2_normalize_512(raw_vec, target_dim=512)


class ReIDExtractor:
    """512-D L2-normalized vehicle appearance feature extractor backed by ONNX Runtime."""

    def __init__(self, cfg: ReIDConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ReIDConfig()
        self.target_dim: int = int(getattr(self.cfg, "embedding_dim", 512))
        self._use_hsv_fallback: bool = False
        self._session: Any | None = None
        self._input_name: str = "input"

        onnx_path = Path(self.cfg.onnx_path)
        is_explicit_hsv = str(self.cfg.onnx_path).strip().lower() == "hsv_hog_test"

        if is_explicit_hsv:
            if not self.cfg.allow_hsv_fallback:
                raise FileNotFoundError(
                    "hsv_hog_test requested while allow_hsv_fallback=False"
                )
            self._use_hsv_fallback = True
        elif onnx_path.is_file():
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.cfg.device.lower() in ("cuda", "gpu"):
                avail = ort.get_available_providers()
                if "CUDAExecutionProvider" in avail:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(onnx_path), providers=providers)
            inputs = self._session.get_inputs()
            if inputs:
                self._input_name = inputs[0].name
        elif self.cfg.allow_hsv_fallback:
            self._use_hsv_fallback = True
        else:
            raise FileNotFoundError(
                f"Deep Re-ID ONNX weights not found at {onnx_path} (and allow_hsv_fallback=False)"
            )

    def _validate_crop(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray):
            raise ValueError("crop_bgr must be a numpy ndarray")
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError(f"crop_bgr must have shape (H, W, 3), got {crop_bgr.shape}")
        if crop_bgr.shape[0] < 1 or crop_bgr.shape[1] < 1 or crop_bgr.size == 0:
            raise ValueError("crop_bgr must have non-zero height and width")
        return crop_bgr

    def _preprocess_onnx(self, crop_bgr: np.ndarray) -> np.ndarray:
        in_h, in_w = int(self.cfg.input_size[0]), int(self.cfg.input_size[1])
        resized = cv2.resize(crop_bgr, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
        norm = (rgb - mean) / std
        chw = np.transpose(norm, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Extract a single (512,) float32 L2-normalized embedding from a vehicle BGR crop."""
        valid = self._validate_crop(crop_bgr)
        if self._use_hsv_fallback or self._session is None:
            return _extract_hsv_hog_512(valid)

        tensor = self._preprocess_onnx(valid)
        outputs = self._session.run(None, {self._input_name: tensor})
        raw_out = np.asarray(outputs[0], dtype=np.float32).ravel()
        return _l2_normalize_512(raw_out, target_dim=self.target_dim)

    def extract_batch(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract an (N, 512) float32 L2-normalized embedding matrix for a list of crops."""
        if not crops:
            return np.empty((0, self.target_dim), dtype=np.float32)
        rows = [self.extract(c) for c in crops]
        return np.vstack(rows).astype(np.float32)

    @staticmethod
    def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
        return cosine_distance(a, b)
