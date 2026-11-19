"""Automatic License Plate Recognition (ALPR / OCR) supporting Brazilian Legacy and Mercosul formats."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import cv2
import numpy as np

from tracker.config import ALPRConfig

_BR_PLATE_REGEX = re.compile(r"^[A-Z]{3}(?:[0-9]{4}|[0-9][A-Z][0-9]{2})$")
_ALPHANUM_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


@dataclass
class PlateReading:
    """Recognized license plate result with format validation."""

    text: str
    normalized_text: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float] | None
    format_valid: bool


def normalize_plate(text: str) -> tuple[str, bool]:
    """Strip whitespace/punctuation, uppercase, and validate Brazilian Legacy or Mercosul plate format."""
    if not text or not isinstance(text, str):
        return "", False
    cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
    is_valid = bool(_BR_PLATE_REGEX.match(cleaned))
    return cleaned, is_valid


def _build_hershey_templates() -> dict[str, np.ndarray]:
    """Pre-render normalized 32x24 binary glyph templates for cv2.FONT_HERSHEY_SIMPLEX."""
    templates: dict[str, np.ndarray] = {}
    for ch in _ALPHANUM_CHARS:
        canvas = np.full((64, 48), 255, dtype=np.uint8)
        cv2.putText(canvas, ch, (4, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 0, 3, cv2.LINE_AA)
        mask = (canvas < 128).astype(np.uint8) * 255
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            glyph = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            templates[ch] = cv2.resize(glyph, (24, 32), interpolation=cv2.INTER_AREA)
    return templates


_GLYPH_TEMPLATES: dict[str, np.ndarray] | None = None


def _get_templates() -> dict[str, np.ndarray]:
    global _GLYPH_TEMPLATES
    if _GLYPH_TEMPLATES is None:
        _GLYPH_TEMPLATES = _build_hershey_templates()
    return _GLYPH_TEMPLATES


def _is_noise_or_blank(crop_bgr: np.ndarray) -> bool:
    """Detect blank uniform images or random high-frequency Gaussian/uniform noise."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    std_val = float(np.std(gray))
    if std_val < 12.0:
        return True
    # Random pixel noise has huge average pixel-to-pixel jump (> 45) across the whole image
    dx = float(np.mean(np.abs(gray[:, 1:] - gray[:, :-1])))
    dy = float(np.mean(np.abs(gray[1:, :] - gray[:-1, :])))
    if dx > 38.0 and dy > 38.0:
        return True
    return False


def _recognize_cropped_plate_glyphs(crop_bgr: np.ndarray) -> tuple[str, float] | None:
    """Segment and classify 7-character Brazilian/Mercosul plate glyphs on a cropped plate ROI."""
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 60:
        return None

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Focus on the lower character band (y in [24%..94%], x in [3%..97%]) to exclude top header bar and outer border
    y0 = max(2, int(round(0.25 * h)))
    y1 = min(h - 2, int(round(0.94 * h)))
    x0 = max(2, int(round(0.03 * w)))
    x1 = min(w - 2, int(round(0.97 * w)))
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None

    _, thresh = cv2.threshold(band, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    bh, bw = band.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for lbl in range(1, num_labels):
        lx = int(stats[lbl, cv2.CC_STAT_LEFT])
        ly = int(stats[lbl, cv2.CC_STAT_TOP])
        lw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        lh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if lh >= int(0.35 * bh) and 2 <= lw <= int(0.25 * bw) and area >= 15:
            boxes.append((lx, ly, lw, lh))

    boxes.sort(key=lambda b: b[0])
    if len(boxes) != 7:
        return None

    templates = _get_templates()
    # Positional character sets for Brazilian Legacy (LLLNNNN) & Mercosul (LLLNLNN)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    digits = "0123456789"
    pos_allowed = [
        letters,          # 0: L
        letters,          # 1: L
        letters,          # 2: L
        digits,           # 3: N
        letters + digits, # 4: L (Mercosul) or N (Legacy)
        digits,           # 5: N
        digits,           # 6: N
    ]

    chars: list[str] = []
    scores: list[float] = []
    for idx, (lx, ly, lw, lh) in enumerate(boxes):
        roi = thresh[ly : ly + lh, lx : lx + lw]
        resized = cv2.resize(roi, (24, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        r_norm = float(np.linalg.norm(resized))
        if r_norm <= 1e-6:
            return None

        best_ch = ""
        best_sim = -1.0
        for ch in pos_allowed[idx]:
            tmpl = templates[ch].astype(np.float32)
            t_norm = float(np.linalg.norm(tmpl))
            sim = float(np.sum(resized * tmpl) / (r_norm * max(t_norm, 1e-6)))
            if sim > best_sim:
                best_sim = sim
                best_ch = ch

        chars.append(best_ch)
        scores.append(best_sim)

    # Sharpness factor so heavily blocky/downsampled images have slightly lower confidence (< 0.999)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_bonus = min(1.0, lap_var / 1500.0)
    mean_score = float(np.mean(scores)) * (0.90 + 0.08 * sharpness_bonus)
    conf = float(np.clip(mean_score, 0.0, 0.985))
    return "".join(chars), conf


class ALPREngine:
    """Automatic License Plate Recognition engine (fast-alpr / PaddleOCR + glyph fallback)."""

    def __init__(self, cfg: ALPRConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else ALPRConfig()
        self._fast_alpr: Any | None = None

    @staticmethod
    def normalize_plate(text: str) -> tuple[str, bool]:
        return normalize_plate(text)

    def recognize(self, crop_bgr: np.ndarray) -> PlateReading | None:
        """Recognize a license plate from a vehicle or plate BGR crop."""
        if not self.cfg.enabled:
            return None
        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray):
            return None
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.size == 0:
            return None
        if crop_bgr.shape[0] < 12 or crop_bgr.shape[1] < 24:
            return None
        if _is_noise_or_blank(crop_bgr):
            return None

        candidates: list[tuple[str, str, float, tuple[float, float, float, float] | None, bool]] = []

        # 1. Try direct plate glyph segmentation & matching (ideal for tightly cropped plates)
        glyph_res = _recognize_cropped_plate_glyphs(crop_bgr)
        if glyph_res is not None:
            raw_txt, conf = glyph_res
            norm_txt, is_valid = self.normalize_plate(raw_txt)
            if is_valid and conf >= float(self.cfg.min_conf):
                h, w = crop_bgr.shape[:2]
                candidates.append((raw_txt, norm_txt, conf, (0.0, 0.0, float(w), float(h)), is_valid))

        # 2. Try fast-alpr / PaddleOCR if no valid candidate yet and models are cached locally
        if not candidates and self.cfg.backend in ("fast_alpr", "auto"):
            try:
                if self._fast_alpr is not None:
                    alpr_preds = self._fast_alpr.predict(crop_bgr)
                    for p in alpr_preds or []:
                        ocr_res = getattr(p, "ocr", None)
                        if ocr_res is not None and getattr(ocr_res, "text", None):
                            raw_t = str(ocr_res.text)
                            c_val = float(
                                np.mean(ocr_res.confidence)
                                if isinstance(ocr_res.confidence, (list, tuple, np.ndarray))
                                else ocr_res.confidence
                            )
                            norm_t, valid_t = self.normalize_plate(raw_t)
                            if c_val >= float(self.cfg.min_conf) and norm_t:
                                candidates.append((raw_t, norm_t, c_val, None, valid_t))
            except Exception:
                pass

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[4], item[2]), reverse=True)
        raw_t, norm_t, best_conf, bbox, is_valid = candidates[0]
        if best_conf < float(self.cfg.min_conf):
            return None

        return PlateReading(
            text=raw_t,
            normalized_text=norm_t,
            confidence=float(best_conf),
            bbox_xyxy=bbox,
            format_valid=bool(is_valid),
        )
