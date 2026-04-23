# -*- coding: utf-8 -*-
"""Lightweight visualization helpers for the ETNA live demo.

All functions consume BGR / grayscale numpy arrays as produced by OpenCV
and return BGR numpy arrays ready to be displayed in Streamlit via
``st.image(..., channels="BGR")``.
"""
from __future__ import annotations

import cv2
import numpy as np


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img is None:
        return None
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) * (255.0 / (hi - lo))
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def load_image(path: str, target_size: int | None = None) -> np.ndarray:
    """Load an image as grayscale uint8. Optionally square-resize to ``target_size``."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(_ensure_bgr(img), cv2.COLOR_BGR2GRAY)
    img = _ensure_uint8(img)
    if target_size is not None and (img.shape[0] != target_size or img.shape[1] != target_size):
        img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return img


def warp_affine(moving: np.ndarray, t_mat_2x3: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray:
    h, w = moving.shape[:2]
    if out_size is None:
        out_size = (w, h)
    return cv2.warpAffine(
        moving, t_mat_2x3.astype(np.float32), out_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def make_fusion(fixed: np.ndarray, warped: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """50/50 fusion tinted red (fixed) + cyan (warped) for contrast."""
    fx = _ensure_uint8(fixed)
    wp = _ensure_uint8(warped)
    if fx.shape != wp.shape:
        wp = cv2.resize(wp, (fx.shape[1], fx.shape[0]), interpolation=cv2.INTER_LINEAR)
    fixed_color = np.zeros((*fx.shape, 3), dtype=np.uint8)
    warped_color = np.zeros((*wp.shape, 3), dtype=np.uint8)
    # BGR: fixed as magenta (B+R), warped as green — gives clear registration error colours.
    fixed_color[..., 0] = fx  # B
    fixed_color[..., 2] = fx  # R
    warped_color[..., 1] = wp  # G
    return cv2.addWeighted(fixed_color, alpha, warped_color, 1.0 - alpha, 0)


def make_checkerboard(fixed: np.ndarray, warped: np.ndarray, block: int = 32) -> np.ndarray:
    fx = _ensure_bgr(_ensure_uint8(fixed))
    wp = _ensure_bgr(_ensure_uint8(warped))
    if fx.shape != wp.shape:
        wp = cv2.resize(wp, (fx.shape[1], fx.shape[0]), interpolation=cv2.INTER_LINEAR)
    h, w = fx.shape[:2]
    out = fx.copy()
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2 == 1:
                out[y:y + block, x:x + block] = wp[y:y + block, x:x + block]
    return out


def make_difference(fixed: np.ndarray, warped: np.ndarray) -> np.ndarray:
    fx = _ensure_uint8(fixed)
    wp = _ensure_uint8(warped)
    if fx.ndim == 3:
        fx = cv2.cvtColor(fx, cv2.COLOR_BGR2GRAY)
    if wp.ndim == 3:
        wp = cv2.cvtColor(wp, cv2.COLOR_BGR2GRAY)
    if fx.shape != wp.shape:
        wp = cv2.resize(wp, (fx.shape[1], fx.shape[0]), interpolation=cv2.INTER_LINEAR)
    diff = cv2.absdiff(fx, wp)
    return cv2.applyColorMap(diff, cv2.COLORMAP_JET)


# OpenCV's Hershey fonts only understand ASCII (0x20..0x7E). Any Unicode
# (em-dash, bullet, arrow, non-breaking space, ...) renders as '?'.  Map a
# handful of common glyphs to their ASCII approximations and drop the rest.
_ASCII_MAP = str.maketrans({
    "—": "-",   # em dash
    "–": "-",   # en dash
    "•": "|",   # bullet
    "→": "->",  # right arrow
    "←": "<-",  # left arrow
    "≤": "<=",  # less-or-equal
    "≥": ">=",  # greater-or-equal
    "×": "x",   # multiplication sign
    " ": " ",   # non-breaking space
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    "°": "deg",
})


def _to_ascii(text: str) -> str:
    """Map common Unicode punctuation to ASCII and drop anything still non-ASCII."""
    translated = str(text).translate(_ASCII_MAP)
    return "".join(ch if 0x20 <= ord(ch) <= 0x7E else "?" for ch in translated)


def annotate(img: np.ndarray, text: str, color=(0, 255, 0)) -> np.ndarray:
    out = _ensure_bgr(img).copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, _to_ascii(text), (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                color, 1, cv2.LINE_AA)
    return out


def draw_landmark_error(fixed: np.ndarray,
                        lm_gt_on_fixed: np.ndarray,
                        lm_predicted_on_fixed: np.ndarray) -> np.ndarray:
    """Overlay GT (green circle) vs predicted (red cross) landmarks
    with cyan error lines joining each pair.
    """
    out = _ensure_bgr(_ensure_uint8(fixed)).copy()
    gt = np.asarray(lm_gt_on_fixed, dtype=np.float32).reshape(-1, 2)
    pr = np.asarray(lm_predicted_on_fixed, dtype=np.float32).reshape(-1, 2)
    n = min(len(gt), len(pr))
    for i in range(n):
        g = (int(round(float(gt[i, 0]))), int(round(float(gt[i, 1]))))
        p = (int(round(float(pr[i, 0]))), int(round(float(pr[i, 1]))))
        cv2.line(out, g, p, (255, 255, 0), 1, cv2.LINE_AA)  # cyan-ish in BGR
        cv2.circle(out, g, 4, (0, 255, 0), 1, cv2.LINE_AA)  # GT — green
        # Predicted — small red tilted cross
        s = 4
        cv2.line(out, (p[0] - s, p[1] - s), (p[0] + s, p[1] + s),
                 (0, 0, 255), 1, cv2.LINE_AA)
        cv2.line(out, (p[0] - s, p[1] + s), (p[0] + s, p[1] - s),
                 (0, 0, 255), 1, cv2.LINE_AA)
    return out


def decompose_affine(t_mat_2x3: np.ndarray) -> dict:
    """Decompose a 2x3 affine matrix into translation, rotation and scale."""
    a, b, tx = float(t_mat_2x3[0, 0]), float(t_mat_2x3[0, 1]), float(t_mat_2x3[0, 2])
    c, d, ty = float(t_mat_2x3[1, 0]), float(t_mat_2x3[1, 1]), float(t_mat_2x3[1, 2])
    sx = float(np.hypot(a, c))
    sy = float(np.hypot(b, d))
    rot = float(np.degrees(np.arctan2(c, a)))
    return {"tx": tx, "ty": ty, "rot_deg": rot, "scale_x": sx, "scale_y": sy}
