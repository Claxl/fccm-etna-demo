# -*- coding: utf-8 -*-
"""Ground-truth .mat loader for the ETNA live demo.

Self-contained extraction of ``starbench_utils.load_ground_truth`` so that
the ``DEMO-FCCM/`` bundle can be shipped on its own without depending on the
full STAR-Bench harness.

The expected MATLAB structure matches the STAR-Bench convention:

- ``I_fix`` — fixed image (informational).
- ``I_move`` — moving image (informational).
- ``Landmarks.I_fix_landmark`` — Nx2 ground-truth points on the fixed image.
- ``Landmarks.I_move_landmark`` — Nx2 corresponding points on the moving image.
- ``T_reg_gt`` — 3x3 homogeneous ground-truth transform.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import scipy.io



def _extract_and_normalize_image(mat_data, image_key):
    if image_key not in mat_data:
        return None
    img = mat_data[image_key]
    if img.dtype != np.uint8:
        mn, mx = np.min(img), np.max(img)
        if mx > mn:
            img = 255 * (img.astype(np.float64) - mn) / (mx - mn)
        img = img.astype(np.uint8)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _extract_landmarks(mat_data) -> Tuple[np.ndarray, np.ndarray]:
    lm = mat_data["Landmarks"]
    # Two flavours seen in the wild: nested struct vs. nested array.
    if lm.shape == (1, 1) and lm[0, 0].dtype.names:
        inner = lm[0, 0]
        fix_raw = inner["I_fix_landmark"][0, 0]
        mov_raw = inner["I_move_landmark"][0, 0]
    elif len(lm[0][0]) == 2:
        fix_raw = lm[0][0][0]
        mov_raw = lm[0][0][1]
    else:
        raise ValueError("Unsupported 'Landmarks' structure in .mat file")
    fix = np.asarray(fix_raw, dtype=np.float32).reshape(-1, 2)
    mov = np.asarray(mov_raw, dtype=np.float32).reshape(-1, 2)
    return fix, mov


def load_ground_truth(mat_file_path: str | Path):
    """Return ``(fix_img, mov_img, lm_fix, lm_mov, T_reg_gt)``.

    Raises ``FileNotFoundError`` / ``IOError`` on failure — callers that want
    graceful degradation should wrap the call in try/except.
    """
    mat_file = Path(mat_file_path)
    if not mat_file.is_file():
        raise FileNotFoundError(f"Ground truth file not found: {mat_file_path}")
    try:
        mat = scipy.io.loadmat(str(mat_file))
        fix_img = _extract_and_normalize_image(mat, "I_fix")
        mov_img = _extract_and_normalize_image(mat, "I_move")
        lm_fix, lm_mov = _extract_landmarks(mat)
        T = mat["T_reg_gt"]
        print("Loaded %d landmark pairs from %s." % (len(lm_fix), mat_file.name))
        return fix_img, mov_img, lm_fix, lm_mov, T
    except Exception as exc:
        print("ERROR: Error parsing GT file '%s': %s" % (mat_file.name, exc))
        raise IOError(f"Failed to load/parse GT from {mat_file.name}") from exc
