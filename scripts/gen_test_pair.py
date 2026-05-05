#!/usr/bin/env python3
"""Generate a synthetic image pair with a known affine transform.

Takes any grayscale or RGB image, applies a random (or user-specified) rigid
affine (translation + rotation), and writes:
  images/<stem>_fixed.png   — the original (fixed)
  images/<stem>_moving.png  — the warped version (moving)
  images/<stem>.mat         — STAR-Bench-compatible ground-truth .mat

The .mat can be loaded by the ETNA demo and the landmark RMSE will be
computed automatically.

Usage
-----
    # Random transform on a single image
    python scripts/gen_test_pair.py path/to/source.png

    # Specify translation (px) and rotation (degrees)
    python scripts/gen_test_pair.py path/to/source.png \\
        --tx 12 --ty -8 --rot 3.5 --name MyPair

    # Generate a batch of N random pairs
    python scripts/gen_test_pair.py path/to/source.png --batch 5

    # Use the generated pairs directly in the ETNA demo:
    #   streamlit run app.py
    #   => the pairs appear in the "Image pair" dropdown automatically
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import scipy.io

# ── paths ────────────────────────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
_IMAGES = _REPO / "images"
_IMAGES.mkdir(exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_affine(tx: float, ty: float, rot_deg: float,
                 cx: float, cy: float) -> np.ndarray:
    """2×3 affine matrix for rotation about the image centre + translation."""
    theta = np.deg2rad(rot_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate about (cx, cy):
    #   [cos  sin  tx + cx(1-cos) - cy·sin]
    #   [-sin cos  ty + cy(1-cos) + cx·sin]
    M = np.array([
        [cos_t,  sin_t, tx + cx * (1 - cos_t) - cy * sin_t],
        [-sin_t, cos_t, ty + cy * (1 - cos_t) + cx * sin_t],
    ], dtype=np.float64)
    return M


def _random_transform(max_tx: float = 20, max_ty: float = 20,
                      max_rot: float = 8) -> tuple[float, float, float]:
    tx  = random.uniform(-max_tx,  max_tx)
    ty  = random.uniform(-max_ty,  max_ty)
    rot = random.uniform(-max_rot, max_rot)
    return tx, ty, rot


def _scatter_landmarks(h: int, w: int, n: int = 64) -> np.ndarray:
    """Uniform grid of (x, y) probe landmarks inside the image."""
    xs = np.linspace(w * 0.1, w * 0.9, int(np.ceil(np.sqrt(n))))
    ys = np.linspace(h * 0.1, h * 0.9, int(np.ceil(np.sqrt(n))))
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return pts[:n].astype(np.float32)


def generate_pair(
    source: Path,
    name: str,
    tx: float, ty: float, rot_deg: float,
    ref_size: int = 256,
    n_landmarks: int = 64,
) -> dict:
    """Generate one fixed/moving pair and write files to images/."""
    img = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"Cannot read image: {source}")

    img = cv2.resize(img, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    M = _make_affine(tx, ty, rot_deg, cx, cy)

    warped = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    fixed_path  = _IMAGES / f"{name}_fixed.png"
    moving_path = _IMAGES / f"{name}_moving.png"
    mat_path    = _IMAGES / f"{name}.mat"

    cv2.imwrite(str(fixed_path),  img)
    cv2.imwrite(str(moving_path), warped)

    # Ground-truth landmarks in the STAR-Bench convention:
    #   lm_fix  = original positions
    #   lm_mov  = positions after applying M (i.e. T_gt @ lm_fix)
    #   T_gt    = the affine that takes fixed→moving  (M itself)
    lm_fix = _scatter_landmarks(h, w, n_landmarks)
    ones   = np.ones((lm_fix.shape[0], 1), dtype=np.float32)
    lm_mov = (M[:, :2] @ lm_fix.T + M[:, 2:]).T   # shape (N, 2)

    T_gt_3x3 = np.vstack([M, [0, 0, 1]]).astype(np.float32)

    scipy.io.savemat(str(mat_path), {
        "lm_fix":  lm_fix,
        "lm_mov":  lm_mov,
        "T_gt":    T_gt_3x3,
        "fix_img": img,
        "mov_img": warped,
    })

    print(f"  {name}: tx={tx:+.1f}px  ty={ty:+.1f}px  rot={rot_deg:+.2f}°")
    print(f"    fixed  → {fixed_path.relative_to(_REPO)}")
    print(f"    moving → {moving_path.relative_to(_REPO)}")
    print(f"    GT     → {mat_path.relative_to(_REPO)}")
    return {"tx": tx, "ty": ty, "rot_deg": rot_deg, "name": name}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate a synthetic ETNA test pair from a source image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage")[0].strip(),
    )
    p.add_argument("source", help="Source image (any format OpenCV can read)")
    p.add_argument("--name",  default=None,
                   help="Output pair name (default: source stem)")
    p.add_argument("--tx",   type=float, default=None,
                   help="Translation X in pixels (default: random)")
    p.add_argument("--ty",   type=float, default=None,
                   help="Translation Y in pixels (default: random)")
    p.add_argument("--rot",  type=float, default=None,
                   help="Rotation in degrees (default: random)")
    p.add_argument("--max-tx",  type=float, default=20.0)
    p.add_argument("--max-ty",  type=float, default=20.0)
    p.add_argument("--max-rot", type=float, default=8.0)
    p.add_argument("--size", type=int, default=256,
                   help="Output image size in pixels (square, default 256)")
    p.add_argument("--landmarks", type=int, default=64,
                   help="Number of ground-truth probe landmarks (default 64)")
    p.add_argument("--batch", type=int, default=1,
                   help="Generate N random pairs (ignores --tx/ty/rot; default 1)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed")
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    source = Path(args.source)
    base_name = args.name or source.stem

    if args.batch > 1:
        print(f"Generating {args.batch} random pairs from {source.name} …\n")
        for i in range(args.batch):
            tx, ty, rot = _random_transform(args.max_tx, args.max_ty, args.max_rot)
            name = f"{base_name}_{i:02d}"
            generate_pair(source, name, tx, ty, rot,
                          ref_size=args.size, n_landmarks=args.landmarks)
            print()
    else:
        tx  = args.tx  if args.tx  is not None else _random_transform(args.max_tx,  args.max_ty,  args.max_rot)[0]
        ty  = args.ty  if args.ty  is not None else _random_transform(args.max_tx,  args.max_ty,  args.max_rot)[1]
        rot = args.rot if args.rot is not None else _random_transform(args.max_tx,  args.max_ty,  args.max_rot)[2]
        generate_pair(source, base_name, tx, ty, rot,
                      ref_size=args.size, n_landmarks=args.landmarks)

    print("\nDone. Start Streamlit to register them:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
