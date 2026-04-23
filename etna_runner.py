# -*- coding: utf-8 -*-
"""Instrumented ETNA runner for the live demo.

Wraps ``ETNA.EtnaMultiMetric`` so that each ``compute_metric`` call emits an
event on a thread-safe queue. The event carries the current pyramid level, the
transform being evaluated and the metric value. The Streamlit app drains the
queue in its main thread and refreshes the dashboard in near real-time.

The ETNA optimizer itself runs unmodified on a worker thread; the metric
subclass is the only instrumentation point.
"""
from __future__ import annotations

import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

# ``ETNA`` now lives next to this file (``DEMO-FCCM/ETNA/``) so the demo is
# self-contained. Make sure the directory holding this script is on
# ``sys.path`` regardless of where Python was invoked from — this covers
# both ``streamlit run app.py`` (cwd == DEMO-FCCM) and
# ``python DEMO-FCCM/etna_runner.py`` invocations from the repo root.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from ETNA import EtnaMultiMetric, EtnaMultiPowell, EtnaMultiOnePlusOne  # noqa: E402
from gt_loader import load_ground_truth as _sb_load  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event schema
# ---------------------------------------------------------------------------

@dataclass
class MetricEvent:
    """Emitted for every ``compute_metric`` call inside the optimizer."""
    kind: str = "metric"
    level: int = -1
    level_size: int = 0
    eval_idx: int = 0
    value: float = 0.0
    transform: np.ndarray | None = None
    wall_time: float = 0.0
    # Landmark RMSE against ground truth (if a .mat GT file was supplied).
    rmse_px: float | None = None


@dataclass
class LevelEvent:
    """Emitted when a new pyramid level starts or ends."""
    kind: str = "level"
    phase: str = "start"  # "start" | "end"
    level: int = -1
    level_size: int = 0
    wall_time: float = 0.0


@dataclass
class StatusEvent:
    """General status / logging / completion event."""
    kind: str = "status"
    message: str = ""
    severity: str = "info"  # "info" | "warning" | "error" | "done"
    payload: dict = field(default_factory=dict)
    wall_time: float = 0.0


@dataclass
class RunResult:
    transform: np.ndarray
    fixed: np.ndarray
    moving: np.ndarray
    warped: np.ndarray
    per_level_time: dict
    total_time: float
    backend: str
    eval_count: int
    # Populated when a .mat ground truth is available.
    landmarks_fix_scaled: np.ndarray | None = None
    landmarks_mov_scaled: np.ndarray | None = None
    gt_transform: np.ndarray | None = None
    gt_transform_ref: np.ndarray | None = None   # rescaled to ref_size grid
    initial_rmse_px: float | None = None
    final_rmse_px: float | None = None


# ---------------------------------------------------------------------------
# Instrumented metric
# ---------------------------------------------------------------------------

class InstrumentedMetric(EtnaMultiMetric):
    """Transparent subclass that publishes every metric evaluation."""

    def __init__(self, *args, event_queue: queue.Queue | None = None,
                 start_time: float = 0.0,
                 landmarks_mov_scaled: np.ndarray | None = None,
                 T_gt_ref: np.ndarray | None = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._event_queue = event_queue
        self._start_time = start_time or time.time()
        self._eval_idx = 0
        self._current_level = -1
        self._current_level_size = 0
        self._last_level_size = -1
        # Ground-truth probe landmarks in ``ref_size`` coordinates and the GT
        # affine rescaled to the same grid (both may be None).
        self._lm_mov = landmarks_mov_scaled
        self._T_gt_ref = T_gt_ref
        # Re-bind the dispatched compute_metric so our override is used.
        self._base_compute_metric = self.compute_metric
        self.compute_metric = self._compute_metric_instrumented  # type: ignore[assignment]

    def set_level_hint(self, level: int, level_size: int) -> None:
        self._current_level = level
        self._current_level_size = level_size

    def _compute_metric_instrumented(self, ref_img, flt_img, t_mat, eref):
        value = self._base_compute_metric(ref_img, flt_img, t_mat, eref)

        if self._event_queue is None:
            return value

        # Derive pyramid level from the ref image size, relative to ``ref_size``.
        try:
            level_size = int(ref_img.shape[-1])
        except Exception:
            level_size = self._current_level_size

        if level_size != self._last_level_size and level_size > 0:
            # Emit a level boundary event so the UI can update the ladder.
            self._emit(LevelEvent(
                phase="start",
                level=self._infer_level(level_size),
                level_size=level_size,
                wall_time=time.time() - self._start_time,
            ))
            self._last_level_size = level_size

        try:
            t_np = t_mat.detach().cpu().numpy().copy() if hasattr(t_mat, "detach") \
                else np.asarray(t_mat).copy()
        except Exception:
            t_np = None

        try:
            val_f = float(value.detach().cpu().item()) if hasattr(value, "detach") \
                else float(value)
        except Exception:
            val_f = float("nan")

        rmse = None
        if t_np is not None and self._lm_mov is not None and self._T_gt_ref is not None:
            rmse = _compute_landmark_rmse(
                t_np, level_size, self.ref_size,
                self._lm_mov, self._T_gt_ref,
            )

        self._eval_idx += 1
        self._emit(MetricEvent(
            level=self._infer_level(level_size),
            level_size=level_size,
            eval_idx=self._eval_idx,
            value=val_f,
            transform=t_np,
            wall_time=time.time() - self._start_time,
            rmse_px=rmse,
        ))
        return value

    def _infer_level(self, level_size: int) -> int:
        """Infer pyramid level index (0 = finest) from the current ref size."""
        full = max(self.ref_size, level_size, 1)
        if level_size <= 0:
            return -1
        ratio = full / float(level_size)
        if ratio < 1.5:
            return 0
        if ratio < 3:
            return 1
        if ratio < 6:
            return 2
        return 3

    def _emit(self, event) -> None:
        try:
            self._event_queue.put_nowait(event)
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def load_ground_truth(mat_path: str | Path) -> dict | None:
    """Return a dict with keys ``fix_lm``, ``mov_lm``, ``T_gt``, ``fix_img``,
    ``mov_img`` loaded from a STAR-Bench ``.mat`` ground-truth file.

    Returns ``None`` (never raises) if the file is missing or malformed — the
    demo degrades gracefully to MI-only mode.
    """
    try:
        fix_img, mov_img, lm_fix, lm_mov, T_gt = _sb_load(str(mat_path))
    except Exception as exc:
        logger.warning("Could not read GT from %s: %s", mat_path, exc)
        return None
    return {
        "fix_img": fix_img,
        "mov_img": mov_img,
        "fix_lm": np.asarray(lm_fix, dtype=np.float32),
        "mov_lm": np.asarray(lm_mov, dtype=np.float32),
        "T_gt": np.asarray(T_gt, dtype=np.float32),
    }


def scale_landmarks_to_ref(landmarks: np.ndarray, orig_hw: tuple[int, int],
                           ref_size: int) -> np.ndarray:
    """Rescale landmarks from the original image grid to the ``ref_size`` grid."""
    orig_h, orig_w = orig_hw
    sx = ref_size / float(orig_w)
    sy = ref_size / float(orig_h)
    out = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2).copy()
    out[:, 0] *= sx
    out[:, 1] *= sy
    return out


def rescale_affine_to_ref(T: np.ndarray, orig_hw: tuple[int, int],
                          ref_size: int) -> np.ndarray:
    """Rescale a 2x3 / 3x3 affine matrix from the original image grid to the
    ``ref_size`` grid (which is what our resized landmarks and transforms live
    in).

    If ``S = diag(sx, sy)`` with ``sx = ref_size/orig_w``, ``sy = ref_size/orig_h``,
    a point ``p`` in original space becomes ``S @ p`` in ref space, so an
    affine ``x -> A x + t`` in original space becomes
    ``x' -> (S A S^-1) x' + S t`` in ref space.  For isotropic resize
    (sx == sy) this collapses to scaling only the translation column, which
    is what you usually want when both input images are square to begin with.
    """
    T = np.asarray(T, dtype=np.float32)
    if T.shape == (3, 3):
        T2 = T[:2, :]
    else:
        T2 = T.astype(np.float32).reshape(2, 3)
    orig_h, orig_w = orig_hw
    sx = ref_size / float(orig_w)
    sy = ref_size / float(orig_h)
    S = np.array([[sx, 0.0], [0.0, sy]], dtype=np.float32)
    S_inv = np.array([[1.0 / sx, 0.0], [0.0, 1.0 / sy]], dtype=np.float32)
    A = T2[:, :2]
    t = T2[:, 2]
    A_ref = S @ A @ S_inv
    t_ref = S @ t
    out = np.zeros((2, 3), dtype=np.float32)
    out[:, :2] = A_ref
    out[:, 2] = t_ref
    return out


def _apply_affine_2x3(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine (or 3x3 homogeneous) matrix to (x, y) points."""
    H = np.asarray(H, dtype=np.float32)
    if H.shape == (3, 3):
        H = H[:2, :]
    pts = np.asarray(points_xy, dtype=np.float32).reshape(-1, 2)
    homog = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float32)])
    return (H @ homog.T).T


def _upscale_level_transform(H_level: np.ndarray, level_size: int,
                             ref_size: int) -> np.ndarray | None:
    """ETNA keeps rotation / scale level-agnostic but expresses translation in
    the current pyramid level's pixel space. Upscale ``tx / ty`` so ``H`` can
    be compared against transforms expressed on the full ``ref_size`` grid.
    """
    if level_size <= 0 or ref_size <= 0:
        return None
    scale_up = float(ref_size) / float(level_size)
    H = np.asarray(H_level, dtype=np.float32).copy()
    if H.shape == (3, 3):
        H = H[:2, :]
    if H.shape != (2, 3):
        return None
    H[0, 2] *= scale_up
    H[1, 2] *= scale_up
    return H


def _compute_landmark_rmse(H_level: np.ndarray, level_size: int, ref_size: int,
                           lm_mov: np.ndarray, T_gt_ref: np.ndarray) -> float:
    """Landmark-space RMSE in ``ref_size`` pixels, following the STAR-Bench
    convention used by ``starbench_registration._calculate_rmse``:

        rmse = || H_est @ lm_mov  -  T_gt @ lm_mov ||

    This probes how far the current estimate is from ground truth using
    ``lm_mov`` as a fixed set of sample points, independent of whether the
    .mat file stores ``lm_mov`` and ``lm_fix`` in the same coordinate frame
    (which happens when the moving image was produced by warping the fixed
    one and the landmark table was not re-projected).
    """
    H = _upscale_level_transform(H_level, level_size, ref_size)
    if H is None:
        return float("nan")
    try:
        est = _apply_affine_2x3(lm_mov, H)
        gt = _apply_affine_2x3(lm_mov, T_gt_ref)
        diff = est - gt
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# FPGA detection
# ---------------------------------------------------------------------------

def detect_fpga_status(want_fpga: bool) -> tuple[bool, str]:
    """Return (fpga_actually_active, human_readable_status)."""
    if not want_fpga:
        return False, "CPU (PyTorch / Kornia)"
    try:
        from ETNA import FaberFPGAAccelerator
    except Exception as exc:
        return False, f"FPGA module import failed: {exc}"
    try:
        from ETNA.fpga_accelerator import PYNQ_AVAILABLE
    except Exception:
        PYNQ_AVAILABLE = False
    if not PYNQ_AVAILABLE:
        return False, "FPGA requested, PYNQ unavailable -> software fallback"
    try:
        accel = FaberFPGAAccelerator()
        if getattr(accel, "enabled", False):
            return True, "FPGA active: wax_mi_accel overlay loaded"
        return False, "FPGA requested, overlay failed -> software fallback"
    except Exception as exc:
        return False, f"FPGA init error: {exc} -> software fallback"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def _prepare_tensors(fixed_np: np.ndarray, moving_np: np.ndarray, ref_size: int
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    import cv2
    if fixed_np.shape[0] != ref_size or fixed_np.shape[1] != ref_size:
        fixed_np = cv2.resize(fixed_np, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    if moving_np.shape[0] != ref_size or moving_np.shape[1] != ref_size:
        moving_np = cv2.resize(moving_np, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
    ref_t = torch.tensor(fixed_np.astype(np.int16), dtype=torch.int16)
    flt_t = torch.tensor(moving_np.astype(np.int16), dtype=torch.int16)
    return ref_t, flt_t


def run_etna(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    *,
    device: str = "cpu",
    metric: str = "mi",
    optimizer: str = "powell",
    ref_size: int = 256,
    event_queue: queue.Queue | None = None,
    gt_mat_path: str | Path | None = None,
) -> RunResult:
    """Run ETNA end-to-end on a pair of numpy grayscale uint8 images.

    When ``gt_mat_path`` points to a STAR-Bench ``.mat`` ground-truth file the
    runner additionally computes the landmark-space RMSE (in ``ref_size``
    pixels) at every metric evaluation and emits it alongside the MI value.

    Emits live events on ``event_queue`` (if provided) and returns a
    ``RunResult`` summarising the final transform and timings.
    """
    want_fpga = (device == "fpga")
    start = time.time()

    fpga_active, fpga_status = detect_fpga_status(want_fpga)
    backend = "FPGA (wax_mi_accel)" if fpga_active else \
              ("CPU fallback" if want_fpga else "CPU")

    if event_queue is not None:
        event_queue.put(StatusEvent(
            message=f"Backend: {backend} — {fpga_status}",
            severity="info" if fpga_active or not want_fpga else "warning",
            payload={"backend": backend, "fpga_active": fpga_active},
            wall_time=0.0,
        ))

    # Optional ground-truth landmarks, rescaled to ref_size coordinates.
    gt = load_ground_truth(gt_mat_path) if gt_mat_path else None
    lm_fix_scaled = lm_mov_scaled = None
    T_gt = None
    T_gt_ref = None
    initial_rmse = None
    if gt is not None:
        orig_hw = fixed_img.shape[:2]
        lm_fix_scaled = scale_landmarks_to_ref(gt["fix_lm"], orig_hw, ref_size)
        lm_mov_scaled = scale_landmarks_to_ref(gt["mov_lm"], orig_hw, ref_size)
        T_gt = gt["T_gt"]
        T_gt_ref = rescale_affine_to_ref(T_gt, orig_hw, ref_size)
        # Baseline RMSE uses the STAR-Bench convention
        # (|| H_est @ lm_mov - T_gt @ lm_mov ||), with H_est = identity.
        initial_rmse = _compute_landmark_rmse(
            np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
            ref_size, ref_size, lm_mov_scaled, T_gt_ref,
        )
        if event_queue is not None:
            event_queue.put(StatusEvent(
                message=(f"Ground truth loaded: {len(lm_mov_scaled)} landmarks, "
                         f"initial RMSE = {initial_rmse:.2f} px"),
                severity="info",
                payload={"landmarks": int(len(lm_mov_scaled)),
                         "initial_rmse_px": initial_rmse},
                wall_time=0.0,
            ))

    ref_t, flt_t = _prepare_tensors(fixed_img, moving_img, ref_size)

    metric_component = InstrumentedMetric(
        ref_size=ref_size, metric=metric, exponential=True,
        use_fpga=want_fpga,
        event_queue=event_queue, start_time=start,
        landmarks_mov_scaled=lm_mov_scaled,
        T_gt_ref=T_gt_ref,
    )

    opt_cls = EtnaMultiPowell if optimizer == "powell" else EtnaMultiOnePlusOne
    opt = opt_cls()

    try:
        H = opt.compute(ref_t, flt_t, metric_component, ref_size, use_pyramid=True)
    except Exception as exc:
        if event_queue is not None:
            event_queue.put(StatusEvent(
                message=f"ETNA failed: {exc}",
                severity="error",
                wall_time=time.time() - start,
            ))
        raise

    total_time = time.time() - start

    # Apply final transform to moving image for display.
    import cv2
    warped = cv2.warpAffine(
        moving_img if moving_img.shape[0] == ref_size else
        cv2.resize(moving_img, (ref_size, ref_size), interpolation=cv2.INTER_AREA),
        H.astype(np.float32), (ref_size, ref_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # Final landmark RMSE (H is already in full ref_size pixel space).
    final_rmse = None
    if T_gt_ref is not None and lm_mov_scaled is not None:
        final_rmse = _compute_landmark_rmse(
            np.asarray(H, dtype=np.float32), ref_size, ref_size,
            lm_mov_scaled, T_gt_ref,
        )

    result = RunResult(
        transform=np.asarray(H, dtype=np.float32),
        fixed=cv2.resize(fixed_img, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
            if fixed_img.shape[0] != ref_size else fixed_img,
        moving=cv2.resize(moving_img, (ref_size, ref_size), interpolation=cv2.INTER_AREA)
            if moving_img.shape[0] != ref_size else moving_img,
        warped=warped,
        per_level_time={},  # populated by the UI from the event stream
        total_time=total_time,
        backend=backend,
        eval_count=metric_component._eval_idx,
        landmarks_fix_scaled=lm_fix_scaled,
        landmarks_mov_scaled=lm_mov_scaled,
        gt_transform=T_gt,
        gt_transform_ref=T_gt_ref,
        initial_rmse_px=initial_rmse,
        final_rmse_px=final_rmse,
    )

    if event_queue is not None:
        event_queue.put(StatusEvent(
            kind="status",
            message="Done",
            severity="done",
            payload={
                "total_time": total_time,
                "eval_count": metric_component._eval_idx,
                "transform": H.tolist() if hasattr(H, "tolist") else list(H),
                "initial_rmse_px": initial_rmse,
                "final_rmse_px": final_rmse,
            },
            wall_time=total_time,
        ))

    return result


# ---------------------------------------------------------------------------
# Threaded wrapper for the Streamlit main loop
# ---------------------------------------------------------------------------

def run_etna_async(fixed_img, moving_img, *, device, metric, optimizer, ref_size,
                   event_queue, gt_mat_path: str | Path | None = None
                   ) -> tuple[threading.Thread, dict]:
    """Launch ``run_etna`` on a background thread.

    The returned ``result_slot`` dict is populated with key ``"result"`` (on
    success) or ``"error"`` (on failure) once the worker finishes.
    """
    result_slot: dict = {}

    def _worker():
        try:
            result_slot["result"] = run_etna(
                fixed_img, moving_img,
                device=device, metric=metric, optimizer=optimizer,
                ref_size=ref_size, event_queue=event_queue,
                gt_mat_path=gt_mat_path,
            )
        except Exception as exc:
            logger.exception("ETNA worker crashed")
            result_slot["error"] = exc

    t = threading.Thread(target=_worker, name="etna-runner", daemon=True)
    t.start()
    return t, result_slot


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import cv2

    p = argparse.ArgumentParser(description="ETNA runner smoke test")
    p.add_argument("fixed")
    p.add_argument("moving")
    p.add_argument("--device", choices=["cpu", "fpga"], default="cpu")
    p.add_argument("--metric", choices=["mi", "mse", "cc"], default="mi")
    p.add_argument("--optimizer", choices=["powell", "oneplusone"], default="powell")
    p.add_argument("--ref-size", type=int, default=256)
    p.add_argument("--gt", help="path to a STAR-Bench .mat ground-truth file")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    fixed = cv2.imread(args.fixed, cv2.IMREAD_GRAYSCALE)
    moving = cv2.imread(args.moving, cv2.IMREAD_GRAYSCALE)
    if fixed is None or moving is None:
        raise SystemExit("Could not read inputs")

    q: queue.Queue = queue.Queue()
    res = run_etna(fixed, moving, device=args.device, metric=args.metric,
                   optimizer=args.optimizer, ref_size=args.ref_size,
                   event_queue=q, gt_mat_path=args.gt)
    print(f"done in {res.total_time:.3f}s, {res.eval_count} metric evals, backend={res.backend}")
    if res.initial_rmse_px is not None:
        print(f"landmark RMSE: initial = {res.initial_rmse_px:.3f} px, "
              f"final = {res.final_rmse_px:.3f} px")
    print("transform =\n", res.transform)
    print("queue events =", q.qsize())
