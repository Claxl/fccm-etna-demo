#!/usr/bin/env python3
"""Run ETNA on every image pair and record events to JSONL files.

Produces one ``runs/<pair>_<device>.jsonl`` per pair × device combination.
These files can be replayed offline (no network, no Streamlit server) with:

    python demo_replay.py runs/<pair>_fpga.jsonl
    python demo_replay.py runs/<pair>_cpu.jsonl --no-gui

Or from the Streamlit sidebar "Replay a recording" section.

Usage
-----
    # Record CPU runs on all pairs
    python scripts/batch_record.py --device cpu

    # Record both CPU and FPGA
    python scripts/batch_record.py --device cpu fpga

    # Only specific pairs
    python scripts/batch_record.py --pair CS01 CS02 --device fpga

    # Then replay the best one offline
    python demo_replay.py runs/CS01_fpga.jsonl
"""
from __future__ import annotations

import argparse
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from etna_runner import run_etna_async  # noqa: E402

_IMAGES = _REPO / "images"
_RUNS   = _REPO / "runs"
_RUNS.mkdir(exist_ok=True)


def _scan_pairs() -> dict[str, dict]:
    """Return {name: {fixed, moving, gt}} from images/."""
    pairs: dict[str, dict] = {}
    img_exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    for fixed in sorted(_IMAGES.glob("**/*_fixed.*")):
        if fixed.suffix.lower() not in img_exts:
            continue
        stem = fixed.stem[: -len("_fixed")]
        rel_dir = fixed.parent.relative_to(_IMAGES)
        pair_key = str(rel_dir / stem) if str(rel_dir) != "." else stem
        for ext in img_exts:
            moving = fixed.parent / f"{stem}_moving{ext}"
            if moving.exists():
                gt = fixed.parent / f"{stem}.mat"
                pairs[pair_key] = {"fixed": fixed, "moving": moving,
                                   "gt": gt if gt.exists() else None}
                break

    sb_tags = ("CS", "DN", "DO", "IO", "MO", "OO", "SO")
    for mat in sorted(_IMAGES.glob("**/*.mat")):
        stem = mat.stem
        if len(stem) < 3 or stem[:2].upper() not in sb_tags:
            continue
        rel_dir = mat.parent.relative_to(_IMAGES)
        pair_key = str(rel_dir / stem) if str(rel_dir) != "." else stem
        if pair_key in pairs:
            continue
        for ext in img_exts:
            cand_m = mat.parent / f"{stem}a{ext}"
            cand_f = mat.parent / f"{stem}b{ext}"
            if cand_m.exists() and cand_f.exists():
                pairs[pair_key] = {"fixed": cand_f, "moving": cand_m, "gt": mat}
                break

    return pairs


def record_pair(name: str, info: dict, device: str, ref_size: int,
                num_levels: int, optimizer: str, metric: str) -> bool:
    fixed_img  = cv2.imread(str(info["fixed"]),  cv2.IMREAD_GRAYSCALE)
    moving_img = cv2.imread(str(info["moving"]), cv2.IMREAD_GRAYSCALE)
    if fixed_img is None or moving_img is None:
        print(f"  ERROR: cannot read images for {name}")
        return False

    out_path = _RUNS / f"{name}_{device}.jsonl"
    print(f"  Recording {name} [{device.upper()}] → {out_path.name} …", flush=True)

    event_q: queue.Queue = queue.Queue()
    t0 = time.time()
    thread, agg, slot = run_etna_async(
        fixed_img, moving_img,
        device=device,
        metric=metric,
        optimizer=optimizer,
        ref_size=ref_size,
        event_queue=event_q,
        gt_mat_path=str(info["gt"]) if info.get("gt") else None,
        num_pyramid_levels=num_levels,
        record_path=str(out_path),
    )
    thread.join()          # wait for ETNA to finish
    agg.stop()
    elapsed = time.time() - t0

    snap = agg.get_snapshot()
    if "error" in slot:
        print(f"  FAILED: {slot['error']}")
        return False

    res = slot.get("result")
    if res is not None:
        rmse_s = (f"  final_rmse={res.final_rmse_px:.3f}px"
                  if res.final_rmse_px is not None else "")
        print(f"  Done  {elapsed:.1f}s  {res.eval_count} evals  "
              f"backend={res.backend}{rmse_s}")
    else:
        print(f"  Done  {elapsed:.1f}s  {snap.total_evals} events replayed")

    return True


def main() -> None:
    all_pairs = _scan_pairs()

    p = argparse.ArgumentParser(
        description="Batch-record ETNA runs to JSONL for offline replay.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pair",   nargs="*", metavar="NAME",
                   default=list(all_pairs.keys()),
                   help="Which pairs to record (default: all found in images/)")
    p.add_argument("--device", nargs="+", choices=["cpu", "fpga"],
                   default=["cpu"],
                   help="Device(s) to record (default: cpu)")
    p.add_argument("--ref-size",  type=int, default=256)
    p.add_argument("--levels",    type=int, default=4)
    p.add_argument("--optimizer", choices=["powell", "oneplusone"], default="powell")
    p.add_argument("--metric",    choices=["mi", "mse", "cc"], default="mi")
    args = p.parse_args()

    if not all_pairs:
        sys.exit("No image pairs found in images/.  "
                 "Run scripts/gen_test_pair.py first.")

    selected = {n: all_pairs[n] for n in args.pair if n in all_pairs}
    if not selected:
        sys.exit(f"None of the requested pairs found.  Available: {list(all_pairs)}")

    print(f"Recording {len(selected)} pair(s) × {len(args.device)} device(s)\n")

    ok = failed = 0
    for name, info in selected.items():
        for device in args.device:
            success = record_pair(name, info, device,
                                  ref_size=args.ref_size,
                                  num_levels=args.levels,
                                  optimizer=args.optimizer,
                                  metric=args.metric)
            if success:
                ok += 1
            else:
                failed += 1
        print()

    print(f"Finished: {ok} OK, {failed} failed.")
    print(f"JSONL files are in {_RUNS.relative_to(_REPO)}/")
    print("\nReplay offline:")
    for name in selected:
        for device in args.device:
            p2 = _RUNS / f"{name}_{device}.jsonl"
            if p2.exists():
                print(f"  python demo_replay.py {p2.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
