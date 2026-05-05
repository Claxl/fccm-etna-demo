#!/usr/bin/env python3
"""Standalone ETNA event replay viewer.

Works completely offline — no Streamlit, no network, no SSH needed.
Reads a JSONL recording produced by running ETNA with "Record run to file"
checked in the sidebar (or with ``--record`` on the CLI).

Usage
-----
    python demo_replay.py runs/CS01.jsonl
    python demo_replay.py runs/CS01.jsonl --speed 2.0
    python demo_replay.py runs/CS01.jsonl --no-gui    # text-only (no display)

Requirements
------------
    matplotlib   (pip install matplotlib)   — for GUI mode
    numpy        (already a project dependency)
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
if str(_THIS) not in sys.path:
    sys.path.insert(0, str(_THIS))

from etna_runner import run_replay_async  # noqa: E402

PALETTE = [
    "#e74c3c", "#f39c12", "#3498db", "#27ae60",
    "#9b59b6", "#1abc9c", "#e67e22", "#2ecc71",
]


def _text_replay(jsonl_path: str, speed: float) -> None:
    """Terminal-only replay — no display required."""
    path = Path(jsonl_path)
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("Recording is empty.")
        return

    total = len(rows)
    print(f"Replaying {path.name}  ({total} events, {speed}× speed)\n")
    t0 = time.time()
    for i, row in enumerate(rows):
        target = row.get("t", 0.0) / max(speed, 1e-6)
        wait = target - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)

        kind = row.get("kind", "")
        if kind == "level":
            print(f"  [{row.get('phase','?').upper():5s}] Level {row.get('level')} "
                  f"@ {row.get('level_size')}px")
        elif kind == "metric":
            rmse = row.get("rmse_px")
            rmse_s = f"  RMSE={rmse:.3f}px" if rmse is not None else ""
            print(f"  eval {row.get('eval_idx'):4d}  L{row.get('level')}  "
                  f"MI={row.get('value', 0):.5f}{rmse_s}")
        elif kind == "status":
            sev = row.get("severity", "info").upper()
            msg = row.get("message", "")
            if msg and not row.get("payload", {}).get("power_w"):
                print(f"  [{sev}] {msg}")

    print("\nReplay complete.")


def _gui_replay(jsonl_path: str, speed: float, num_levels: int) -> None:
    """Matplotlib-based live replay."""
    try:
        import matplotlib
        import os as _os
        # Prefer interactive backends; fall back gracefully.
        if _os.environ.get("DISPLAY") or sys.platform in ("darwin", "win32"):
            for _be in ("TkAgg", "Qt5Agg", "Qt6Agg", "WXAgg", "MacOSX"):
                try:
                    matplotlib.use(_be)
                    break
                except Exception:
                    continue
        else:
            matplotlib.use("Agg")   # headless — will save a PNG instead
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not found — falling back to text mode (pip install matplotlib)")
        _text_replay(jsonl_path, speed)
        return

    _q: queue.Queue = queue.Queue()
    t, agg, _slot = run_replay_async(
        jsonl_path,
        num_pyramid_levels=num_levels,
        speed=speed,
        event_queue=_q,
    )

    headless = matplotlib.get_backend().lower() == "agg"

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    ax_mi   = fig.add_subplot(gs[0, :2])
    ax_rmse = fig.add_subplot(gs[1, :2])
    ax_info = fig.add_subplot(gs[:, 2])
    ax_info.axis("off")

    ax_mi.set_title("Similarity metric (MI)")
    ax_mi.set_xlabel("eval #")
    ax_mi.set_ylabel("MI")
    ax_rmse.set_title("Landmark RMSE (px)")
    ax_rmse.set_xlabel("eval #")
    ax_rmse.set_ylabel("px")

    info_text = ax_info.text(
        0.05, 0.97, "Starting…",
        transform=ax_info.transAxes,
        va="top", fontfamily="monospace", fontsize=9,
        color="white" if matplotlib.rcParams.get("figure.facecolor") != "white" else "black",
    )
    fig.suptitle(f"ETNA Replay — {Path(jsonl_path).name}", fontsize=11)

    if not headless:
        plt.ion()
        plt.show(block=False)

    last_version = -1
    while True:
        snap = agg.get_snapshot()
        if snap.version != last_version:
            last_version = snap.version

            ax_mi.cla()
            ax_mi.set_title("Similarity metric (MI)")
            ax_mi.set_xlabel("eval #")
            ax_mi.set_ylabel("MI")
            for lvl, series in snap.series.items():
                if series:
                    xs, ys = zip(*series)
                    ax_mi.plot(xs, ys, color=PALETTE[lvl % len(PALETTE)],
                               label=f"L{lvl}", linewidth=1.5)
            ax_mi.legend(loc="upper right", fontsize=7)

            ax_rmse.cla()
            ax_rmse.set_title("Landmark RMSE (px)")
            ax_rmse.set_xlabel("eval #")
            ax_rmse.set_ylabel("px")
            any_rmse = False
            for lvl, series in snap.rmse_series.items():
                if series:
                    any_rmse = True
                    xs, ys = zip(*series)
                    ax_rmse.plot(xs, ys, color=PALETTE[lvl % len(PALETTE)],
                                 label=f"L{lvl}", linewidth=1.5)
            if snap.initial_rmse is not None:
                ax_rmse.axhline(snap.initial_rmse, color="#888", linestyle="--",
                                label=f"initial={snap.initial_rmse:.1f}px")
            if any_rmse or snap.initial_rmse:
                ax_rmse.legend(loc="upper right", fontsize=7)
            else:
                ax_rmse.text(0.5, 0.5, "no ground truth in recording",
                             ha="center", va="center",
                             transform=ax_rmse.transAxes, color="#888")

            rmse_s = (f"{snap.best_rmse:.3f} px"
                      if snap.best_rmse != float("inf") else "—")
            lines = [
                f"Backend : {snap.backend or '?'}",
                f"Level   : L{snap.active_level}",
                f"Evals   : {snap.total_evals}",
                f"ms/step : " + (f"{snap.last_step_ms:.1f}" if snap.last_step_ms else "—"),
                f"RMSE    : {rmse_s}",
            ]
            if snap.status_messages:
                for ev in snap.status_messages[-3:]:
                    if ev.message and not ev.payload.get("power_w"):
                        lines.append(f"\n[{ev.severity.upper()}]")
                        lines.append(f"  {ev.message[:55]}")
            info_text.set_text("\n".join(lines))

            if not headless:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

        if snap.done:
            break
        time.sleep(0.04)

    t.join(timeout=2.0)

    # Final summary
    snap = agg.get_snapshot()
    rmse_s = (f"{snap.best_rmse:.3f} px"
              if snap.best_rmse != float("inf") else "—")
    summary = [
        "═══ REPLAY COMPLETE ═══",
        f"Backend : {snap.backend or '?'}",
        f"Evals   : {snap.total_evals}",
        f"Best RMSE: {rmse_s}",
        "",
        "Close window to exit.",
    ]
    info_text.set_text("\n".join(summary))
    if not headless:
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.ioff()

    if headless:
        out = Path(jsonl_path).with_suffix(".png")
        plt.savefig(out, dpi=120, bbox_inches="tight")
        print(f"Saved replay chart to {out}")
    else:
        print("Replay complete. Close the window to exit.")
        plt.show(block=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Standalone ETNA event replay viewer (no Streamlit / network needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("recording", help="Path to a .jsonl file recorded by ETNA")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Replay speed multiplier (default 1.0 = real-time, 0 = instant)")
    p.add_argument("--levels", type=int, default=4,
                   help="Number of pyramid levels used in the recording")
    p.add_argument("--no-gui", action="store_true",
                   help="Text-only output (no matplotlib window required)")
    args = p.parse_args()

    path = Path(args.recording)
    if not path.exists():
        sys.exit(f"Error: recording not found: {path}")

    if args.no_gui:
        _text_replay(str(path), args.speed)
    else:
        _gui_replay(str(path), args.speed, args.levels)


if __name__ == "__main__":
    main()
